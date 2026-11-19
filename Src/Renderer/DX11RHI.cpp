#include "Renderer/DX11RHI.h"
#include "Renderer/DX11Shading.h"
#include "Renderer/DX11MicropolygonDicing.h"
#include "Renderer/RealtimeShaderContract.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max
#include <d3d11.h>
#include <wrl/client.h>
#include "DX11ShaderBytecode.h"
#endif

namespace Arvo::Renderer
{
#if defined(_WIN32)
	using Microsoft::WRL::ComPtr;

	namespace
	{

		std::string HResultMessage(const char* operation, HRESULT result)
		{
			std::ostringstream stream;
			stream << operation << " failed with HRESULT 0x"
				<< std::hex << std::uppercase << static_cast<unsigned long>(result);
			return stream.str();
		}

		float LinearToSRGBChannel(float value)
		{
			value = std::max(value, 0.0f);
			return value <= 0.0031308f
				? value * 12.92f
				: 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
		}

		float SRGBToLinearChannel(float value)
		{
			value = std::clamp(value, 0.0f, 1.0f);
			return value <= 0.04045f
				? value / 12.92f
				: std::pow((value + 0.055f) / 1.055f, 2.4f);
		}

		Matrix4 BuildNormalMatrix(const Matrix4& world)
		{
			const float a00 = world.Values[0];
			const float a01 = world.Values[1];
			const float a02 = world.Values[2];
			const float a10 = world.Values[4];
			const float a11 = world.Values[5];
			const float a12 = world.Values[6];
			const float a20 = world.Values[8];
			const float a21 = world.Values[9];
			const float a22 = world.Values[10];
			const float determinant =
				a00 * (a11 * a22 - a12 * a21) -
				a01 * (a10 * a22 - a12 * a20) +
				a02 * (a10 * a21 - a11 * a20);
			if (!std::isfinite(determinant) || std::fabs(determinant) <= 1.0e-12f)
			{
				return Matrix4::Identity();
			}
			const float inverseDeterminant = 1.0f / determinant;
			Matrix4 result;
			result.Values = {
				(a11 * a22 - a12 * a21) * inverseDeterminant,
				(a12 * a20 - a10 * a22) * inverseDeterminant,
				(a10 * a21 - a11 * a20) * inverseDeterminant,
				0.0f,
				(a02 * a21 - a01 * a22) * inverseDeterminant,
				(a00 * a22 - a02 * a20) * inverseDeterminant,
				(a01 * a20 - a00 * a21) * inverseDeterminant,
				0.0f,
				(a01 * a12 - a02 * a11) * inverseDeterminant,
				(a02 * a10 - a00 * a12) * inverseDeterminant,
				(a00 * a11 - a01 * a10) * inverseDeterminant,
				0.0f,
				0.0f, 0.0f, 0.0f, 1.0f };
			return result;
		}

		struct GPULight
		{
			Float4 PositionType;
			Float4 DirectionIntensity;
			Float4 ColorCastShadows;
			Float4 AreaSizePadding;
		};

		struct UInt4
		{
			std::uint32_t X = 0;
			std::uint32_t Y = 0;
			std::uint32_t Z = 0;
			std::uint32_t W = 0;
		};

		struct DrawConstants
		{
			Matrix4 World;
			Matrix4 NormalWorld;
			Matrix4 ViewProjection;
			Matrix4 LightViewProjection;
			Float4 PipelineColor;
			Float4 MaterialDiffuseColor;
			Float4 MaterialAmbientColor;
			Float4 MaterialSpecularColor;
			Float4 MaterialEmissiveColor;
			Float4 MaterialParameters;
			Float4 AmbientColorStrength;
			Float4 CameraLightCount;
			Float4 RenderParameters;
			UInt4 TextureFlags;
			std::array<GPULight, MaximumSceneLights> Lights;
		};

		static_assert(sizeof(Vertex) == 48, "The D3D11 input layout must match Renderer::Vertex");
		static_assert(sizeof(GPULight) == 64, "The D3D11 light constants must match HLSL");
		static_assert(offsetof(DrawConstants, NormalWorld) == 64
			&& offsetof(DrawConstants, ViewProjection) == 128
			&& offsetof(DrawConstants, LightViewProjection) == 192
			&& offsetof(DrawConstants, PipelineColor) == 256
			&& offsetof(DrawConstants, MaterialDiffuseColor) == 272
			&& offsetof(DrawConstants, MaterialParameters) == 336
			&& offsetof(DrawConstants, TextureFlags) == 400
			&& offsetof(DrawConstants, Lights) == 416,
			"The D3D11 draw constants must match HLSL packing");
		static_assert(sizeof(DrawConstants) == 672,
			"The D3D11 constant buffer size must match HLSL");
	}

	struct DX11RHI::Impl
	{
		struct BufferResource
		{
			BufferDesc Description;
			std::vector<std::uint8_t> Bytes;
			ComPtr<ID3D11Buffer> Buffer;
		};

		struct TextureResource
		{
			TextureDesc Description;
			ComPtr<ID3D11Texture2D> Texture;
			ComPtr<ID3D11RenderTargetView> RenderTargetView;
			ComPtr<ID3D11DepthStencilView> DepthStencilView;
			ComPtr<ID3D11ShaderResourceView> ShaderResourceView;
			ComPtr<ID3D11ShaderResourceView> SRGBShaderResourceView;
			ComPtr<ID3D11Texture2D> StagingTexture;
		};

		bool Fail(const std::string& error)
		{
			LastError = error;
			return false;
		}

		bool CheckThread(const char* operation)
		{
			if (!Initialized) return Fail("DX11 RHI is not initialized");
			if (OwnerThread != std::this_thread::get_id())
			{
				return Fail(std::string(operation) + " must execute on the owning DX11 RHI thread");
			}
			return true;
		}

		bool CreateDepthTexture(int width, int height, bool shaderResource,
			ComPtr<ID3D11Texture2D>& texture, ComPtr<ID3D11DepthStencilView>& depthView,
			ComPtr<ID3D11ShaderResourceView>& shaderView, ComPtr<ID3D11Texture2D>& staging)
		{
			D3D11_TEXTURE2D_DESC desc = {};
			desc.Width = static_cast<UINT>(width);
			desc.Height = static_cast<UINT>(height);
			desc.MipLevels = 1;
			desc.ArraySize = 1;
			desc.Format = DXGI_FORMAT_R32_TYPELESS;
			desc.SampleDesc.Count = 1;
			desc.Usage = D3D11_USAGE_DEFAULT;
			desc.BindFlags = D3D11_BIND_DEPTH_STENCIL | (shaderResource ? D3D11_BIND_SHADER_RESOURCE : 0);
			HRESULT result = Device->CreateTexture2D(&desc, nullptr, texture.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateTexture2D(depth)", result));

			D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
			dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
			dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
			result = Device->CreateDepthStencilView(texture.Get(), &dsvDesc, depthView.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateDepthStencilView", result));
			if (shaderResource)
			{
				D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
				srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
				srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
				srvDesc.Texture2D.MipLevels = 1;
				result = Device->CreateShaderResourceView(texture.Get(), &srvDesc, shaderView.GetAddressOf());
				if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateShaderResourceView(depth)", result));
			}

			desc.Usage = D3D11_USAGE_STAGING;
			desc.BindFlags = 0;
			desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			result = Device->CreateTexture2D(&desc, nullptr, staging.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateTexture2D(depth staging)", result));
			return true;
		}

		bool CreateOffscreenTargets(int width, int height)
		{
			D3D11_TEXTURE2D_DESC colorDesc = {};
			colorDesc.Width = static_cast<UINT>(width);
			colorDesc.Height = static_cast<UINT>(height);
			colorDesc.MipLevels = 1;
			colorDesc.ArraySize = 1;
			colorDesc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
			colorDesc.SampleDesc.Count = 1;
			colorDesc.Usage = D3D11_USAGE_DEFAULT;
			colorDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
			ComPtr<ID3D11Texture2D> colorTexture;
			ComPtr<ID3D11RenderTargetView> colorTarget;
			ComPtr<ID3D11Texture2D> colorStaging;
			HRESULT result = Device->CreateTexture2D(&colorDesc, nullptr, colorTexture.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateTexture2D(color)", result));
			D3D11_RENDER_TARGET_VIEW_DESC colorTargetDesc = {};
			colorTargetDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			colorTargetDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
			result = Device->CreateRenderTargetView(
				colorTexture.Get(), &colorTargetDesc, colorTarget.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateRenderTargetView", result));
			colorDesc.Usage = D3D11_USAGE_STAGING;
			colorDesc.BindFlags = 0;
			colorDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			result = Device->CreateTexture2D(&colorDesc, nullptr, colorStaging.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateTexture2D(color staging)", result));

			colorDesc.Usage = D3D11_USAGE_DEFAULT;
			colorDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
			colorDesc.CPUAccessFlags = 0;
			colorDesc.Format = DXGI_FORMAT_R16G16B16A16_UNORM;
			ComPtr<ID3D11Texture2D> outlineMaskTexture;
			ComPtr<ID3D11RenderTargetView> outlineMaskTarget;
			ComPtr<ID3D11Texture2D> outlineMaskStaging;
			result = Device->CreateTexture2D(&colorDesc, nullptr, outlineMaskTexture.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateTexture2D(outline mask)", result));
			result = Device->CreateRenderTargetView(
				outlineMaskTexture.Get(), nullptr, outlineMaskTarget.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage(
				"ID3D11Device::CreateRenderTargetView(outline mask)", result));
			colorDesc.Usage = D3D11_USAGE_STAGING;
			colorDesc.BindFlags = 0;
			colorDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			result = Device->CreateTexture2D(
				&colorDesc, nullptr, outlineMaskStaging.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage(
				"ID3D11Device::CreateTexture2D(outline mask staging)", result));

			ComPtr<ID3D11Texture2D> depthTexture;
			ComPtr<ID3D11DepthStencilView> depthTarget;
			ComPtr<ID3D11Texture2D> depthStaging;
			ComPtr<ID3D11ShaderResourceView> unusedShaderView;
			if (!CreateDepthTexture(width, height, false, depthTexture, depthTarget,
				unusedShaderView, depthStaging)) return false;

			ColorTexture = std::move(colorTexture);
			ColorTarget = std::move(colorTarget);
			ColorStaging = std::move(colorStaging);
			OutlineMaskTexture = std::move(outlineMaskTexture);
			OutlineMaskTarget = std::move(outlineMaskTarget);
			OutlineMaskStaging = std::move(outlineMaskStaging);
			DepthTexture = std::move(depthTexture);
			DepthTarget = std::move(depthTarget);
			DepthStaging = std::move(depthStaging);
			return true;
		}

		bool CreatePipelineObjects()
		{
			// Build-time bytecode retains the full DX11 shading model and avoids
			// repeating FXC optimization when the editor creates a preview device.
			HRESULT result = Device->CreateVertexShader(DX11Shaders::Vertex,
				std::size(DX11Shaders::Vertex), nullptr, VertexShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateVertexShader", result));
			result = Device->CreatePixelShader(DX11Shaders::Fragment,
				std::size(DX11Shaders::Fragment), nullptr, PixelShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreatePixelShader", result));
			result = Device->CreatePixelShader(DX11Shaders::Shadow,
				std::size(DX11Shaders::Shadow), nullptr, ShadowPixelShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreatePixelShader(shadow)", result));
			result = Device->CreatePixelShader(DX11Shaders::Outline,
				std::size(DX11Shaders::Outline), nullptr, OutlinePixelShader.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreatePixelShader(outline)", result));

			const D3D11_INPUT_ELEMENT_DESC inputElements[] = {
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 40, D3D11_INPUT_PER_VERTEX_DATA, 0 }
			};
			result = Device->CreateInputLayout(inputElements, 4, DX11Shaders::Vertex,
				std::size(DX11Shaders::Vertex), InputLayout.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateInputLayout", result));

			D3D11_BUFFER_DESC constantDesc = {};
			constantDesc.ByteWidth = sizeof(DrawConstants);
			constantDesc.Usage = D3D11_USAGE_DYNAMIC;
			constantDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			constantDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			result = Device->CreateBuffer(&constantDesc, nullptr, ConstantBuffer.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateBuffer(constants)", result));
			constantDesc.ByteWidth = sizeof(DX11PhysicalConstants);
			result = Device->CreateBuffer(&constantDesc, nullptr, PhysicalConstantBuffer.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateBuffer(physical constants)", result));
			constantDesc.ByteWidth = sizeof(DX11MediumConstants);
			result = Device->CreateBuffer(&constantDesc, nullptr, MediumConstantBuffer.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateBuffer(medium constants)", result));

			D3D11_RASTERIZER_DESC rasterDesc = {};
			rasterDesc.FillMode = D3D11_FILL_SOLID;
			rasterDesc.CullMode = D3D11_CULL_NONE;
			rasterDesc.FrontCounterClockwise = TRUE;
			rasterDesc.DepthClipEnable = TRUE;
			result = Device->CreateRasterizerState(&rasterDesc, SolidRasterState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateRasterizerState(solid)", result));
			D3D11_RASTERIZER_DESC shadowRasterDesc = rasterDesc;
			shadowRasterDesc.DepthBias = ShadowDepthBiasUnits;
			shadowRasterDesc.DepthBiasClamp = ShadowDepthBiasClamp;
			shadowRasterDesc.SlopeScaledDepthBias = ShadowSlopeScaledDepthBias;
			result = Device->CreateRasterizerState(
				&shadowRasterDesc, ShadowRasterState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage(
				"ID3D11Device::CreateRasterizerState(shadow)", result));
			rasterDesc.FillMode = D3D11_FILL_WIREFRAME;
			result = Device->CreateRasterizerState(&rasterDesc, WireRasterState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateRasterizerState(wire)", result));
			// The outline pass writes the visible selected surface into a mask.
			// Pixel-sized dilation is applied to that mask during presentation.
			rasterDesc.FillMode = D3D11_FILL_SOLID;
			rasterDesc.CullMode = D3D11_CULL_NONE;
			rasterDesc.FrontCounterClockwise = FALSE;
			rasterDesc.DepthBias = 0;
			rasterDesc.SlopeScaledDepthBias = 0.0f;
			result = Device->CreateRasterizerState(&rasterDesc, OutlineRasterState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateRasterizerState(outline)", result));

			D3D11_DEPTH_STENCIL_DESC depthDesc = {};
			depthDesc.DepthEnable = TRUE;
			depthDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
			depthDesc.DepthFunc = D3D11_COMPARISON_LESS;
			result = Device->CreateDepthStencilState(&depthDesc, DepthWriteState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateDepthStencilState(write)", result));
			depthDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
			depthDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
			result = Device->CreateDepthStencilState(&depthDesc, DepthReadState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateDepthStencilState(read)", result));
			depthDesc.DepthEnable = FALSE;
			result = Device->CreateDepthStencilState(&depthDesc, DepthDisabledState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateDepthStencilState(disabled)", result));

			D3D11_BLEND_DESC blendDesc = {};
			blendDesc.RenderTarget[0].RenderTargetWriteMask = 0;
			result = Device->CreateBlendState(&blendDesc, ColorWriteDisabledState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateBlendState", result));
			D3D11_BLEND_DESC alphaBlendDesc = {};
			D3D11_RENDER_TARGET_BLEND_DESC& alphaBlend =
				alphaBlendDesc.RenderTarget[0];
			alphaBlend.BlendEnable = TRUE;
			alphaBlend.SrcBlend = D3D11_BLEND_SRC_ALPHA;
			alphaBlend.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
			alphaBlend.BlendOp = D3D11_BLEND_OP_ADD;
			alphaBlend.SrcBlendAlpha = D3D11_BLEND_ONE;
			alphaBlend.DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
			alphaBlend.BlendOpAlpha = D3D11_BLEND_OP_ADD;
			alphaBlend.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
			result = Device->CreateBlendState(
				&alphaBlendDesc, AlphaBlendState.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage(
				"ID3D11Device::CreateBlendState(alpha)", result));

			D3D11_SAMPLER_DESC samplerDesc = {};
			samplerDesc.Filter = D3D11_FILTER_ANISOTROPIC;
			samplerDesc.MaxAnisotropy = 8;
			samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
			samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
			samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
			samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
			result = Device->CreateSamplerState(&samplerDesc, SurfaceSampler.GetAddressOf());
			if (FAILED(result)) return Fail(HResultMessage("ID3D11Device::CreateSamplerState(surface)", result));
			return true;
		}

		bool ReadColor(std::vector<std::uint8_t>& output)
		{
			Context->CopyResource(ColorStaging.Get(), ColorTexture.Get());
			D3D11_MAPPED_SUBRESOURCE mapped = {};
			const HRESULT result = Context->Map(ColorStaging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
			if (FAILED(result)) return Fail(HResultMessage("ID3D11DeviceContext::Map(color)", result));
			output.resize(static_cast<std::size_t>(CreateInfo.Width) * CreateInfo.Height * 4);
			const std::size_t rowSize = static_cast<std::size_t>(CreateInfo.Width) * 4;
			for (int row = 0; row < CreateInfo.Height; ++row)
			{
				std::memcpy(output.data() + static_cast<std::size_t>(row) * rowSize,
					static_cast<const std::uint8_t*>(mapped.pData) + static_cast<std::size_t>(row) * mapped.RowPitch,
					rowSize);
			}
			Context->Unmap(ColorStaging.Get(), 0);
			return true;
		}

		bool ReadOutlineMask(std::vector<std::uint16_t>& output)
		{
			Context->CopyResource(OutlineMaskStaging.Get(), OutlineMaskTexture.Get());
			D3D11_MAPPED_SUBRESOURCE mapped = {};
			const HRESULT result = Context->Map(
				OutlineMaskStaging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
			if (FAILED(result)) return Fail(HResultMessage(
				"ID3D11DeviceContext::Map(outline mask)", result));
			output.resize(static_cast<std::size_t>(CreateInfo.Width) * CreateInfo.Height * 4);
			const std::size_t rowSize =
				static_cast<std::size_t>(CreateInfo.Width) * 4 * sizeof(std::uint16_t);
			for (int row = 0; row < CreateInfo.Height; ++row)
			{
				std::memcpy(
					reinterpret_cast<std::uint8_t*>(output.data())
						+ static_cast<std::size_t>(row) * rowSize,
					static_cast<const std::uint8_t*>(mapped.pData)
						+ static_cast<std::size_t>(row) * mapped.RowPitch,
					rowSize);
			}
			Context->Unmap(OutlineMaskStaging.Get(), 0);
			return true;
		}

		void ApplyOutline(
			const std::vector<std::uint16_t>& mask,
			std::vector<std::uint8_t>& color) const
		{
			const int width = CreateInfo.Width;
			const int height = CreateInfo.Height;
			const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
			if (width <= 0 || height <= 0
				|| mask.size() != pixelCount * 4
				|| color.size() != pixelCount * 4)
			{
				return;
			}

			std::vector<std::uint32_t> bestDistance(
				pixelCount, std::numeric_limits<std::uint32_t>::max());
			std::vector<std::uint32_t> bestSource(
				pixelCount, std::numeric_limits<std::uint32_t>::max());
			const auto radiusAt = [&mask](std::size_t pixel)
			{
				return static_cast<int>(mask[pixel * 4 + 3] >> 8);
			};
			for (int sourceY = 0; sourceY < height; ++sourceY)
			{
				for (int sourceX = 0; sourceX < width; ++sourceX)
				{
					const std::size_t sourcePixel =
						static_cast<std::size_t>(sourceY) * width + sourceX;
					const int radius = radiusAt(sourcePixel);
					if (radius <= 0)
					{
						continue;
					}

					bool boundary = false;
					for (int neighborY = -1; neighborY <= 1 && !boundary; ++neighborY)
					{
						for (int neighborX = -1; neighborX <= 1; ++neighborX)
						{
							if (neighborX == 0 && neighborY == 0)
							{
								continue;
							}
							const int x = sourceX + neighborX;
							const int y = sourceY + neighborY;
							if (x < 0 || x >= width || y < 0 || y >= height
								|| radiusAt(static_cast<std::size_t>(y) * width + x) == 0)
							{
								boundary = true;
								break;
							}
						}
					}
					if (!boundary)
					{
						continue;
					}

					const int minimumX = std::max(0, sourceX - radius);
					const int maximumX = std::min(width - 1, sourceX + radius);
					const int minimumY = std::max(0, sourceY - radius);
					const int maximumY = std::min(height - 1, sourceY + radius);
					const std::uint32_t radiusSquared =
						static_cast<std::uint32_t>(radius * radius);
					for (int targetY = minimumY; targetY <= maximumY; ++targetY)
					{
						for (int targetX = minimumX; targetX <= maximumX; ++targetX)
						{
							const int deltaX = targetX - sourceX;
							const int deltaY = targetY - sourceY;
							const std::uint32_t distance =
								static_cast<std::uint32_t>(
									deltaX * deltaX + deltaY * deltaY);
							if (distance > radiusSquared)
							{
								continue;
							}
							const std::size_t targetPixel =
								static_cast<std::size_t>(targetY) * width + targetX;
							if (radiusAt(targetPixel) != 0
								|| distance >= bestDistance[targetPixel])
							{
								continue;
							}
							bestDistance[targetPixel] = distance;
							bestSource[targetPixel] =
								static_cast<std::uint32_t>(sourcePixel);
						}
					}
				}
			}

			for (std::size_t targetPixel = 0;
				targetPixel < pixelCount;
				++targetPixel)
			{
				if (bestSource[targetPixel] ==
					std::numeric_limits<std::uint32_t>::max())
				{
					continue;
				}
				const std::size_t sourcePixel = bestSource[targetPixel];
				const std::uint16_t packedOutline =
					mask[sourcePixel * 4 + 3];
				const float opacity =
					static_cast<float>(packedOutline & 0xffu) / 255.0f;
				if (opacity <= 0.0f)
				{
					continue;
				}
				for (std::size_t channel = 0; channel < 3; ++channel)
				{
					const float sourceSRGB =
						static_cast<float>(mask[sourcePixel * 4 + channel])
						/ 65535.0f;
					const float destinationSRGB =
						static_cast<float>(color[targetPixel * 4 + channel])
						/ 255.0f;
					const float blendedLinear =
						SRGBToLinearChannel(sourceSRGB) * opacity +
						SRGBToLinearChannel(destinationSRGB) *
							(1.0f - opacity);
					color[targetPixel * 4 + channel] =
						static_cast<std::uint8_t>(std::lround(
							std::clamp(
								LinearToSRGBChannel(blendedLinear),
								0.0f,
								1.0f) * 255.0f));
				}
				const float destinationAlpha =
					static_cast<float>(color[targetPixel * 4 + 3]) / 255.0f;
				color[targetPixel * 4 + 3] =
					static_cast<std::uint8_t>(std::lround(
						std::clamp(
							opacity + destinationAlpha * (1.0f - opacity),
							0.0f,
							1.0f) * 255.0f));
			}
		}

		bool ReadDepth(ID3D11Texture2D* source, ID3D11Texture2D* staging,
			int width, int height, std::vector<float>& output)
		{
			Context->CopyResource(staging, source);
			D3D11_MAPPED_SUBRESOURCE mapped = {};
			const HRESULT result = Context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
			if (FAILED(result)) return Fail(HResultMessage("ID3D11DeviceContext::Map(depth)", result));
			output.resize(static_cast<std::size_t>(width) * height);
			const std::size_t rowSize = static_cast<std::size_t>(width) * sizeof(float);
			for (int row = 0; row < height; ++row)
			{
				std::memcpy(output.data() + static_cast<std::size_t>(row) * width,
					static_cast<const std::uint8_t*>(mapped.pData) + static_cast<std::size_t>(row) * mapped.RowPitch,
					rowSize);
			}
			Context->Unmap(staging, 0);
			return true;
		}

		void PresentToWindow(const FrameCapture& capture) const
		{
			if (CreateInfo.NativeWindow == nullptr || !capture.HasColorOutput()) return;
			HWND window = static_cast<HWND>(CreateInfo.NativeWindow);
			if (!IsWindow(window)) return;
			std::vector<std::uint8_t> bgra(capture.ColorRGBA8.size());
			for (std::size_t pixel = 0; pixel < capture.ColorRGBA8.size() / 4; ++pixel)
			{
				bgra[pixel * 4] = capture.ColorRGBA8[pixel * 4 + 2];
				bgra[pixel * 4 + 1] = capture.ColorRGBA8[pixel * 4 + 1];
				bgra[pixel * 4 + 2] = capture.ColorRGBA8[pixel * 4];
				bgra[pixel * 4 + 3] = 255;
			}
			BITMAPINFO bitmap = {};
			bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
			bitmap.bmiHeader.biWidth = static_cast<LONG>(capture.Width);
			bitmap.bmiHeader.biHeight = -static_cast<LONG>(capture.Height);
			bitmap.bmiHeader.biPlanes = 1;
			bitmap.bmiHeader.biBitCount = 32;
			bitmap.bmiHeader.biCompression = BI_RGB;
			RECT client = {};
			GetClientRect(window, &client);
			HDC dc = GetDC(window);
			if (dc != nullptr)
			{
				StretchDIBits(dc, 0, 0, client.right, client.bottom, 0, 0,
					static_cast<int>(capture.Width), static_cast<int>(capture.Height), bgra.data(),
					&bitmap, DIB_RGB_COLORS, SRCCOPY);
				ReleaseDC(window, dc);
			}
		}

		RHICreateInfo CreateInfo;
		ComPtr<ID3D11Device> Device;
		ComPtr<ID3D11DeviceContext> Context;
		std::unique_ptr<DX11MicropolygonDicer> MicropolygonDicer;
		D3D_FEATURE_LEVEL FeatureLevel = D3D_FEATURE_LEVEL_11_0;
		D3D_DRIVER_TYPE DriverType = D3D_DRIVER_TYPE_UNKNOWN;
		ComPtr<ID3D11Texture2D> ColorTexture;
		ComPtr<ID3D11RenderTargetView> ColorTarget;
		ComPtr<ID3D11Texture2D> ColorStaging;
		ComPtr<ID3D11Texture2D> OutlineMaskTexture;
		ComPtr<ID3D11RenderTargetView> OutlineMaskTarget;
		ComPtr<ID3D11Texture2D> OutlineMaskStaging;
		ComPtr<ID3D11Texture2D> DepthTexture;
		ComPtr<ID3D11DepthStencilView> DepthTarget;
		ComPtr<ID3D11Texture2D> DepthStaging;
		ComPtr<ID3D11VertexShader> VertexShader;
		ComPtr<ID3D11PixelShader> PixelShader;
		ComPtr<ID3D11PixelShader> ShadowPixelShader;
		ComPtr<ID3D11PixelShader> OutlinePixelShader;
		ComPtr<ID3D11InputLayout> InputLayout;
		ComPtr<ID3D11Buffer> ConstantBuffer;
		ComPtr<ID3D11Buffer> PhysicalConstantBuffer;
		ComPtr<ID3D11Buffer> MediumConstantBuffer;
		ComPtr<ID3D11RasterizerState> SolidRasterState;
		ComPtr<ID3D11RasterizerState> ShadowRasterState;
		ComPtr<ID3D11RasterizerState> WireRasterState;
		ComPtr<ID3D11RasterizerState> OutlineRasterState;
		ComPtr<ID3D11DepthStencilState> DepthWriteState;
		ComPtr<ID3D11DepthStencilState> DepthReadState;
		ComPtr<ID3D11DepthStencilState> DepthDisabledState;
		ComPtr<ID3D11BlendState> ColorWriteDisabledState;
		ComPtr<ID3D11BlendState> AlphaBlendState;
		ComPtr<ID3D11SamplerState> SurfaceSampler;
		std::unordered_map<std::uint32_t, BufferResource> Buffers;
		std::unordered_map<std::uint32_t, TextureResource> Textures;
		FrameCapture PendingFrame;
		FrameCapture LastFrame;
		PassDesc CurrentPass;
		PipelineState BoundPipeline;
		Matrix4 BoundWorld;
		Matrix4 BoundViewProjection;
		Matrix4 BoundLightViewProjection;
		BufferHandle BoundVertexBuffer;
		BufferHandle BoundIndexBuffer;
		TextureHandle LastShadowTexture;
		std::string LastError;
		std::thread::id OwnerThread;
		std::uint32_t NextBuffer = 0x11000001u;
		std::uint32_t NextTexture = 0x11800001u;
		std::uint64_t FrameNumber = 0;
		bool Initialized = false;
		bool FrameOpen = false;
		bool PassOpen = false;
		bool PipelineBound = false;
		bool WorldBound = false;
		bool ViewProjectionBound = false;
		bool LightViewProjectionBound = false;
		bool HasOutlineMaskThisFrame = false;
	};

	DX11RHI::DX11RHI() : m_Impl(std::make_unique<Impl>()) {}
	DX11RHI::~DX11RHI() { Shutdown(); }

	bool DX11RHI::Initialize(const RHICreateInfo& createInfo)
	{
		if (m_Impl->Initialized) return m_Impl->Fail("DX11 RHI is already initialized");
		if (createInfo.Width <= 0 || createInfo.Height <= 0) return m_Impl->Fail("DX11 dimensions must be positive");
		const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
		const D3D_DRIVER_TYPE drivers[] = { D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP };
		const UINT baseFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
		const UINT requestedFlags = baseFlags
			| (createInfo.EnableValidation ? D3D11_CREATE_DEVICE_DEBUG : 0u);
		auto createDevice = [&](D3D_DRIVER_TYPE driver, UINT flags)
		{
			m_Impl->Device.Reset();
			m_Impl->Context.Reset();
			HRESULT result = D3D11CreateDevice(nullptr, driver, nullptr, flags,
				levels, static_cast<UINT>(std::size(levels)), D3D11_SDK_VERSION,
				m_Impl->Device.GetAddressOf(), &m_Impl->FeatureLevel,
				m_Impl->Context.GetAddressOf());
			if (result == E_INVALIDARG)
			{
				m_Impl->Device.Reset();
				m_Impl->Context.Reset();
				result = D3D11CreateDevice(nullptr, driver, nullptr, flags,
					&levels[1], 1, D3D11_SDK_VERSION, m_Impl->Device.GetAddressOf(),
					&m_Impl->FeatureLevel, m_Impl->Context.GetAddressOf());
			}
			return result;
		};
		HRESULT lastResult = E_FAIL;
		for (D3D_DRIVER_TYPE driver : drivers)
		{
			lastResult = createDevice(driver, requestedFlags);
			if (lastResult == DXGI_ERROR_SDK_COMPONENT_MISSING
				&& createInfo.EnableValidation)
			{
				// The debug layer is optional on current Windows installations.
				// Validation still requests it, but absence must not disable DX11.
				lastResult = createDevice(driver, baseFlags);
			}
			if (SUCCEEDED(lastResult))
			{
				m_Impl->DriverType = driver;
				break;
			}
		}
		if (FAILED(lastResult)) return m_Impl->Fail(HResultMessage("D3D11CreateDevice", lastResult));
		m_Impl->CreateInfo = createInfo;
		m_Impl->OwnerThread = std::this_thread::get_id();
		m_Impl->Initialized = true;
		if (!m_Impl->CreatePipelineObjects() || !m_Impl->CreateOffscreenTargets(createInfo.Width, createInfo.Height))
		{
			const std::string error = m_Impl->LastError;
			Shutdown();
			m_Impl->LastError = error;
			return false;
		}
		m_Impl->LastError.clear();
		return true;
	}

	void DX11RHI::Shutdown()
	{
		if (!m_Impl) return;
		if (m_Impl->Context) m_Impl->Context->ClearState();
		m_Impl->MicropolygonDicer.reset();
		m_Impl->Buffers.clear();
		m_Impl->Textures.clear();
		m_Impl->ColorTexture.Reset();
		m_Impl->ColorTarget.Reset();
		m_Impl->ColorStaging.Reset();
		m_Impl->OutlineMaskTexture.Reset();
		m_Impl->OutlineMaskTarget.Reset();
		m_Impl->OutlineMaskStaging.Reset();
		m_Impl->DepthTexture.Reset();
		m_Impl->DepthTarget.Reset();
		m_Impl->DepthStaging.Reset();
		m_Impl->VertexShader.Reset();
		m_Impl->PixelShader.Reset();
		m_Impl->ShadowPixelShader.Reset();
		m_Impl->OutlinePixelShader.Reset();
		m_Impl->InputLayout.Reset();
		m_Impl->ConstantBuffer.Reset();
		m_Impl->PhysicalConstantBuffer.Reset();
		m_Impl->MediumConstantBuffer.Reset();
		m_Impl->SolidRasterState.Reset();
		m_Impl->ShadowRasterState.Reset();
		m_Impl->WireRasterState.Reset();
		m_Impl->OutlineRasterState.Reset();
		m_Impl->DepthWriteState.Reset();
		m_Impl->DepthReadState.Reset();
		m_Impl->DepthDisabledState.Reset();
		m_Impl->ColorWriteDisabledState.Reset();
		m_Impl->AlphaBlendState.Reset();
		m_Impl->SurfaceSampler.Reset();
		m_Impl->Context.Reset();
		m_Impl->Device.Reset();
		m_Impl->Initialized = false;
		m_Impl->FrameOpen = false;
		m_Impl->PassOpen = false;
		m_Impl->HasOutlineMaskThisFrame = false;
		m_Impl->LastShadowTexture = {};
		m_Impl->OwnerThread = {};
	}

	RayTracing::MicropolygonResult DX11RHI::PrepareMicropolygons(const RayTracing::Scene& scene,
																 const RayTracing::Camera& camera,
																 const RayTracing::RenderSettings& settings,
																 const RayTracing::CancellationToken& token)
	{
		RayTracing::MicropolygonResult result;
		if (!m_Impl->CheckThread("PrepareMicropolygons"))
		{
			result.Error = m_Impl->LastError;
			return result;
		}
		if (m_Impl->FrameOpen)
		{
			result.Error = "DX11 geometry preparation must precede raster passes";
			return result;
		}
		if (!m_Impl->MicropolygonDicer)
			m_Impl->MicropolygonDicer =
				std::make_unique<DX11MicropolygonDicer>(m_Impl->Device.Get(), m_Impl->Context.Get());
		return m_Impl->MicropolygonDicer->Prepare(scene, camera, settings, token);
	}
	bool DX11RHI::IsInitialized() const { return m_Impl->Initialized; }
	RHIBackend DX11RHI::GetBackend() const { return RHIBackend::DX11; }
	const char* DX11RHI::GetName() const { return "DirectX 11 Native RHI"; }
	RHICapabilities DX11RHI::GetCapabilities() const
	{
		RHICapabilities capabilities;
		capabilities.NativeAPI = true;
		capabilities.Wireframe = true;
		capabilities.ShadowMaps = true;
		capabilities.Outline = true;
		capabilities.AsyncCommandRecording = false;
		capabilities.MaximumTextureSize = 16384;
		return capabilities;
	}

	bool DX11RHI::Resize(int width, int height)
	{
		if (!m_Impl->CheckThread("Resize") || m_Impl->FrameOpen || width <= 0 || height <= 0)
			return m_Impl->Fail("DX11 cannot resize in its current state");
		if (width == m_Impl->CreateInfo.Width && height == m_Impl->CreateInfo.Height
			&& m_Impl->ColorTexture && m_Impl->ColorTarget && m_Impl->ColorStaging
			&& m_Impl->OutlineMaskTexture && m_Impl->OutlineMaskTarget
			&& m_Impl->OutlineMaskStaging && m_Impl->DepthTexture
			&& m_Impl->DepthTarget && m_Impl->DepthStaging)
		{
			m_Impl->LastError.clear();
			return true;
		}
		if (!m_Impl->CreateOffscreenTargets(width, height)) return false;
		m_Impl->CreateInfo.Width = width;
		m_Impl->CreateInfo.Height = height;
		return true;
	}

	BufferHandle DX11RHI::CreateBuffer(const BufferDesc& desc, const void* initialData)
	{
		if (!m_Impl->CheckThread("CreateBuffer") || desc.SizeInBytes == 0
			|| desc.SizeInBytes > std::numeric_limits<UINT>::max())
		{
			m_Impl->Fail("DX11 buffer description is invalid");
			return {};
		}
		Impl::BufferResource resource;
		resource.Description = desc;
		resource.Bytes.resize(desc.SizeInBytes);
		if (initialData) std::memcpy(resource.Bytes.data(), initialData, desc.SizeInBytes);
		D3D11_BUFFER_DESC nativeDesc = {};
		nativeDesc.ByteWidth = static_cast<UINT>(desc.SizeInBytes);
		nativeDesc.Usage = desc.Dynamic ? D3D11_USAGE_DYNAMIC : D3D11_USAGE_DEFAULT;
		nativeDesc.CPUAccessFlags = desc.Dynamic ? D3D11_CPU_ACCESS_WRITE : 0;
		switch (desc.Usage)
		{
		case BufferUsage::Vertex: nativeDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER; break;
		case BufferUsage::Index: nativeDesc.BindFlags = D3D11_BIND_INDEX_BUFFER; break;
		case BufferUsage::Uniform: nativeDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER; break;
		}
		D3D11_SUBRESOURCE_DATA data = {};
		data.pSysMem = initialData;
		const HRESULT result = m_Impl->Device->CreateBuffer(&nativeDesc,
			initialData ? &data : nullptr, resource.Buffer.GetAddressOf());
		if (FAILED(result))
		{
			m_Impl->Fail(HResultMessage("ID3D11Device::CreateBuffer", result));
			return {};
		}
		const BufferHandle handle { m_Impl->NextBuffer++ };
		m_Impl->Buffers.emplace(handle.Value, std::move(resource));
		return handle;
	}

	bool DX11RHI::UpdateBuffer(BufferHandle buffer, std::size_t offset, const void* data, std::size_t size)
	{
		if (!m_Impl->CheckThread("UpdateBuffer")) return false;
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || data == nullptr || size == 0
			|| offset > iterator->second.Bytes.size() || size > iterator->second.Bytes.size() - offset)
			return m_Impl->Fail("DX11 buffer update is invalid");
		std::memcpy(iterator->second.Bytes.data() + offset, data, size);
		if (iterator->second.Description.Dynamic)
		{
			D3D11_MAPPED_SUBRESOURCE mapped = {};
			const HRESULT result = m_Impl->Context->Map(iterator->second.Buffer.Get(), 0,
				D3D11_MAP_WRITE_DISCARD, 0, &mapped);
			if (FAILED(result)) return m_Impl->Fail(HResultMessage("ID3D11DeviceContext::Map(buffer)", result));
			std::memcpy(mapped.pData, iterator->second.Bytes.data(), iterator->second.Bytes.size());
			m_Impl->Context->Unmap(iterator->second.Buffer.Get(), 0);
		}
		else
		{
			D3D11_BOX box = { static_cast<UINT>(offset), 0, 0,
				static_cast<UINT>(offset + size), 1, 1 };
			m_Impl->Context->UpdateSubresource(iterator->second.Buffer.Get(), 0, &box, data, 0, 0);
		}
		return true;
	}

	bool DX11RHI::DestroyBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("DestroyBuffer")) return false;
		if (!buffer || m_Impl->Buffers.erase(buffer.Value) == 0)
			return m_Impl->Fail("DX11 buffer handle is invalid");
		return true;
	}

	TextureHandle DX11RHI::CreateTexture(const TextureDesc& desc, const void* initialData)
	{
		if (!m_Impl->CheckThread("CreateTexture") || desc.Width <= 0 || desc.Height <= 0)
		{
			m_Impl->Fail("DX11 texture description is invalid");
			return {};
		}
		Impl::TextureResource resource;
		resource.Description = desc;
		if (desc.Format == TextureFormat::Depth32)
		{
			if (!m_Impl->CreateDepthTexture(desc.Width, desc.Height, true, resource.Texture,
				resource.DepthStencilView, resource.ShaderResourceView, resource.StagingTexture)) return {};
			if (initialData != nullptr)
			{
				m_Impl->Context->UpdateSubresource(resource.Texture.Get(), 0, nullptr, initialData,
					static_cast<UINT>(desc.Width * sizeof(float)), 0);
			}
		}
		else
		{
			const bool materialMips = initialData && !desc.RenderTarget && (desc.Width > 1 || desc.Height > 1);
			D3D11_TEXTURE2D_DESC nativeDesc = {};
			nativeDesc.Width = static_cast<UINT>(desc.Width);
			nativeDesc.Height = static_cast<UINT>(desc.Height);
			nativeDesc.MipLevels = materialMips ? 0 : 1;
			nativeDesc.ArraySize = 1;
			nativeDesc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
			nativeDesc.SampleDesc.Count = 1;
			nativeDesc.Usage = D3D11_USAGE_DEFAULT;
			nativeDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE
				| ((desc.RenderTarget || materialMips) ? D3D11_BIND_RENDER_TARGET : 0u);
			nativeDesc.MiscFlags = materialMips ? D3D11_RESOURCE_MISC_GENERATE_MIPS : 0u;
			HRESULT result = m_Impl->Device->CreateTexture2D(&nativeDesc, nullptr,
				resource.Texture.GetAddressOf());
			if (FAILED(result))
			{
				m_Impl->Fail(HResultMessage("ID3D11Device::CreateTexture2D", result));
				return {};
			}
			if (initialData)
				m_Impl->Context->UpdateSubresource(resource.Texture.Get(), 0, nullptr, initialData,
					static_cast<UINT>(desc.Width * 4), 0);
			if (desc.RenderTarget)
			{
				D3D11_RENDER_TARGET_VIEW_DESC renderTargetDesc = {};
				renderTargetDesc.Format = desc.ColorSpace == TextureColorSpace::SRGB
					? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
					: DXGI_FORMAT_R8G8B8A8_UNORM;
				renderTargetDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
				result = m_Impl->Device->CreateRenderTargetView(
					resource.Texture.Get(), &renderTargetDesc,
					resource.RenderTargetView.GetAddressOf());
				if (FAILED(result))
				{
					m_Impl->Fail(HResultMessage("ID3D11Device::CreateRenderTargetView(texture)", result));
					return {};
				}
			}
			D3D11_SHADER_RESOURCE_VIEW_DESC linearViewDesc = {};
			linearViewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			linearViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
			linearViewDesc.Texture2D.MipLevels = UINT(-1);
			result = m_Impl->Device->CreateShaderResourceView(
				resource.Texture.Get(), &linearViewDesc,
				resource.ShaderResourceView.GetAddressOf());
			if (FAILED(result))
			{
				m_Impl->Fail(HResultMessage("ID3D11Device::CreateShaderResourceView(texture)", result));
				return {};
			}
			D3D11_SHADER_RESOURCE_VIEW_DESC srgbViewDesc = linearViewDesc;
			srgbViewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
			result = m_Impl->Device->CreateShaderResourceView(
				resource.Texture.Get(), &srgbViewDesc,
				resource.SRGBShaderResourceView.GetAddressOf());
			if (FAILED(result))
			{
				m_Impl->Fail(HResultMessage(
					"ID3D11Device::CreateShaderResourceView(texture sRGB)",
					result));
				return {};
			}
			// Filter albedo in linear light and keep normal/opacity channels linear.
			if (materialMips)
				m_Impl->Context->GenerateMips(desc.ColorSpace == TextureColorSpace::SRGB
					? resource.SRGBShaderResourceView.Get() : resource.ShaderResourceView.Get());
		}
		const TextureHandle handle { m_Impl->NextTexture++ };
		m_Impl->Textures.emplace(handle.Value, std::move(resource));
		return handle;
	}

	bool DX11RHI::DestroyTexture(TextureHandle texture)
	{
		if (!m_Impl->CheckThread("DestroyTexture")) return false;
		if (!texture || m_Impl->Textures.erase(texture.Value) == 0)
			return m_Impl->Fail("DX11 texture handle is invalid");
		if (m_Impl->LastShadowTexture == texture) m_Impl->LastShadowTexture = {};
		return true;
	}

	bool DX11RHI::BeginFrame()
	{
		if (!m_Impl->CheckThread("BeginFrame") || m_Impl->FrameOpen)
			return m_Impl->Fail("DX11 frame cannot begin in the current state");
		m_Impl->PendingFrame = {};
		m_Impl->PendingFrame.Backend = RHIBackend::DX11;
		m_Impl->PendingFrame.Statistics.FrameNumber = m_Impl->FrameNumber + 1;
		m_Impl->FrameOpen = true;
		m_Impl->HasOutlineMaskThisFrame = false;
		m_Impl->LastShadowTexture = {};
		m_Impl->LastError.clear();
		return true;
	}

	void DX11RHI::AbortFrame()
	{
		if (!m_Impl || !m_Impl->Initialized)
		{
			return;
		}
		if (m_Impl->OwnerThread != std::this_thread::get_id())
		{
			m_Impl->LastError =
				"AbortFrame must execute on the owning DX11 RHI thread";
			return;
		}
		ID3D11ShaderResourceView* nullViews[4] = {};
		m_Impl->Context->PSSetShaderResources(0, 4, nullViews);
		m_Impl->Context->OMSetRenderTargets(0, nullptr, nullptr);
		m_Impl->PendingFrame = {};
		m_Impl->CurrentPass = {};
		m_Impl->BoundPipeline = {};
		m_Impl->BoundVertexBuffer = {};
		m_Impl->BoundIndexBuffer = {};
		m_Impl->LastShadowTexture = {};
		m_Impl->FrameOpen = false;
		m_Impl->PassOpen = false;
		m_Impl->PipelineBound = false;
		m_Impl->WorldBound = false;
		m_Impl->ViewProjectionBound = false;
		m_Impl->LightViewProjectionBound = false;
		m_Impl->HasOutlineMaskThisFrame = false;
	}

	bool DX11RHI::BeginPass(const PassDesc& desc)
	{
		if (!m_Impl->CheckThread("BeginPass") || !m_Impl->FrameOpen || m_Impl->PassOpen)
			return m_Impl->Fail("DX11 pass cannot begin in the current state");
		ID3D11ShaderResourceView* nullViews[4] = {};
		m_Impl->Context->PSSetShaderResources(0, 4, nullViews);
		ID3D11RenderTargetView* renderTarget = m_Impl->ColorTarget.Get();
		ID3D11DepthStencilView* depthTarget = m_Impl->DepthTarget.Get();
		int width = m_Impl->CreateInfo.Width;
		int height = m_Impl->CreateInfo.Height;
		if (desc.Type == PassType::Outline)
		{
			if (desc.Target)
			{
				return m_Impl->Fail("DX11 outline passes use the internal visibility mask");
			}
			renderTarget = m_Impl->OutlineMaskTarget.Get();
			if (!m_Impl->HasOutlineMaskThisFrame)
			{
				const float clearMask[] = { 0.0f, 0.0f, 0.0f, 0.0f };
				m_Impl->Context->ClearRenderTargetView(renderTarget, clearMask);
			}
			m_Impl->HasOutlineMaskThisFrame = true;
		}
		else if (desc.Target)
		{
			auto iterator = m_Impl->Textures.find(desc.Target.Value);
			if (iterator == m_Impl->Textures.end()) return m_Impl->Fail("DX11 pass target is invalid");
			width = iterator->second.Description.Width;
			height = iterator->second.Description.Height;
			if (desc.Type == PassType::Shadow)
			{
				if (!iterator->second.DepthStencilView) return m_Impl->Fail("DX11 shadow target is not a depth texture");
				renderTarget = nullptr;
				depthTarget = iterator->second.DepthStencilView.Get();
				m_Impl->LastShadowTexture = desc.Target;
			}
			else
			{
				if (!iterator->second.RenderTargetView) return m_Impl->Fail("DX11 color target is not renderable");
				if (width != m_Impl->CreateInfo.Width || height != m_Impl->CreateInfo.Height)
				{
					return m_Impl->Fail(
						"DX11 custom color target dimensions must match the default depth target");
				}
				renderTarget = iterator->second.RenderTargetView.Get();
			}
		}
		else if (desc.Type == PassType::Shadow)
		{
			return m_Impl->Fail("DX11 shadow pass requires a depth texture");
		}
		m_Impl->Context->OMSetRenderTargets(renderTarget ? 1u : 0u,
			renderTarget ? &renderTarget : nullptr, depthTarget);
		if (desc.Type != PassType::Outline && desc.ClearColorEnabled && renderTarget)
		{
			const float color[] = {
				desc.ClearColor.X,
				desc.ClearColor.Y,
				desc.ClearColor.Z,
				desc.ClearColor.W };
			m_Impl->Context->ClearRenderTargetView(renderTarget, color);
		}
		if (desc.Type != PassType::Outline && desc.ClearDepthEnabled && depthTarget)
			m_Impl->Context->ClearDepthStencilView(depthTarget, D3D11_CLEAR_DEPTH, 1.0f, 0);
		D3D11_VIEWPORT viewport = {};
		viewport.Width = static_cast<float>(width);
		viewport.Height = static_cast<float>(height);
		viewport.MaxDepth = 1.0f;
		m_Impl->Context->RSSetViewports(1, &viewport);
		m_Impl->CurrentPass = desc;
		m_Impl->PendingFrame.Passes.push_back({ desc, {} });
		m_Impl->PassOpen = true;
		m_Impl->PipelineBound = false;
		m_Impl->WorldBound = false;
		m_Impl->ViewProjectionBound = false;
		m_Impl->LightViewProjectionBound = false;
		m_Impl->BoundVertexBuffer = {};
		m_Impl->BoundIndexBuffer = {};
		return true;
	}

	bool DX11RHI::BindPipeline(const PipelineState& pipeline)
	{
		if (!m_Impl->CheckThread("BindPipeline") || !m_Impl->PassOpen)
			return m_Impl->Fail("DX11 pipeline must be bound inside a pass");
		std::string materialError;
		if (!RayTracing::ValidateMaterialScattering(pipeline.Scattering, materialError))
			return m_Impl->Fail(materialError);
		if (!RayTracing::ValidateMediumOptions(RayTracing::Algorithm::PathTracing, pipeline.Medium, materialError))
			return m_Impl->Fail(materialError);
		if (pipeline.MediumBackground && (!pipeline.Medium.Enabled || m_Impl->CurrentPass.Type != PassType::Color))
			return m_Impl->Fail("Medium background requires an enabled medium in a color pass");
		m_Impl->BoundPipeline = pipeline;
		m_Impl->PipelineBound = true;
		return true;
	}

	bool DX11RHI::BindWorldTransform(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindWorldTransform") || !m_Impl->PassOpen) return false;
		m_Impl->BoundWorld = value;
		m_Impl->WorldBound = true;
		return true;
	}

	bool DX11RHI::BindViewProjection(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindViewProjection") || !m_Impl->PassOpen) return false;
		m_Impl->BoundViewProjection = value;
		m_Impl->ViewProjectionBound = true;
		return true;
	}

	bool DX11RHI::BindLightViewProjection(const Matrix4& value)
	{
		if (!m_Impl->CheckThread("BindLightViewProjection") || !m_Impl->PassOpen) return false;
		m_Impl->BoundLightViewProjection = value;
		m_Impl->LightViewProjectionBound = true;
		return true;
	}

	bool DX11RHI::BindVertexBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("BindVertexBuffer") || !m_Impl->PassOpen) return false;
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || iterator->second.Description.Usage != BufferUsage::Vertex)
			return m_Impl->Fail("DX11 vertex buffer is invalid");
		m_Impl->BoundVertexBuffer = buffer;
		return true;
	}

	bool DX11RHI::BindIndexBuffer(BufferHandle buffer)
	{
		if (!m_Impl->CheckThread("BindIndexBuffer") || !m_Impl->PassOpen) return false;
		auto iterator = m_Impl->Buffers.find(buffer.Value);
		if (iterator == m_Impl->Buffers.end() || iterator->second.Description.Usage != BufferUsage::Index)
			return m_Impl->Fail("DX11 index buffer is invalid");
		m_Impl->BoundIndexBuffer = buffer;
		return true;
	}

	bool DX11RHI::DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex)
	{
		if (!m_Impl->CheckThread("DrawIndexed") || !m_Impl->PassOpen || !m_Impl->PipelineBound
			|| !m_Impl->WorldBound || !m_Impl->ViewProjectionBound || !m_Impl->LightViewProjectionBound
			|| !m_Impl->BoundVertexBuffer || !m_Impl->BoundIndexBuffer || indexCount == 0)
			return m_Impl->Fail("DX11 DrawIndexed is missing required state");
		auto vertex = m_Impl->Buffers.find(m_Impl->BoundVertexBuffer.Value);
		auto index = m_Impl->Buffers.find(m_Impl->BoundIndexBuffer.Value);
		if (vertex == m_Impl->Buffers.end() || index == m_Impl->Buffers.end())
			return m_Impl->Fail("DX11 DrawIndexed buffers are invalid");
		const std::size_t availableIndices = index->second.Bytes.size() / sizeof(std::uint32_t);
		if (firstIndex > availableIndices || indexCount > availableIndices - firstIndex)
			return m_Impl->Fail("DX11 DrawIndexed exceeds the index buffer");
		if (vertex->second.Bytes.size() % sizeof(Vertex) != 0
			|| index->second.Bytes.size() % sizeof(std::uint32_t) != 0
			|| indexCount % 3 != 0)
			return m_Impl->Fail("DX11 DrawIndexed requires complete vertices and triangle-list indices");
		const std::size_t availableVertices = vertex->second.Bytes.size() / sizeof(Vertex);
		const auto* indices = reinterpret_cast<const std::uint32_t*>(index->second.Bytes.data());
		for (std::size_t indexOffset = firstIndex;
			indexOffset < static_cast<std::size_t>(firstIndex) + indexCount; ++indexOffset)
		{
			const std::int64_t resolvedVertex = static_cast<std::int64_t>(indices[indexOffset]) + baseVertex;
			if (resolvedVertex < 0 || static_cast<std::uint64_t>(resolvedVertex) >= availableVertices)
				return m_Impl->Fail("DX11 DrawIndexed references a vertex outside the vertex buffer");
		}

		ID3D11ShaderResourceView* diffuseView = nullptr;
		ID3D11ShaderResourceView* normalView = nullptr;
		ID3D11ShaderResourceView* opacityView = nullptr;
		auto resolveSurfaceTexture = [&](TextureHandle handle, const char* usage,
			ID3D11ShaderResourceView*& view)
		{
			if (!handle)
			{
				return true;
			}
			auto texture = m_Impl->Textures.find(handle.Value);
			if (texture == m_Impl->Textures.end()
				|| texture->second.Description.Format != TextureFormat::RGBA8
				|| !texture->second.ShaderResourceView
				|| !texture->second.SRGBShaderResourceView)
			{
				return m_Impl->Fail(std::string("DX11 ") + usage
					+ " texture is invalid");
			}
			view = texture->second.Description.ColorSpace ==
				TextureColorSpace::SRGB
				? texture->second.SRGBShaderResourceView.Get()
				: texture->second.ShaderResourceView.Get();
			return true;
		};
		if (!resolveSurfaceTexture(
				m_Impl->BoundPipeline.DiffuseTexture, "diffuse", diffuseView)
			|| !resolveSurfaceTexture(
				m_Impl->BoundPipeline.NormalTexture, "normal", normalView)
			|| !resolveSurfaceTexture(
				m_Impl->BoundPipeline.OpacityTexture, "opacity", opacityView))
		{
			return false;
		}

		ID3D11ShaderResourceView* shadowView = nullptr;
		if (m_Impl->CurrentPass.Type != PassType::Shadow
			&& m_Impl->BoundPipeline.SamplesShadowMap
			&& m_Impl->LastShadowTexture)
		{
			auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
			if (shadow != m_Impl->Textures.end()
				&& shadow->second.Description.Format == TextureFormat::Depth32)
			{
				shadowView = shadow->second.ShaderResourceView.Get();
			}
		}

		DrawConstants constants {};
		constants.World = m_Impl->BoundWorld;
		constants.NormalWorld = BuildNormalMatrix(m_Impl->BoundWorld);
		constants.ViewProjection = m_Impl->BoundViewProjection;
		constants.LightViewProjection = m_Impl->BoundLightViewProjection;
		constants.PipelineColor = m_Impl->BoundPipeline.Color;
		constants.MaterialDiffuseColor =
			m_Impl->BoundPipeline.MaterialDiffuseColor;
		constants.MaterialAmbientColor =
			m_Impl->BoundPipeline.MaterialAmbientColor;
		constants.MaterialSpecularColor =
			m_Impl->BoundPipeline.MaterialSpecularColor;
		constants.MaterialEmissiveColor =
			m_Impl->BoundPipeline.MaterialEmissiveColor;
		const float outlineThickness = std::isfinite(
			m_Impl->BoundPipeline.OutlineThickness)
			? std::clamp(m_Impl->BoundPipeline.OutlineThickness, 0.0f, 255.0f)
			: 0.0f;
		constants.MaterialParameters = {
			m_Impl->BoundPipeline.MaterialShininess,
			m_Impl->BoundPipeline.MaterialOpacity,
			m_Impl->BoundPipeline.AlphaCutoff,
			outlineThickness };
		const LightingState& lighting = m_Impl->BoundPipeline.Lighting;
		const std::uint32_t lightCount = std::min(
			lighting.LightCount, MaximumSceneLights);
		constants.AmbientColorStrength = {
			lighting.AmbientColor.X,
			lighting.AmbientColor.Y,
			lighting.AmbientColor.Z,
			lighting.AmbientStrength };
		constants.CameraLightCount = {
			lighting.CameraPosition.X,
			lighting.CameraPosition.Y,
			lighting.CameraPosition.Z,
			static_cast<float>(lightCount) };
		const float renderMode = m_Impl->CurrentPass.Type == PassType::Outline
			|| m_Impl->BoundPipeline.OutlineExtrusion
			? 2.0f
			: (m_Impl->BoundPipeline.Fill == FillMode::Wireframe ? 1.0f : 0.0f);
		constants.RenderParameters = {
			shadowView != nullptr ? 1.0f : 0.0f,
			renderMode,
			static_cast<float>(lighting.ShadowLightIndex),
			m_Impl->BoundPipeline.BlendEnabled ? 1.0f : 0.0f };
		constants.TextureFlags = {
			diffuseView != nullptr ? 1u : 0u,
			normalView != nullptr ? 1u : 0u,
			opacityView != nullptr ? 1u : 0u,
			0u };
		for (std::uint32_t lightIndex = 0; lightIndex < lightCount; ++lightIndex)
		{
			const SceneLight& light = lighting.Lights[lightIndex];
			GPULight& gpuLight = constants.Lights[lightIndex];
			gpuLight.PositionType = {
				light.Position.X,
				light.Position.Y,
				light.Position.Z,
				static_cast<float>(light.Type) };
			gpuLight.DirectionIntensity = {
				light.Direction.X,
				light.Direction.Y,
				light.Direction.Z,
				light.Intensity };
			gpuLight.ColorCastShadows = {
				light.Color.X,
				light.Color.Y,
				light.Color.Z,
				light.CastShadows ? 1.0f : 0.0f };
			gpuLight.AreaSizePadding = {
				light.AreaSize.X,
				light.AreaSize.Y,
				0.0f,
				0.0f };
		}

		D3D11_MAPPED_SUBRESOURCE mapped = {};
		HRESULT result = m_Impl->Context->Map(m_Impl->ConstantBuffer.Get(), 0,
			D3D11_MAP_WRITE_DISCARD, 0, &mapped);
		if (FAILED(result)) return m_Impl->Fail(HResultMessage("ID3D11DeviceContext::Map(constants)", result));
		std::memcpy(mapped.pData, &constants, sizeof(constants));
		m_Impl->Context->Unmap(m_Impl->ConstantBuffer.Get(), 0);
		const auto physical = BuildDX11PhysicalConstants(m_Impl->BoundPipeline);
		result = m_Impl->Context->Map(m_Impl->PhysicalConstantBuffer.Get(), 0,
			D3D11_MAP_WRITE_DISCARD, 0, &mapped);
		if (FAILED(result)) return m_Impl->Fail(HResultMessage("ID3D11DeviceContext::Map(physical constants)", result));
		std::memcpy(mapped.pData, &physical, sizeof(physical));
		m_Impl->Context->Unmap(m_Impl->PhysicalConstantBuffer.Get(), 0);
		const auto medium = BuildDX11MediumConstants(m_Impl->BoundPipeline);
		result = m_Impl->Context->Map(m_Impl->MediumConstantBuffer.Get(), 0,
			D3D11_MAP_WRITE_DISCARD, 0, &mapped);
		if (FAILED(result)) return m_Impl->Fail(HResultMessage("ID3D11DeviceContext::Map(medium constants)", result));
		std::memcpy(mapped.pData, &medium, sizeof(medium));
		m_Impl->Context->Unmap(m_Impl->MediumConstantBuffer.Get(), 0);

		const UINT stride = sizeof(Vertex);
		const UINT offset = 0;
		ID3D11Buffer* vertexBuffer = vertex->second.Buffer.Get();
		m_Impl->Context->IASetInputLayout(m_Impl->InputLayout.Get());
		m_Impl->Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_Impl->Context->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
		m_Impl->Context->IASetIndexBuffer(index->second.Buffer.Get(), DXGI_FORMAT_R32_UINT, 0);
		m_Impl->Context->VSSetShader(m_Impl->VertexShader.Get(), nullptr, 0);
		ID3D11Buffer* constantBuffer = m_Impl->ConstantBuffer.Get();
		m_Impl->Context->VSSetConstantBuffers(0, 1, &constantBuffer);
		m_Impl->Context->PSSetConstantBuffers(0, 1, &constantBuffer);
		ID3D11Buffer* physicalBuffer = m_Impl->PhysicalConstantBuffer.Get();
		m_Impl->Context->PSSetConstantBuffers(DX11PhysicalConstantSlot, 1, &physicalBuffer);
		ID3D11Buffer* mediumBuffer = m_Impl->MediumConstantBuffer.Get();
		m_Impl->Context->VSSetConstantBuffers(DX11MediumConstantSlot, 1, &mediumBuffer);
		m_Impl->Context->PSSetConstantBuffers(DX11MediumConstantSlot, 1, &mediumBuffer);
		ID3D11PixelShader* pixelShader = m_Impl->PixelShader.Get();
		if (m_Impl->CurrentPass.Type == PassType::Shadow)
		{
			pixelShader = m_Impl->ShadowPixelShader.Get();
		}
		else if (m_Impl->CurrentPass.Type == PassType::Outline)
		{
			pixelShader = m_Impl->OutlinePixelShader.Get();
		}
		m_Impl->Context->PSSetShader(pixelShader, nullptr, 0);
		m_Impl->Context->RSSetState(m_Impl->CurrentPass.Type == PassType::Shadow
			? m_Impl->ShadowRasterState.Get()
			: (m_Impl->BoundPipeline.OutlineExtrusion
				? m_Impl->OutlineRasterState.Get()
				: (m_Impl->BoundPipeline.Fill == FillMode::Wireframe
					? m_Impl->WireRasterState.Get() : m_Impl->SolidRasterState.Get())));
		m_Impl->Context->OMSetDepthStencilState(!m_Impl->BoundPipeline.DepthTest
			? m_Impl->DepthDisabledState.Get()
			: (m_Impl->BoundPipeline.DepthWrite ? m_Impl->DepthWriteState.Get() : m_Impl->DepthReadState.Get()), 0);
		const float blendFactor[] = { 0.0f, 0.0f, 0.0f, 0.0f };
		ID3D11BlendState* blendState = nullptr;
		if (!m_Impl->BoundPipeline.ColorWrite)
		{
			blendState = m_Impl->ColorWriteDisabledState.Get();
		}
		else if (m_Impl->CurrentPass.Type == PassType::Color
			&& m_Impl->BoundPipeline.BlendEnabled)
		{
			blendState = m_Impl->AlphaBlendState.Get();
		}
		m_Impl->Context->OMSetBlendState(
			blendState, blendFactor, 0xFFFFFFFFu);
		ID3D11ShaderResourceView* shaderViews[] = {
			shadowView,
			diffuseView,
			normalView,
			opacityView
		};
		m_Impl->Context->PSSetShaderResources(0, 4, shaderViews);
		ID3D11SamplerState* sampler = m_Impl->SurfaceSampler.Get();
		m_Impl->Context->PSSetSamplers(0, 1, &sampler);
		m_Impl->Context->DrawIndexed(indexCount, firstIndex, baseVertex);

		DrawCallRecord record;
		record.VertexBuffer = m_Impl->BoundVertexBuffer;
		record.IndexBuffer = m_Impl->BoundIndexBuffer;
		record.Pipeline = m_Impl->BoundPipeline;
		record.WorldTransform = m_Impl->BoundWorld;
		record.FirstIndex = firstIndex;
		record.IndexCount = indexCount;
		record.BaseVertex = baseVertex;
		m_Impl->PendingFrame.Passes.back().DrawCalls.push_back(record);
		FrameStatistics& stats = m_Impl->PendingFrame.Statistics;
		++stats.DrawCallCount;
		stats.SubmittedTriangleCount += indexCount / 3;
		stats.ShadowDrawCallCount += m_Impl->CurrentPass.Type == PassType::Shadow ? 1u : 0u;
		stats.OutlineDrawCallCount += m_Impl->CurrentPass.Type == PassType::Outline ? 1u : 0u;
		return true;
	}

	bool DX11RHI::EndPass()
	{
		if (!m_Impl->CheckThread("EndPass") || !m_Impl->PassOpen)
			return m_Impl->Fail("DX11 has no open pass");
		ID3D11ShaderResourceView* nullViews[4] = {};
		m_Impl->Context->PSSetShaderResources(0, 4, nullViews);
		m_Impl->PassOpen = false;
		return true;
	}

	bool DX11RHI::Present()
	{
		if (!m_Impl->CheckThread("Present") || !m_Impl->FrameOpen || m_Impl->PassOpen)
			return m_Impl->Fail("DX11 frame cannot be presented in the current state");
		m_Impl->PendingFrame.Statistics.PassCount = static_cast<std::uint32_t>(m_Impl->PendingFrame.Passes.size());
		m_Impl->PendingFrame.Width = static_cast<std::uint32_t>(m_Impl->CreateInfo.Width);
		m_Impl->PendingFrame.Height = static_cast<std::uint32_t>(m_Impl->CreateInfo.Height);
		const bool captureDepth = IsDepthCaptureRequired() || m_Impl->CreateInfo.CaptureDiagnosticBuffers ||
			m_Impl->PendingFrame.Statistics.OutlineDrawCallCount > 0;
		if (!m_Impl->ReadColor(m_Impl->PendingFrame.ColorRGBA8)) return false;
		if (captureDepth &&
			!m_Impl->ReadDepth(m_Impl->DepthTexture.Get(), m_Impl->DepthStaging.Get(),
				m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
				m_Impl->PendingFrame.Depth32)) return false;
		if (m_Impl->HasOutlineMaskThisFrame)
		{
			std::vector<std::uint16_t> outlineMask;
			if (!m_Impl->ReadOutlineMask(outlineMask)) return false;
			m_Impl->ApplyOutline(outlineMask, m_Impl->PendingFrame.ColorRGBA8);
		}
		if (m_Impl->CreateInfo.CaptureDiagnosticBuffers && m_Impl->LastShadowTexture)
		{
			auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
			if (shadow != m_Impl->Textures.end())
			{
				m_Impl->PendingFrame.ShadowWidth = static_cast<std::uint32_t>(shadow->second.Description.Width);
				m_Impl->PendingFrame.ShadowHeight = static_cast<std::uint32_t>(shadow->second.Description.Height);
				if (!m_Impl->ReadDepth(shadow->second.Texture.Get(), shadow->second.StagingTexture.Get(),
					shadow->second.Description.Width, shadow->second.Description.Height,
					m_Impl->PendingFrame.ShadowDepth32)) return false;
			}
		}
		m_Impl->LastFrame = std::move(m_Impl->PendingFrame);
		m_Impl->FrameNumber = m_Impl->LastFrame.Statistics.FrameNumber;
		m_Impl->FrameOpen = false;
		m_Impl->PresentToWindow(m_Impl->LastFrame);
		return true;
	}

	FrameCapture DX11RHI::GetLastFrameCapture() const { return m_Impl->LastFrame; }
	std::string DX11RHI::GetLastError() const { return m_Impl->LastError; }
#else
	struct DX11RHI::Impl
	{
		std::string LastError = "DirectX 11 is only available on Windows";
	};

	DX11RHI::DX11RHI() : m_Impl(std::make_unique<Impl>()) {}
	DX11RHI::~DX11RHI() = default;
	bool DX11RHI::Initialize(const RHICreateInfo&) { return false; }
	void DX11RHI::Shutdown() {}
	bool DX11RHI::IsInitialized() const { return false; }
	RayTracing::MicropolygonResult DX11RHI::PrepareMicropolygons(const RayTracing::Scene&,
																 const RayTracing::Camera&,
																 const RayTracing::RenderSettings&,
																 const RayTracing::CancellationToken& token)
	{
		RayTracing::MicropolygonResult result;
		if (token.IsCancellationRequested())
			result.Status = RayTracing::RenderStatus::Cancelled;
		else
			result.Error = "DX11 micropolygons require Windows";
		return result;
	}
	RHIBackend DX11RHI::GetBackend() const { return RHIBackend::DX11; }
	const char* DX11RHI::GetName() const { return "DirectX 11 Unavailable"; }
	RHICapabilities DX11RHI::GetCapabilities() const { return {}; }
	bool DX11RHI::Resize(int, int) { return false; }
	BufferHandle DX11RHI::CreateBuffer(const BufferDesc&, const void*) { return {}; }
	bool DX11RHI::UpdateBuffer(BufferHandle, std::size_t, const void*, std::size_t) { return false; }
	bool DX11RHI::DestroyBuffer(BufferHandle) { return false; }
	TextureHandle DX11RHI::CreateTexture(const TextureDesc&, const void*) { return {}; }
	bool DX11RHI::DestroyTexture(TextureHandle) { return false; }
	bool DX11RHI::BeginFrame() { return false; }
	void DX11RHI::AbortFrame() {}
	bool DX11RHI::BeginPass(const PassDesc&) { return false; }
	bool DX11RHI::BindPipeline(const PipelineState&) { return false; }
	bool DX11RHI::BindWorldTransform(const Matrix4&) { return false; }
	bool DX11RHI::BindViewProjection(const Matrix4&) { return false; }
	bool DX11RHI::BindLightViewProjection(const Matrix4&) { return false; }
	bool DX11RHI::BindVertexBuffer(BufferHandle) { return false; }
	bool DX11RHI::BindIndexBuffer(BufferHandle) { return false; }
	bool DX11RHI::DrawIndexed(std::uint32_t, std::uint32_t, std::int32_t) { return false; }
	bool DX11RHI::EndPass() { return false; }
	bool DX11RHI::Present() { return false; }
	FrameCapture DX11RHI::GetLastFrameCapture() const { return {}; }
	std::string DX11RHI::GetLastError() const { return m_Impl->LastError; }
#endif
}

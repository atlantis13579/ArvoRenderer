#include "Renderer/DX11Shading.h"
#include "Renderer/RealtimeShader.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#define NOMINMAX
#include <Windows.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

// Compile this backend's own HLSL during the build, using the same source and
// optimization as the renderer previously used at startup. No numerical code is
// translated between APIs and no shader assets are required at runtime.
extern "C" __declspec(dllexport) int CompileDX11(const wchar_t* outputPath, int debug)
try
{
    if (!outputPath || !*outputPath) throw std::runtime_error("Missing DX11 output header path");
    namespace R = Arvo::Renderer;
    std::string source, error;
    if (!R::BuildDX11ShaderSource(source, error)) throw std::runtime_error(error);
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
    if (debug) flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
    else flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS;
    struct Stage { R::RealtimeShaderStage Id; const char* Name; const char* Target; };
    const Stage stages[] = {
        {R::RealtimeShaderStage::Vertex, "Vertex", "vs_5_0"},
        {R::RealtimeShaderStage::Fragment, "Fragment", "ps_5_0"},
        {R::RealtimeShaderStage::Shadow, "Shadow", "ps_5_0"},
        {R::RealtimeShaderStage::Outline, "Outline", "ps_5_0"}};
    std::ostringstream header;
    header << "// Built from the independent DX11 source. Do not edit.\n#pragma once\n"
              "namespace Arvo::Renderer::DX11Shaders\n{\n";
    constexpr size_t BytesPerLine = 16;
    for (const auto& stage : stages)
    {
        const auto* entry = R::BuiltinRealtimeShader().Program.FindStage(stage.Id);
        Microsoft::WRL::ComPtr<ID3DBlob> bytecode, diagnostics;
        const auto begin = std::chrono::steady_clock::now();
        if (FAILED(D3DCompile(source.data(), source.size(), "ArvoDX11.hlsl", nullptr, nullptr,
                             entry->HLSL.c_str(), stage.Target, flags, 0, &bytecode, &diagnostics)))
            throw std::runtime_error(diagnostics ? static_cast<const char*>(diagnostics->GetBufferPointer())
                                                 : "DX11 shader compilation failed");
        if (diagnostics) std::cerr << static_cast<const char*>(diagnostics->GetBufferPointer());
        header << "inline constexpr unsigned char " << stage.Name << "[] = {\n";
        const auto* bytes = static_cast<const unsigned char*>(bytecode->GetBufferPointer());
        for (size_t index = 0; index < bytecode->GetBufferSize(); ++index)
        {
            header << "0x" << std::hex << std::setfill('0') << std::setw(2) << unsigned(bytes[index]) << ',';
            if ((index + 1) % BytesPerLine == 0) header << '\n';
        }
        header << "\n};\n";
        std::cout << entry->HLSL << ": " << bytecode->GetBufferSize() << " bytes, "
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count()
                  << " ms\n";
    }
    header << "}\n";
    const auto destination = std::filesystem::absolute(outputPath);
    std::filesystem::create_directories(destination.parent_path());
    auto temporary = destination;
    temporary += ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output << header.str();
        output.close();
        if (!output) throw std::runtime_error("Could not write the DX11 bytecode header");
    }
    if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Could not publish the DX11 bytecode header");
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}

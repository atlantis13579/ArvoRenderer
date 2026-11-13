"""Compile handwritten native shaders and embed bytecode; never translate algorithms."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('compute', 'cuda', 'dxr'), default='compute')
    parser.add_argument('--dxc', type=Path)
    parser.add_argument('--nvcc', type=Path)
    parser.add_argument('--host-compiler', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compute_programs = (
        ('ComputeMicropolygonRootSignature', 'MicropolygonRootSignature.hlsl', 'ArvoComputeMicropolygonRootSignature'),
        ('ComputeMicropolygonDxil', 'MicropolygonDicing.hlsl', 'DiceMicropolygons'),
        ('ComputeRootSignature', 'RootSignature.hlsl', 'ArvoComputeRootSignature'),
        ('ComputeSpectralRootSignature', 'SpectralRootSignature.hlsl', 'ArvoComputeSpectralRootSignature'),
        ('ComputeSpectralResponseDxil', 'SpectralPrepare.hlsl', 'SpectralPrepareResponse'),
        ('ComputeSpectralPaletteDxil', 'SpectralPrepare.hlsl', 'SpectralPreparePalette'),
        ('ComputeClearPixelsDxil', 'Film.hlsl', 'ClearPixelsKernel'),
        ('ComputeClearStatisticsDxil', 'Film.hlsl', 'ClearStatisticsKernel'),
        ('ComputeResolvePixelsDxil', 'Film.hlsl', 'ResolvePixelsKernel'),
        ('ComputeWhittedDxil', 'Whitted.hlsl', 'TracePixelsKernel'),
        ('ComputePathTracingDxil', 'PathTracing.hlsl', 'TracePixelsKernel'),
        ('ComputeDistributedRayDxil', 'DistributedRayTracing.hlsl', 'TracePixelsKernel'),
        ('ComputeQueryRaysDxil', 'Whitted.hlsl', 'QueryRaysKernel'),
        ('ComputeBidirectionalRootSignature', 'BidirectionalRootSignature.hlsl', 'ArvoComputeBidirectionalRootSignature'),
        ('ComputeBidirectionalPairsDxil', 'BidirectionalPathTracing.hlsl', 'BidirectionalPairsKernel'),
        ('ComputeBidirectionalClearDxil', 'BidirectionalPathTracing.hlsl', 'BidirectionalClearKernel'),
        ('ComputeBidirectionalCommitDxil', 'BidirectionalPathTracing.hlsl', 'BidirectionalCommitKernel'),
        ('ComputeMetropolisRootSignature', 'MetropolisRootSignature.hlsl', 'ArvoComputeMetropolisRootSignature'),
        ('ComputeMetropolisBootstrapDxil', 'MetropolisLightTransport.hlsl', 'MetropolisBootstrapKernel'),
        ('ComputeMetropolisDistributionDxil', 'MetropolisLightTransport.hlsl', 'MetropolisDistributionKernel'),
        ('ComputeMetropolisInitializeDxil', 'MetropolisLightTransport.hlsl', 'MetropolisInitializeKernel'),
        ('ComputeMetropolisMutateDxil', 'MetropolisLightTransport.hlsl', 'MetropolisMutateKernel'),
        ('ComputeMetropolisClearDxil', 'MetropolisLightTransport.hlsl', 'MetropolisClearKernel'),
        ('ComputeMetropolisCommitDxil', 'MetropolisLightTransport.hlsl', 'MetropolisCommitKernel'),
        ('ComputePpmRootSignature', 'ProgressivePhotonRootSignature.hlsl', 'ArvoComputePpmRootSignature'),
        ('ComputePpmClearDxil', 'ProgressivePhotonMapping.hlsl', 'PpmClearKernel'),
        ('ComputePpmInitializeDxil', 'ProgressivePhotonMapping.hlsl', 'PpmInitializeKernel'),
        ('ComputePpmCameraDxil', 'ProgressivePhotonMapping.hlsl', 'PpmCameraKernel'),
        ('ComputePpmEmitDxil', 'ProgressivePhotonMapping.hlsl', 'PpmEmitKernel'),
        ('ComputePpmIndexDxil', 'ProgressivePhotonMapping.hlsl', 'PpmIndexKernel'),
        ('ComputePpmGatherDxil', 'ProgressivePhotonMapping.hlsl', 'PpmGatherKernel'),
        ('ComputePpmUpdateDxil', 'ProgressivePhotonMapping.hlsl', 'PpmUpdateKernel'),
        ('ComputePhotonRootSignature', 'PhotonRootSignature.hlsl', 'ArvoComputePhotonRootSignature'),
        ('ComputePhotonClearDxil', 'PhotonMapping.hlsl', 'PhotonClearKernel'),
        ('ComputePhotonEmitDxil', 'PhotonMapping.hlsl', 'PhotonEmitKernel'),
        ('ComputePhotonIndexDxil', 'PhotonMapping.hlsl', 'PhotonIndexKernel'),
        ('ComputePhotonCameraDxil', 'PhotonMapping.hlsl', 'PhotonCameraKernel'),
        ('ComputeVcmRootSignature', 'VertexConnectionMergingRootSignature.hlsl', 'ArvoComputeVcmRootSignature'),
        ('ComputeVcmCameraWalksDxil', 'VertexConnectionMerging.hlsl', 'VcmTraceCameraPaths'),
        ('ComputeVcmLightWalksDxil', 'VertexConnectionMerging.hlsl', 'VcmTraceLightPaths'),
        ('ComputeVcmCouplingsDxil', 'VertexConnectionMerging.hlsl', 'VcmEvaluateCouplings'),
        ('ComputeVcmInitializeDxil', 'VertexConnectionMerging.hlsl', 'VcmInitializeFrame'),
        ('ComputeVcmBuildGridDxil', 'VertexConnectionMerging.hlsl', 'VcmBuildGrid'),
        ('ComputeVcmSortIndexDxil', 'VertexConnectionMerging.hlsl', 'VcmSortIndex'),
        ('ComputeVcmCameraEstimateDxil', 'VertexConnectionMerging.hlsl', 'VcmEvaluateCameraFrame'),
        ('ComputeVcmCameraMergeDxil', 'VertexConnectionMerging.hlsl', 'VcmEvaluateCameraMerges'),
        ('ComputeVcmSplatEstimateDxil', 'VertexConnectionMerging.hlsl', 'VcmEvaluateSplats'),
        ('ComputeVcmCommitDxil', 'VertexConnectionMerging.hlsl', 'VcmCommitFrame'),
        ('ComputePointGiRootSignature', 'PointGiRootSignature.hlsl', 'ArvoComputePointGiRootSignature'),
        ('ComputePointGiInitializeDxil', 'PointBasedGlobalIllumination.hlsl', 'PointGiInitialize'),
        ('ComputePointGiBakeDxil', 'PointBasedGlobalIllumination.hlsl', 'PointGiBake'),
        ('ComputePointGiCameraDxil', 'PointBasedGlobalIllumination.hlsl', 'PointGiCamera'),
        ('ComputeSpatialDenoiseRootSignature', 'SpatialDenoiseRootSignature.hlsl', 'ArvoComputeSpatialDenoiseRootSignature'),
        ('ComputeSpatialDenoiseDxil', 'SpatialDenoise.hlsl', 'SpatialDenoise'),
        ('ComputeAdaptiveDenoiseRootSignature', 'AdaptiveDenoiseRootSignature.hlsl', 'ArvoComputeAdaptiveDenoiseRootSignature'),
        ('ComputeAdaptiveDenoiseDxil', 'AdaptiveDenoise.hlsl', 'AdaptiveDenoiseMain'),
        ('ComputeGeometryGuidesRootSignature', 'GeometryGuidesRootSignature.hlsl', 'ArvoComputeGeometryGuidesRootSignature'),
        ('ComputeGenerateGeometryGuideRaysDxil', 'GeometryGuides.hlsl', 'GenerateGeometryGuideRays'),
        ('ComputeResolveGeometryGuideHitsDxil', 'GeometryGuides.hlsl', 'ResolveGeometryGuideHits'),
        ('ComputeTemporalDenoiseRootSignature', 'TemporalDenoiseRootSignature.hlsl', 'ArvoComputeTemporalDenoiseRootSignature'),
        ('ComputeTemporalDenoiseDxil', 'TemporalDenoise.hlsl', 'TemporalDenoise'),
    )
    cuda_programs = (
        ('CudaMicropolygonPtx', 'MicropolygonDicing.cu', None),
        ('CudaSpectralPreparePtx', 'SpectralPrepare.cu', None),
        ('CudaWhittedPtx', 'Whitted.cu', None),
        ('CudaPathTracingPtx', 'PathTracing.cu', None),
        ('CudaDistributedRayPtx', 'DistributedRayTracing.cu', None),
        ('CudaBidirectionalPtx', 'BidirectionalPathTracing.cu', None),
        ('CudaMetropolisPtx', 'MetropolisLightTransport.cu', None),
        ('CudaPhotonPtx', 'PhotonMapping.cu', None),
        ('CudaProgressivePhotonPtx', 'ProgressivePhotonMapping.cu', None),
        ('CudaVcmPtx', 'VertexConnectionMerging.cu', None),
        ('CudaPointGiPtx', 'PointBasedGlobalIllumination.cu', None),
        ('CudaSpatialDenoisePtx', 'SpatialDenoise.cu', None),
        ('CudaAdaptiveDenoisePtx', 'AdaptiveDenoise.cu', None),
        ('CudaGeometryGuidesPtx', 'GeometryGuides.cu', None),
        ('CudaTemporalDenoisePtx', 'TemporalDenoise.cu', None),
        ('CudaFilmPtx', 'Film.cu', None),
    )
    dxr_programs = (
        ('DxrMicropolygonRootSignature', 'MicropolygonRootSignature.hlsl', 'ArvoDxrMicropolygonRootSignature'),
        ('DxrMicropolygonDxil', 'MicropolygonDicing.hlsl', None),
        ('DxrRootSignature', 'RootSignature.hlsl', 'ArvoDxrRootSignature'),
        ('DxrSpectralRootSignature', 'SpectralRootSignature.hlsl', 'ArvoDxrSpectralRootSignature'),
        ('DxrSpectralPrepareDxil', 'SpectralPrepare.hlsl', None),
        ('DxrWhittedDxil', 'Whitted.hlsl', None),
        ('DxrPathTracingDxil', 'PathTracing.hlsl', None),
        ('DxrDistributedRayDxil', 'DistributedRayTracing.hlsl', None),
        ('DxrRayQueriesDxil', 'QueryRays.hlsl', None),
        ('DxrBidirectionalDxil', 'BidirectionalPathTracing.hlsl', None),
        ('DxrMetropolisRootSignature', 'MetropolisRootSignature.hlsl', 'ArvoDxrMetropolisRootSignature'),
        ('DxrMetropolisDxil', 'MetropolisLightTransport.hlsl', None),
        ('DxrPhotonRootSignature', 'PhotonRootSignature.hlsl', 'ArvoDxrPhotonRootSignature'),
        ('DxrPhotonDxil', 'PhotonMapping.hlsl', None),
        ('DxrPpmRootSignature', 'ProgressivePhotonRootSignature.hlsl', 'ArvoDxrPpmRootSignature'),
        ('DxrPpmDxil', 'ProgressivePhotonMapping.hlsl', None),
        ('DxrVcmRootSignature', 'VertexConnectionMergingRootSignature.hlsl', 'ArvoDxrVcmRootSignature'),
        ('DxrVcmDxil', 'VertexConnectionMerging.hlsl', None),
        ('DxrPointGiRootSignature', 'PointGiRootSignature.hlsl', 'ArvoDxrPointGiRootSignature'),
        ('DxrPointGiDxil', 'PointBasedGlobalIllumination.hlsl', None),
        ('DxrSpatialDenoiseRootSignature', 'SpatialDenoiseRootSignature.hlsl', 'ArvoDxrSpatialDenoiseRootSignature'),
        ('DxrSpatialDenoiseDxil', 'SpatialDenoise.hlsl', None),
        ('DxrAdaptiveDenoiseRootSignature', 'AdaptiveDenoiseRootSignature.hlsl', 'ArvoDxrAdaptiveDenoiseRootSignature'),
        ('DxrAdaptiveDenoiseDxil', 'AdaptiveDenoise.hlsl', None),
        ('DxrGeometryGuidesRootSignature', 'GeometryGuidesRootSignature.hlsl', 'ArvoDxrGeometryGuidesRootSignature'),
        ('DxrGeometryGuidesDxil', 'GeometryGuides.hlsl', None),
        ('DxrTemporalDenoiseRootSignature', 'TemporalDenoiseRootSignature.hlsl', 'ArvoDxrTemporalDenoiseRootSignature'),
        ('DxrTemporalDenoiseDxil', 'TemporalDenoise.hlsl', None),
    )
    programs = {'compute': compute_programs, 'cuda': cuda_programs, 'dxr': dxr_programs}[args.backend]
    folder = {'compute': 'Compute', 'cuda': 'CUDA', 'dxr': 'DXR'}[args.backend]
    shader_root = args.root / 'Src/RayTracer/Backends' / folder
    cuda = args.backend == 'cuda'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = ['#pragma once', '// Compiled from native shader sources by Build/CompileShaders.py.',
              'namespace Arvo::RayTracing::Detail', '{']
    for symbol, filename, entry in programs:
        binary = args.output.parent / (symbol + ('.ptx' if cuda else '.dxil'))
        if not cuda:
            if args.dxc is None: parser.error('--dxc is required for HLSL shaders')
            profile = 'lib_6_3' if args.backend == 'dxr' else 'cs_6_0'
            if filename.endswith('RootSignature.hlsl'): profile = 'rootsig_1_0'
            command = [str(args.dxc), '-T', profile, '-HV', '2021', '-O3', '-Ges',
                       '-Fo', str(binary), str(shader_root / filename)]
            if entry: command += ['-E', entry]
            if filename == 'VertexConnectionMerging.hlsl':
                # Preserve cancellation of common log-density factors in VCM MIS.
                command += ['-Gis', '-WX']
        else:
            if args.nvcc is None: parser.error('--nvcc is required for CUDA kernels')
            command = [str(args.nvcc), '--ptx', '--std=c++17', '-O3',
                       '--gpu-architecture=compute_75', '-o', str(binary), str(shader_root / filename)]
            if args.host_compiler:
                command += ['--compiler-bindir', str(args.host_compiler.parent)]
        subprocess.run(command, check=True)
        if args.backend == 'compute' and not filename.endswith('RootSignature.hlsl'):
            # Verify the exact production binding blob without creating a device.
            # HLSL compilation alone does not catch an oversized/incompatible root.
            signature = {
                'MicropolygonDicing.hlsl': 'ComputeMicropolygonRootSignature.dxil',
                'SpectralPrepare.hlsl': 'ComputeSpectralRootSignature.dxil',
                'BidirectionalPathTracing.hlsl': 'ComputeBidirectionalRootSignature.dxil',
                'MetropolisLightTransport.hlsl': 'ComputeMetropolisRootSignature.dxil',
                'PhotonMapping.hlsl': 'ComputePhotonRootSignature.dxil',
                'ProgressivePhotonMapping.hlsl': 'ComputePpmRootSignature.dxil',
                'VertexConnectionMerging.hlsl': 'ComputeVcmRootSignature.dxil',
                'PointBasedGlobalIllumination.hlsl': 'ComputePointGiRootSignature.dxil',
                'SpatialDenoise.hlsl': 'ComputeSpatialDenoiseRootSignature.dxil',
                'AdaptiveDenoise.hlsl': 'ComputeAdaptiveDenoiseRootSignature.dxil',
                'GeometryGuides.hlsl': 'ComputeGeometryGuidesRootSignature.dxil',
                'TemporalDenoise.hlsl': 'ComputeTemporalDenoiseRootSignature.dxil',
            }.get(filename, 'ComputeRootSignature.dxil')
            subprocess.run([str(args.dxc), '-dumpbin', '-verifyrootsignature',
                            str(args.output.parent / signature), str(binary)], check=True)
        # PTX is passed to the driver as a terminated string, without rewriting
        # its version header. Drivers must support the selected toolkit's ISA.
        data = binary.read_bytes() + (b'\0' if cuda else b'')
        header.append(f'inline constexpr unsigned char {symbol}[] = {{')
        bytes_per_line = 16
        for offset in range(0, len(data), bytes_per_line):
            header.append('    ' + ', '.join(f'0x{value:02x}' for value in data[offset:offset + bytes_per_line]) + ',')
        header.append('};')
    header.append('}')
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text('\n'.join(header) + '\n', encoding='utf-8')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()

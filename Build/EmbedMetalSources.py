"""Embed native MSL compilation units, resolving only same-backend includes."""
import argparse
from pathlib import Path
import re


def compilation_unit(path, root, active=()):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or path in active:
        raise ValueError('Metal include must be local and acyclic: ' + str(path))
    source = path.read_text(encoding='utf-8')
    # Metal's runtime compiler receives a string and has no source-directory
    # include search path. Expand native leaf files exactly as a preprocessor;
    # never read or translate another backend's source.
    def include(match):
        dependency = path.parent / match.group(1)
        return compilation_unit(dependency, root, (*active, path))
    return re.sub(r'^\s*#include "([^"]+)"\s*$', include, source, flags=re.MULTILINE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.root / 'Src/RayTracer/Backends/Metal'
    programs = (
        ('MetalMicropolygonSource', 'MicropolygonDicing.metal'),
        ('MetalSpectralPrepareSource', 'SpectralPrepare.metal'),
        ('MetalWhittedSource', 'Whitted.metal'),
        ('MetalPathTracingSource', 'PathTracing.metal'),
        ('MetalDistributedRaySource', 'DistributedRayTracing.metal'),
        ('MetalBidirectionalSource', 'BidirectionalPathTracing.metal'),
        ('MetalMetropolisSource', 'MetropolisLightTransport.metal'),
        ('MetalPhotonSource', 'PhotonMapping.metal'),
        ('MetalPpmSource', 'ProgressivePhotonMapping.metal'),
        ('MetalPointGiSource', 'PointBasedGlobalIllumination.metal'),
        ('MetalSpatialDenoiseSource', 'SpatialDenoise.metal'),
        ('MetalAdaptiveDenoiseSource', 'AdaptiveDenoise.metal'),
        ('MetalGeometryGuidesSource', 'GeometryGuides.metal'),
        ('MetalTemporalDenoiseSource', 'TemporalDenoise.metal'),
        ('MetalFilmSource', 'Film.metal'),
        ('MetalVcmSource', 'VertexConnectionMerging.metal'),
    )
    delimiter = 'ARVO_METAL'
    header = ['#pragma once', '// Native MSL with local includes expanded; no algorithm translation.',
              'namespace Arvo::RayTracing::Detail', '{']
    for symbol, filename in programs:
        source = compilation_unit(folder / filename, folder)
        if f'){delimiter}"' in source:
            raise ValueError('MSL contains the embedding delimiter')
        header.append(f'inline constexpr const char {symbol}[] = R"{delimiter}({source}){delimiter}";')
    header.append('}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text('\n'.join(header) + '\n', encoding='utf-8')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()

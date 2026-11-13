# Asset attribution

The code in Arvo is self-contained. The meshes and textures under `Contents`
are test content distributed under their own licenses.

## Cornell Box

- Files: `Scenes/CornellBox/`
- OBJ variants: Guedis Cardenas, Morgan McGuire, and Michael Mara
- Source: [McGuire Computer Graphics Archive](https://casual-effects.com/g3d/data10/common/model/CornellBox/)
- License: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)
- The source archive's original notice is retained as `copyright.txt`.

## Crytek Sponza Atrium

- Files: `Scenes/Sponza/`
- Model: Frank Meinl, Crytek (2010), inspired by Marko Dabrovic's Sponza
- Archive conversion and corrections: Morgan McGuire
- Canonical source: [McGuire Computer Graphics Archive](https://casual-effects.com/g3d/data10/common/model/crytek_sponza/)
- Download mirror: [MatthewBieda/VKEngine, commit d1f485b](https://github.com/MatthewBieda/VKEngine/tree/d1f485b9d02253a8db81577d7d9ae626422833fb/Models/Sponza)
- Missing opacity masks restored from [McNopper/OpenGL, commit b5a0247](https://github.com/McNopper/OpenGL/tree/b5a0247536ddad953556bc0f4ab29170317d3aa5/Binaries/sponza)
- License declared by the archive and mirror: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)

When redistributing Cornell Box or Sponza content, retain this file and the
attribution required by CC BY 3.0.

## Stanford Dragon

- Files: `Scenes/Dragon/`
- Original scan/reconstruction: Stanford University Computer Graphics Laboratory
- Source model and terms: [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)
- Download: [Sebastian Lague's Coding Adventure ray tracer, commit e8981ea](https://github.com/SebLague/Ray-Tracing/blob/e8981eae5e4bc392070a6fdbc7e1a74e295c86fa/Assets/Graphics/Dragon_80K.obj)
- The source OBJ has 87,130 triangles and is preserved unchanged, apart from possible Git line-ending normalization.
- Stanford permits research use and free redistribution with acknowledgment; commercial use requires permission. See `Scenes/Dragon/copyright.txt` and the original terms.
- Arvo authors the clear-glass, frosted-glass and subsurface materials, floor and backdrop in code.

cmake_minimum_required(VERSION 3.20)
project(Arvo VERSION 0.1.0 LANGUAGES CXX)
find_package(Threads REQUIRED)
get_filename_component(ARVO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(ARVO_SRC "${ARVO_ROOT}/Src")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_XCODE_ATTRIBUTE_OBJROOT "${ARVO_ROOT}/Intermediate/Xcode")
    set(CMAKE_XCODE_ATTRIBUTE_SYMROOT "${ARVO_ROOT}/Intermediate/Xcode")
    set(ARVO_METAL_SOURCE "${ARVO_SRC}/RayTracer/MetalRayTracingRHI.mm")
else()
    set(ARVO_METAL_SOURCE "${ARVO_SRC}/RayTracer/MetalRayTracingRHIStub.cpp")
endif()
foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${upper} "${ARVO_ROOT}/Binaries/${config}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${upper} "${ARVO_ROOT}/Binaries/${config}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${upper} "${ARVO_ROOT}/Intermediate/Lib/${config}")
    set(CMAKE_PDB_OUTPUT_DIRECTORY_${upper} "${ARVO_ROOT}/Intermediate/Symbols/${config}")
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY_${upper} "${ARVO_ROOT}/Intermediate/Symbols/${config}")
endforeach()
function(arvo_target target)
    target_include_directories(${target} PUBLIC "${ARVO_SRC}")
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()
# Offline compute only: this target never links the renderer or graphics APIs.
add_library(ArvoLightmapCore STATIC "${ARVO_SRC}/Lightmap/Baker.cpp" "${ARVO_SRC}/Lightmap/CudaCompute.cpp")
arvo_target(ArvoLightmapCore)
target_compile_definitions(ArvoLightmapCore PUBLIC ARVO_RENDERER_STATIC ARVO_RAYTRACER_STATIC)
target_link_libraries(ArvoLightmapCore PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
option(ARVO_LIGHTMAP_CUDA "Compile the default offline CUDA lightmap backend" ON)
if(ARVO_LIGHTMAP_CUDA)
    find_program(ARVO_NVCC NAMES nvcc HINTS "$ENV{CUDA_PATH}/bin")
    if(ARVO_NVCC)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        get_filename_component(ARVO_CUDA_BIN "${ARVO_NVCC}" DIRECTORY)
        find_path(ARVO_CUDA_INCLUDE cuda.h HINTS "${ARVO_CUDA_BIN}/../include")
        if(NOT ARVO_CUDA_INCLUDE)
            message(FATAL_ERROR "CUDA headers not found for ArvoLightmap")
        endif()
        set(ARVO_LIGHTMAP_PTX "${CMAKE_CURRENT_BINARY_DIR}/LightmapCuda/LightmapPtx.h")
        add_custom_command(OUTPUT "${ARVO_LIGHTMAP_PTX}"
            COMMAND "${Python3_EXECUTABLE}" "${ARVO_ROOT}/Build/CompileLightmap.py"
                --nvcc "${ARVO_NVCC}" --host-compiler "${CMAKE_CXX_COMPILER}"
                --root "${ARVO_ROOT}" --output "${ARVO_LIGHTMAP_PTX}"
            DEPENDS "${ARVO_ROOT}/Build/CompileLightmap.py" "${ARVO_SRC}/Lightmap/Kernel.cu"
                "${ARVO_SRC}/Lightmap/Transport.h" "${ARVO_SRC}/Lightmap/Exchange.h"
            COMMENT "Compiling offline CUDA lightmap transport" VERBATIM)
        target_sources(ArvoLightmapCore PRIVATE "${ARVO_LIGHTMAP_PTX}")
        target_include_directories(ArvoLightmapCore PRIVATE "${ARVO_CUDA_INCLUDE}" "${CMAKE_CURRENT_BINARY_DIR}/LightmapCuda")
        target_compile_definitions(ArvoLightmapCore PRIVATE ARVO_LIGHTMAP_CUDA)
        set(ARVO_LIGHTMAP_CUDA_AVAILABLE ON)
    else()
        message(WARNING "CUDA Toolkit not found: default lightmap backend will report unavailable; --backend cpu remains available.")
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(ArvoLightmapTests "${ARVO_ROOT}/Tests/LightmapTests.cpp")
    arvo_target(ArvoLightmapTests)
    target_link_libraries(ArvoLightmapTests PRIVATE ArvoLightmapCore)
    add_test(NAME LightmapPhysicsCPU COMMAND ArvoLightmapTests cpu)
    if(ARVO_LIGHTMAP_CUDA_AVAILABLE)
        add_test(NAME LightmapPhysicsCUDA COMMAND ArvoLightmapTests cuda)
    endif()
endif()

add_library(ArvoMaths STATIC "${ARVO_SRC}/Maths/StaticAABBTree.cpp" "${ARVO_SRC}/Maths/TriMesh.cpp")
arvo_target(ArvoMaths)
add_library(ArvoRenderer STATIC
    "${ARVO_SRC}/RayTracer/Backends/DXR/EmitterData.cpp"
    "${ARVO_SRC}/RayTracer/Backends/DXR/SpectralScene.cpp"
    "${ARVO_SRC}/RayTracer/Backends/DXR/SpectralBindings.cpp"
    "${ARVO_SRC}/RayTracer/Backends/DXR/PointBasedGlobalIllumination.cpp"
    "${ARVO_SRC}/RayTracer/Backends/DXR/EmissionCatalog.cpp"
    "${ARVO_SRC}/RayTracer/Backends/DXR/InstancingBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/DXR/MotionPreparation.cpp"
    "${ARVO_SRC}/RayTracer/Backends/DXR/HardwareBvh.cpp"
    "${ARVO_SRC}/RayTracer/Backends/DXR/SpatialBvhBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/DXR/BihBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/DXR/KdTreeBuilder.cpp"
    "${ARVO_SRC}/Renderer/RenderTechniques.cpp"
    "${ARVO_SRC}/Renderer/Reconstruction.cpp"
    "${ARVO_SRC}/Renderer/AdaptiveReconstruction.cpp"
    "${ARVO_SRC}/Renderer/ParticipatingMedia.cpp"
    "${ARVO_SRC}/Renderer/Radiometry.cpp"
    "${ARVO_SRC}/Renderer/GeometryGuides.cpp"
    "${ARVO_SRC}/Renderer/TemporalReconstruction.cpp"
    "${ARVO_SRC}/Renderer/TemporalSession.cpp"
    "${ARVO_SRC}/Renderer/MaterialParameters.cpp"
    "${ARVO_SRC}/Renderer/CameraParameters.cpp"
    "${ARVO_SRC}/Renderer/CameraMotionParameters.cpp"
    "${ARVO_SRC}/Renderer/SceneMotionParameters.cpp"
    "${ARVO_SRC}/Renderer/DX11CameraExposure.cpp"
    "${ARVO_SRC}/Renderer/DX11InstanceExposure.cpp"
    "${ARVO_SRC}/Renderer/SubsurfaceValidation.cpp"
    "${ARVO_SRC}/Renderer/MicropolygonPreparation.cpp"
    "${ARVO_SRC}/Renderer/DX11MicropolygonDicing.cpp"
    "${ARVO_SRC}/Renderer/DX11RHI.cpp" "${ARVO_SRC}/Renderer/MetalRHI.cpp"
    "${ARVO_SRC}/Renderer/DxrRayTracingRHI.cpp" "${ARVO_SRC}/Renderer/RayTracing.cpp"
    "${ARVO_SRC}/Renderer/RayTracingBackendCommon.cpp" "${ARVO_SRC}/Renderer/RealtimeShader.cpp"
    "${ARVO_SRC}/Renderer/Renderer.cpp" "${ARVO_SRC}/Renderer/RHIBackends.cpp"
    "${ARVO_SRC}/Renderer/RHIThread.cpp")
# Embed the DX11 algorithm verbatim; offline backends never consume this source.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ARVO_SRC}/Renderer/DX11PhysicalShading.hlsl")
file(READ "${ARVO_SRC}/Renderer/DX11PhysicalShading.hlsl" ARVO_DX11_PHYSICAL_SOURCE)
configure_file("${ARVO_SRC}/Renderer/DX11PhysicalSource.h.in"
    "${ARVO_ROOT}/Intermediate/Shaders/DX11/DX11PhysicalSource.h" @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ARVO_SRC}/Renderer/DX11HomogeneousMedium.hlsl")
file(READ "${ARVO_SRC}/Renderer/DX11HomogeneousMedium.hlsl" ARVO_DX11_MEDIUM_SOURCE)
configure_file("${ARVO_SRC}/Renderer/DX11MediumSource.h.in"
    "${ARVO_ROOT}/Intermediate/Shaders/DX11/DX11MediumSource.h" @ONLY)
target_sources(ArvoRenderer PRIVATE "${ARVO_SRC}/Renderer/DX11Shading.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ARVO_SRC}/Renderer/DX11SubsurfaceScattering.hlsl")
file(READ "${ARVO_SRC}/Renderer/DX11SubsurfaceScattering.hlsl" ARVO_DX11_SUBSURFACE_SOURCE)
configure_file("${ARVO_SRC}/Renderer/DX11SubsurfaceSource.h.in"
    "${ARVO_ROOT}/Intermediate/Shaders/DX11/DX11SubsurfaceSource.h" @ONLY)
target_include_directories(ArvoRenderer PRIVATE "${ARVO_ROOT}/Intermediate/Shaders/DX11")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ARVO_SRC}/Renderer/DX11MicropolygonDicing.hlsl")
file(READ "${ARVO_SRC}/Renderer/DX11MicropolygonDicing.hlsl" ARVO_DX11_MICROPOLYGON_SOURCE)
configure_file("${ARVO_SRC}/Renderer/DX11MicropolygonSource.h.in"
    "${ARVO_ROOT}/Intermediate/Shaders/DX11/DX11MicropolygonSource.h" @ONLY)
arvo_target(ArvoRenderer)
target_compile_definitions(ArvoRenderer PUBLIC ARVO_RENDERER_STATIC)
target_link_libraries(ArvoRenderer PUBLIC Threads::Threads)
if(WIN32)
    target_link_libraries(ArvoRenderer PUBLIC d3d11 d3d12 dxgi d3dcompiler)
    # This build-only module compiles the backend's own HLSL without adding
    # another application executable or a runtime dependency for the editor.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_library(ArvoShaderCompiler MODULE "${ARVO_ROOT}/Build/CompileDX11.cpp"
        "${ARVO_SRC}/Renderer/DX11Shading.cpp" "${ARVO_SRC}/Renderer/RealtimeShader.cpp")
    arvo_target(ArvoShaderCompiler)
    target_compile_definitions(ArvoShaderCompiler PRIVATE ARVO_RENDERER_STATIC ARVO_RAYTRACER_STATIC)
    target_include_directories(ArvoShaderCompiler PRIVATE "${ARVO_ROOT}/Intermediate/Shaders/DX11")
    target_link_libraries(ArvoShaderCompiler PRIVATE d3dcompiler)
    set_target_properties(ArvoShaderCompiler PROPERTIES FOLDER "Build tools")
    foreach(config Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${config}" upper)
        foreach(kind RUNTIME LIBRARY ARCHIVE PDB COMPILE_PDB)
            set_target_properties(ArvoShaderCompiler PROPERTIES
                ${kind}_OUTPUT_DIRECTORY_${upper} "${ARVO_ROOT}/Intermediate/Tools/${config}")
        endforeach()
    endforeach()
    set(ARVO_DX11_BYTECODE_DIR "${ARVO_ROOT}/Intermediate/Shaders/DX11/$<CONFIG>")
    set(ARVO_DX11_BYTECODE "${ARVO_DX11_BYTECODE_DIR}/DX11ShaderBytecode.h")
    add_custom_command(OUTPUT "${ARVO_DX11_BYTECODE}"
        COMMAND "${Python3_EXECUTABLE}" "${ARVO_ROOT}/Build/CompileDX11.py"
            "$<TARGET_FILE:ArvoShaderCompiler>" "${ARVO_DX11_BYTECODE}" "$<CONFIG:Debug>"
        DEPENDS ArvoShaderCompiler "${ARVO_ROOT}/Build/CompileDX11.py"
        COMMENT "Compiling independent DX11 preview shaders"
        VERBATIM)
    target_sources(ArvoRenderer PRIVATE "${ARVO_DX11_BYTECODE}")
    target_include_directories(ArvoRenderer PRIVATE "${ARVO_DX11_BYTECODE_DIR}")
endif()
add_library(ArvoIntegrator STATIC
    "${ARVO_SRC}/RayTracer/Backends/CPU/MicropolygonDicing.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/ShutterSampling.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/CameraMotion.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/PathTime.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/MotionInterpolation.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/MotionBounds.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/MotionPreparation.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/MotionInstancing.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/WavelengthSampling.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/SpectralColor.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/AnalyticSpectrum.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/RgbSpectrum.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/SpectralScene.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/SpectralPreparation.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/SpectralFiltering.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/SpectralTransport.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Compute/SpectralScene.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Compute/SpectralBindings.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CUDA/SpectralScene.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CUDA/SpectralBindings.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/SpectralScene.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/SpectralBindings.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/BidirectionalPathTracing.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/MetropolisLightTransport.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/PointKdTree.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/PhotonMapping.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/ProgressivePhotonMapping.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/BidirectionalMis.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/VertexConnectionMergingMis.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/VertexConnectionMerging.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/PointBasedGlobalIllumination.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/SpatialDenoise.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/GeometryGuides.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/TemporalDenoise.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/SubsurfaceScattering.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/CameraEndpoint.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/EmitterData.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/EmitterEndpoint.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Compute/EmitterData.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Compute/PointBasedGlobalIllumination.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CUDA/EmitterData.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CUDA/PointBasedGlobalIllumination.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/EmitterData.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/PointBasedGlobalIllumination.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/VertexConnectionMergingFrame.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Fresnel.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Diffuse.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Conductor.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Dielectric.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/SurfaceScattering.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/HomogeneousMedium.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/VolumeLighting.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/LightEmission.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/SurfaceEmission.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/EmissionCatalog.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/LightSelection.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Compute/EmissionCatalog.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CUDA/EmissionCatalog.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/EmissionCatalog.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Ggx.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/ScenePacking.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/InstancingBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/MotionPreparation.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Compute/InstancingBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Compute/MotionPreparation.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CUDA/InstancingBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CUDA/MotionPreparation.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Instancing.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Compute/BvhBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CUDA/BvhBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/BvhBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CUDA/SpatialBvhBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/SpatialBvhBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/SpatialBvh.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Compute/SpatialBvhBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Bih.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Compute/BihBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CUDA/BihBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/BihBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Bvh.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/KdTree.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Select.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Compute/KdTreeBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CUDA/KdTreeBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/Metal/KdTreeBuilder.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Triangle.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Sphere.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Pinhole.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/ThinLens.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Whitted.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/PathTracing.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/DistributedRayTracing.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/DirectLighting.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/FilmMoments.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Independent.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Sequence.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Adaptive.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/AdaptiveDenoise.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Stratified.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Sobol.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/BlueNoise.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Importance.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/MultipleImportance.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/RussianRoulette.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Statistics.cpp"
    "${ARVO_SRC}/RayTracer/Backends/CPU/Surface.cpp"
    "${ARVO_SRC}/RayTracer/ComputeShaderRayTracingRHI.cpp"
    "${ARVO_SRC}/RayTracer/CpuRayTracingRHI.cpp" "${ARVO_SRC}/RayTracer/CudaRayTracingRHI.cpp"
    "${ARVO_METAL_SOURCE}" "${ARVO_SRC}/RayTracer/RayTracer.cpp" "${ARVO_SRC}/RayTracer/TracingCore.cpp")
arvo_target(ArvoIntegrator)
target_compile_definitions(ArvoIntegrator PUBLIC ARVO_RAYTRACER_STATIC)
target_link_libraries(ArvoIntegrator PUBLIC ArvoRenderer Threads::Threads)
if(WIN32)
    find_program(ARVO_DXC NAMES dxc HINTS
        "$ENV{WindowsSdkDir}/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
        "C:/Program Files (x86)/Windows Kits/10/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
        REQUIRED)
    set(ARVO_COMPUTE_SHADERS "${ARVO_SRC}/RayTracer/Backends/Compute")
    file(GLOB ARVO_COMPUTE_SOURCES CONFIGURE_DEPENDS "${ARVO_COMPUTE_SHADERS}/*.hlsl" "${ARVO_COMPUTE_SHADERS}/*.h" "${ARVO_COMPUTE_SHADERS}/*.hlsli")
    set(ARVO_COMPUTE_BYTECODE "${ARVO_ROOT}/Intermediate/Shaders/Compute/ComputeShaderBytecode.h")
    add_custom_command(OUTPUT "${ARVO_COMPUTE_BYTECODE}"
        COMMAND "${Python3_EXECUTABLE}" "${ARVO_ROOT}/Build/CompileShaders.py"
            --dxc "${ARVO_DXC}" --root "${ARVO_ROOT}" --output "${ARVO_COMPUTE_BYTECODE}"
        DEPENDS "${ARVO_ROOT}/Build/CompileShaders.py"
            ${ARVO_COMPUTE_SOURCES}
        COMMENT "Compiling independent native compute integrators"
        VERBATIM)
    target_sources(ArvoIntegrator PRIVATE "${ARVO_COMPUTE_BYTECODE}")
    target_include_directories(ArvoIntegrator PRIVATE "${ARVO_ROOT}/Intermediate/Shaders/Compute")
    set(ARVO_DXR_SHADERS "${ARVO_SRC}/RayTracer/Backends/DXR")
    file(GLOB ARVO_DXR_SOURCES CONFIGURE_DEPENDS "${ARVO_DXR_SHADERS}/*.hlsl" "${ARVO_DXR_SHADERS}/*.h" "${ARVO_DXR_SHADERS}/*.hlsli")
    set(ARVO_DXR_BYTECODE "${ARVO_ROOT}/Intermediate/Shaders/DXR/DxrShaderBytecode.h")
    add_custom_command(OUTPUT "${ARVO_DXR_BYTECODE}"
        COMMAND "${Python3_EXECUTABLE}" "${ARVO_ROOT}/Build/CompileShaders.py"
            --backend dxr --dxc "${ARVO_DXC}" --root "${ARVO_ROOT}" --output "${ARVO_DXR_BYTECODE}"
        DEPENDS "${ARVO_ROOT}/Build/CompileShaders.py"
            ${ARVO_DXR_SOURCES}
        COMMENT "Compiling independent native DXR integrators"
        VERBATIM)
    target_sources(ArvoRenderer PRIVATE "${ARVO_DXR_BYTECODE}")
    target_include_directories(ArvoRenderer PRIVATE "${ARVO_ROOT}/Intermediate/Shaders/DXR")
    find_program(ARVO_NVCC NAMES nvcc HINTS "$ENV{CUDA_PATH}/bin")
    if(ARVO_NVCC)
        set(ARVO_CUDA_SHADERS "${ARVO_SRC}/RayTracer/Backends/CUDA")
        file(GLOB ARVO_CUDA_SOURCES CONFIGURE_DEPENDS "${ARVO_CUDA_SHADERS}/*.cu" "${ARVO_CUDA_SHADERS}/*.h" "${ARVO_CUDA_SHADERS}/*.cuh")
        set(ARVO_CUDA_BYTECODE "${ARVO_ROOT}/Intermediate/Shaders/CUDA/CudaShaderBytecode.h")
        add_custom_command(OUTPUT "${ARVO_CUDA_BYTECODE}"
            COMMAND "${Python3_EXECUTABLE}" "${ARVO_ROOT}/Build/CompileShaders.py"
                --backend cuda --nvcc "${ARVO_NVCC}" --host-compiler "${CMAKE_CXX_COMPILER}"
                --root "${ARVO_ROOT}" --output "${ARVO_CUDA_BYTECODE}"
            DEPENDS "${ARVO_ROOT}/Build/CompileShaders.py"
                ${ARVO_CUDA_SOURCES}
            COMMENT "Compiling independent native CUDA integrators"
            VERBATIM)
        target_sources(ArvoIntegrator PRIVATE "${ARVO_CUDA_BYTECODE}")
        target_include_directories(ArvoIntegrator PRIVATE "${ARVO_ROOT}/Intermediate/Shaders/CUDA")
        target_compile_definitions(ArvoIntegrator PRIVATE ARVO_CUDA_KERNELS)
    else()
        message(STATUS "CUDA Toolkit not found: CUDA backend will report unavailable")
    endif()
endif()
if(APPLE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_program(ARVO_XCRUN NAMES xcrun REQUIRED)
    set(ARVO_METAL_SHADERS "${ARVO_SRC}/RayTracer/Backends/Metal")
    file(GLOB ARVO_METAL_SOURCES CONFIGURE_DEPENDS "${ARVO_METAL_SHADERS}/*.metal" "${ARVO_METAL_SHADERS}/*.h")
    # Compile native MSL before embedding it for runtime pipeline creation.
    set(ARVO_METAL_AIR_FILES)
    foreach(program Whitted PathTracing DistributedRayTracing BidirectionalPathTracing Film SpectralPrepare)
        set(air "${ARVO_ROOT}/Intermediate/Shaders/Metal/${program}.air")
        add_custom_command(OUTPUT "${air}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${ARVO_ROOT}/Intermediate/Shaders/Metal"
            COMMAND "${ARVO_XCRUN}" -sdk macosx metal -fno-fast-math -c
                "${ARVO_METAL_SHADERS}/${program}.metal" -o "${air}"
            DEPENDS ${ARVO_METAL_SOURCES}
            COMMENT "Checking native Metal ${program}"
            VERBATIM)
        list(APPEND ARVO_METAL_AIR_FILES "${air}")
    endforeach()
    foreach(program MetropolisLightTransport PhotonMapping ProgressivePhotonMapping VertexConnectionMerging PointBasedGlobalIllumination SpatialDenoise GeometryGuides TemporalDenoise AdaptiveDenoise MicropolygonDicing)
        set(air "${ARVO_ROOT}/Intermediate/Shaders/Metal/${program}.air")
        add_custom_command(OUTPUT "${air}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${ARVO_ROOT}/Intermediate/Shaders/Metal"
            COMMAND "${ARVO_XCRUN}" -sdk macosx metal -std=macos-metal2.3 -fno-fast-math -c
                "${ARVO_METAL_SHADERS}/${program}.metal" -o "${air}"
            DEPENDS ${ARVO_METAL_SOURCES}
            COMMENT "Checking native Metal ${program}"
            VERBATIM)
        list(APPEND ARVO_METAL_AIR_FILES "${air}")
    endforeach()
    add_custom_target(ArvoMetalShaders DEPENDS ${ARVO_METAL_AIR_FILES})
    add_dependencies(ArvoIntegrator ArvoMetalShaders)
    set(ARVO_METAL_HEADER "${ARVO_ROOT}/Intermediate/Shaders/Metal/MetalShaderSources.h")
    add_custom_command(OUTPUT "${ARVO_METAL_HEADER}"
        COMMAND "${Python3_EXECUTABLE}" "${ARVO_ROOT}/Build/EmbedMetalSources.py"
            --root "${ARVO_ROOT}" --output "${ARVO_METAL_HEADER}"
        DEPENDS "${ARVO_ROOT}/Build/EmbedMetalSources.py"
            ${ARVO_METAL_SOURCES}
        COMMENT "Embedding independent native Metal integrators"
        VERBATIM)
    target_sources(ArvoIntegrator PRIVATE "${ARVO_METAL_HEADER}")
    target_include_directories(ArvoIntegrator PRIVATE "${ARVO_ROOT}/Intermediate/Shaders/Metal")
    target_link_libraries(ArvoIntegrator PUBLIC "-framework Foundation" "-framework Metal")
endif()
add_library(ArvoScene STATIC
    "${ARVO_SRC}/Editor/Demos.cpp" "${ARVO_SRC}/Editor/DragonDemo.cpp" "${ARVO_SRC}/Editor/Editor.cpp"
    "${ARVO_SRC}/Editor/ObjectMotion.cpp"
    "${ARVO_SRC}/Editor/ModuleAdapters.cpp" "${ARVO_SRC}/Editor/DxrPreview.cpp" "${ARVO_SRC}/Editor/RenderAdapter.cpp"
    "${ARVO_SRC}/Editor/Resources.cpp")
arvo_target(ArvoScene)
target_link_libraries(ArvoScene PUBLIC ArvoMaths ArvoRenderer ArvoIntegrator)
target_compile_definitions(ArvoScene PUBLIC ARVO_CONTENT_ROOT="${ARVO_ROOT}/Contents" ARVO_BINARY_ROOT="${ARVO_ROOT}/Binaries")
if(WIN32)
    target_link_libraries(ArvoScene PUBLIC ole32 windowscodecs)
elseif(APPLE)
    target_link_libraries(ArvoScene PUBLIC "-framework CoreGraphics" "-framework ImageIO")
endif()
add_library(ArvoDistributed STATIC
    "${ARVO_SRC}/Distributed/LocalWorkers.cpp"
    "${ARVO_SRC}/Distributed/OfflineRenderer.cpp"
    "${ARVO_SRC}/Distributed/InstanceMotion.cpp"
    "${ARVO_SRC}/Distributed/MicropolygonTransfer.cpp"
    "${ARVO_SRC}/Distributed/Distributed.cpp" "${ARVO_SRC}/Distributed/FilmTransfer.cpp"
    "${ARVO_SRC}/Distributed/TemporalTransfer.cpp" "${ARVO_SRC}/Distributed/Http.cpp"
    "${ARVO_SRC}/Distributed/Png.cpp" "${ARVO_SRC}/Distributed/Worker.cpp")
arvo_target(ArvoDistributed)
target_link_libraries(ArvoDistributed PUBLIC ArvoScene ${CMAKE_DL_LIBS})
if(WIN32)
    target_link_libraries(ArvoDistributed PUBLIC ws2_32)
endif()
add_executable(ArvoWorker "${ARVO_SRC}/Server/WorkerMain.cpp")
arvo_target(ArvoWorker)
target_link_libraries(ArvoWorker PRIVATE ArvoDistributed)
set(ARVO_EDITOR_SOURCES "${ARVO_SRC}/Editor/Main.cpp" "${ARVO_SRC}/Lightmap/Command.cpp" "${ARVO_SRC}/Editor/EditorPresenter.cpp"
    "${ARVO_SRC}/Editor/RenderPanel.cpp" "${ARVO_SRC}/Editor/ObjectMotionPanel.cpp" "${ARVO_SRC}/Editor/Imgui.cpp")
add_executable(ArvoEditor ${ARVO_EDITOR_SOURCES})
if(WIN32)
    set_target_properties(ArvoEditor PROPERTIES WIN32_EXECUTABLE TRUE)
    target_sources(ArvoEditor PRIVATE
        "${ARVO_SRC}/Editor/EditorPresenterDX11.cpp" "${ARVO_SRC}/Editor/ImguiRenderDX11.cpp")
    target_link_libraries(ArvoEditor PRIVATE d3d11 dxgi d3dcompiler gdi32 user32 shell32)
    target_compile_definitions(ArvoEditor PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif(APPLE)
    target_sources(ArvoEditor PRIVATE
        "${ARVO_SRC}/Editor/EditorPresenterMetal.mm" "${ARVO_SRC}/Editor/MainMac.mm")
    target_link_libraries(ArvoEditor PRIVATE "-framework Cocoa" "-framework Metal" "-framework QuartzCore")
endif()
arvo_target(ArvoEditor)
target_link_libraries(ArvoEditor PRIVATE ArvoDistributed ArvoLightmapCore)
add_dependencies(ArvoEditor ArvoWorker)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ArvoEditor)

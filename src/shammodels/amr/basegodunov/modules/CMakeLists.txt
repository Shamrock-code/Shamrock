
set(Sources
    GhostZones.cpp
    AMRTree.cpp
    StencilGenerator.cpp
    AMRGraphGen.cpp
    ComputeGradient.cpp
    ComputeCellInfos.cpp
    FaceInterpolate.cpp
    ComputeFlux.cpp
    ComputeTimeDerivative.cpp
    TimeIntegrator.cpp
    ConsToPrim.cpp
    ComputeCFL.cpp
)
set(Headers
)

target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )

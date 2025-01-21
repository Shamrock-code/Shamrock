target_sources(shamrock_lib
    PRIVATE
        BasicSPHGhosts.cpp
        SPHSolverImpl.cpp
        Model.cpp
        Solver.cpp
        SPHUtilities.cpp
        SolverLog.cpp
        io/PhantomDump.cpp
        SolverConfig.cpp
        pySPHModel.cpp
        pySPHKernels.cpp
        pyPhantomDump.cpp
    PUBLIC
    )



#add_subdirectory(algs)
#add_subdirectory(physics)
#add_subdirectory(setup)
add_subdirectory(legacy)
add_subdirectory(modules)

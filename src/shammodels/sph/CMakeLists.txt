target_sources(shamrock_lib
    PRIVATE
        BasicSPHGhosts.cpp
        SPHSolverImpl.cpp
        Model.cpp
        Solver.cpp
    PUBLIC
    )



#add_subdirectory(algs)
#add_subdirectory(physics)
#add_subdirectory(setup)
add_subdirectory(legacy)
add_subdirectory(modules)
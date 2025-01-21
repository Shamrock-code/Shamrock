target_sources(shamrock_lib
    PRIVATE
        Model.cpp
        Solver.cpp
        pyAMRZeusModel.cpp
    PUBLIC
    )

    add_subdirectory(modules)

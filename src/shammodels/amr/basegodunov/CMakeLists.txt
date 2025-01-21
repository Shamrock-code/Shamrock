target_sources(shamrock_lib
    PRIVATE
        Model.cpp
        Solver.cpp
        pyAMRGodunovModel.cpp
    PUBLIC
    )

    add_subdirectory(modules)

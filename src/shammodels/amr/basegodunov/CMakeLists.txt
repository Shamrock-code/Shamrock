target_sources(shamrock_lib
    PRIVATE
        Model.cpp
        Solver.cpp
    PUBLIC
    )

    add_subdirectory(modules)
    
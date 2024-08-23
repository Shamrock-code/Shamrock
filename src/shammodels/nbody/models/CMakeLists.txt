

target_sources(shamrock
    PRIVATE
        #scheduler.cpp

        nbody_selfgrav.cpp



    PUBLIC
        #patch.hpp

        #patchdata.hpp
        #scheduler.hpp
    )


if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE


    nbody_selfgrav.cpp


    PUBLIC
        #patch.hpp
        #patchdata.hpp
        #scheduler.hpp
    )


endif()

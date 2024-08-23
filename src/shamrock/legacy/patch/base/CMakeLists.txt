


target_sources(shamrock
    PRIVATE
        #scheduler.cpp




        patchdata.cpp
        patchdata_field.cpp



    PUBLIC

        #patchdata.hpp
        #scheduler.hpp
    )




if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE

    patchdata.cpp
    patchdata_field.cpp


    PUBLIC
        #patch.hpp
        #patchdata.hpp
        #scheduler.hpp
    )


endif()




target_sources(shamrock
    PRIVATE
        #scheduler.cpp
        
        
        
        #patchdata_buffer.cpp 


        
    PUBLIC
        #patch.hpp
        
        #patchdata.hpp
        #scheduler.hpp
    )




if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE

        
    #patchdata_buffer.cpp 


    PUBLIC
        #patch.hpp
        #patchdata.hpp
        #scheduler.hpp
    )


endif()

add_subdirectory(base)
add_subdirectory(interfaces)
add_subdirectory(utility)

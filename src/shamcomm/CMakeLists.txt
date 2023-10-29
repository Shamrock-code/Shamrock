
target_sources(shamrock_lib
    PRIVATE
        worldInfo.cpp
        mpiErrorCheck.cpp
        mpiInfo.cpp

        details/CommunicationBufferImpl.cpp
    PUBLIC
    )


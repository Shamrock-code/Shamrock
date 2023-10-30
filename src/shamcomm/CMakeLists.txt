
target_sources(shamrock_lib
    PRIVATE
        worldInfo.cpp
        mpiErrorCheck.cpp
        mpiInfo.cpp
        collectives.cpp
        io.cpp

        details/CommunicationBufferImpl.cpp
    PUBLIC
    )


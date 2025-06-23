## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shambackends CXX C)

set(Sources
        src/comm/details/CommunicationBufferImpl.cpp
        src/comm/CommunicationBuffer.cpp
        src/details/BufferEventHandler.cpp
        src/details/memoryHandle.cpp
        src/details/internal_alloc.cpp
        src/USMPtrHolder.cpp
        src/DeviceBuffer.cpp
        src/Device.cpp
        src/DeviceContext.cpp
        src/DeviceQueue.cpp
        src/DeviceScheduler.cpp
        src/SyclMpiTypes.cpp
        src/EventList.cpp
        src/sysinfo.cpp
    )

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shambackends target (Shared lib)")
add_library(shambackends SHARED ${Sources})
else()
message("-- adding shambackends target (object lib)")
add_library(shambackends OBJECT ${Sources})
endif()

add_library(shambackends::shambackends ALIAS shambackends)

target_link_libraries(shambackends PUBLIC shamcomm)

target_include_directories(shambackends PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BufferEventHandler.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/container/BufferEventHandler.hpp"
#include "shamcomm/logs.hpp"
#include <random>

u32 shamalgs::gen_buf_hash() {
    constexpr u32 u32_max = 0xFFFFFFFF;
    static std::mt19937 gengine{0};
    return std::uniform_int_distribution<u32>(0, u32_max)(gengine);
}

void shamalgs::BufferEventHandler::add_read_dependancies(std::vector<sycl::event> &depends_list) {

    if (!up_to_date_events) {
        std::string err
            = get_hash_log()
              + "you want to create a event depedancy, but the event state was not updated "
                "after last event usage";

        throw shambase::make_except_with_loc<std::runtime_error>(err);
    }

    up_to_date_events = false;
    last_event_create = READ;

    depends_list.push_back(event_last_write);

    shamlog_debug_sycl_ln("[USMBuffer]", get_hash_log(), "add read dependancy");
}

void shamalgs::BufferEventHandler::add_read_write_dependancies(
    std::vector<sycl::event> &depends_list) {

    if (!up_to_date_events) {
        std::string err
            = get_hash_log()
              + "you want to create a event depedancy, but the event state was not updated "
                "after last event usage";

        throw shambase::make_except_with_loc<std::runtime_error>(err);
    }

    up_to_date_events = false;
    last_event_create = READ_WRITE;

    depends_list.push_back(event_last_write);
    for (sycl::event e : event_last_read) {
        depends_list.push_back(e);
    }
    shamlog_debug_sycl_ln("[USMBuffer]", get_hash_log(), "add read write dependancy");

    event_last_read  = {};
    event_last_write = {};

    shamlog_debug_sycl_ln("[USMBuffer]", get_hash_log(), "reset event list");
}

void shamalgs::BufferEventHandler::register_read_event(sycl::event e) {

    if (up_to_date_events) {
        std::string err
            = (get_hash_log()
               + "you are trying to register an event without having fetched one previoulsy");

        throw shambase::make_except_with_loc<std::runtime_error>(err);
    }

    if (last_event_create != READ) {
        std::string err
            = (get_hash_log()
               + "you want to register a read event but the last dependcy was not in read mode");

        throw shambase::make_except_with_loc<std::runtime_error>(err);
    }

    up_to_date_events = true;
    event_last_read.push_back(e);

    shamlog_debug_sycl_ln("[USMBuffer]", get_hash_log(), "append last read");
}

void shamalgs::BufferEventHandler::register_read_write_event(sycl::event e) {
    shamlog_debug_sycl_ln("[USMBuffer]", get_hash_log(), "set last write");
    if (up_to_date_events) {
        std::string err
            = (get_hash_log()
               + "you are trying to register an event without having fetched one previoulsy");

        throw shambase::make_except_with_loc<std::runtime_error>(err);
    }

    if (last_event_create != READ_WRITE) {
        std::string err
            = (get_hash_log()
               + "you want to register a read event but the last dependcy was not in read mode");

        throw shambase::make_except_with_loc<std::runtime_error>(err);
    }

    up_to_date_events = true;
    event_last_write  = e;
}

void shamalgs::BufferEventHandler::synchronize() {

    shamlog_debug_sycl_ln("[USMBuffer]", get_hash_log(), "synchronize");

    if (!up_to_date_events) {
        std::string err = (get_hash_log() + "the events are not up to date");

        throw shambase::make_except_with_loc<std::runtime_error>(err);
    }

    event_last_write.wait_and_throw();
    for (sycl::event e : event_last_read) {
        e.wait_and_throw();
    }

    event_last_read  = {};
    event_last_write = {};
}

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file serialize.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/serialize.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shamcomm/logs.hpp"

// Layout of the SerializeHelper is
// aligned on base 64 bits

// pre head (header length)
// header
// content

// The idea is to move pre head and the header to the host
// to avoid multiplying querries to the device

u64 extract_preahead(sham::DeviceQueue &q, sham::DeviceBuffer<u8> &storage) {

    if (storage.get_size() == 0) {
        throw shambase::make_except_with_loc<std::runtime_error>(
            ("the buffer is not allocated, the head cannot be moved"));
    }

    using Helper = shamalgs::details::SerializeHelperMember<u64>;

    u64 ret;
    {

        sham::EventList depends_list;
        auto accbufstg = storage.get_read_access(depends_list);

        auto dest   = &ret;
        auto source = accbufstg;

        auto e = q.submit(depends_list, [dest, source](sycl::handler &cgh) {
            cgh.memcpy(dest, source, sizeof(u64));
        });

        e.wait_and_throw();
        storage.complete_event_state(sycl::event{});
    }

    return ret;
}

void write_prehead(sham::DeviceQueue &q, u64 prehead, sham::DeviceBuffer<u8> &storage) {

    sham::EventList depends_list;
    auto accbufstg = storage.get_write_access(depends_list);

    auto dest   = accbufstg;
    auto source = &prehead;

    auto e = q.submit(depends_list, [dest, source](sycl::handler &cgh) {
        cgh.memcpy(dest, source, sizeof(u64));
    });

    e.wait_and_throw();
    storage.complete_event_state(sycl::event{});
}

// this is the real fix this time
// race conditions in compilers
// i will never miss those ...
#ifdef SYCL_COMP_ACPP
    #define ACPP_WAIT .wait()
#else
    #define ACPP_WAIT
#endif

std::vector<u8> extract_header(
    sham::DeviceQueue &q, sham::DeviceBuffer<u8> &storage, u64 header_size, u64 pre_head_length) {

    std::vector<u8> storage_header = std::vector<u8>(header_size);

    if (header_size > 0) {
        sycl::buffer<u8> attach(storage_header.data(), header_size);

        sham::EventList depends_list;
        auto accbufstg = storage.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&, pre_head_length](sycl::handler &cgh) {
            sycl::accessor buf_header{attach, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{header_size}, [=](sycl::item<1> id) {
                buf_header[id] = accbufstg[id + pre_head_length];
            });
        });

        e.wait_and_throw();
        storage.complete_event_state(sycl::event{});
    }

    // std::cout << "extract header" << std::endl;

    return storage_header;
}

void write_header(
    sham::DeviceQueue &q,
    sham::DeviceBuffer<u8> &storage,
    std::vector<u8> &storage_header,
    u64 header_size,
    u64 pre_head_length) {

    if (header_size > 0) {
        sycl::buffer<u8> attach(storage_header.data(), header_size);

        sham::EventList depends_list;
        auto accbufstg = storage.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&, pre_head_length](sycl::handler &cgh) {
            sycl::accessor buf_header{attach, cgh, sycl::read_only};

            cgh.parallel_for(sycl::range<1>{header_size}, [=](sycl::item<1> id) {
                accbufstg[id + pre_head_length] = buf_header[id];
            });
        });

        e.wait_and_throw();
        storage.complete_event_state(sycl::event{});
    }
    // std::cout << "write header" << std::endl;
}

u64 shamalgs::SerializeHelper::pre_head_length() {
    return shamalgs::details::serialize_byte_size<shamalgs::SerializeHelper::alignment, u64>()
        .head_size;
}

void shamalgs::SerializeHelper::allocate(SerializeSize szinfo) {
    StackEntry stack_loc{false};
    u64 bytelen = szinfo.head_size + szinfo.content_size + pre_head_length();

    storage.resize(bytelen);
    header_size = szinfo.head_size;
    storage_header.resize(header_size);

    shamlog_debug_sycl_ln("SerializeHelper", "allocate()", bytelen, header_size);

    write_prehead(dev_sched->get_queue(), szinfo.head_size, storage);
    // std::cout << "prehead write :" << szinfo.head_size << std::endl;

    head_device = pre_head_length() + header_size;
}

sham::DeviceBuffer<u8> shamalgs::SerializeHelper::finalize() {
    StackEntry stack_loc{false};

    shamlog_debug_sycl_ln("SerializeHelper", "finalize()", storage.get_size(), header_size);

    write_header(dev_sched->get_queue(), storage, storage_header, header_size, pre_head_length());

    return std::move(storage);
}

shamalgs::SerializeHelper::SerializeHelper(std::shared_ptr<sham::DeviceScheduler> _dev_sched)
    : dev_sched(std::move(_dev_sched)), storage(0, _dev_sched) {}

shamalgs::SerializeHelper::SerializeHelper(
    std::shared_ptr<sham::DeviceScheduler> _dev_sched, sham::DeviceBuffer<u8> &&input)
    : dev_sched(std::move(_dev_sched)), storage(std::forward<sham::DeviceBuffer<u8>>(input)) {

    header_size = extract_preahead(dev_sched->get_queue(), storage);

    shamlog_debug_sycl_ln(
        "SerializeHelper",
        shambase::format(
            "Init SerializeHelper from buffer\n    storage size : {},\n    header_size : {}",
            storage.get_size(),
            header_size));

    storage_header
        = extract_header(dev_sched->get_queue(), storage, header_size, pre_head_length());

    head_device = pre_head_length() + header_size;
}

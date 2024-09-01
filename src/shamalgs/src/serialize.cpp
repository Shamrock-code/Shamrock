// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file serialize.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/serialize.hpp"
#include "shamcomm/logs.hpp"

// Layout of the SerializeHelper is
// aligned on base 64 bits

// pre head (header lenght)
// header
// content

// The idea is to move pre head and the header to the host
// to avoid multiplying querries to the device

u64 extract_preahead(std::unique_ptr<sycl::buffer<u8>> &storage) {

    if (!storage) {
        throw shambase::make_except_with_loc<std::runtime_error>(
            ("the buffer is not allocated, the head cannot be moved"));
    }

    using Helper = shamalgs::details::SerializeHelperMember<u64>;

    u64 ret;
    { // using host_acc rather than anything else since other options causes addition latency
        sycl::host_accessor accbuf{*storage, sycl::read_only};
        ret = Helper::load(&accbuf[0]);
    }

    return ret;
}

void write_prehead(sycl::queue &q, u64 prehead, sycl::buffer<u8> &buf) {
    using Helper = shamalgs::details::SerializeHelperMember<u64>;
    q.submit([&, prehead](sycl::handler &cgh) {
        sycl::accessor accbuf{buf, cgh, sycl::write_only, sycl::no_init};
        cgh.single_task([=]() {
            Helper::store(&accbuf[0], prehead);
        });
    });
}

// this is the real fix this time
// race conditions in compilers
// i will never miss those ...
#ifdef SYCL_COMP_ACPP
    #define ACPP_WAIT .wait()
#else
    #define ACPP_WAIT
#endif

std::unique_ptr<std::vector<u8>> extract_header(
    sycl::queue &q,
    std::unique_ptr<sycl::buffer<u8>> &storage,
    u64 header_size,
    u64 pre_head_lenght) {

    std::unique_ptr<std::vector<u8>> storage_header
        = std::make_unique<std::vector<u8>>(header_size);

    if (header_size > 0) {
        sycl::buffer<u8> attach(storage_header->data(), header_size);

        q.submit([&, pre_head_lenght](sycl::handler &cgh) {
            sycl::accessor accbufstg{*storage, cgh, sycl::read_only};
            sycl::accessor buf_header{attach, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{header_size}, [=](sycl::item<1> id) {
                buf_header[id] = accbufstg[id + pre_head_lenght];
            });
        }) ACPP_WAIT;
    }

    // std::cout << "extract header" << std::endl;

    return storage_header;
}

void write_header(
    sycl::queue &q,
    std::unique_ptr<sycl::buffer<u8>> &storage,
    std::unique_ptr<std::vector<u8>> &storage_header,
    u64 header_size,
    u64 pre_head_lenght) {

    if (header_size > 0) {
        sycl::buffer<u8> attach(storage_header->data(), header_size);

        q.submit([&, pre_head_lenght](sycl::handler &cgh) {
            sycl::accessor accbufstg{*storage, cgh, sycl::write_only};
            sycl::accessor buf_header{attach, cgh, sycl::read_only};

            cgh.parallel_for(sycl::range<1>{header_size}, [=](sycl::item<1> id) {
                accbufstg[id + pre_head_lenght] = buf_header[id];
            });
        }) ACPP_WAIT;
    }
    // std::cout << "write header" << std::endl;
}

u64 shamalgs::SerializeHelper::pre_head_lenght() {
    return shamalgs::details::serialize_byte_size<shamalgs::SerializeHelper::alignment, u64>()
        .head_size;
}

void shamalgs::SerializeHelper::allocate(SerializeSize szinfo) {
    StackEntry stack_loc{false};
    u64 bytelen = szinfo.head_size + szinfo.content_size + pre_head_lenght();

    storage        = std::make_unique<sycl::buffer<u8>>(bytelen);
    header_size    = szinfo.head_size;
    storage_header = std::make_unique<std::vector<u8>>(header_size);

    logger::debug_sycl_ln("SerializeHelper", "allocate()", bytelen, header_size);

    write_prehead(dev_sched->get_queue().q, szinfo.head_size, *storage);
    // std::cout << "prehead write :" << szinfo.head_size << std::endl;

    head_device = pre_head_lenght() + header_size;
}

std::unique_ptr<sycl::buffer<u8>> shamalgs::SerializeHelper::finalize() {
    StackEntry stack_loc{false};

    logger::debug_sycl_ln("SerializeHelper", "finalize()", storage->size(), header_size);

    write_header(dev_sched->get_queue().q, storage, storage_header, header_size, pre_head_lenght());

    std::unique_ptr<sycl::buffer<u8>> ret;
    std::swap(ret, storage);
    return ret;
}

shamalgs::SerializeHelper::SerializeHelper(std::shared_ptr<sham::DeviceScheduler> _dev_sched)
    : dev_sched(std::move(_dev_sched)) {}

shamalgs::SerializeHelper::SerializeHelper(
    std::shared_ptr<sham::DeviceScheduler> _dev_sched, std::unique_ptr<sycl::buffer<u8>> &&input)
    : dev_sched(std::move(_dev_sched)),
      storage(std::forward<std::unique_ptr<sycl::buffer<u8>>>(input)) {

    header_size = extract_preahead(storage);

    logger::debug_sycl_ln(
        "SerializeHelper",
        shambase::format(
            "Init SerializeHelper from buffer\n    storage size : {},\n    header_size : {}",
            storage->size(),
            header_size));

    storage_header
        = extract_header(dev_sched->get_queue().q, storage, header_size, pre_head_lenght());

    head_device = pre_head_lenght() + header_size;
}

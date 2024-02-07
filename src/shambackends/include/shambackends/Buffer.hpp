// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file math.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */
#include "shambackends/queues.hpp"
#include "shamcomm/logs.hpp"

namespace sham::buffer::details {

    template<class T>
    class BufferAllocBasic {

        QueueId queue_id;
        std::unique_ptr<sycl::buffer<T>> storage;
        u64 size;

        public:
        inline QueueId get_queue_id() { return queue_id; }

        inline sycl::queue &get_queue() { queue_id.get_queue(); }

        BufferAllocBasic(u64 sz, QueueId qid) : queue_id(qid), size(sz) {
            shamcomm::logs::debug_alloc_ln("BufferImpl Basic", "alloc size =",sz);
            storage = std::make_unique<sycl::buffer<T>>(size);
        }

        ~BufferAllocBasic() noexcept {
            shamcomm::logs::debug_alloc_ln("BufferImpl Basic", "free size =",size);
        }

        inline sycl::buffer<T> &get_buf() const { return *storage; }
    };

} // namespace sham::buffer::details

namespace sham {

    template<class T, class AllocImpl = buffer::details::BufferAllocBasic<T>>
    class Buffer {

        AllocImpl impl;

        public:
        Buffer(u64 size, QueueId qid) : impl(size, qid) {}

        inline QueueId get_queue_id() { return impl.get_queue_id(); }

        inline sycl::queue &get_queue() { return impl.get_queue(); }

        inline sycl::buffer<T> &get_buf() const { return impl.get_buf(); }
    };

} // namespace sham
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

namespace sham::buffer::details {

    template<class T>
    class BufferImplBasic {

        QueueId queue_id;
        std::unique_ptr<sycl::buffer<T>> storage;

        public:
        inline QueueId get_queue_id() { return queue_id; }
        inline sycl::queue &get_queue() { queue_id.get_queue(); }


        bool is_allocated() { return bool(storage); }

        inline u64 get_capacity() { return (is_allocated()) ? storage->size() : 0; }

        void alloc(u64 min_size) { storage = std::make_unique<sycl::buffer<T>>(min_size); }

        void free() { storage.reset(); }

        inline const std::unique_ptr<sycl::buffer<T>> &get_buf() const { return storage; }

    };

} // namespace sham::buffer::details

namespace sham {

    template<class T, class Impl = buffer::details::BufferImplBasic<T>>
    class Buffer {

        Impl impl;

        public:

        inline QueueId get_queue_id() { return impl.get_queue_id(); }

        inline sycl::queue &get_queue() { return impl.get_queue(); }

        bool is_allocated() { return impl.is_allocated(); }

        inline u64 get_capacity() { return impl.get_capacity(); }

        void alloc(u64 min_size) { impl.alloc(min_size); }

        void free() { impl.free(); }

        inline const std::unique_ptr<sycl::buffer<T>> &get_buf() const { return impl.get_buf(); }

    };

} // namespace sham
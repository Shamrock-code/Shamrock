// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file queues.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/queues.hpp"
#include "shambase/memory.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////////////////////

namespace sham::impl {
    /**
     * @brief validate a sycl queue
     *
     * @param q
     */
    void check_queue_working(sycl::queue &q) {

        auto test_kernel = [](sycl::queue &q) {
            sycl::buffer<u32> b(1000);

            q.submit([&](sycl::handler &cgh) {
                sycl::accessor acc{b, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range<1>{1000}, [=](sycl::item<1> i) {
                    acc[i] = i.get_linear_id();
                });
            });

            q.wait();

            {
                sycl::host_accessor acc{b, sycl::read_only};
                if (acc[999] != 999) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "The chosen SYCL queue cannot execute a basic kernel");
                }
            }
        };

        std::exception_ptr eptr;
        try {
            test_kernel(q);
            // logger::info_ln("NodeInstance", "selected queue
            // :",q.get_device().get_info<sycl::info::device::name>()," working !");
        } catch (...) {
            eptr = std::current_exception(); // capture
        }

        if (eptr) {
            // logger::err_ln("NodeInstance", "selected queue
            // :",q.get_device().get_info<sycl::info::device::name>(),"does not function properly");
            std::rethrow_exception(eptr);
        }
    }
} // namespace sham::impl

////////////////////////////////////////////////////////////////////////////////////////////////
// Backend data
////////////////////////////////////////////////////////////////////////////////////////////////

namespace sham::queues {
    bool initialized = false;
    std::vector<QueueDetails> compute;
    std::vector<QueueDetails> alternative;
} // namespace sham::queues

////////////////////////////////////////////////////////////////////////////////////////////////
// Init close related routines
////////////////////////////////////////////////////////////////////////////////////////////////

void sham::backend::init_manual(
    std::vector<queues::QueueDetails> &&compute,
    std::vector<queues::QueueDetails> &&alternative) {

    using fT = std::vector<queues::QueueDetails>;

    if(queues::initialized){
        shambase::throw_with_loc<std::runtime_error>("queue are already initiliazed");
    }

    queues::initialized = true;
    queues::compute     = std::forward<fT>(compute);
    queues::alternative = std::forward<fT>(alternative);

    // test that all queues work fine
    for (auto &queue : queues::compute) {
        if (queue.queue) {
            impl::check_queue_working(*(queue.queue));
        }
    }
    for (auto &queue : queues::alternative) {
        if (queue.queue) {
            impl::check_queue_working(*(queue.queue));
        }
    }
}

void sham::backend::close() {
    queues::initialized = false;
    queues::compute.clear();
    queues::alternative.clear();
}

bool sham::backend::is_initialized() { return queues::initialized; }

////////////////////////////////////////////////////////////////////////////////////////////////
// User side impl
////////////////////////////////////////////////////////////////////////////////////////////////

namespace sham::impl {

    sham::queues::QueueDetails &_int_get_compute_queue(u32 id) {
        if (id >= sham::queues::compute.size()) {
            shambase::throw_with_loc<std::invalid_argument>("the id is larger than the queue list");
        }

        return sham::queues::compute[id];
    }

    sham::queues::QueueDetails &_int_get_alternative_queue(u32 id) {
        if (id >= sham::queues::alternative.size()) {
            shambase::throw_with_loc<std::invalid_argument>("the id is larger than the queue list");
        }

        return sham::queues::alternative[id];
    }

} // namespace sham::impl

sham::queues::QueueDetails &sham::get_queue_details(u32 id, sham::queues::QueueKind kind) {
    switch (kind) {
    case sham::queues::Compute: return sham::impl::_int_get_compute_queue(id);
    case sham::queues::Alternative: return sham::impl::_int_get_alternative_queue(id);
    default: shambase::throw_unimplemented();
    }
}

sycl::queue & sham::get_queue(u32 id, sham::queues::QueueKind kind){
    return shambase::get_check_ref(get_queue_details(id, kind).queue);
}
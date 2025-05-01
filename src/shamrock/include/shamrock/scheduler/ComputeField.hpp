// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ComputeField.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/DistributedData.hpp"
#include "shambase/memory.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/math/integrators.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock {

    template<class T>
    class ComputeField {

        public:
        shambase::DistributedData<PatchDataField<T>> field_data;

        inline void generate(PatchScheduler &sched, std::string name, u32 nvar = 1) {
            StackEntry stack_loc{};

            using namespace shamrock::patch;

            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                field_data.add_obj(id_patch, PatchDataField<T>(name, nvar));
                field_data.get(id_patch).resize(pdat.get_obj_cnt());
            });
        }

        inline sham::DeviceBuffer<T> &get_buf(u64 id_patch) {
            return field_data.get(id_patch).get_buf();
        }

        inline PatchDataField<T> &get_field(u64 id_patch) { return field_data.get(id_patch); }

        inline sham::DeviceBuffer<T> &get_buf_check(u64 id) { return get_buf(id); }

        template<u32 nvar>
        shambase::DistributedData<PatchDataFieldSpan<T, nvar>> get_field_span() {
            return field_data.template map<PatchDataFieldSpan<T, nvar>>(
                [&](u64 id, PatchDataField<T> &cfield) {
                    return cfield.template get_span<nvar>();
                });
        }

        shambase::DistributedData<PatchDataFieldSpan<T, shamrock::dynamic_nvar>>
        get_field_span_nvar_dynamic() {
            return field_data.template map<PatchDataFieldSpan<T, shamrock::dynamic_nvar>>(
                [&](u64 id, PatchDataField<T> &cfield) {
                    return cfield.get_field_span_nvar_dynamic();
                });
        }

        inline T compute_rank_max() {
            StackEntry stack_loc{};
            T ret = shambase::VectorProperties<T>::get_min();
            field_data.for_each([&](u64 id, PatchDataField<T> &cfield) {
                if (!cfield.is_empty()) {
                    ret = sham::max(ret, cfield.compute_max());
                }
            });

            return ret;
        }

        inline T compute_rank_min() {
            StackEntry stack_loc{};
            T ret = shambase::VectorProperties<T>::get_max();
            field_data.for_each([&](u64 id, PatchDataField<T> &cfield) {
                if (!cfield.is_empty()) {
                    ret = sham::min(ret, cfield.compute_min());
                }
            });

            return ret;
        }

        inline T compute_rank_sum() {
            StackEntry stack_loc{};
            T ret = shambase::VectorProperties<T>::get_zero();
            field_data.for_each([&](u64 id, PatchDataField<T> &cfield) {
                if (!cfield.is_empty()) {
                    ret += cfield.compute_sum();
                }
            });

            return ret;
        }

        inline T compute_rank_dot_sum() {
            StackEntry stack_loc{};
            T ret = shambase::VectorProperties<T>::get_zero();
            field_data.for_each([&](u64 id, PatchDataField<T> &cfield) {
                if (!cfield.is_empty()) {
                    ret += cfield.compute_dot_sum();
                }
            });

            return ret;
        }

        inline u32 get_nvar() {

            std::optional<u32> nvar = std::nullopt;

            field_data.for_each([&](u64 id, PatchDataField<T> &cfield) {
                u32 loc_nvar = cfield.get_nvar();
                if (!bool(nvar)) {
                    nvar = loc_nvar;
                }

                if (nvar != loc_nvar) {
                    shambase::throw_with_loc<std::runtime_error>(
                        shambase::format("mismatch in nvar excepted={} found={}", *nvar, loc_nvar));
                }
            });

            if (!bool(nvar)) {
                shambase::throw_with_loc<std::runtime_error>(
                    "you cannot querry this function when you have no fields");
            }

            return *nvar;
        }

        inline std::unique_ptr<sycl::buffer<T>> rankgather_computefield(PatchScheduler &sched) {
            StackEntry stack_loc{};

            std::unique_ptr<sycl::buffer<T>> ret;

            u64 num_obj = sched.get_rank_count();
            u64 nvar    = get_nvar();

            if (num_obj > 0) {
                ret = std::make_unique<sycl::buffer<T>>(num_obj * nvar);

                using namespace shamrock::patch;

                u64 ptr = 0; // TODO accumulate_field() in scheduler ?
                sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                    using namespace shamalgs::memory;
                    using namespace shambase;

                    if (pdat.get_obj_cnt() > 0) {
                        write_with_offset_into(
                            shamsys::instance::get_compute_scheduler().get_queue(),
                            get_check_ref(ret),
                            get_buf(id_patch),
                            ptr,
                            pdat.get_obj_cnt() * nvar);

                        ptr += pdat.get_obj_cnt() * nvar;
                    }
                });
            }

            return ret;
        }

        inline void reset() { field_data.reset(); }
    };
} // namespace shamrock

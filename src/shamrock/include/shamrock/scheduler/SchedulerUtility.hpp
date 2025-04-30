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
 * @file SchedulerUtility.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/DistributedData.hpp"
#include "shambase/memory.hpp"
#include "ComputeField.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/math/integrators.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
namespace shamrock {

    class SchedulerUtility {
        PatchScheduler &sched;

        public:
        SchedulerUtility(PatchScheduler &sched) : sched(sched) {}

        template<class T, class flt>
        inline void fields_forward_euler(u32 field_idx, u32 derfield_idx, flt dt) {
            StackEntry stack_loc{};
            using namespace shamrock::patch;
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                integrators::forward_euler(
                    shamsys::instance::get_compute_scheduler().get_queue(),
                    pdat.get_field<T>(field_idx).get_buf(),
                    pdat.get_field<T>(derfield_idx).get_buf(),
                    pdat.get_obj_cnt(),
                    dt);
            });
        }

        template<class T, class flt>
        inline void
        fields_leapfrog_corrector(u32 field_idx, u32 derfield_idx, u32 derfield_old_idx, flt hdt) {
            StackEntry stack_loc{};
            using namespace shamrock::patch;
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                integrators::leapfrog_corrector(
                    shamsys::instance::get_compute_scheduler().get_queue(),
                    pdat.get_field<T>(field_idx).get_buf(),
                    pdat.get_field<T>(derfield_idx).get_buf(),
                    pdat.get_field<T>(derfield_old_idx).get_buf(),
                    pdat.get_obj_cnt(),
                    hdt);
            });
        }

        template<class T, class flt>
        inline void fields_leapfrog_corrector(
            u32 field_idx,
            u32 derfield_idx,
            ComputeField<T> &derfield_old,
            ComputeField<flt> &field_epsilon,
            flt hdt) {
            StackEntry stack_loc{};
            using namespace shamrock::patch;
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                integrators::leapfrog_corrector(
                    shamsys::instance::get_compute_scheduler().get_queue(),
                    pdat.get_field<T>(field_idx).get_buf(),
                    pdat.get_field<T>(derfield_idx).get_buf(),
                    derfield_old.get_field(cur_p.id_patch).get_buf(),
                    field_epsilon.get_field(cur_p.id_patch).get_buf(),
                    pdat.get_obj_cnt(),
                    hdt);
            });
        }

        template<class T>
        inline void fields_apply_periodicity(u32 field_idx, std::pair<T, T> box) {
            StackEntry stack_loc{};
            using namespace shamrock::patch;
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                utilities::sycl_position_modulo(
                    shamsys::instance::get_compute_scheduler().get_queue(),
                    pdat.get_field<T>(field_idx).get_buf(),
                    pdat.get_obj_cnt(),
                    box);
            });
        }

        template<class T>
        inline void fields_apply_shearing_periodicity(
            u32 field_idx,
            u32 field_velocity,
            std::pair<T, T> box,
            i32_3 shear_base,
            i32_3 shear_dir,
            shambase::VecComponent<T> shear_value,
            shambase::VecComponent<T> shear_speed) {

            StackEntry stack_loc{};
            using namespace shamrock::patch;
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                utilities::sycl_position_sheared_modulo(
                    shamsys::instance::get_compute_scheduler().get_queue(),
                    pdat.get_field<T>(field_idx).get_buf(),
                    pdat.get_field<T>(field_velocity).get_buf(),
                    pdat.get_obj_cnt(),
                    box,
                    shear_base,
                    shear_dir,
                    shear_value,
                    shear_speed);
            });
        }

        template<class T>
        inline void fields_swap(u32 field_idx1, u32 field_idx2) {
            StackEntry stack_loc{};
            using namespace shamrock::patch;
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                utilities::swap_fields(
                    shamsys::instance::get_compute_scheduler().get_queue(),
                    pdat.get_field<T>(field_idx1).get_buf(),
                    pdat.get_field<T>(field_idx2).get_buf(),
                    pdat.get_obj_cnt());
            });
        }

        template<class T>
        inline T compute_rank_max(u32 field_idx) {
            StackEntry stack_loc{};
            using namespace shamrock::patch;
            T ret = shambase::VectorProperties<T>::get_min();
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                ret = sham::max(ret, pdat.get_field<T>(field_idx).compute_max());
            });

            return ret;
        }

        template<class T>
        inline T compute_rank_min(u32 field_idx) {
            StackEntry stack_loc{};
            using namespace shamrock::patch;
            T ret = shambase::VectorProperties<T>::get_max();
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                ret = sham::min(ret, pdat.get_field<T>(field_idx).compute_min());
            });

            return ret;
        }

        template<class T>
        inline T compute_rank_sum(u32 field_idx) {
            StackEntry stack_loc{};
            using namespace shamrock::patch;
            T ret = shambase::VectorProperties<T>::get_zero();
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                ret += pdat.get_field<T>(field_idx).compute_sum();
            });

            return ret;
        }

        template<class T>
        inline shambase::VecComponent<T> compute_rank_dot_sum(u32 field_idx) {
            StackEntry stack_loc{};
            using namespace shamrock::patch;
            shambase::VecComponent<T> ret = 0;
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                ret += pdat.get_field<T>(field_idx).compute_dot_sum();
            });

            return ret;
        }

        /**
         * @brief save a field in patchdata to a compute field
         *
         * @tparam T
         * @param field_idx
         * @param new_name
         * @return ComputeField<T>
         */
        template<class T>
        inline ComputeField<T> save_field(u32 field_idx, std::string new_name) {
            StackEntry stack_loc{};
            ComputeField<T> cfield;
            using namespace shamrock::patch;
            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                PatchDataField<T> &pdat_field = pdat.get_field<T>(field_idx);
                cfield.field_data.add_obj(id_patch, pdat_field.duplicate(new_name));
            });
            return cfield;
        }

        template<class T>
        inline ComputeField<T> save_field_custom(
            std::string new_name, std::function<PatchDataField<T> &(u64)> field_getter) {
            StackEntry stack_loc{};
            ComputeField<T> cfield;
            using namespace shamrock::patch;
            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                PatchDataField<T> &pdat_field = field_getter(id_patch);
                cfield.field_data.add_obj(id_patch, pdat_field.duplicate(new_name));
            });
            return cfield;
        }

        /**
         * @brief create a compute field and init it to zeros
         *
         * @tparam T
         * @param new_name
         * @param nvar
         * @return ComputeField<T>
         */
        template<class T>
        inline ComputeField<T> make_compute_field(std::string new_name, u32 nvar) {
            StackEntry stack_loc{};
            ComputeField<T> cfield;
            using namespace shamrock::patch;
            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                if (pdat.get_obj_cnt() == 0) {
                    return;
                }
                auto it = cfield.field_data.add_obj(
                    id_patch, PatchDataField<T>(new_name, nvar, pdat.get_obj_cnt()));

                PatchDataField<T> &ins = it->second;
                ins.field_raz();
            });
            return cfield;
        }

        /**
         * @brief create a compute field and init it to zeros, and specify size for each cases
         *
         * @tparam T
         * @param new_name
         * @param nvar
         * @return ComputeField<T>
         */
        template<class T>
        inline ComputeField<T>
        make_compute_field(std::string new_name, u32 nvar, std::function<u32(u64)> size_getter) {
            StackEntry stack_loc{};
            ComputeField<T> cfield;
            using namespace shamrock::patch;
            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                if (pdat.get_obj_cnt() == 0) {
                    return;
                }
                auto it = cfield.field_data.add_obj(
                    id_patch, PatchDataField<T>(new_name, nvar, size_getter(id_patch)));

                PatchDataField<T> &ins = it->second;
                ins.field_raz();
            });
            return cfield;
        }

        /**
         * @brief create a compute field and init it to the set value
         *
         * @tparam T
         * @param new_name
         * @param nvar
         * @param value_init
         * @return ComputeField<T>
         */
        template<class T>
        inline ComputeField<T> make_compute_field(std::string new_name, u32 nvar, T value_init) {
            StackEntry stack_loc{};
            ComputeField<T> cfield;
            using namespace shamrock::patch;
            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                auto it = cfield.field_data.add_obj(
                    id_patch, PatchDataField<T>(new_name, nvar, pdat.get_obj_cnt()));

                PatchDataField<T> &ins = it->second;
                ins.override(value_init);
            });
            return cfield;
        }
    };

} // namespace shamrock

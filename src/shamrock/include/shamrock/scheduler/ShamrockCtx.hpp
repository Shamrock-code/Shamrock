// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ShamrockCtx.hpp
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include <map>
#include <memory>
#include <tuple>
#include <vector>

class ShamAPIException : public std::exception {
    public:
    explicit ShamAPIException(const char *message) : msg_(message) {}

    explicit ShamAPIException(const std::string &message) : msg_(message) {}

    virtual ~ShamAPIException() noexcept {}

    virtual const char *what() const noexcept { return msg_.c_str(); }

    protected:
    std::string msg_;
};

class ShamrockCtx {
    public:
    shamrock::patch::PatchDataLayout pdl;
    std::unique_ptr<PatchScheduler> sched;

    inline void pdata_layout_new(size_t nlayers = 1) {
        if (sched) {
            throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
        }
        pdl.create_layers(nlayers);
    }

    // inline void pdata_layout_do_double_prec_mode(){
    //     if(sched){
    //         throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
    //     }
    //     pdl->xyz_mode = xyz64;
    // }
    //
    // inline void pdata_layout_do_single_prec_mode(){
    //    if(sched){
    //        throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
    //    }
    //    pdl->xyz_mode = xyz32;
    //}

    inline shamrock::patch::PatchDataLayerLayout &get_pdl_write(size_t layer_idx = 0) {
        if (sched) {
            throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
        }
        return pdl.get_layer_layout_ptr(layer_idx);
    }

    template<class T>
    inline void pdata_layout_add_field(std::string fname, u32 nvar, size_t layer_idx = 0) {
        if (sched) {
            throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
        }
        pdl.get_layer_layout_ptr(layer_idx).add_field<T>(fname, nvar);
    }

    inline void pdata_layout_add_field_t(
        std::string fname, u32 nvar, std::string type, size_t layer_idx = 0) {
        if (sched) {
            throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
        }
        pdl.get_layer_layout_ptr(layer_idx).add_field_t(fname, nvar, type);
    }

    inline void pdata_layout_print(size_t layer_idx = 0) {
        if (!pdl.get_layer_layout(layer_idx)) {
            throw ShamAPIException(
                shambase::format("patch data layout of index {} is not initialized", layer_idx));
        }
        std::cout << pdl.get_layer_layout_ptr(layer_idx).get_description_str() << std::endl;
    }

    inline void dump_status() {
        if (!sched) {
            throw ShamAPIException("scheduler is not initialized");
        }

        logger::raw_ln(sched->dump_status());
    }

    inline void init_sched(u64 crit_split, u64 crit_merge) {

        for (size_t layer_idx = 0; layer_idx < pdl.get_layer_count(); layer_idx++) {
            if (!pdl.get_layer_layout(layer_idx)) {
                throw ShamAPIException(
                    shambase::format(
                        "patch data layout of index {} is not initialized", layer_idx));
            }
        }

        sched = std::make_unique<PatchScheduler>(pdl, crit_split, crit_merge);
        sched->init_mpi_required_types();
    }

    inline void close_sched() { sched.reset(); }

    inline ShamrockCtx() {
        // logfiles::open_log_files();
    }

    inline ~ShamrockCtx() {
        // logfiles::close_log_files();
    }

    inline std::vector<std::unique_ptr<shamrock::patch::PatchDataLayer>> gather_data(u32 rank) {
        return sched->gather_data(rank);
    }

    inline std::vector<std::unique_ptr<shamrock::patch::PatchDataLayer>> allgather_data() {

        using namespace shamsys::instance;
        using namespace shamrock::patch;

        std::vector<std::unique_ptr<PatchDataLayer>> recv_data;

        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            if (i == shamcomm::world_rank()) {
                recv_data = sched->gather_data(i);
            } else {
                sched->gather_data(i);
            }
        }

        return recv_data;
    }

    void set_coord_domain_bound(std::tuple<f64_3, f64_3> box, size_t layer_idx = 0) {

        if (!pdl.get_layer_layout(layer_idx)) {
            throw ShamAPIException(
                shambase::format("patch data layout of index {} is not initialized", layer_idx));
        }

        if (!sched) {
            throw ShamAPIException("scheduler is not initialized");
        }

        auto [a, b] = box;

        if (pdl.get_layer_layout_ptr(layer_idx).check_main_field_type<f32_3>()) {
            auto conv_vec = [](f64_3 v) -> f32_3 {
                return {v.x(), v.y(), v.z()};
            };

            f32_3 vec0 = conv_vec(a);
            f32_3 vec1 = conv_vec(b);

            sched->set_coord_domain_bound<f32_3>(vec0, vec1);
        } else if (pdl.get_layer_layout_ptr(layer_idx).check_main_field_type<f64_3>()) {
            sched->set_coord_domain_bound<f64_3>(a, b);
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "the chosen type for the main field is not handled");
        }
    }

    inline void scheduler_step(bool do_split_merge, bool do_load_balancing) {
        if (!sched) {
            throw ShamAPIException("scheduler is not initialized");
        }
        sched->scheduler_step(do_split_merge, do_load_balancing);
    }

    inline std::vector<shamrock::patch::Patch> get_patch_list_global() {
        if (!sched) {
            throw ShamAPIException("scheduler is not initialized");
        }
        return sched->patch_list.global;
    }

    /// returns true if the scheduler is initialized
    inline bool is_scheduler_initialized() { return bool(sched); }
};

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ShamrockCtx.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
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
    std::unique_ptr<shamrock::patch::PatchDataLayout> pdl;
    std::unique_ptr<PatchScheduler> sched;

    inline void pdata_layout_new() {
        if (sched) {
            throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
        }
        pdl = std::make_unique<shamrock::patch::PatchDataLayout>();
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

    template<class T>
    inline void pdata_layout_add_field(std::string fname, u32 nvar) {
        if (sched) {
            throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
        }
        pdl->add_field<T>(fname, nvar);
    }

    inline void pdata_layout_add_field_t(std::string fname, u32 nvar, std::string type) {
        if (sched) {
            throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
        }
        pdl->add_field_t(fname, nvar, type);
    }

    inline void pdata_layout_print() {
        if (!pdl) {
            throw ShamAPIException("patch data layout is not initialized");
        }
        std::cout << pdl->get_description_str() << std::endl;
    }

    inline void dump_status() {
        if (!sched) {
            throw ShamAPIException("scheduler is not initialized");
        }

        logger::raw_ln(sched->dump_status());
    }

    inline void init_sched(u64 crit_split, u64 crit_merge) {

        if (!pdl) {
            throw ShamAPIException("patch data layout is not initialized");
        }

        sched = std::make_unique<PatchScheduler>(*pdl, crit_split, crit_merge);
        sched->init_mpi_required_types();
    }

    inline void close_sched() { sched.reset(); }

    inline ShamrockCtx() {
        // logfiles::open_log_files();
    }

    inline ~ShamrockCtx() {
        // logfiles::close_log_files();
    }

    inline std::vector<std::unique_ptr<shamrock::patch::PatchData>> gather_data(u32 rank) {
        return sched->gather_data(rank);
    }

    inline std::vector<std::unique_ptr<shamrock::patch::PatchData>> allgather_data() {

        using namespace shamsys::instance;
        using namespace shamrock::patch;

        std::vector<std::unique_ptr<PatchData>> recv_data;

        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            if (i == shamcomm::world_rank()) {
                recv_data = sched->gather_data(i);
            } else {
                sched->gather_data(i);
            }
        }

        return recv_data;
    }

    void set_coord_domain_bound(std::tuple<f64_3, f64_3> box) {

        if (!pdl) {
            throw ShamAPIException("patch data layout is not initialized");
        }

        if (!sched) {
            throw ShamAPIException("scheduler is not initialized");
        }

        auto [a, b] = box;

        if (pdl->check_main_field_type<f32_3>()) {
            auto conv_vec = [](f64_3 v) -> f32_3 {
                return {v.x(), v.y(), v.z()};
            };

            f32_3 vec0 = conv_vec(a);
            f32_3 vec1 = conv_vec(b);

            sched->set_coord_domain_bound<f32_3>(vec0, vec1);
        } else if (pdl->check_main_field_type<f64_3>()) {

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
};

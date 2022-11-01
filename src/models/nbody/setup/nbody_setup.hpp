// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


//%Impl status : Good

#pragma once

#include "aliases.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"

#include "core/patch/base/patch.hpp"
#include "core/patch/utility/serialpatchtree.hpp"

#include "models/generic/setup/generators.hpp"
#include "models/generic/setup/modifiers.hpp"

#include "models/sph/algs/smoothing_lenght.hpp"
#include <tuple>

namespace models::nbody {

    template<class flt>
    class NBodySetup {

        using vec = sycl::vec<flt,3>;

        bool periodic_mode;
        u64 part_cnt = 0;

        flt part_mass;

        public:

        void init(PatchScheduler & sched);

        void set_boundaries(bool periodic){
            periodic_mode = periodic;
        }


        inline vec get_box_dim(flt dr, u32 xcnt, u32 ycnt, u32 zcnt){
            return generic::setup::generators::get_box_dim(dr, xcnt, ycnt, zcnt);
        }

        inline std::tuple<vec,vec> get_ideal_box(flt dr, std::tuple<vec,vec> box){
            return generic::setup::generators::get_ideal_fcc_box(dr, box);
        }

        template<class T> 
        inline void set_value_in_box(PatchScheduler & sched, T val, std::string name, std::tuple<vec,vec> box){
            generic::setup::modifiers::set_value_in_box(sched, val,  name, box);
        }

        inline void pertub_eigenmode_wave(PatchScheduler &sched, std::tuple<flt,flt> ampls, vec k, flt phase){
            generic::setup::modifiers::pertub_eigenmode_wave(sched, ampls, k, phase);
        }
        

        void add_particules_fcc(PatchScheduler & sched, flt dr, std::tuple<vec,vec> box);

        inline void set_total_mass(flt tot_mass){
            u64 part = 0;
            mpi::allreduce(&part_cnt, &part, 1, mpi_type_u64, MPI_SUM, MPI_COMM_WORLD);
            part_mass = tot_mass/part;
        }

        inline flt get_part_mass(){
            return part_mass;
        }

    };


} // namespace models::sph
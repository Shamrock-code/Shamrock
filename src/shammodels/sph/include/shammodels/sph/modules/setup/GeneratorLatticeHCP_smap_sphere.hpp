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
 * @file GeneratorLatticeHCP_smap_sphere.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author David Fang (david.fang@ikmail.com)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammath/AABB.hpp"
#include "shammath/crystalLattice_smap_sphere.hpp"
#include "shammath/integrator.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <hipSYCL/sycl/libkernel/builtins.hpp>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class GeneratorLatticeHCP_smap_sphere : public ISPHSetupNode {
        using Tscal               = shambase::VecComponent<Tvec>;
        static constexpr u32 dim  = shambase::VectorProperties<Tvec>::dimension;
        using Lattice_smap_sphere = shammath::LatticeHCP_smap_sphere<Tvec>;
        using LatticeIter_smap_sphere =
            typename shammath::LatticeHCP_smap_sphere<Tvec>::IteratorDiscontinuous;
        using Config = SolverConfig<Tvec, SPHKernel>;
        using Kernel = SPHKernel<Tscal>;

        static constexpr Tscal pi = shambase::constants::pi<Tscal>;

        ShamrockCtx &context;
        Config &solver_config;
        Tscal dr;
        shammath::AABB<Tvec> box;
        Tscal rmin;
        Tscal rmax;
        std::function<Tscal(Tscal)> rhoprofile;
        std::function<Tscal(Tscal)> rhodS;

        Tscal integral_profile;

        LatticeIter_smap_sphere generator;

        static auto init_gen(
            Tscal dr,
            std::pair<Tvec, Tvec> box,
            std::function<Tscal(Tscal)> rhoprofile,
            Tscal integral_profile,
            Tscal rmin,
            Tscal rmax) {

            auto [idxs_min, idxs_max]
                = Lattice_smap_sphere::get_box_index_bounds(dr, box.first, box.second);
            u32 idx_gen = 0;
            Tvec center = (box.first + box.second) / 2;
            return LatticeIter_smap_sphere(
                dr, idxs_min, idxs_max, rhoprofile, integral_profile, rmin, rmax, center);
        };

        public:
        GeneratorLatticeHCP_smap_sphere(
            ShamrockCtx &context,
            Config &solver_config,
            Tscal dr,
            std::pair<Tvec, Tvec> box,
            std::function<Tscal(Tscal)> rhoprofile)
            : context(context), solver_config(solver_config), dr(dr), box(box),
              rhoprofile(rhoprofile), rhodS([&rhoprofile](Tscal r) {
                  return 4 * pi * sycl::pow(r, 2) * rhoprofile(r);
              }),
              rmin(0), // There's a particle at r=0 because the HCP is centered on (0,0,0)
              //   rmax(sycl::length(box.second)),
              rmax(sycl::fabs((box.second - box.first).x()) / 2.), // if it's cubic box
              integral_profile(
                  shammath::integ_riemann_sum(rmin, rmax, (rmax - rmin) / 2000, rhodS)),
              generator(init_gen(dr, box, rhoprofile, integral_profile, rmin, rmax)) {};

        bool is_done() { return generator.is_done(); }

        shamrock::patch::PatchDataLayer next_n(u32 nmax) {
            StackEntry stack_loc{};

            using namespace shamrock::patch;
            PatchScheduler &sched = shambase::get_check_ref(context.sched);

            auto has_pdat = [&]() {
                bool ret = false;
                sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
                    ret = true;
                });
                return ret;
            };

            std::vector<Tvec> pos_data;
            std::vector<Tscal> h_data;

            // Fill pos_data and h_data if the scheduler has some patchdata in this rank
            if (!is_done()) {
                u64 loc_gen_count = (has_pdat()) ? nmax : 0;

                auto gen_info = shamalgs::collective::fetch_view(loc_gen_count);

                u64 skip_start = gen_info.head_offset;
                u64 gen_cnt    = loc_gen_count;
                u64 skip_end   = gen_info.total_byte_count - loc_gen_count - gen_info.head_offset;

                shamlog_debug_ln(
                    "GeneratorLatticeHCP_smap_sphere",
                    "generate : ",
                    skip_start,
                    gen_cnt,
                    skip_end,
                    "total",
                    skip_start + gen_cnt + skip_end);

                generator.skip(skip_start);
                auto tmp = generator.next_n(gen_cnt);
                generator.skip(skip_end);

                for (Tvec r : tmp) {
                    if (Patch::is_in_patch_converted(r, box.lower, box.upper)) {
                        pos_data.push_back(r);
                    }
                    // if (Patch::is_in_patch_converted_sphere(r, box.upper)) {
                    //     pos_data.push_back(r);
                    // }
                }
            }

            Tscal mpart   = solver_config.gpart_mass;
            u64 npart     = pos_data.size();
            Tscal totmass = mpart * npart;
            Tscal hfact   = Kernel::hfactd;
            for (Tvec &r : pos_data) {
                Tscal rhoa = rhoprofile(sycl::length(r)) * totmass / integral_profile;
                Tscal h    = hfact * sycl::pow(mpart / rhoa, 1. / 3);
                h_data.push_back(h);
            }

            // Make a patchdata from pos_data
            PatchDataLayer tmp(sched.get_layout_ptr());
            if (!pos_data.empty()) {
                tmp.resize(pos_data.size());
                tmp.fields_raz();

                {
                    u32 len = pos_data.size();
                    PatchDataField<Tvec> &f
                        = tmp.get_field<Tvec>(sched.pdl().get_field_idx<Tvec>("xyz"));
                    // sycl::buffer<Tvec> buf(pos_data.data(), len);
                    f.override(pos_data, len);
                }

                {
                    u32 len = pos_data.size();
                    PatchDataField<Tscal> &f
                        = tmp.get_field<Tscal>(sched.pdl().get_field_idx<Tscal>("hpart"));
                    f.override(h_data, len);
                    // sycl::buffer<Tscal> buf(h_data.data(), len);
                    // f.override(buf, len);
                }
            }
            return tmp;
        }

        std::string get_name() { return "GeneratorLatticeHCP_smap_sphere"; }
        ISPHSetupNode_Dot get_dot_subgraph() { return ISPHSetupNode_Dot{get_name(), 0, {}}; }
    };

} // namespace shammodels::sph::modules

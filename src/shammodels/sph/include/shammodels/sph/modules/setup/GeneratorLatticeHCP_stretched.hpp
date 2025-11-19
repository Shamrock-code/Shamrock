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
 * @file GeneratorLatticeHCP_stretched.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author David Fang (david.fang@ikmail.com)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammath/AABB.hpp"
#include "shammath/crystalLattice_stretched.hpp"
#include "shammath/integrator.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec>
    class GeneratorLatticeHCP_stretched : public ISPHSetupNode {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Lattice_stretched  = shammath::LatticeHCP_stretched<Tvec>;
        using LatticeIter_stretched =
            typename shammath::LatticeHCP_stretched<Tvec>::IteratorDiscontinuous;

        static constexpr Tscal pi = shambase::constants::pi<Tscal>;

        ShamrockCtx &context;
        Tscal dr;
        shammath::AABB<Tvec> box;

        std::function<Tscal(Tscal)> rhoprofile;
        Tscal integral_profile;

        LatticeIter_stretched generator;

        static auto init_gen(
            Tscal dr,
            std::pair<Tvec, Tvec> box,
            std::function<Tscal(Tscal)> rhoprofile,
            Tscal integral_profile) {

            auto [idxs_min, idxs_max]
                = Lattice_stretched::get_box_index_bounds(dr, box.first, box.second);
            u32 idx_gen = 0;
            Tscal rmin  = 0; // There's a particle at r=0 because the HCP is centered on (0,0,0)
            Tscal rmax  = sycl::length(box.second);
            return LatticeIter_stretched(
                dr, idxs_min, idxs_max, rhoprofile, integral_profile, rmin, rmax);
        };

        public:
        GeneratorLatticeHCP_stretched(
            ShamrockCtx &context,
            Tscal dr,
            std::pair<Tvec, Tvec> box,
            std::function<Tscal(Tscal)> rhoprofile)
            : context(context), dr(dr), box(box), rhoprofile(rhoprofile),
              integral_profile(
                  shammath::integ_riemann_sum(
                      dr,
                      sycl::length(box.second),
                      (sycl::length(box.second) - dr) / 5000,
                      [&rhoprofile](Tscal r) {
                          return 4 * pi * sycl::pow(r, 2) * rhoprofile(r);
                      })),
              generator(init_gen(dr, box, rhoprofile, integral_profile)) {};

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

            // Fill pos_data if the scheduler has some patchdata in this rank
            if (!is_done()) {
                u64 loc_gen_count = (has_pdat()) ? nmax : 0;

                auto gen_info = shamalgs::collective::fetch_view(loc_gen_count);

                u64 skip_start = gen_info.head_offset;
                u64 gen_cnt    = loc_gen_count;
                u64 skip_end   = gen_info.total_byte_count - loc_gen_count - gen_info.head_offset;

                shamlog_debug_ln(
                    "GeneratorLatticeHCP_stretched",
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
                }
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
                    PatchDataField<Tscal> &f
                        = tmp.get_field<Tscal>(sched.pdl().get_field_idx<Tscal>("hpart"));
                    f.override(dr);
                }
            }
            return tmp;
        }

        std::string get_name() { return "GeneratorLatticeHCP_stretched"; }
        ISPHSetupNode_Dot get_dot_subgraph() { return ISPHSetupNode_Dot{get_name(), 0, {}}; }
    };

} // namespace shammodels::sph::modules

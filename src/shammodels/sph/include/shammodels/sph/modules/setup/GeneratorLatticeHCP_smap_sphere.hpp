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
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammath/AABB.hpp"
#include "shammath/DiscontinuousIterator.hpp"
#include "shammath/crystalLattice_smap_sphere.hpp"
#include "shammath/integrator.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <shambackends/sycl.hpp>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class GeneratorLatticeHCP_smap_sphere : public ISPHSetupNode {
        using Tscal               = shambase::VecComponent<Tvec>;
        static constexpr u32 dim  = shambase::VectorProperties<Tvec>::dimension;
        using Lattice_smap_sphere = shammath::LatticeHCP_smap_sphere<Tvec>;
        using LatticeIterDiscont_smap_sphere =
            typename shammath::LatticeHCP_smap_sphere<Tvec>::IteratorDiscontinuous;
        using Config = SolverConfig<Tvec, SPHKernel>;
        using Kernel = SPHKernel<Tscal>;

        static constexpr Tscal pi = shambase::constants::pi<Tscal>;

        ShamrockCtx &context;
        Config &solver_config;
        shammath::AABB<Tvec> box;

        Lattice_smap_sphere lattice;
        LatticeIterDiscont_smap_sphere discont_iterator;

        static auto init_lattice(
            Tscal dr, std::pair<Tvec, Tvec> box, std::function<Tscal(Tscal)> rhoprofile) {
            static constexpr u32 ngrid = 2000;

            Tscal rmin = 0; // There's a particle at r=0 because the HCP is centered on (0,0,0)
            Tscal rmax = sham::abs((box.second - box.first).x()) / 2.;
            auto [idxs_min, idxs_max]
                = Lattice_smap_sphere::get_box_index_bounds(dr, box.first, box.second);
            Tvec center = (box.first + box.second) / 2;

            // geometry dependant
            auto S = [](Tscal r) {
                return 4 * pi * sycl::pown(r, 2);
            };
            auto rhodS = [&rhoprofile, &S](Tscal r) {
                return rhoprofile(r) * S(r);
            };
            auto a_from_pos = [](Tvec pos) {
                return sycl::length(pos);
            };
            auto a_to_pos = [&a_from_pos](Tscal new_a, Tvec pos) {
                return pos * (new_a / a_from_pos(pos));
            };

            Tscal step = (rmax - rmin) / ngrid;

            Tscal integral_profile = shammath::integ_riemann_sum(rmin, rmax, step, rhodS);

            return Lattice_smap_sphere(
                dr,
                idxs_min,
                idxs_max,
                rhoprofile,
                S,
                a_from_pos,
                a_to_pos,
                integral_profile,
                rmin,
                rmax,
                center,
                step);
        }

        public:
        GeneratorLatticeHCP_smap_sphere(
            ShamrockCtx &context,
            Config &solver_config,
            Tscal dr,
            std::pair<Tvec, Tvec> box,
            std::function<Tscal(Tscal)> rhoprofile)
            : context(context), solver_config(solver_config), box(box),
              lattice(init_lattice(dr, box, rhoprofile)),
              discont_iterator(lattice.get_IteratorDiscontinuous()) {};

        bool is_done() { return discont_iterator.is_done(); }

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

                discont_iterator.skip(skip_start);
                auto tmp = discont_iterator.next_n(gen_cnt);
                discont_iterator.skip(skip_end);

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
                Tscal rhoa
                    = lattice.rhoprofile(sycl::length(r)) * totmass / lattice.integral_profile;
                Tscal h = shamrock::sph::h_rho(mpart, rhoa, hfact);
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

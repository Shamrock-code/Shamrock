// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file GeneratorLattice.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Generic SPH setup generator placing particles on a crystal lattice
 *
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammath/AABB.hpp"
#include "shammath/crystalLattice.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <type_traits>

namespace shammodels::sph::modules {

    /**
     * @brief Per-lattice properties of the generator (specialized per lattice)
     *
     * - `name` : name of the setup node
     * - `default_init_h_factor` :
     */
    template<class Lattice>
    struct LatticeProperties;

    /// Neighbours are 2 dr apart, V = (2 dr)^3 / sqrt(2) = 4 sqrt(2) dr^3, so h = 2^(5/6) dr
    template<class Tvec>
    struct LatticeProperties<shammath::LatticeHCP<Tvec>> {
        static constexpr const char *name          = "GeneratorLatticeHCP";
        static constexpr f64 default_init_h_factor = 1.7817974362806785; // 2^(5/6)
    };

    /// Same density as HCP (only the stacking differs), so h = 2^(5/6) dr
    template<class Tvec>
    struct LatticeProperties<shammath::LatticeFCC<Tvec>> {
        static constexpr const char *name          = "GeneratorLatticeFCC";
        static constexpr f64 default_init_h_factor = 1.7817974362806785; // 2^(5/6)
    };

    /// Neighbours are dr apart, V = dr^3, so h = dr
    template<class Tvec>
    struct LatticeProperties<shammath::LatticeCubic<Tvec>> {
        static constexpr const char *name          = "GeneratorLatticeCubic";
        static constexpr f64 default_init_h_factor = 1.0;
    };

    /**
     * @brief Setup node generating particles on a lattice
     *
     * @tparam Tvec position vector type
     * @tparam Lattice lattice class (shammath::LatticeHCP, LatticeFCC or LatticeCubic)
     * @tparam discontinuous if true, use the discontinuous lattice iterator
     */
    template<class Tvec, class Lattice, bool discontinuous = true>
    class GeneratorLattice : public ISPHSetupNode {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using LatticeIter        = std::conditional_t<
            discontinuous,
            typename Lattice::IteratorDiscontinuous,
            typename Lattice::Iterator>;

        ShamrockCtx &context;
        Tscal dr;
        shammath::AABB<Tvec> box;
        Tscal init_h_factor;

        LatticeIter generator;

        public:
        /// Default ratio between the initial smoothing length and dr (see LatticeProperties)
        static constexpr Tscal default_init_h_factor
            = LatticeProperties<Lattice>::default_init_h_factor;

        private:
        static auto init_gen(Tscal dr, std::pair<Tvec, Tvec> box) {

            auto [idxs_min, idxs_max] = Lattice::get_box_index_bounds(dr, box.first, box.second);
            u32 idx_gen               = 0;
            return LatticeIter(dr, idxs_min, idxs_max);
        };

        public:
        GeneratorLattice(
            ShamrockCtx &context,
            Tscal dr,
            std::pair<Tvec, Tvec> box,
            Tscal init_h_factor = default_init_h_factor)
            : context(context), dr(dr), box(box), init_h_factor(init_h_factor),
              generator(init_gen(dr, box)) {}

        bool is_done() { return generator.is_done(); }

        shamrock::patch::PatchDataLayer next_n(u32 nmax) {
            StackEntry stack_loc{};

            using namespace shamrock::patch;
            PatchScheduler &sched = shambase::get_check_ref(context.sched);

            std::vector<Tvec> pos_data;

            // Fill pos_data if the scheduler has some patchdata in this rank
            if (!is_done()) {
                u64 loc_gen_count = nmax;

                auto gen_info = shamalgs::collective::fetch_view(loc_gen_count);

                u64 skip_start = gen_info.head_offset;
                u64 gen_cnt    = loc_gen_count;
                u64 skip_end   = gen_info.total_byte_count - loc_gen_count - gen_info.head_offset;

                shamlog_debug_ln(
                    get_name(),
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
            PatchDataLayer tmp(sched.get_layout_ptr_old());
            if (!pos_data.empty()) {
                tmp.resize(pos_data.size());
                tmp.fields_raz();

                {
                    u32 len = pos_data.size();
                    PatchDataField<Tvec> &f
                        = tmp.get_field<Tvec>(sched.pdl_old().get_field_idx<Tvec>("xyz"));
                    // sycl::buffer<Tvec> buf(pos_data.data(), len);
                    f.override(pos_data, len);
                }

                {
                    PatchDataField<Tscal> &f
                        = tmp.get_field<Tscal>(sched.pdl_old().get_field_idx<Tscal>("hpart"));
                    f.override(init_h_factor * dr);
                }
            }
            return tmp;
        }

        std::string get_name() { return LatticeProperties<Lattice>::name; }
        ISPHSetupNode_Dot get_dot_subgraph() { return ISPHSetupNode_Dot{get_name(), 0, {}}; }
    };

} // namespace shammodels::sph::modules

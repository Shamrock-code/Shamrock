// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SinkParticlesAccreteDustMass.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/narrowing.hpp"
#include "shambase/string.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/SinkParticlesAccreteDustMass.hpp"
#include "shamsys/NodeInstance.hpp"
#include <experimental/mdspan>
#include <algorithm>
#include <string>

namespace shammodels::sph::modules {

    template<class Tvec>
    void SinkParticlesAccreteDustMass<Tvec>::_impl_evaluate_internal() {

        __shamrock_stack_entry();

        using mdspan_rank_2       = std::mdspan<Tscal, std::dextents<u32, 2>>;
        using const_mdspan_rank_2 = std::mdspan<const Tscal, std::dextents<u32, 2>>;

        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        auto &q        = shambase::get_check_ref(dev_sched).get_queue();

        Tscal gpart_mass = edges.gpart_mass.data;
        Tscal hfactd     = edges.hfactd.data;
        u32 ndust        = this->ndust;

        u32 sink_count = shambase::narrow_or_throw<u32>(edges.sink_positions.data.size());

        auto &sink_dust_mass_vec = edges.sink_dust_mass.data;
        if (sink_dust_mass_vec.size() != usize(sink_count) * ndust) {
            shambase::throw_with_loc<std::runtime_error>(sham::format(
                "sink_dust_mass has {} entries, expected sink_count * ndust = {} * {}",
                sink_dust_mass_vec.size(),
                sink_count,
                ndust));
        }

        // one row of ndust bins per sink : | bin0, bin1, ... | bin0, bin1, ... |
        mdspan_rank_2 sink_dust_mass(sink_dust_mass_vec.data(), sink_count, ndust);

        sham::DeviceBuffer<u32> acc_flag(0, dev_sched);

        std::vector<Tscal> s_acc_dust_mass(ndust);

        for (u32 i_sink = 0; i_sink < sink_count; i_sink++) {

            std::fill(s_acc_dust_mass.begin(), s_acc_dust_mass.end(), Tscal{0});

            edges.part_counts.indexes.for_each([&](u64 id_patch, u32 Nobj) {
                acc_flag.resize(Nobj);

                auto &acc_table = edges.sink_accretion_table.get_spans().get(id_patch);

                sham::kernel_call(
                    q,
                    sham::MultiRef{acc_table},
                    sham::MultiRef{acc_flag},
                    Nobj,
                    [i_sink](u32 id_a, const u32 *__restrict acc_table, u32 *__restrict acc_flag) {
                        acc_flag[id_a] = (acc_table[id_a] == i_sink) ? 1 : 0;
                    });

                auto id_list_accrete = shamalgs::stream_compact(dev_sched, acc_flag, Nobj);

                if (id_list_accrete.get_size() == 0) {
                    return;
                }

                u32 Naccrete = shambase::narrow_or_throw<u32>(id_list_accrete.get_size());

                auto &hpart_data = edges.hpart.get_spans().get(id_patch);
                auto &s_j_data   = edges.s_j.get_spans().get(id_patch);

                // bin-major layout, such that the accreted dust mass of a bin is a contiguous
                // range that can be summed directly
                sham::DeviceBuffer<Tscal> dust_mass_acc(ndust * Naccrete, dev_sched);

                sham::kernel_call(
                    q,
                    sham::MultiRef{hpart_data, s_j_data, id_list_accrete},
                    sham::MultiRef{dust_mass_acc},
                    Naccrete,
                    [gpart_mass, hfactd, ndust, Nobj, Naccrete](
                        u32 id_acc,
                        const Tscal *__restrict hpart,
                        const Tscal *__restrict s_j_ptr,
                        const u32 *__restrict id_list,
                        Tscal *__restrict dust_mass_acc_ptr) {
                        const_mdspan_rank_2 s_j(s_j_ptr, Nobj, ndust);
                        mdspan_rank_2 dust_mass_acc(dust_mass_acc_ptr, ndust, Naccrete);

                        u32 i_a     = id_list[id_acc];
                        Tscal rho_a = shamrock::sph::rho_h(gpart_mass, hpart[i_a], hfactd);

                        for (u32 j = 0; j < ndust; j++) {
                            Tscal s_ja = s_j(i_a, j);
                            // m_a * epsilon_j with epsilon_j = S_j^2 / rho
                            dust_mass_acc(j, id_acc) = gpart_mass * s_ja * s_ja / rho_a;
                        }
                    });

                for (u32 j = 0; j < ndust; j++) {
                    s_acc_dust_mass[j] += shamalgs::primitives::sum(
                        dev_sched, dust_mass_acc, j * Naccrete, (j + 1) * Naccrete);
                }
            });

            for (u32 j = 0; j < ndust; j++) {
                sink_dust_mass(i_sink, j)
                    += shamalgs::collective::allreduce_sum(s_acc_dust_mass[j]);
            }
        }
    }

    template<class Tvec>
    std::string SinkParticlesAccreteDustMass<Tvec>::_impl_get_tex() const {
        std::string tex = R"tex(
            SinkParticlesAccreteDustMass

            For each sink $s$ and dust bin $j$, with $\mathcal{A}_s$ the particles flagged
            for accretion onto $s$ in ${sink_accretion_table}$ and
            $\rho_a = \rho({hpart}_a)$:

            \begin{align}
            {sink_dust_mass}_{s,j} &\leftarrow {sink_dust_mass}_{s,j}
                + \sum_{a \in \mathcal{A}_s} m \frac{{s_j}_{j,a}^2}{\rho_a}
            \end{align}

            $j \in [0,{ndust})$, $m = {gpart_mass}$, $h_{{\rm fact}} = {hfactd}$.
        )tex";

        replace_edges_tex_symbols(tex);

        shambase::replace_all(tex, "{ndust}", sham::format("{}", ndust));

        return tex;
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::SinkParticlesAccreteDustMass<f64_3>;

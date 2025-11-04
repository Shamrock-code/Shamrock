// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AMRSetup.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/AMRSetup.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammodels/ramses/modules/AMRSortBlocks.hpp"
#include "shamrock/scheduler/DataInserterUtility.hpp"

template<class TgridVec>
class CellGenIterator {
    using Tgridscal          = shambase::VecComponent<TgridVec>;
    static constexpr u32 dim = shambase::VectorProperties<TgridVec>::dimension;

    private:
    u32 cnt_x;
    u32 cnt_y;
    u32 cnt_z;
    u32 cnt_xy;
    u32 tot_count;
    TgridVec sz;
    bool done;
    u32 current_iterator;

    public:
    CellGenIterator(std::array<u32, dim> cell_count, TgridVec cell_size) {

        cnt_x  = cell_count[0];
        cnt_y  = cell_count[1];
        cnt_z  = cell_count[2];
        cnt_xy = cnt_x * cnt_y;

        tot_count = cnt_x * cnt_y * cnt_z;

        current_iterator = 0;
        done             = (current_iterator == tot_count);

        sz = cell_size;
    }

    std::pair<TgridVec, TgridVec> next() {

        u32 idx = current_iterator % cnt_x;
        u32 idy = (current_iterator / cnt_x) % cnt_y;
        u32 idz = current_iterator / cnt_xy;

        u64 id_a = idx + cnt_x * idy + cnt_xy * idz;

        assert(id_a < tot_count);
        assert(idx + cnt_x * idy + cnt_xy * idz == current_iterator);

        TgridVec acc_min = sz * TgridVec{idx, idy, idz};
        TgridVec acc_max = sz * TgridVec{idx + 1, idy + 1, idz + 1};

        current_iterator++;
        if (current_iterator == tot_count) {
            done = true;
        }

        return {acc_min, acc_max};
    }

    std::vector<std::pair<TgridVec, TgridVec>> next_n(u32 n) {

        std::vector<std::pair<TgridVec, TgridVec>> res;
        for (u32 i = 0; i < n; i++) {
            if (done) {
                break;
            }
            res.push_back(next());
        }
        return res;
    }

    void skip(u32 n) { next_n(n); }

    bool is_done() { return done; }
};

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::AMRSetup<Tvec, TgridVec>::make_base_grid(
    TgridVec bmin, TgridVec cell_size, std::array<u32, dim> cell_count) {

    PatchScheduler &sched = scheduler();

    TgridVec bmax{
        bmin.x() + cell_size.x() * (cell_count[0]),
        bmin.y() + cell_size.y() * (cell_count[1]),
        bmin.z() + cell_size.z() * (cell_count[2])};

    sched.set_coord_domain_bound(bmin, bmax);

    if ((cell_size.x() != cell_size.y()) || (cell_size.y() != cell_size.z())) {
        ON_RANK_0(logger::warn_ln("AMR Grid", "your cells aren't cube"));
    }

    static_assert(dim == 3, "this is not implemented for dim != 3");

    std::array<u32, dim> patch_count;

    constexpr u32 gcd_pow2 = 1U << 31U;
    u32 gcd_cell_count;
    {
        gcd_cell_count = std::gcd(cell_count[0], cell_count[1]);
        gcd_cell_count = std::gcd(gcd_cell_count, cell_count[2]);
        gcd_cell_count = std::gcd(gcd_cell_count, gcd_pow2);
    }

    shamlog_debug_ln(
        "AMRGrid",
        "patch grid :",
        cell_count[0] / gcd_cell_count,
        cell_count[1] / gcd_cell_count,
        cell_count[2] / gcd_cell_count);

    sched.make_patch_base_grid<3>(
        {{cell_count[0] / gcd_cell_count,
          cell_count[1] / gcd_cell_count,
          cell_count[2] / gcd_cell_count}});

    sched.for_each_patch([](u64 id_patch, shamrock::patch::Patch p) {
        // TODO implement check to verify that patch a cubes of size 2^n
    });

    u32 cell_tot_count = cell_count[0] * cell_count[1] * cell_count[2];

    auto has_pdat = [&]() {
        using namespace shamrock::patch;
        bool ret = false;
        sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
            ret = true;
        });
        return ret;
    };

    CellGenIterator cell_gen_iter(cell_count, cell_size);

    auto next_n_patch = [&]() {
        u32 nmax = scheduler().crit_patch_split;

        u64 loc_gen_count = (has_pdat()) ? nmax : 0;

        auto gen_info = shamalgs::collective::fetch_view(loc_gen_count);

        u64 skip_start = gen_info.head_offset;
        u64 gen_cnt    = loc_gen_count;
        u64 skip_end   = gen_info.total_byte_count - loc_gen_count - gen_info.head_offset;

        shamlog_debug_ln(
            "AMRSetup",
            "generate : ",
            skip_start,
            gen_cnt,
            skip_end,
            "total",
            skip_start + gen_cnt + skip_end);
        cell_gen_iter.skip(skip_start);
        auto tmp_out = cell_gen_iter.next_n(gen_cnt);
        cell_gen_iter.skip(skip_end);

        std::vector<TgridVec> bmin;
        std::vector<TgridVec> bmax;

        for (auto [m, M] : tmp_out) {
            bmin.push_back(m);
            bmax.push_back(M);
        }

        // Make a patchdata from pos_data
        shamrock::patch::PatchDataLayer tmp(sched.get_layout_ptr());
        if (!tmp_out.empty()) {
            tmp.resize(tmp_out.size());
            tmp.fields_raz();

            tmp.get_field<TgridVec>(0).override(bmin, tmp_out.size());
            tmp.get_field<TgridVec>(1).override(bmax, tmp_out.size());
        }
        return tmp;
    };

    // mutli step injection routine
    shamrock::DataInserterUtility inserter(sched);
    u32 nmax = scheduler().crit_patch_split;
    while (!cell_gen_iter.is_done()) {

        shamrock::patch::PatchDataLayer pdat = next_n_patch();

        inserter.push_patch_data<TgridVec>(pdat, "cell_min", sched.crit_patch_split * 8, [&]() {
            scheduler().update_local_load_value([&](shamrock::patch::Patch p) {
                return scheduler().patch_data.owned_data.get(p.id_patch).get_obj_cnt();
            });
        });
    }

    // Ensure that the blocks are sorted in each patches
    AMRSortBlocks block_sorter(context, solver_config, storage);
    block_sorter.reorder_amr_blocks();
}

template class shammodels::basegodunov::modules::AMRSetup<f64_3, i64_3>;

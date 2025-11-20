// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GridRender.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/render/GridRender.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamtree/CompressedLeafBVH.hpp"

namespace shammodels::ramses::modules {

    // implement render slice function
    template<class Tvec, class TgridVec, class Tfield>
    sham::DeviceBuffer<Tfield> GridRender<Tvec, TgridVec, Tfield>::compute_slice(
        std::function<field_getter_t> field_getter, const sham::DeviceBuffer<Tvec> &positions) {

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        auto &q        = shambase::get_check_ref(dev_sched).get_queue();

        sham::DeviceBuffer<Tfield> ret{positions.get_size(), dev_sched};
        ret.fill(sham::VectorProperties<Tfield>::get_zero());

        using u_morton = u64;
        using RTree    = shamtree::CompressedLeafBVH<u_morton, Tvec, 3>;

        shamrock::patch::PatchCoordTransform<TgridVec> transf
            = scheduler().get_sim_box().template get_patch_transform<TgridVec>();

        scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch cur_p,
                                                    shamrock::patch::PatchDataLayer &pdat) {
            shammath::CoordRange<TgridVec> box = transf.to_obj_coord(cur_p);

            PatchDataField<TgridVec> &block_min = pdat.get_field<TgridVec>(0);
            PatchDataField<TgridVec> &block_max = pdat.get_field<TgridVec>(1);

            auto &buf_block_min = block_min.get_buf();
            auto &buf_block_max = block_max.get_buf();

            using Block    = typename Config::AMRBlock;
            u32 block_size = Block::block_size;

            u64 num_obj = block_min.get_obj_cnt();

            sham::DeviceBuffer<Tvec> pos_max_cell(num_obj * block_size, dev_sched);
            sham::DeviceBuffer<Tvec> pos_min_cell(num_obj * block_size, dev_sched);

            Tscal dxfact = solver_config.grid_coord_to_pos_fact;

            sham::kernel_call(
                q,
                sham::MultiRef{buf_block_min, buf_block_max},
                sham::MultiRef{pos_min_cell, pos_max_cell},
                num_obj,
                [dxfact](
                    u32 id_a,
                    const TgridVec *__restrict ptr_block_min,
                    const TgridVec *__restrict ptr_block_max,
                    Tvec *cell_min,
                    Tvec *cell_max) {
                    Tvec block_min = ptr_block_min[id_a].template convert<Tscal>() * dxfact;
                    Tvec block_max = ptr_block_max[id_a].template convert<Tscal>() * dxfact;

                    Tvec delta_cell = (block_max - block_min) / Block::side_size;
                    for (u32 ix = 0; ix < Block::side_size; ix++) {
                        for (u32 iy = 0; iy < Block::side_size; iy++) {
                            for (u32 iz = 0; iz < Block::side_size; iz++) {
                                u32 i          = Block::get_index({ix, iy, iz});
                                Tvec delta_val = delta_cell * Tvec{ix, iy, iz};
                                cell_min[id_a * Block::block_size + i] = block_min + delta_val;
                                cell_max[id_a * Block::block_size + i]
                                    = block_min + (delta_cell) + delta_val;
                            }
                        }
                    }
                });

            Tvec min_pos
                = shamalgs::primitives::min(dev_sched, pos_min_cell, 0, num_obj * block_size);
            Tvec max_pos
                = shamalgs::primitives::max(dev_sched, pos_max_cell, 0, num_obj * block_size);

            auto &buf_field_to_render = field_getter(cur_p, pdat);

            shammath::AABB<Tvec> tree_box(min_pos, max_pos);
            u32 reduction_level = 0;

            RTree tree = RTree::make_empty(dev_sched);
            tree.rebuild_from_position_range(pos_min_cell, pos_max_cell, tree_box, reduction_level);

            auto leaf_iterator = tree.get_object_iterator();

            sham::kernel_call(
                q,
                sham::MultiRef{
                    positions, pos_min_cell, pos_max_cell, leaf_iterator, buf_field_to_render},
                sham::MultiRef{ret},
                positions.get_size(),
                [](u32 id_a,
                   const Tvec *positions,
                   const Tvec *min_pos,
                   const Tvec *max_pos,
                   auto cell_finder,
                   const Tfield *buf_field_to_render,
                   Tfield *ret) {
                    Tvec pos_a = positions[id_a];

                    Tfield accumulator = {};

                    cell_finder.rtree_for(
                        [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                            return node_aabb.contains_asymmetric(pos_a);
                        },
                        [&](u32 id_b) {
                            shammath::AABB<Tvec> cell_b = {min_pos[id_b], max_pos[id_b]};

                            if (cell_b.contains_asymmetric(pos_a)) {
                                Tfield val_cell = buf_field_to_render[id_b];
                                accumulator += val_cell;
                            }
                        });

                    ret[id_a] += accumulator;
                });
        });

        shamalgs::collective::reduce_buffer_in_place_sum(ret, MPI_COMM_WORLD);

        return ret;
    }

    // implement render slice function
    template<class Tvec, class TgridVec, class Tfield>
    sham::DeviceBuffer<Tfield> GridRender<Tvec, TgridVec, Tfield>::compute_slice(
        std::string field_name, const sham::DeviceBuffer<Tvec> &positions) {
        auto field_source_getter
            = [&](const shamrock::patch::Patch cur_p,
                  shamrock::patch::PatchDataLayer &pdat) -> const sham::DeviceBuffer<Tfield> & {
            return pdat.get_field<Tfield>(pdat.pdl().get_field_idx<Tfield>(field_name)).get_buf();
        };
        return compute_slice(field_source_getter, positions);
    }
} // namespace shammodels::ramses::modules

using namespace shammath;
template class shammodels::ramses::modules::GridRender<f64_3, i64_3, f64>;
template class shammodels::ramses::modules::GridRender<f64_3, i64_3, f64_3>;

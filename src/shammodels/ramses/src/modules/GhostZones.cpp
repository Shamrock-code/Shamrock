// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GhostZones.cpp
 * @author Benoit Commercon (benoit.commercon@ens-lyon.fr)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedDataShared.hpp"
#include "shambase/exception.hpp"
#include "shambase/logs/loglevels.hpp"
#include "shambase/memory.hpp"
#include "shambase/print.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamalgs/numeric.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammath/AABB.hpp"
#include "shammath/CoordRange.hpp"
#include "shammodels/ramses/GhostZoneData.hpp"
#include "shammodels/ramses/modules/ExtractGhostLayer.hpp"
#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shammodels/ramses/modules/FindGhostLayerIndices.hpp"
#include "shammodels/ramses/modules/FuseGhostLayer.hpp"
#include "shammodels/ramses/modules/GhostZones.hpp"
#include "shammodels/ramses/modules/TransformGhostLayer.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/solvergraph/CopyPatchDataLayerFields.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/ExchangeGhostLayer.hpp"
#include "shamrock/solvergraph/ExtractCounts.hpp"
#include "shamrock/solvergraph/GetFieldRefFromLayer.hpp"
#include "shamrock/solvergraph/ITDataEdge.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/PatchDataLayerEdge.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <functional>
#include <stdexcept>
#include <vector>

namespace shammodels::basegodunov::modules {
    /**
     * @brief find interfaces corresponding to shared surface between domains
     *
     * @tparam Tvec
     * @tparam TgridVec
     */
    template<class Tvec, class TgridVec>
    auto find_interfaces(PatchScheduler &sched, SerialPatchTree<TgridVec> &sptree) {

        using namespace shamrock::patch;
        using namespace shammath;

        using GZData              = shammodels::basegodunov::GhostZonesData<Tvec, TgridVec>;
        static constexpr u32 dim  = shambase::VectorProperties<Tvec>::dimension;
        using InterfaceBuildInfos = typename GZData::InterfaceBuildInfos;
        using GeneratorMap        = typename GZData::GeneratorMap;

        StackEntry stack_loc{};

        i32 repetition_x = 1;
        i32 repetition_y = 1;
        i32 repetition_z = 1;

        GeneratorMap results;

        shamrock::patch::SimulationBoxInfo &sim_box = sched.get_sim_box();

        PatchCoordTransform<TgridVec> patch_coord_transf = sim_box.get_patch_transform<TgridVec>();
        TgridVec bsize = sim_box.get_bounding_box_size<TgridVec>();

        for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
            for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
                for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {

                    // sender translation
                    TgridVec periodic_offset
                        = TgridVec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                    sched.for_each_local_patch([&](const Patch psender) {
                        CoordRange<TgridVec> sender_bsize
                            = patch_coord_transf.to_obj_coord(psender);
                        CoordRange<TgridVec> sender_bsize_off
                            = sender_bsize.add_offset(periodic_offset);

                        shammath::AABB<TgridVec> sender_bsize_off_aabb{
                            sender_bsize_off.lower, sender_bsize_off.upper};

                        using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

                        shamlog_debug_sycl_ln(
                            "AMR:interf",
                            "find_interfaces -",
                            psender.id_patch,
                            sender_bsize_off_aabb.lower,
                            sender_bsize_off_aabb.upper);

                        sptree.host_for_each_leafs(
                            [&](u64 tree_id, PtNode n) {
                                shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};

                                bool result
                                    = tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();

                                return result;
                            },
                            [&](u64 id_found, PtNode n) {
                                if ((id_found == psender.id_patch) && (xoff == 0) && (yoff == 0)
                                    && (zoff == 0)) {
                                    return;
                                }

                                InterfaceBuildInfos ret{
                                    periodic_offset,
                                    {xoff, yoff, zoff},
                                    shammath::AABB<TgridVec>{
                                        n.box_min - periodic_offset, n.box_max - periodic_offset}};

                                results.add_obj(psender.id_patch, id_found, std::move(ret));
                            });
                    });
                }
            }
        }

        return results;
    }
} // namespace shammodels::basegodunov::modules

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::GhostZones<Tvec, TgridVec>::build_ghost_cache() {

    StackEntry stack_loc{};

    using GZData = GhostZonesData<Tvec, TgridVec>;

    storage.ghost_zone_infos.set(GZData{});
    GZData &gen_ghost = storage.ghost_zone_infos.get();

    using InterfaceBuildInfos = typename GZData::InterfaceBuildInfos;
    using InterfaceIdTable    = typename GZData::InterfaceIdTable;

    FindGhostLayerCandidates<TgridVec> find_ghost_layer_candidates(
        GhostLayerGenMode{GhostType::Periodic, GhostType::Periodic, GhostType::Periodic});
    find_ghost_layer_candidates.set_edges(
        storage.local_patch_ids,
        storage.sim_box_edge,
        storage.sptree_edge,
        storage.global_patch_boxes_edge,
        storage.ghost_layers_candidates_edge);
    find_ghost_layer_candidates.evaluate();

    FindGhostLayerIndices<TgridVec> find_ghost_layer_indices(
        GhostLayerGenMode{GhostType::Periodic, GhostType::Periodic, GhostType::Periodic});
    find_ghost_layer_indices.set_edges(
        storage.sim_box_edge,
        storage.source_patches,
        storage.ghost_layers_candidates_edge,
        storage.global_patch_boxes_edge,
        storage.idx_in_ghost);
    find_ghost_layer_indices.evaluate();
}

template<class Tvec, class TgridVec>
shambase::DistributedDataShared<shamrock::patch::PatchDataLayer>
shammodels::basegodunov::modules::GhostZones<Tvec, TgridVec>::communicate_pdat(
    const std::shared_ptr<shamrock::patch::PatchDataLayerLayout> &ghost_layout_ptr,
    shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> &&interf) {
    StackEntry stack_loc{};

    shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> recv_dat;

    shamalgs::collective::serialize_sparse_comm<shamrock::patch::PatchDataLayer>(
        shamsys::instance::get_compute_scheduler_ptr(),
        std::forward<shambase::DistributedDataShared<shamrock::patch::PatchDataLayer>>(interf),
        recv_dat,
        [&](u64 id) {
            return scheduler().get_patch_rank_owner(id);
        },
        [](shamrock::patch::PatchDataLayer &pdat) {
            shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
            ser.allocate(pdat.serialize_buf_byte_size());
            pdat.serialize_buf(ser);
            return ser.finalize();
        },
        [&](sham::DeviceBuffer<u8> &&buf) {
            // exchange the buffer held by the distrib data and give it to the serializer
            shamalgs::SerializeHelper ser(
                shamsys::instance::get_compute_scheduler_ptr(),
                std::forward<sham::DeviceBuffer<u8>>(buf));
            return shamrock::patch::PatchDataLayer::deserialize_buf(ser, ghost_layout_ptr);
        });

    return recv_dat;
}

template<class Tvec, class TgridVec>
template<class T>
shambase::DistributedDataShared<PatchDataField<T>>
shammodels::basegodunov::modules::GhostZones<Tvec, TgridVec>::communicate_pdat_field(
    shambase::DistributedDataShared<PatchDataField<T>> &&interf) {
    StackEntry stack_loc{};

    shambase::DistributedDataShared<PatchDataField<T>> recv_dat;

    shamalgs::collective::serialize_sparse_comm<PatchDataField<T>>(
        shamsys::instance::get_compute_scheduler_ptr(),
        std::forward<shambase::DistributedDataShared<PatchDataField<T>>>(interf),
        recv_dat,
        [&](u64 id) {
            return scheduler().get_patch_rank_owner(id);
        },
        [](PatchDataField<T> &pdat) {
            shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
            ser.allocate(pdat.serialize_full_byte_size());
            pdat.serialize_buf(ser);
            return ser.finalize();
        },
        [&](sham::DeviceBuffer<u8> &&buf) {
            // exchange the buffer held by the distrib data and give it to the serializer
            shamalgs::SerializeHelper ser(
                shamsys::instance::get_compute_scheduler_ptr(),
                std::forward<sham::DeviceBuffer<u8>>(buf));
            return PatchDataField<T>::deserialize_full(ser);
        });

    return recv_dat;
}

template<class Tvec, class TgridVec>
template<class T, class Tmerged>
shambase::DistributedData<Tmerged>
shammodels::basegodunov::modules::GhostZones<Tvec, TgridVec>::merge_native(
    shambase::DistributedDataShared<T> &&interfs,
    std::function<Tmerged(const shamrock::patch::Patch, shamrock::patch::PatchDataLayer &pdat)>
        init,
    std::function<void(Tmerged &, T &)> appender) {

    StackEntry stack_loc{};

    shambase::DistributedData<Tmerged> merge_f;

    scheduler().for_each_patchdata_nonempty(
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
            Tmerged tmp_merge = init(p, pdat);

            interfs.for_each([&](u64 sender, u64 receiver, T &interface) {
                if (receiver == p.id_patch) {
                    appender(tmp_merge, interface);
                }
            });

            merge_f.add_obj(p.id_patch, std::move(tmp_merge));
        });

    return merge_f;
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::GhostZones<Tvec, TgridVec>::exchange_ghost() {}

template<class Tvec, class TgridVec>
template<class T>
shamrock::ComputeField<T>
shammodels::basegodunov::modules::GhostZones<Tvec, TgridVec>::exchange_compute_field(
    shamrock::ComputeField<T> &in) {

    StackEntry stack_loc{};

    shambase::Timer timer_interf;
    timer_interf.start();

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    using GZData              = GhostZonesData<Tvec, TgridVec>;
    using InterfaceBuildInfos = typename GZData::InterfaceBuildInfos;
    using InterfaceIdTable    = typename GZData::InterfaceIdTable;

    using AMRBlock = typename Config::AMRBlock;

    // generate send buffers
    GZData &gen_ghost = storage.ghost_zone_infos.get();
    auto pdat_interf  = gen_ghost.template build_interface_native<PatchDataField<T>>(
        [&](u64 sender, u64, InterfaceBuildInfos binfo, sycl::buffer<u32> &buf_idx, u32 cnt) {
            PatchDataField<T> &sender_patch = in.get_field(sender);

            PatchDataField<T> pdat(sender_patch.get_name(), sender_patch.get_nvar(), cnt);

            return pdat;
        });

    // communicate buffers
    shambase::DistributedDataShared<PatchDataField<T>> interf_pdat
        = communicate_pdat_field<T>(std::move(pdat_interf));

    std::map<u64, u64> sz_interf_map;
    interf_pdat.for_each([&](u64 s, u64 r, PatchDataField<T> &pdat_interf) {
        sz_interf_map[r] += pdat_interf.get_obj_cnt();
    });

    ComputeField<T> out;
    scheduler().for_each_patchdata_nonempty(
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
            PatchDataField<T> &receiver_patch = in.get_field(p.id_patch);

            PatchDataField<T> new_pdat(receiver_patch);

            interf_pdat.for_each([&](u64 sender, u64 receiver, PatchDataField<T> &interface) {
                if (receiver == p.id_patch) {
                    new_pdat.insert(interface);
                }
            });

            out.field_data.add_obj(p.id_patch, std::move(new_pdat));
        });

    timer_interf.end();
    storage.timings_details.interface += timer_interf.elasped_sec();
    return out;
}

// doxygen does not have a clue of what is happenning here
// like ... come on ...
#ifndef DOXYGEN
namespace shammodels::basegodunov::modules {

    /// Explicit instanciation of the GhostZones class to exchange
    /// compute fields of f64_8
    template class GhostZones<f64_3, i64_3>;
    template shamrock::ComputeField<f64_8>
    GhostZones<f64_3, i64_3>::exchange_compute_field<f64_8>(shamrock::ComputeField<f64_8> &in);

    /// Explicit instanciation of the GhostZones class to communicate
    /// compute fields of f64_8
    template shambase::DistributedDataShared<PatchDataField<f64_8>>
    GhostZones<f64_3, i64_3>::communicate_pdat_field<f64_8>(
        shambase::DistributedDataShared<PatchDataField<f64_8>> &&interf);

} // namespace shammodels::basegodunov::modules
#endif

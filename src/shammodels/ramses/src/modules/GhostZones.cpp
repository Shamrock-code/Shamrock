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

    // get ids of cells that will be on the surface of another patch.
    // for cells corresponding to fixed boundary they will be generated after the exhange
    // and appended to the interface list a posteriori

    gen_ghost.ghost_gen_infos
        = find_interfaces<Tvec, TgridVec>(scheduler(), storage.serial_patch_tree.get());

    using InterfaceBuildInfos = typename GZData::InterfaceBuildInfos;
    using InterfaceIdTable    = typename GZData::InterfaceIdTable;

    // if(logger::log_debug);
    gen_ghost.ghost_gen_infos.for_each([&](u64 sender, u64 receiver, InterfaceBuildInfos &build) {
        std::string log;

        log = shambase::format(
            "{} -> {} : off = {}, {} -> {}",
            sender,
            receiver,
            build.offset,
            build.volume_target.lower,
            build.volume_target.upper);

        shamlog_debug_ln("AMRgodunov", log);
    });

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    gen_ghost.ghost_gen_infos.for_each([&](u64 sender, u64 receiver, InterfaceBuildInfos &build) {
        shamrock::patch::PatchDataLayer &src = scheduler().patch_data.get_pdat(sender);

        sycl::buffer<u32> is_in_interf{src.get_obj_cnt()};

        sham::EventList depends_list;

        auto cell_min = src.get_field_buf_ref<TgridVec>(0).get_read_access(depends_list);
        auto cell_max = src.get_field_buf_ref<TgridVec>(1).get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            sycl::accessor flag{is_in_interf, cgh, sycl::write_only, sycl::no_init};

            shammath::AABB<TgridVec> check_volume = build.volume_target;

            shambase::parallel_for(cgh, src.get_obj_cnt(), "check if in interf", [=](u32 id_a) {
                flag[id_a] = shammath::AABB<TgridVec>(cell_min[id_a], cell_max[id_a])
                                 .get_intersect(check_volume)
                                 .is_not_empty();
            });
        });

        src.get_field_buf_ref<TgridVec>(0).complete_event_state(e);
        src.get_field_buf_ref<TgridVec>(1).complete_event_state(e);

        auto resut = shamalgs::numeric::stream_compact(q.q, is_in_interf, src.get_obj_cnt());
        f64 ratio  = f64(std::get<1>(resut)) / f64(src.get_obj_cnt());

        std::string s = shambase::format(
            "{} -> {} : off = {}, test volume = {} -> {}",
            sender,
            receiver,
            build.offset,
            build.volume_target.lower,
            build.volume_target.upper);
        s += shambase::format("\n    found N = {}, ratio = {} %", std::get<1>(resut), ratio);

        shamlog_debug_ln("AMR interf", s);

        std::unique_ptr<sycl::buffer<u32>> ids
            = std::make_unique<sycl::buffer<u32>>(shambase::extract_value(std::get<0>(resut)));

        gen_ghost.ghost_id_build_map.add_obj(
            sender, receiver, InterfaceIdTable{build, std::move(ids), ratio});
    });

    gen_ghost.ghost_id_build_map.for_each([&](u64 sender, u64 receiver, InterfaceIdTable &build) {
        storage.ghost_layers_candidates_edge->values.add_obj(
            sender,
            receiver,
            GhostLayerCandidateInfos{
                i32(build.build_infos.periodicity_index[0]),
                i32(build.build_infos.periodicity_index[1]),
                i32(build.build_infos.periodicity_index[2]),
            });
    });

#if true
    // Compare the new one with the old one

    std::shared_ptr<shamrock::solvergraph::SerialPatchTreeRefEdge<TgridVec>> sptree_edge
        = std::make_shared<shamrock::solvergraph::SerialPatchTreeRefEdge<TgridVec>>("", "");
    sptree_edge->patch_tree = std::ref(storage.serial_patch_tree.get());

    auto &sim_box_edge = storage.sim_box_edge;

    std::shared_ptr<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>
        global_patch_boxes_edge
        = std::make_shared<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>(
            "global_patch_boxes", "global_patch_boxes");
    {
        auto &sim_box = scheduler().get_sim_box();
        auto transf   = sim_box.template get_patch_transform<TgridVec>();

        scheduler().for_each_global_patch([&](const shamrock::patch::Patch p) {
            auto pbounds = transf.to_obj_coord(p);
            global_patch_boxes_edge->values.add_obj(
                p.id_patch, shammath::AABB<TgridVec>{pbounds.lower, pbounds.upper});
        });
    }

    std::shared_ptr<shamrock::solvergraph::ITDataEdge<std::vector<u64>>> local_patch_ids
        = std::make_shared<shamrock::solvergraph::ITDataEdge<std::vector<u64>>>("", "");
    {
        auto &sim_box = scheduler().get_sim_box();
        auto transf   = sim_box.template get_patch_transform<TgridVec>();

        scheduler().for_each_local_patch([&](const shamrock::patch::Patch p) {
            local_patch_ids->data.push_back(p.id_patch);
        });
    }

    std::shared_ptr<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>
        ghost_layers_candidates_edge2
        = std::make_shared<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>(
            "ghost_layers_candidates", "ghost_layers_candidates");

    FindGhostLayerCandidates<TgridVec> find_ghost_layer_candidates(
        GhostLayerGenMode{GhostType::Periodic, GhostType::Periodic, GhostType::Periodic});
    find_ghost_layer_candidates.set_edges(
        local_patch_ids,
        sim_box_edge,
        sptree_edge,
        global_patch_boxes_edge,
        ghost_layers_candidates_edge2);
    find_ghost_layer_candidates.evaluate();

    {
        auto &ghost_layers_candidates2 = ghost_layers_candidates_edge2->values;
        auto &ghost_layers_candidates  = storage.ghost_layers_candidates_edge->values;

        std::vector<std::tuple<u64, u64, GhostLayerCandidateInfos>> ghost_layers_candidates_vec1;
        std::vector<std::tuple<u64, u64, GhostLayerCandidateInfos>> ghost_layers_candidates_vec2;

        ghost_layers_candidates.for_each(
            [&](u64 sender, u64 receiver, GhostLayerCandidateInfos &info) {
                ghost_layers_candidates_vec1.push_back(std::make_tuple(sender, receiver, info));
            });

        ghost_layers_candidates2.for_each(
            [&](u64 sender, u64 receiver, GhostLayerCandidateInfos &info) {
                ghost_layers_candidates_vec2.push_back(std::make_tuple(sender, receiver, info));
            });

        if (ghost_layers_candidates_vec1.size() != ghost_layers_candidates_vec2.size()) {
            shambase::throw_with_loc<std::runtime_error>(shambase::format(
                "ghost_layers_candidates sizes are different, {} != {}",
                ghost_layers_candidates_vec1.size(),
                ghost_layers_candidates_vec2.size()));
        }

        for (u32 i = 0; i < ghost_layers_candidates_vec1.size(); i++) {
            auto &[sender, receiver, info]    = ghost_layers_candidates_vec1[i];
            auto &[sender2, receiver2, info2] = ghost_layers_candidates_vec2[i];

            shamlog_normal_ln(
                "tmp",
                "case 1 sender = {}, receiver = {}, info = {} {} {}",
                sender,
                receiver,
                info.xoff,
                info.yoff,
                info.zoff);
            shamlog_normal_ln(
                "tmp",
                "case 2 sender = {}, receiver = {}, info = {} {} {}",
                sender2,
                receiver2,
                info2.xoff,
                info2.yoff,
                info2.zoff);

            if (sender != sender2) {
                shambase::throw_with_loc<std::runtime_error>("sender are different");
            }

            if (receiver != receiver2) {
                shambase::throw_with_loc<std::runtime_error>("receiver are different");
            }

            if (info.xoff != info2.xoff) {
                shambase::throw_with_loc<std::runtime_error>("xoff are different");
            }

            if (info.yoff != info2.yoff) {
                shambase::throw_with_loc<std::runtime_error>("yoff are different");
            }

            if (info.zoff != info2.zoff) {
                shambase::throw_with_loc<std::runtime_error>("zoff are different");
            }
        }
    }

#endif

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    gen_ghost.ghost_id_build_map.for_each(
        [&](u64 sender, u64 receiver, InterfaceIdTable &build_table) {
            auto buf = sham::DeviceBuffer<u32>(build_table.ids_interf->size(), sched);
            buf.copy_from_sycl_buffer(shambase::get_check_ref(build_table.ids_interf));
            storage.idx_in_ghost->buffers.add_obj(sender, receiver, std::move(buf));
        });

#if true

    auto idx_in_ghost2 = std::make_shared<shamrock::solvergraph::DDSharedBuffers<u32>>("", "");

    FindGhostLayerIndices<TgridVec> find_ghost_layer_indices(
        GhostLayerGenMode{GhostType::Periodic, GhostType::Periodic, GhostType::Periodic});
    find_ghost_layer_indices.set_edges(
        sim_box_edge,
        storage.source_patches,
        storage.ghost_layers_candidates_edge,
        global_patch_boxes_edge,
        idx_in_ghost2);
    find_ghost_layer_indices.evaluate();

    {
        auto &_idx_in_ghost2 = idx_in_ghost2->buffers;
        auto &_idx_in_ghost  = storage.idx_in_ghost->buffers;

        using vec_t
            = std::vector<std::tuple<u64, u64, std::reference_wrapper<sham::DeviceBuffer<u32>>>>;

        vec_t ghost_layers_candidates_vec1;
        vec_t ghost_layers_candidates_vec2;

        _idx_in_ghost.for_each([&](u64 sender, u64 receiver, sham::DeviceBuffer<u32> &idx) {
            ghost_layers_candidates_vec1.push_back(
                std::make_tuple(sender, receiver, std::ref(idx)));
        });

        _idx_in_ghost2.for_each([&](u64 sender, u64 receiver, sham::DeviceBuffer<u32> &idx) {
            ghost_layers_candidates_vec2.push_back(
                std::make_tuple(sender, receiver, std::ref(idx)));
        });

        if (ghost_layers_candidates_vec1.size() != ghost_layers_candidates_vec2.size()) {
            shambase::throw_with_loc<std::runtime_error>(
                "ghost_layers_candidates sizes are different");
        }

        for (u32 i = 0; i < ghost_layers_candidates_vec1.size(); i++) {
            auto &[sender, receiver, idx]    = ghost_layers_candidates_vec1[i];
            auto &[sender2, receiver2, idx2] = ghost_layers_candidates_vec2[i];

            shamlog_normal_ln(
                "tmp",
                shambase::format(
                    "case 1 sender = {}, receiver = {}, idx = {}",
                    sender,
                    receiver,
                    idx.get().get_size()));
            shamlog_normal_ln(
                "tmp",
                shambase::format(
                    "case 2 sender = {}, receiver = {}, idx = {}, idx1==idx2 = {}",
                    sender2,
                    receiver2,
                    idx2.get().get_size(),
                    idx.get().copy_to_stdvec() == idx2.get().copy_to_stdvec()));

            if (sender != sender2) {
                shambase::throw_with_loc<std::runtime_error>("sender are different");
            }

            if (receiver != receiver2) {
                shambase::throw_with_loc<std::runtime_error>("receiver are different");
            }

            if (idx.get().get_size() != idx2.get().get_size()) {
                shambase::throw_with_loc<std::runtime_error>("idx sizes are different");
            }

            std::vector<u32> idx_vec1 = idx.get().copy_to_stdvec();
            std::vector<u32> idx_vec2 = idx2.get().copy_to_stdvec();

            if (idx_vec1 != idx_vec2) {
                shambase::throw_with_loc<std::runtime_error>("idx are different");
            }
        }
    }

#endif
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

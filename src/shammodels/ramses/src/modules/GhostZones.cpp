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
#include "shambase/memory.hpp"
#include "shambase/print.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamalgs/numeric.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammath/AABB.hpp"
#include "shammath/CoordRange.hpp"
#include "shammodels/ramses/GhostZoneData.hpp"
#include "shammodels/ramses/modules/ExchangeGhostLayer.hpp"
#include "shammodels/ramses/modules/ExtractGhostLayer.hpp"
#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shammodels/ramses/modules/FuseGhostLayer.hpp"
#include "shammodels/ramses/modules/GhostZones.hpp"
#include "shammodels/ramses/modules/TransformGhostLayer.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/solvergraph/CopyPatchDataLayerFields.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/PatchDataLayerEdge.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"

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
void shammodels::basegodunov::modules::GhostZones<Tvec, TgridVec>::exchange_ghost() {

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

    auto &ghost_layout_ptr = storage.ghost_layout;
    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout);
    u32 icell_min_interf = ghost_layout.get_field_idx<TgridVec>("cell_min");
    u32 icell_max_interf = ghost_layout.get_field_idx<TgridVec>("cell_max");
    u32 irho_interf      = ghost_layout.get_field_idx<Tscal>("rho");
    u32 irhoetot_interf  = ghost_layout.get_field_idx<Tscal>("rhoetot");
    u32 irhovel_interf   = ghost_layout.get_field_idx<Tvec>("rhovel");

    u32 irho_d_interf, irhovel_d_interf, iphi_interf, irho_gas_pscal_interf;
    if (solver_config.is_dust_on()) {
        irho_d_interf    = ghost_layout.get_field_idx<Tscal>("rho_dust");
        irhovel_d_interf = ghost_layout.get_field_idx<Tvec>("rhovel_dust");
    }

    if (solver_config.is_gravity_on()) {
        iphi_interf = ghost_layout.get_field_idx<Tscal>("phi");
    }

    if (solver_config.is_gas_passive_scalar_on()) {
        irho_gas_pscal_interf = ghost_layout.get_field_idx<Tscal>("rho_gas_pscal");
    }

    // load layout info
    PatchDataLayerLayout &pdl = scheduler().pdl();

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");

    u32 irho_d, irhovel_d, iphi, irho_gas_pscal;
    if (solver_config.is_dust_on()) {
        irho_d    = pdl.get_field_idx<Tscal>("rho_dust");
        irhovel_d = pdl.get_field_idx<Tvec>("rhovel_dust");
    }

    if (solver_config.is_gravity_on()) {
        iphi = pdl.get_field_idx<Tscal>("phi");
    }

    if (solver_config.is_gas_passive_scalar_on()) {
        irho_gas_pscal = pdl.get_field_idx<Tscal>("rho_gas_pscal");
    }

    GZData &gen_ghost = storage.ghost_zone_infos.get();

    // ----------------------------------------------------------------------------------------
    // load the sim box into an edge
    shamrock::patch::SimulationBoxInfo &sim_box      = scheduler().get_sim_box();
    PatchCoordTransform<TgridVec> patch_coord_transf = sim_box.get_patch_transform<TgridVec>();
    auto [bmin, bmax]                                = sim_box.get_bounding_box<TgridVec>();

    std::shared_ptr<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>> sim_box_edge
        = std::make_shared<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>>(
            "sim_box", "sim_box");
    sim_box_edge->value = shammath::AABB<TgridVec>(bmin, bmax);
    // ----------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------
    std::shared_ptr<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>
        ghost_layers_candidates_edge
        = std::make_shared<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>(
            "ghost_layers_candidates", "ghost_layers_candidates");
    gen_ghost.ghost_id_build_map.for_each([&](u64 sender, u64 receiver, InterfaceIdTable &build) {
        ghost_layers_candidates_edge->values.add_obj(
            sender,
            receiver,
            GhostLayerCandidateInfos{
                i32(build.build_infos.periodicity_index[0]),
                i32(build.build_infos.periodicity_index[1]),
                i32(build.build_infos.periodicity_index[2]),
            });
    });
    // ----------------------------------------------------------------------------------------

    auto compare_paving = [&](shambase::DistributedDataShared<InterfaceIdTable> &build_table) {
        auto paving_function = get_paving(
            GhostLayerGenMode{GhostType::Periodic, GhostType::Periodic, GhostType::Periodic},
            sim_box_edge->value);

        build_table.for_each([&](u64 sender, u64 receiver, InterfaceIdTable &build) {
            i32 xoff = build.build_infos.periodicity_index[0];
            i32 yoff = build.build_infos.periodicity_index[1];
            i32 zoff = build.build_infos.periodicity_index[2];

            TgridVec ref_vec = {};
            TgridVec offset  = paving_function.f(ref_vec, xoff, yoff, zoff) - ref_vec;

            shamlog_debug_ln("Offset", offset, build.build_infos.offset);
        });
    };

    // compare_paving(gen_ghost.ghost_id_build_map);

    auto print_debug = [](shambase::DistributedDataShared<PatchDataLayer> &gz) {
        gz.for_each([&](u64 sender, u64 receiver, PatchDataLayer &pdat) {
            shamlog_debug_ln(
                "Ghost zone",
                sender,
                receiver,
                pdat.get_field_buf_ref<TgridVec>(0).copy_to_stdvec(),
                pdat.get_field_buf_ref<TgridVec>(1).copy_to_stdvec());
        });
        throw std::runtime_error("debug");
    };

    // ----------------------------------------------------------------------------------------
    // temporary wrapper to slowly migrate to the new solvergraph
    std::shared_ptr<shamrock::solvergraph::PatchDataLayerRefs> source_patches
        = std::make_shared<shamrock::solvergraph::PatchDataLayerRefs>("", "");

    scheduler().for_each_patchdata_nonempty([&](const Patch &p, PatchDataLayer &pdat) {
        source_patches->patchdatas.add_obj(p.id_patch, std::ref(pdat));
    });

    std::shared_ptr<shamrock::solvergraph::PatchDataLayerEdge> merged_patches
        = std::make_shared<shamrock::solvergraph::PatchDataLayerEdge>("", "", ghost_layout_ptr);
    merged_patches->set_patchdatas({});

    std::shared_ptr<shamrock::solvergraph::CopyPatchDataLayerFields> copy_fields
        = std::make_shared<shamrock::solvergraph::CopyPatchDataLayerFields>(
            scheduler().get_layout_ptr(), ghost_layout_ptr);

    copy_fields->set_edges(source_patches, merged_patches);
    copy_fields->evaluate();

    std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> exchange_gz_edge
        = std::make_shared<shamrock::solvergraph::PatchDataLayerDDShared>("", "");

#if false
    exchange_gz_edge->patchdatas = gen_ghost.template build_interface_native<PatchDataLayer>(
        [&](u64 sender, u64, InterfaceBuildInfos binfo, sycl::buffer<u32> &buf_idx, u32 cnt) {
            PatchDataLayer &sender_patch = scheduler().patch_data.get_pdat(sender);

            PatchDataLayer pdat(ghost_layout_ptr);

            pdat.reserve(cnt);

            sender_patch.get_field<TgridVec>(icell_min).append_subset_to(
                buf_idx, cnt, pdat.get_field<TgridVec>(icell_min_interf));

            sender_patch.get_field<TgridVec>(icell_max).append_subset_to(
                buf_idx, cnt, pdat.get_field<TgridVec>(icell_max_interf));

            sender_patch.get_field<Tscal>(irho).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tscal>(irho_interf));

            sender_patch.get_field<Tscal>(irhoetot).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tscal>(irhoetot_interf));

            sender_patch.get_field<Tvec>(irhovel).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tvec>(irhovel_interf));

            if (solver_config.is_dust_on()) {
                sender_patch.get_field<Tscal>(irho_d).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tscal>(irho_d_interf));

                sender_patch.get_field<Tvec>(irhovel_d).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tvec>(irhovel_d_interf));
            }

            if (solver_config.is_gravity_on()) {
                sender_patch.get_field<Tscal>(iphi).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tscal>(iphi_interf));
            }

            if (solver_config.is_gas_passive_scalar_on()) {
                sender_patch.get_field<Tscal>(irho_gas_pscal)
                    .append_subset_to(buf_idx, cnt, pdat.get_field<Tscal>(irho_gas_pscal_interf));
            }
            pdat.check_field_obj_cnt_match();

            // pdat.get_field<TgridVec>(icell_min_interf).apply_offset(binfo.offset);
            // pdat.get_field<TgridVec>(icell_max_interf).apply_offset(binfo.offset);

            return pdat;
        });
#else
    auto sched = shamsys::instance::get_compute_scheduler_ptr();
    std::shared_ptr<shamrock::solvergraph::DDSharedBuffers<u32>> idx_in_ghost
        = std::make_shared<shamrock::solvergraph::DDSharedBuffers<u32>>(
            "idx_in_ghost", "idx_in_ghost");

    gen_ghost.ghost_id_build_map.for_each(
        [&](u64 sender, u64 receiver, InterfaceIdTable &build_table) {
            auto buf = sham::DeviceBuffer<u32>(build_table.ids_interf->size(), sched);
            buf.copy_from_sycl_buffer(shambase::get_check_ref(build_table.ids_interf));
            idx_in_ghost->buffers.add_obj(sender, receiver, std::move(buf));
        });

    std::shared_ptr<shammodels::basegodunov::modules::ExtractGhostLayer> extract_gz_node
        = std::make_shared<shammodels::basegodunov::modules::ExtractGhostLayer>(ghost_layout_ptr);

    extract_gz_node->set_edges(merged_patches, idx_in_ghost, exchange_gz_edge);
    extract_gz_node->evaluate();

#endif

    // to see the values of the ghost zones
    // print_debug(exchange_gz_edge->patchdatas);

    std::shared_ptr<shammodels::basegodunov::modules::TransformGhostLayer<Tvec, TgridVec>>
        transform_gz_node
        = std::make_shared<shammodels::basegodunov::modules::TransformGhostLayer<Tvec, TgridVec>>(
            GhostLayerGenMode{GhostType::Periodic, GhostType::Periodic, GhostType::Periodic},
            ghost_layout_ptr);

    transform_gz_node->set_edges(sim_box_edge, ghost_layers_candidates_edge, exchange_gz_edge);
    transform_gz_node->evaluate();

    // to see the values of the ghost zones
    // print_debug(exchange_gz_edge->patchdatas);

    std::shared_ptr<ExchangeGhostLayer> exchange_gz_node
        = std::make_shared<ExchangeGhostLayer>(ghost_layout_ptr);
    exchange_gz_node->set_edges(storage.patch_rank_owner, exchange_gz_edge);

    exchange_gz_node->evaluate();

    std::shared_ptr<shammodels::basegodunov::modules::FuseGhostLayer> fuse_gz_node
        = std::make_shared<shammodels::basegodunov::modules::FuseGhostLayer>();
    fuse_gz_node->set_edges(exchange_gz_edge, merged_patches);
    fuse_gz_node->evaluate();

    // ----------------------------------------------------------------------------------------

#if false
    shambase::DistributedDataShared<PatchDataLayer> interf_pdat
        = std::move(exchange_gz_edge->patchdatas);

    std::map<u64, u64> sz_interf_map;
    interf_pdat.for_each([&](u64 s, u64 r, PatchDataLayer &pdat_interf) {
        sz_interf_map[r] += pdat_interf.get_obj_cnt();
    });

    storage.merged_patchdata_ghost.set(merge_native<PatchDataLayer, PatchDataLayer>(
        std::move(interf_pdat),
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
            shamlog_debug_ln("Merged patch init", p.id_patch);

            PatchDataLayer pdat_new(ghost_layout_ptr);

            u32 or_elem = pdat.get_obj_cnt();
            pdat_new.reserve(or_elem + sz_interf_map[p.id_patch]);
            u32 total_elements = or_elem;

            pdat_new.get_field<TgridVec>(icell_min_interf)
                .insert(pdat.get_field<TgridVec>(icell_min));
            pdat_new.get_field<TgridVec>(icell_max_interf)
                .insert(pdat.get_field<TgridVec>(icell_max));
            pdat_new.get_field<Tscal>(irho_interf).insert(pdat.get_field<Tscal>(irho));
            pdat_new.get_field<Tscal>(irhoetot_interf).insert(pdat.get_field<Tscal>(irhoetot));
            pdat_new.get_field<Tvec>(irhovel_interf).insert(pdat.get_field<Tvec>(irhovel));

            if (solver_config.is_dust_on()) {
                pdat_new.get_field<Tscal>(irho_d_interf).insert(pdat.get_field<Tscal>(irho_d));
                pdat_new.get_field<Tvec>(irhovel_d_interf).insert(pdat.get_field<Tvec>(irhovel_d));
            }

            if (solver_config.is_gravity_on()) {
                pdat_new.get_field<Tscal>(iphi_interf).insert(pdat.get_field<Tscal>(iphi));
            }

            if (solver_config.is_gas_passive_scalar_on()) {
                pdat_new.get_field<Tscal>(irho_gas_pscal_interf)
                    .insert(pdat.get_field<Tscal>(irho_gas_pscal));
            }

            pdat_new.check_field_obj_cnt_match();

            return std::move(pdat_new);
        },
        [](PatchDataLayer &mpdat, PatchDataLayer &pdat_interf) {
            mpdat.insert_elements(pdat_interf);
        }));
#else
    storage.merged_patchdata_ghost.set(merged_patches->extract_patchdatas());
#endif

    timer_interf.end();
    storage.timings_details.interface += timer_interf.elasped_sec();

    // TODO this should be output nodes from basic ghost ideally

    { // set element counts
        using MergedPDat = shamrock::MergedPatchData;

        shambase::get_check_ref(storage.block_counts).indexes
            = storage.merged_patchdata_ghost.get().template map<u32>(
                [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                    u32 cnt = scheduler().patch_data.get_pdat(id).get_obj_cnt();
                    return cnt;
                });
    }

    { // set element counts
        using MergedPDat = shamrock::MergedPatchData;

        shambase::get_check_ref(storage.block_counts_with_ghost).indexes
            = storage.merged_patchdata_ghost.get().template map<u32>(
                [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                    u32 cnt = mpdat.get_obj_cnt();
                    return cnt;
                });
    }

    { // Attach spans to block coords
        using MergedPDat = shamrock::MergedPatchData;
        storage.refs_block_min->set_refs(
            storage.merged_patchdata_ghost.get()
                .template map<std::reference_wrapper<PatchDataField<TgridVec>>>(
                    [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                        return std::ref(mpdat.get_field<TgridVec>(0));
                    }));

        storage.refs_block_max->set_refs(
            storage.merged_patchdata_ghost.get()
                .template map<std::reference_wrapper<PatchDataField<TgridVec>>>(
                    [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                        return std::ref(mpdat.get_field<TgridVec>(1));
                    }));
    }

    { // attach spans to gas field with ghosts
        using MergedPDat = shamrock::MergedPatchData;
        shamrock::patch::PatchDataLayerLayout &ghost_layout
            = shambase::get_check_ref(storage.ghost_layout);
        u32 irho_ghost  = ghost_layout.get_field_idx<Tscal>("rho");
        u32 irhov_ghost = ghost_layout.get_field_idx<Tvec>("rhovel");
        u32 irhoe_ghost = ghost_layout.get_field_idx<Tscal>("rhoetot");

        storage.refs_rho->set_refs(storage.merged_patchdata_ghost.get()
                                       .template map<std::reference_wrapper<PatchDataField<Tscal>>>(
                                           [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                                               return std::ref(mpdat.get_field<Tscal>(irho_ghost));
                                           }));

        storage.refs_rhov->set_refs(storage.merged_patchdata_ghost.get()
                                        .template map<std::reference_wrapper<PatchDataField<Tvec>>>(
                                            [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                                                return std::ref(mpdat.get_field<Tvec>(irhov_ghost));
                                            }));

        storage.refs_rhoe->set_refs(
            storage.merged_patchdata_ghost.get()
                .template map<std::reference_wrapper<PatchDataField<Tscal>>>(
                    [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                        return std::ref(mpdat.get_field<Tscal>(irhoe_ghost));
                    }));
    }

    if (solver_config.is_dust_on()) { // attach spans to dust field with ghosts
        using MergedPDat = shamrock::MergedPatchData;
        u32 ndust        = solver_config.dust_config.ndust;
        shamrock::patch::PatchDataLayerLayout &ghost_layout
            = shambase::get_check_ref(storage.ghost_layout);

        u32 irho_dust_ghost  = ghost_layout.get_field_idx<Tscal>("rho_dust");
        u32 irhov_dust_ghost = ghost_layout.get_field_idx<Tvec>("rhovel_dust");

        storage.refs_rho_dust->set_refs(
            storage.merged_patchdata_ghost.get()
                .template map<std::reference_wrapper<PatchDataField<Tscal>>>(
                    [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                        return std::ref(mpdat.get_field<Tscal>(irho_dust_ghost));
                    }));

        storage.refs_rhov_dust->set_refs(
            storage.merged_patchdata_ghost.get()
                .template map<std::reference_wrapper<PatchDataField<Tvec>>>(
                    [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                        return std::ref(mpdat.get_field<Tvec>(irhov_dust_ghost));
                    }));
    }
}

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

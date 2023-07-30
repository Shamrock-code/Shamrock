// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/basegodunov/modules/GhostZones.hpp"
#include "shamalgs/numeric/numeric.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/sycl_utils.hpp"
#include "shammath/AABB.hpp"
#include "shammath/CoordRange.hpp"
#include "shammodels/amr/basegodunov/GhostZoneData.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::basegodunov::modules::GhostZones<Tvec, TgridVec>;

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
    TgridVec bsize                                   = sim_box.get_bounding_box_size<TgridVec>();

    for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
        for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
            for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {

                // sender translation
                TgridVec periodic_offset =
                    TgridVec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                sched.for_each_local_patch([&](const Patch psender) {
                    CoordRange<TgridVec> sender_bsize = patch_coord_transf.to_obj_coord(psender);
                    CoordRange<TgridVec> sender_bsize_off =
                        sender_bsize.add_offset(periodic_offset);

                    shammath::AABB<TgridVec> sender_bsize_off_aabb{sender_bsize_off.lower,
                                                                   sender_bsize_off.upper};

                    using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

                    logger::debug_sycl_ln("AMR:interf",
                                          "find_interfaces -",
                                          psender.id_patch,
                                          sender_bsize_off_aabb.lower,
                                          sender_bsize_off_aabb.upper);

                    sptree.host_for_each_leafs(
                        [&](u64 tree_id, PtNode n) {
                            shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};

                            bool result = tree_cell.get_intersect(sender_bsize_off_aabb)
                                              .is_surface_or_volume();

                            return result;
                        },
                        [&](u64 id_found, PtNode n) {
                            if ((id_found == psender.id_patch) && (xoff == 0) && (yoff == 0) &&
                                (zoff == 0)) {
                                return;
                            }

                            InterfaceBuildInfos ret{
                                periodic_offset,
                                {xoff, yoff, zoff},
                                shammath::AABB<TgridVec>{n.box_min - periodic_offset,
                                                         n.box_max - periodic_offset}};

                            results.add_obj(psender.id_patch, id_found, std::move(ret));
                        });
                });
            }
        }
    }

    return results;
}

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::build_ghost_cache() {

    using GZData = GhostZonesData<Tvec, TgridVec>;

    storage.ghost_zone_infos.set(GZData{});
    GZData &gen_ghost = storage.ghost_zone_infos.get();

    // get ids of cells that will be on the surface of another patch.
    // for cells corresponding to fixed boundary they will be generated after the exhange
    // and appended to the interface list a poosteriori

    gen_ghost.ghost_gen_infos =
        find_interfaces<Tvec, TgridVec>(scheduler(), storage.serial_patch_tree.get());

    using InterfaceBuildInfos = typename GZData::InterfaceBuildInfos;
    using InterfaceIdTable    = typename GZData::InterfaceIdTable;

    // if(logger::log_debug);
    gen_ghost.ghost_gen_infos.for_each([&](u64 sender, u64 receiver, InterfaceBuildInfos &build) {
        std::string log;

        log = shambase::format("{} -> {} : off = {}, {} -> {}",
                               sender,
                               receiver,
                               build.offset,
                               build.volume_target.lower,
                               build.volume_target.upper);

        logger::debug_ln("AMRGodunov", log);
    });

    sycl::queue &q = shamsys::instance::get_compute_queue();

    gen_ghost.ghost_gen_infos.for_each([&](u64 sender, u64 receiver, InterfaceBuildInfos &build) {
        shamrock::patch::PatchData &src = scheduler().patch_data.get_pdat(sender);

        sycl::buffer<u32> is_in_interf{src.get_obj_cnt()};

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{src.get_field_buf_ref<TgridVec>(0), cgh, sycl::read_only};
            sycl::accessor cell_max{src.get_field_buf_ref<TgridVec>(1), cgh, sycl::read_only};
            sycl::accessor flag{is_in_interf, cgh, sycl::write_only, sycl::no_init};

            shammath::AABB<TgridVec> check_volume = build.volume_target;

            shambase::parralel_for(cgh, src.get_obj_cnt(), "check if in interf", [=](u32 id_a) {
                flag[id_a] = shammath::AABB<TgridVec>(cell_min[id_a], cell_max[id_a])
                                 .get_intersect(check_volume)
                                 .is_surface_or_volume();
            });
        });

        auto resut = shamalgs::numeric::stream_compact(q, is_in_interf, src.get_obj_cnt());
        f64 ratio  = f64(std::get<1>(resut)) / f64(src.get_obj_cnt());

        std::string s = shambase::format("{} -> {} : off = {}, test volume = {} -> {}",
                                         sender,
                                         receiver,
                                         build.offset,
                                         build.volume_target.lower,
                                         build.volume_target.upper);
        s += shambase::format("\n    found N = {}, ratio = {} %", std::get<1>(resut), ratio);

        logger::debug_ln("AMR interf", s);

        std::unique_ptr<sycl::buffer<u32>> ids =
            std::make_unique<sycl::buffer<u32>>(shambase::extract_value(std::get<0>(resut)));

        gen_ghost.ghost_id_build_map.add_obj(
            sender, receiver, InterfaceIdTable{build, std::move(ids), ratio});
    });
}

template class shammodels::basegodunov::modules::GhostZones<f64_3, i64_3>;
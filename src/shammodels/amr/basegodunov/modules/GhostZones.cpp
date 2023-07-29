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
#include "shambase/sycl_utils.hpp"
#include "shammath/AABB.hpp"
#include "shammath/CoordRange.hpp"
#include "shammodels/amr/basegodunov/GhostZoneData.hpp"
#include "shamsys/NodeInstance.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::basegodunov::modules::GhostZones<Tvec, TgridVec>;

/**
 * @brief find interfaces corresponding to shared surface between domains
 *
 * @tparam Tvec
 * @tparam TgridVec
 */
template<class Tvec, class TgridVec>
auto find_interfaces(PatchScheduler &sched, SerialPatchTree<Tvec> &sptree) {

    using GZData = shammodels::basegodunov::GhostZonesData<Tvec, TgridVec>;

    StackEntry stack_loc{};

    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    using InterfaceBuildInfos = typename GZData::InterfaceBuildInfos;

    using namespace shamrock::patch;
    using namespace shammath;

    i32 repetition_x = 1;
    i32 repetition_y = 1;
    i32 repetition_z = 1;

    using GeneratorMap = typename GZData::GeneratorMap;
    GeneratorMap results;

    shamrock::patch::SimulationBoxInfo &sim_box = sched.get_sim_box();

    PatchCoordTransform<Tvec> patch_coord_transf = sim_box.get_patch_transform<Tvec>();
    Tvec bsize                                   = sim_box.get_bounding_box_size<Tvec>();

    for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
        for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
            for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {

                // sender translation
                Tvec periodic_offset = Tvec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                sched.for_each_local_patch([&](const Patch psender) {
                    CoordRange<Tvec> sender_bsize     = patch_coord_transf.to_obj_coord(psender);
                    CoordRange<Tvec> sender_bsize_off = sender_bsize.add_offset(periodic_offset);

                    shammath::AABB<Tvec> sender_bsize_off_aabb{sender_bsize_off.lower,
                                                               sender_bsize_off.upper};

                    using PtNode = typename SerialPatchTree<Tvec>::PtNode;

                    sptree.host_for_each_leafs(
                        [&](u64 tree_id, PtNode n) {
                            shammath::AABB<Tvec> tree_cell{n.box_min, n.box_max};

                            return tree_cell.get_intersect(sender_bsize_off_aabb)
                                .is_surface_or_volume();
                        },
                        [&](u64 id_found, PtNode n) {
                            if ((id_found == psender.id_patch) && (xoff == 0) && (yoff == 0) &&
                                (zoff == 0)) {
                                return;
                            }

                            InterfaceBuildInfos ret{
                                periodic_offset, {xoff, yoff, zoff}, sender_bsize_off_aabb};

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
    GZData & gen_ghost = storage.ghost_zone_infos.get();

    // get ids of cells that will be on the surface of another patch.
    // for cells corresponding to fixed boundary they will be generated after the exhange
    // and appended to the interface list a poosteriori

    gen_ghost.ghost_gen_infos = find_interfaces<Tvec, TgridVec>(scheduler(), storage.serial_patch_tree.get());






    using InterfaceBuildInfos = typename GZData::InterfaceBuildInfos;
    struct InterfaceIdTable {
        InterfaceBuildInfos build_infos;
        std::unique_ptr<sycl::buffer<u32>> ids_interf;
        f64 cell_count_ratio;
    };

    shambase::DistributedDataShared<InterfaceIdTable> res;

    sycl::queue & q = shamsys::instance::get_compute_queue();


    gen_ghost.ghost_gen_infos.for_each([&](u64 sender, u64 receiver, InterfaceBuildInfos &build) {
        shamrock::patch::PatchData &src = scheduler().patch_data.get_pdat(sender);

        sycl::buffer<u32> is_in_interf {src.get_obj_cnt()};

        q.submit([&](sycl::handler & cgh){
            sycl::accessor cell_min {src.get_field_buf_ref<Tvec>(0), cgh, sycl::read_only};
            sycl::accessor cell_max {src.get_field_buf_ref<Tvec>(1), cgh, sycl::read_only};
            sycl::accessor flag{is_in_interf, cgh ,sycl::write_only, sycl::no_init};

            shammath::AABB<Tvec> check_volume = build.volume_target;

            shambase::parralel_for(cgh, src.get_obj_cnt(), "check if in interf", [=](u32 id_a){
                flag[id_a] = shammath::AABB<Tvec>(cell_min[id_a], cell_max[id_a]).get_intersect(check_volume).is_surface_or_volume();
            });
        });

        auto resut = shamalgs::numeric::stream_compact(q, is_in_interf, src.get_obj_cnt());
        f64 ratio = f64(std::get<1>(resut)) / f64(src.get_obj_cnt());

        std::unique_ptr<sycl::buffer<u32>> ids = std::make_unique<sycl::buffer<u32>>(shambase::extract_value(std::get<0>(resut)));

        res.add_obj(sender, receiver, InterfaceIdTable{build, std::move(ids),ratio});

    });

}

template class shammodels::basegodunov::modules::GhostZones<f64_3, i64_3>;
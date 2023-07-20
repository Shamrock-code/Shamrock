// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/basegodunov/modules/GhostZones.hpp"
#include "shammath/AABB.hpp"
#include "shammath/CoordRange.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::basegodunov::modules::GhostZones<Tvec, TgridVec>;

/**
 * @brief find interfaces corresponding to shared surface between domains
 *
 * @tparam Tvec
 * @tparam TgridVec
 */
template<class Tvec, class TgridVec>
void find_interfaces(PatchScheduler &sched, SerialPatchTree<Tvec> &sptree) {
    StackEntry stack_loc{};

    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    struct InterfaceBuildInfos {
        Tvec offset;
        sycl::vec<i32, dim> periodicity_index;
        shammath::AABB<Tvec> volume_target;
    };

    using namespace shamrock::patch;
    using namespace shammath;

    i32 repetition_x = 1;
    i32 repetition_y = 1;
    i32 repetition_z = 1;

    using GeneratorMap = shambase::DistributedDataShared<InterfaceBuildInfos>;
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

                            results.add_obj(psender.id_patch, id_found, ret);
                        });
                });
            }
        }
    }
}

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::build_ghost_cache() {

    // get ids of cells that will be on the surface of another patch.
    // for cells corresponding to fixed boundary they will be generated after the exhange
    // and appended to the interface list a poosteriori

    using namespace shamrock;
    using namespace shamrock::patch;

    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData &pdat) {

    });
}
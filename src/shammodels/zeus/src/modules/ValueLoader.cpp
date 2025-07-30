// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ValueLoader.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/zeus/modules/ValueLoader.hpp"
#include "shammodels/zeus/modules/FaceFlagger.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_internal_block_xm(
    u32 nobj, u32 nvar, sham::DeviceBuffer<T> &buf_src, sham::DeviceBuffer<T> &buf_dest) {

    StackEntry stack_loc{};
    using Block = typename Config::AMRBlock;

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto val_out = buf_dest.get_write_access(depends_list);
    auto src     = buf_src.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        shambase::parallel_for(cgh, nobj * Block::block_size, "compute xm val (1)", [=](u64 id_a) {
            const u32 base_idx = id_a;
            const u32 lid      = id_a % Block::block_size;

            static_assert(dim == 3, "implemented only in dim 3");
            std::array<u32, 3> lid_coord = Block::get_coord(lid);

            if (lid_coord[0] > 0) {
                lid_coord[0] -= 1;
                val_out[base_idx] = src[base_idx - lid + Block::get_index(lid_coord)];
            }
        });
    });

    buf_dest.complete_event_state(e);
    buf_src.complete_event_state(e);
}

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_internal_block_xp(
    u32 nobj, u32 nvar, sham::DeviceBuffer<T> &buf_src, sham::DeviceBuffer<T> &buf_dest) {

    StackEntry stack_loc{};
    using Block = typename Config::AMRBlock;

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto val_out = buf_dest.get_write_access(depends_list);
    auto src     = buf_src.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        shambase::parallel_for(cgh, nobj * Block::block_size, "compute xp val (1)", [=](u64 id_a) {
            const u32 base_idx = id_a;
            const u32 lid      = id_a % Block::block_size;

            static_assert(dim == 3, "implemented only in dim 3");
            std::array<u32, 3> lid_coord = Block::get_coord(lid);

            if (lid_coord[0] < Block::Nside - 1) {
                lid_coord[0] += 1;
                val_out[base_idx] = src[base_idx - lid + Block::get_index(lid_coord)];
            }
        });
    });

    buf_dest.complete_event_state(e);
    buf_src.complete_event_state(e);
}

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_internal_block_ym(
    u32 nobj, u32 nvar, sham::DeviceBuffer<T> &buf_src, sham::DeviceBuffer<T> &buf_dest) {

    StackEntry stack_loc{};
    using Block = typename Config::AMRBlock;

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto val_out = buf_dest.get_write_access(depends_list);
    auto src     = buf_src.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        shambase::parallel_for(cgh, nobj * Block::block_size, "compute ym val (1)", [=](u64 id_a) {
            const u32 base_idx = id_a;
            const u32 lid      = id_a % Block::block_size;

            static_assert(dim == 3, "implemented only in dim 3");
            std::array<u32, 3> lid_coord = Block::get_coord(lid);

            if (lid_coord[1] > 0) {
                lid_coord[1] -= 1;
                val_out[base_idx] = src[base_idx - lid + Block::get_index(lid_coord)];
            }
        });
    });

    buf_dest.complete_event_state(e);
    buf_src.complete_event_state(e);
}

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_internal_block_yp(
    u32 nobj, u32 nvar, sham::DeviceBuffer<T> &buf_src, sham::DeviceBuffer<T> &buf_dest) {

    StackEntry stack_loc{};
    using Block = typename Config::AMRBlock;

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto val_out = buf_dest.get_write_access(depends_list);
    auto src     = buf_src.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        shambase::parallel_for(cgh, nobj * Block::block_size, "compute yp val (1)", [=](u64 id_a) {
            const u32 base_idx = id_a;
            const u32 lid      = id_a % Block::block_size;

            static_assert(dim == 3, "implemented only in dim 3");
            std::array<u32, 3> lid_coord = Block::get_coord(lid);

            if (lid_coord[1] < Block::Nside - 1) {
                lid_coord[1] += 1;
                val_out[base_idx] = src[base_idx - lid + Block::get_index(lid_coord)];
            }
        });
    });

    buf_dest.complete_event_state(e);
    buf_src.complete_event_state(e);
}

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_internal_block_zm(
    u32 nobj, u32 nvar, sham::DeviceBuffer<T> &buf_src, sham::DeviceBuffer<T> &buf_dest) {

    StackEntry stack_loc{};
    using Block = typename Config::AMRBlock;

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto val_out = buf_dest.get_write_access(depends_list);
    auto src     = buf_src.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        shambase::parallel_for(cgh, nobj * Block::block_size, "compute ym val (1)", [=](u64 id_a) {
            const u32 base_idx = id_a;
            const u32 lid      = id_a % Block::block_size;

            static_assert(dim == 3, "implemented only in dim 3");
            std::array<u32, 3> lid_coord = Block::get_coord(lid);

            if (lid_coord[2] > 0) {
                lid_coord[2] -= 1;
                val_out[base_idx] = src[base_idx - lid + Block::get_index(lid_coord)];
            }
        });
    });

    buf_dest.complete_event_state(e);
    buf_src.complete_event_state(e);
}

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_internal_block_zp(
    u32 nobj, u32 nvar, sham::DeviceBuffer<T> &buf_src, sham::DeviceBuffer<T> &buf_dest) {

    StackEntry stack_loc{};
    using Block = typename Config::AMRBlock;

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto val_out = buf_dest.get_write_access(depends_list);
    auto src     = buf_src.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        shambase::parallel_for(cgh, nobj * Block::block_size, "compute ym val (1)", [=](u64 id_a) {
            const u32 base_idx = id_a;
            const u32 lid      = id_a % Block::block_size;

            static_assert(dim == 3, "implemented only in dim 3");
            std::array<u32, 3> lid_coord = Block::get_coord(lid);

            if (lid_coord[2] < Block::Nside - 1) {
                lid_coord[2] += 1;
                val_out[base_idx] = src[base_idx - lid + Block::get_index(lid_coord)];
            }
        });
    });

    buf_dest.complete_event_state(e);
    buf_src.complete_event_state(e);
}

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_internal_block(
    std::array<Tgridscal, dim> offset,
    u32 nobj,
    u32 nvar,
    sham::DeviceBuffer<T> &buf_src,
    sham::DeviceBuffer<T> &buf_dest) {

    StackEntry stack_loc{};
    using Block = typename Config::AMRBlock;

    if constexpr (dim == 3) {
        if (offset[0] == -1 && offset[1] == 0 && offset[2] == 0) {

            load_patch_internal_block_xm(nobj, nvar, buf_src, buf_dest);

        } else if (offset[0] == 0 && offset[1] == -1 && offset[2] == 0) {

            load_patch_internal_block_ym(nobj, nvar, buf_src, buf_dest);

        } else if (offset[0] == 0 && offset[1] == 0 && offset[2] == -1) {

            load_patch_internal_block_zm(nobj, nvar, buf_src, buf_dest);

        } else if (offset[0] == 1 && offset[1] == 0 && offset[2] == 0) {

            load_patch_internal_block_xp(nobj, nvar, buf_src, buf_dest);

        } else if (offset[0] == 0 && offset[1] == 1 && offset[2] == 0) {

            load_patch_internal_block_yp(nobj, nvar, buf_src, buf_dest);

        } else if (offset[0] == 0 && offset[1] == 0 && offset[2] == 1) {

            load_patch_internal_block_zp(nobj, nvar, buf_src, buf_dest);

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "offset : ({},{},{}) is invalid", offset[0], offset[1], offset[2]));
        }
    } else {
        shambase::throw_unimplemented();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_neigh_same_level_xm(

    std::array<Tgridscal, dim> offset,
    sham::DeviceBuffer<TgridVec> &buf_cell_min,
    sham::DeviceBuffer<TgridVec> &buf_cell_max,
    shammodels::zeus::NeighFaceList<Tvec> &face_lists,
    u32 nobj,
    u32 nvar,
    sham::DeviceBuffer<T> &buf_src,
    sham::DeviceBuffer<T> &buf_dest

) {
    StackEntry stack_loc{};

    using Block = typename Config::AMRBlock;
    using namespace shamrock;

    OrientedNeighFaceList<Tvec> &face_xm = face_lists.xm();

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto val_out  = buf_dest.get_write_access(depends_list);
    auto src      = buf_src.get_read_access(depends_list);
    auto cell_min = buf_cell_min.get_read_access(depends_list);
    auto cell_max = buf_cell_max.get_read_access(depends_list);

    auto fptr = face_xm.neigh_info.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        tree::ObjectCacheIterator faces_xm(fptr);

        shambase::parallel_for(cgh, nobj * Block::block_size, "compute xm val (2)", [=](u64 id_a) {
            const u32 base_idx = id_a;
            const u32 block_id = id_a / Block::block_size;
            const u32 lid      = id_a % Block::block_size;

            std::array<u32, 3> lid_coord = Block::get_coord(lid);

            if (lid_coord[0] == 0) {
                auto tmp = cell_max[block_id] - cell_min[block_id];
                i32 Va   = tmp.x() * tmp.y() * tmp.z();

                static_assert(dim == 3, "implemented only in dim 3");
                faces_xm.for_each_object(block_id, [&](u32 block_id_b) {
                    auto tmp = cell_max[block_id_b] - cell_min[block_id_b];
                    i32 nV   = tmp.x() * tmp.y() * tmp.z();

                    if (nV == Va) { // same level
                        val_out[base_idx] = src
                            [block_id_b * Block::block_size
                             + Block::get_index({Block::Nside - 1, lid_coord[1], lid_coord[2]})];
                    }
                });
            }
        });
    });

    buf_dest.complete_event_state(e);
    buf_src.complete_event_state(e);
    buf_cell_min.complete_event_state(e);
    buf_cell_max.complete_event_state(e);

    sham::EventList resulting_events;
    resulting_events.add_event(e);
    face_xm.neigh_info.complete_event_state(resulting_events);
}

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_neigh_same_level_xp(

    std::array<Tgridscal, dim> offset,
    sham::DeviceBuffer<TgridVec> &buf_cell_min,
    sham::DeviceBuffer<TgridVec> &buf_cell_max,
    shammodels::zeus::NeighFaceList<Tvec> &face_lists,
    u32 nobj,
    u32 nvar,
    sham::DeviceBuffer<T> &buf_src,
    sham::DeviceBuffer<T> &buf_dest

) {
    StackEntry stack_loc{};

    using Block = typename Config::AMRBlock;
    using namespace shamrock;

    OrientedNeighFaceList<Tvec> &face_xp = face_lists.xp();

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto val_out  = buf_dest.get_write_access(depends_list);
    auto src      = buf_src.get_read_access(depends_list);
    auto cell_min = buf_cell_min.get_read_access(depends_list);
    auto cell_max = buf_cell_max.get_read_access(depends_list);

    auto fptr = face_xp.neigh_info.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        tree::ObjectCacheIterator faces_xp(fptr);

        shambase::parallel_for(cgh, nobj * Block::block_size, "compute xm val (2)", [=](u64 id_a) {
            const u32 base_idx = id_a;
            const u32 block_id = id_a / Block::block_size;
            const u32 lid      = id_a % Block::block_size;

            std::array<u32, 3> lid_coord = Block::get_coord(lid);

            if (lid_coord[0] == Block::Nside - 1) {
                auto tmp = cell_max[block_id] - cell_min[block_id];
                i32 Va   = tmp.x() * tmp.y() * tmp.z();

                static_assert(dim == 3, "implemented only in dim 3");
                faces_xp.for_each_object(block_id, [&](u32 block_id_b) {
                    auto tmp = cell_max[block_id_b] - cell_min[block_id_b];
                    i32 nV   = tmp.x() * tmp.y() * tmp.z();

                    if (nV == Va) { // same level
                        auto val = src
                            [block_id_b * Block::block_size
                             + Block::get_index({0, lid_coord[1], lid_coord[2]})];

                        // if constexpr (std::is_same_v<T, Tvec>){
                        // sycl::ext::oneapi::experimental::printf("%d %f %f %f\n",block_id_b *
                        // Block::block_size +
                        //         Block::get_index({0, lid_coord[1],
                        //         lid_coord[2]}),val.x(),val.y(),val.z());
                        // }

                        val_out[base_idx] = val;
                    }
                });
            }
        });
    });

    buf_dest.complete_event_state(e);
    buf_src.complete_event_state(e);
    buf_cell_min.complete_event_state(e);
    buf_cell_max.complete_event_state(e);

    sham::EventList resulting_events;
    resulting_events.add_event(e);
    face_xp.neigh_info.complete_event_state(resulting_events);
}

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_neigh_same_level_ym(

    std::array<Tgridscal, dim> offset,
    sham::DeviceBuffer<TgridVec> &buf_cell_min,
    sham::DeviceBuffer<TgridVec> &buf_cell_max,
    shammodels::zeus::NeighFaceList<Tvec> &face_lists,
    u32 nobj,
    u32 nvar,
    sham::DeviceBuffer<T> &buf_src,
    sham::DeviceBuffer<T> &buf_dest

) {
    StackEntry stack_loc{};

    using Block = typename Config::AMRBlock;
    using namespace shamrock;

    OrientedNeighFaceList<Tvec> &face_ym = face_lists.ym();

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto val_out  = buf_dest.get_write_access(depends_list);
    auto src      = buf_src.get_read_access(depends_list);
    auto cell_min = buf_cell_min.get_read_access(depends_list);
    auto cell_max = buf_cell_max.get_read_access(depends_list);

    auto fptr = face_ym.neigh_info.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        tree::ObjectCacheIterator faces_ym(fptr);

        shambase::parallel_for(cgh, nobj * Block::block_size, "compute ym val (2)", [=](u64 id_a) {
            const u32 base_idx = id_a;
            const u32 block_id = id_a / Block::block_size;
            const u32 lid      = id_a % Block::block_size;

            std::array<u32, 3> lid_coord = Block::get_coord(lid);

            if (lid_coord[1] == 0) {
                auto tmp = cell_max[block_id] - cell_min[block_id];
                i32 Va   = tmp.x() * tmp.y() * tmp.z();

                static_assert(dim == 3, "implemented only in dim 3");
                faces_ym.for_each_object(block_id, [&](u32 block_id_b) {
                    auto tmp = cell_max[block_id_b] - cell_min[block_id_b];
                    i32 nV   = tmp.x() * tmp.y() * tmp.z();

                    if (nV == Va) { // same level
                        val_out[base_idx] = src
                            [block_id_b * Block::block_size
                             + Block::get_index({lid_coord[0], Block::Nside - 1, lid_coord[2]})];
                    }
                });
            }
        });
    });

    buf_dest.complete_event_state(e);
    buf_src.complete_event_state(e);
    buf_cell_min.complete_event_state(e);
    buf_cell_max.complete_event_state(e);

    sham::EventList resulting_events;
    resulting_events.add_event(e);
    face_ym.neigh_info.complete_event_state(resulting_events);
}

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_neigh_same_level_yp(

    std::array<Tgridscal, dim> offset,
    sham::DeviceBuffer<TgridVec> &buf_cell_min,
    sham::DeviceBuffer<TgridVec> &buf_cell_max,
    shammodels::zeus::NeighFaceList<Tvec> &face_lists,
    u32 nobj,
    u32 nvar,
    sham::DeviceBuffer<T> &buf_src,
    sham::DeviceBuffer<T> &buf_dest

) {
    StackEntry stack_loc{};

    using Block = typename Config::AMRBlock;
    using namespace shamrock;

    OrientedNeighFaceList<Tvec> &face_yp = face_lists.yp();

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto val_out  = buf_dest.get_write_access(depends_list);
    auto src      = buf_src.get_read_access(depends_list);
    auto cell_min = buf_cell_min.get_read_access(depends_list);
    auto cell_max = buf_cell_max.get_read_access(depends_list);

    auto fptr = face_yp.neigh_info.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        tree::ObjectCacheIterator faces_yp(fptr);

        shambase::parallel_for(cgh, nobj * Block::block_size, "compute ym val (2)", [=](u64 id_a) {
            const u32 base_idx = id_a;
            const u32 block_id = id_a / Block::block_size;
            const u32 lid      = id_a % Block::block_size;

            std::array<u32, 3> lid_coord = Block::get_coord(lid);

            if (lid_coord[1] == Block::Nside - 1) {
                auto tmp = cell_max[block_id] - cell_min[block_id];
                i32 Va   = tmp.x() * tmp.y() * tmp.z();

                static_assert(dim == 3, "implemented only in dim 3");
                faces_yp.for_each_object(block_id, [&](u32 block_id_b) {
                    auto tmp = cell_max[block_id_b] - cell_min[block_id_b];
                    i32 nV   = tmp.x() * tmp.y() * tmp.z();

                    if (nV == Va) { // same level
                        val_out[base_idx] = src
                            [block_id_b * Block::block_size
                             + Block::get_index({lid_coord[0], 0, lid_coord[2]})];
                    }
                });
            }
        });
    });

    buf_dest.complete_event_state(e);
    buf_src.complete_event_state(e);
    buf_cell_min.complete_event_state(e);
    buf_cell_max.complete_event_state(e);

    sham::EventList resulting_events;
    resulting_events.add_event(e);
    face_yp.neigh_info.complete_event_state(resulting_events);
}

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_neigh_same_level_zm(

    std::array<Tgridscal, dim> offset,
    sham::DeviceBuffer<TgridVec> &buf_cell_min,
    sham::DeviceBuffer<TgridVec> &buf_cell_max,
    shammodels::zeus::NeighFaceList<Tvec> &face_lists,
    u32 nobj,
    u32 nvar,
    sham::DeviceBuffer<T> &buf_src,
    sham::DeviceBuffer<T> &buf_dest

) {
    StackEntry stack_loc{};

    using Block = typename Config::AMRBlock;
    using namespace shamrock;

    OrientedNeighFaceList<Tvec> &face_zm = face_lists.zm();

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto val_out  = buf_dest.get_write_access(depends_list);
    auto src      = buf_src.get_read_access(depends_list);
    auto cell_min = buf_cell_min.get_read_access(depends_list);
    auto cell_max = buf_cell_max.get_read_access(depends_list);

    auto fptr = face_zm.neigh_info.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        tree::ObjectCacheIterator faces_zm(fptr);

        shambase::parallel_for(cgh, nobj * Block::block_size, "compute zm val (2)", [=](u64 id_a) {
            const u32 base_idx = id_a;
            const u32 block_id = id_a / Block::block_size;
            const u32 lid      = id_a % Block::block_size;

            std::array<u32, 3> lid_coord = Block::get_coord(lid);

            if (lid_coord[2] == 0) {
                auto tmp = cell_max[block_id] - cell_min[block_id];
                i32 Va   = tmp.x() * tmp.y() * tmp.z();

                static_assert(dim == 3, "implemented only in dim 3");
                faces_zm.for_each_object(block_id, [&](u32 block_id_b) {
                    auto tmp = cell_max[block_id_b] - cell_min[block_id_b];
                    i32 nV   = tmp.x() * tmp.y() * tmp.z();

                    if (nV == Va) { // same level
                        val_out[base_idx] = src
                            [block_id_b * Block::block_size
                             + Block::get_index({lid_coord[0], lid_coord[1], Block::Nside - 1})];
                    }
                });
            }
        });
    });

    buf_dest.complete_event_state(e);
    buf_src.complete_event_state(e);
    buf_cell_min.complete_event_state(e);
    buf_cell_max.complete_event_state(e);

    sham::EventList resulting_events;
    resulting_events.add_event(e);
    face_zm.neigh_info.complete_event_state(resulting_events);
}

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_neigh_same_level_zp(

    std::array<Tgridscal, dim> offset,
    sham::DeviceBuffer<TgridVec> &buf_cell_min,
    sham::DeviceBuffer<TgridVec> &buf_cell_max,
    shammodels::zeus::NeighFaceList<Tvec> &face_lists,
    u32 nobj,
    u32 nvar,
    sham::DeviceBuffer<T> &buf_src,
    sham::DeviceBuffer<T> &buf_dest

) {
    StackEntry stack_loc{};

    using Block = typename Config::AMRBlock;
    using namespace shamrock;

    OrientedNeighFaceList<Tvec> &face_zp = face_lists.zp();

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto val_out  = buf_dest.get_write_access(depends_list);
    auto src      = buf_src.get_read_access(depends_list);
    auto cell_min = buf_cell_min.get_read_access(depends_list);
    auto cell_max = buf_cell_max.get_read_access(depends_list);

    auto fptr = face_zp.neigh_info.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        tree::ObjectCacheIterator faces_zp(fptr);

        shambase::parallel_for(cgh, nobj * Block::block_size, "compute zm val (2)", [=](u64 id_a) {
            const u32 base_idx = id_a;
            const u32 block_id = id_a / Block::block_size;
            const u32 lid      = id_a % Block::block_size;

            std::array<u32, 3> lid_coord = Block::get_coord(lid);

            if (lid_coord[2] == Block::Nside - 1) {
                auto tmp = cell_max[block_id] - cell_min[block_id];
                i32 Va   = tmp.x() * tmp.y() * tmp.z();

                static_assert(dim == 3, "implemented only in dim 3");
                faces_zp.for_each_object(block_id, [&](u32 block_id_b) {
                    auto tmp = cell_max[block_id_b] - cell_min[block_id_b];
                    i32 nV   = tmp.x() * tmp.y() * tmp.z();

                    if (nV == Va) { // same level
                        val_out[base_idx] = src
                            [block_id_b * Block::block_size
                             + Block::get_index({lid_coord[0], lid_coord[1], 0})];
                    }
                });
            }
        });
    });

    buf_dest.complete_event_state(e);
    buf_src.complete_event_state(e);
    buf_cell_min.complete_event_state(e);
    buf_cell_max.complete_event_state(e);

    sham::EventList resulting_events;
    resulting_events.add_event(e);
    face_zp.neigh_info.complete_event_state(resulting_events);
}

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_neigh_same_level(

    std::array<Tgridscal, dim> offset,
    sham::DeviceBuffer<TgridVec> &buf_cell_min,
    sham::DeviceBuffer<TgridVec> &buf_cell_max,
    shammodels::zeus::NeighFaceList<Tvec> &face_lists,
    u32 nobj,
    u32 nvar,
    sham::DeviceBuffer<T> &buf_src,
    sham::DeviceBuffer<T> &buf_dest

) {
    StackEntry stack_loc{};
    using Block = typename Config::AMRBlock;

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    if constexpr (dim == 3) {
        if (offset[0] == -1 && offset[1] == 0 && offset[2] == 0) {

            load_patch_neigh_same_level_xm(
                offset, buf_cell_min, buf_cell_max, face_lists, nobj, nvar, buf_src, buf_dest);

        } else if (offset[0] == 0 && offset[1] == -1 && offset[2] == 0) {

            load_patch_neigh_same_level_ym(
                offset, buf_cell_min, buf_cell_max, face_lists, nobj, nvar, buf_src, buf_dest);

        } else if (offset[0] == 0 && offset[1] == 0 && offset[2] == -1) {

            load_patch_neigh_same_level_zm(
                offset, buf_cell_min, buf_cell_max, face_lists, nobj, nvar, buf_src, buf_dest);

        } else if (offset[0] == 1 && offset[1] == 0 && offset[2] == 0) {

            load_patch_neigh_same_level_xp(
                offset, buf_cell_min, buf_cell_max, face_lists, nobj, nvar, buf_src, buf_dest);

        } else if (offset[0] == 0 && offset[1] == 1 && offset[2] == 0) {

            load_patch_neigh_same_level_yp(
                offset, buf_cell_min, buf_cell_max, face_lists, nobj, nvar, buf_src, buf_dest);

        } else if (offset[0] == 0 && offset[1] == 0 && offset[2] == 1) {

            load_patch_neigh_same_level_zp(
                offset, buf_cell_min, buf_cell_max, face_lists, nobj, nvar, buf_src, buf_dest);

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "offset : ({},{},{}) is invalid", offset[0], offset[1], offset[2]));
        }
    } else {
        shambase::throw_unimplemented();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_neigh_level_up(

    std::array<Tgridscal, dim> offset,
    sham::DeviceBuffer<TgridVec> &buf_cell_min,
    sham::DeviceBuffer<TgridVec> &buf_cell_max,
    shammodels::zeus::NeighFaceList<Tvec> &face_lists,
    u32 nobj,
    u32 nvar,
    sham::DeviceBuffer<T> &buf_src,
    sham::DeviceBuffer<T> &buf_dest

) {

    StackEntry stack_loc{};
    using Block = typename Config::AMRBlock;

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    if constexpr (dim == 3) {
        if (offset[0] == -1 && offset[1] == 0 && offset[2] == 0) {

            OrientedNeighFaceList<Tvec> &face_xm = face_lists.xm();

        } else if (offset[0] == 0 && offset[1] == -1 && offset[2] == 0) {

            OrientedNeighFaceList<Tvec> &face_ym = face_lists.ym();

        } else if (offset[0] == 0 && offset[1] == 0 && offset[2] == -1) {

            OrientedNeighFaceList<Tvec> &face_zm = face_lists.zm();

        } else if (offset[0] == 1 && offset[1] == 0 && offset[2] == 0) {

            OrientedNeighFaceList<Tvec> &face_xp = face_lists.xp();

        } else if (offset[0] == 0 && offset[1] == 1 && offset[2] == 0) {

            OrientedNeighFaceList<Tvec> &face_yp = face_lists.yp();

        } else if (offset[0] == 0 && offset[1] == 0 && offset[2] == 1) {

            OrientedNeighFaceList<Tvec> &face_zp = face_lists.zp();

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "offset : ({},{},{}) is invalid", offset[0], offset[1], offset[2]));
        }
    } else {
        shambase::throw_unimplemented();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, class TgridVec, class T>
void shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_patch_neigh_level_down(

    std::array<Tgridscal, dim> offset,
    sham::DeviceBuffer<TgridVec> &buf_cell_min,
    sham::DeviceBuffer<TgridVec> &buf_cell_max,
    shammodels::zeus::NeighFaceList<Tvec> &face_lists,
    u32 nobj,
    u32 nvar,
    sham::DeviceBuffer<T> &buf_src,
    sham::DeviceBuffer<T> &buf_dest

) {
    StackEntry stack_loc{};
    using Block = typename Config::AMRBlock;

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    if constexpr (dim == 3) {
        if (offset[0] == -1 && offset[1] == 0 && offset[2] == 0) {

            OrientedNeighFaceList<Tvec> &face_xm = face_lists.xm();

        } else if (offset[0] == 0 && offset[1] == -1 && offset[2] == 0) {

            OrientedNeighFaceList<Tvec> &face_ym = face_lists.ym();

        } else if (offset[0] == 0 && offset[1] == 0 && offset[2] == -1) {

            OrientedNeighFaceList<Tvec> &face_zm = face_lists.zm();

        } else if (offset[0] == 1 && offset[1] == 0 && offset[2] == 0) {

            OrientedNeighFaceList<Tvec> &face_xp = face_lists.xp();

        } else if (offset[0] == 0 && offset[1] == 1 && offset[2] == 0) {

            OrientedNeighFaceList<Tvec> &face_yp = face_lists.yp();

        } else if (offset[0] == 0 && offset[1] == 0 && offset[2] == 1) {

            OrientedNeighFaceList<Tvec> &face_zp = face_lists.zp();

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "offset : ({},{},{}) is invalid", offset[0], offset[1], offset[2]));
        }
    } else {
        shambase::throw_unimplemented();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, class TgridVec, class T>
shamrock::ComputeField<T>
shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_value_with_gz(
    std::string field_name, std::array<Tgridscal, dim> offset, std::string result_name) {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;
    using Flagger    = FaceFlagger<Tvec, TgridVec>;
    using Block      = typename Config::AMRBlock;

    shamrock::SchedulerUtility utility(scheduler());
    ComputeField<T> tmp
        = utility.make_compute_field<T>(result_name, Block::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ifield                                     = ghost_layout.get_field_idx<T>(field_name);
    u32 nvar                                       = ghost_layout.get_field<T>(ifield).nvar;

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sham::DeviceBuffer<T> &buf_src  = mpdat.pdat.get_field_buf_ref<T>(ifield);
        sham::DeviceBuffer<T> &buf_dest = tmp.get_buf_check(p.id_patch);

        load_patch_internal_block(offset, mpdat.total_elements, nvar, buf_src, buf_dest);
    });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sham::DeviceBuffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sham::DeviceBuffer<T> &buf_src  = mpdat.pdat.get_field_buf_ref<T>(ifield);
        sham::DeviceBuffer<T> &buf_dest = tmp.get_buf_check(p.id_patch);

        shammodels::zeus::NeighFaceList<Tvec> &face_lists
            = storage.face_lists.get().get(p.id_patch);

        load_patch_neigh_same_level(
            offset,
            buf_cell_min,
            buf_cell_max,
            face_lists,
            mpdat.total_elements,
            nvar,
            buf_src,
            buf_dest);
    });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sham::DeviceBuffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sham::DeviceBuffer<T> &buf_src  = mpdat.pdat.get_field_buf_ref<T>(ifield);
        sham::DeviceBuffer<T> &buf_dest = tmp.get_buf_check(p.id_patch);

        shammodels::zeus::NeighFaceList<Tvec> &face_lists
            = storage.face_lists.get().get(p.id_patch);

        load_patch_neigh_level_up(
            offset,
            buf_cell_min,
            buf_cell_max,
            face_lists,
            mpdat.total_elements,
            nvar,
            buf_src,
            buf_dest);
    });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sham::DeviceBuffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sham::DeviceBuffer<T> &buf_src  = mpdat.pdat.get_field_buf_ref<T>(ifield);
        sham::DeviceBuffer<T> &buf_dest = tmp.get_buf_check(p.id_patch);

        shammodels::zeus::NeighFaceList<Tvec> &face_lists
            = storage.face_lists.get().get(p.id_patch);

        load_patch_neigh_level_down(
            offset,
            buf_cell_min,
            buf_cell_max,
            face_lists,
            mpdat.total_elements,
            nvar,
            buf_src,
            buf_dest);
    });

    return tmp;
}

template<class Tvec, class TgridVec, class T>
shamrock::ComputeField<T>
shammodels::zeus::modules::ValueLoader<Tvec, TgridVec, T>::load_value_with_gz(
    shamrock::ComputeField<T> &compute_field,
    std::array<Tgridscal, dim> offset,
    std::string result_name) {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;
    using Flagger    = FaceFlagger<Tvec, TgridVec>;
    using Block      = typename Config::AMRBlock;

    shamrock::SchedulerUtility utility(scheduler());
    ComputeField<T> tmp
        = utility.make_compute_field<T>(result_name, Block::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sham::DeviceBuffer<T> &buf_src  = compute_field.get_buf_check(p.id_patch);
        sham::DeviceBuffer<T> &buf_dest = tmp.get_buf_check(p.id_patch);

        load_patch_internal_block(
            offset,
            mpdat.total_elements,
            compute_field.get_field(p.id_patch).get_nvar(),
            buf_src,
            buf_dest);
    });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sham::DeviceBuffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sham::DeviceBuffer<T> &buf_src  = compute_field.get_buf_check(p.id_patch);
        sham::DeviceBuffer<T> &buf_dest = tmp.get_buf_check(p.id_patch);

        shammodels::zeus::NeighFaceList<Tvec> &face_lists
            = storage.face_lists.get().get(p.id_patch);

        load_patch_neigh_same_level(
            offset,
            buf_cell_min,
            buf_cell_max,
            face_lists,
            mpdat.total_elements,
            compute_field.get_field(p.id_patch).get_nvar(),
            buf_src,
            buf_dest);
    });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sham::DeviceBuffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sham::DeviceBuffer<T> &buf_src  = compute_field.get_buf_check(p.id_patch);
        sham::DeviceBuffer<T> &buf_dest = tmp.get_buf_check(p.id_patch);

        shammodels::zeus::NeighFaceList<Tvec> &face_lists
            = storage.face_lists.get().get(p.id_patch);

        load_patch_neigh_level_up(
            offset,
            buf_cell_min,
            buf_cell_max,
            face_lists,
            mpdat.total_elements,
            compute_field.get_field(p.id_patch).get_nvar(),
            buf_src,
            buf_dest);
    });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sham::DeviceBuffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sham::DeviceBuffer<T> &buf_src  = compute_field.get_buf_check(p.id_patch);
        sham::DeviceBuffer<T> &buf_dest = tmp.get_buf_check(p.id_patch);

        shammodels::zeus::NeighFaceList<Tvec> &face_lists
            = storage.face_lists.get().get(p.id_patch);

        load_patch_neigh_level_down(
            offset,
            buf_cell_min,
            buf_cell_max,
            face_lists,
            mpdat.total_elements,
            compute_field.get_field(p.id_patch).get_nvar(),
            buf_src,
            buf_dest);
    });

    return tmp;
}

template class shammodels::zeus::modules::ValueLoader<f64_3, i64_3, f64>;
template class shammodels::zeus::modules::ValueLoader<f64_3, i64_3, f64_3>;
template class shammodels::zeus::modules::ValueLoader<f64_3, i64_3, f64_8>;

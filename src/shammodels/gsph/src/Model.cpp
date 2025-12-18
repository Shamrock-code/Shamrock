// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Model.cpp
 * @author Guo (guo.yansong@optimind.tech)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief GSPH Model implementation
 */

#include "shambase/aliases_float.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/CoordRange.hpp"
#include "shammath/crystalLattice.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/common/setup/generators.hpp"
#include "shammodels/gsph/Model.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/DataInserterUtility.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <functional>
#include <utility>
#include <vector>

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::Model<Tvec, SPHKernel>::init_scheduler(u32 crit_split, u32 crit_merge) {
    solver.init_required_fields();
    ctx.init_sched(crit_split, crit_merge);

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    sched.add_root_patch();

    shamlog_debug_ln("Sys", "build local scheduler tables");
    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_load_value([&](shamrock::patch::Patch p) {
        return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });
    solver.init_ghost_layout();

    solver.init_solver_graph();
}

template<class Tvec, template<class> class SPHKernel>
u64 shammodels::gsph::Model<Tvec, SPHKernel>::get_total_part_count() {
    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
    return shamalgs::collective::allreduce_sum(sched.get_rank_count());
}

template<class Tvec, template<class> class SPHKernel>
f64 shammodels::gsph::Model<Tvec, SPHKernel>::total_mass_to_part_mass(f64 totmass) {
    return totmass / get_total_part_count();
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::gsph::Model<Tvec, SPHKernel>::get_ideal_fcc_box(
    Tscal dr, std::pair<Tvec, Tvec> box) -> std::pair<Tvec, Tvec> {
    StackEntry stack_loc{};
    auto [a, b] = generic::setup::generators::get_ideal_fcc_box<Tscal>(
        dr, std::make_tuple(box.first, box.second));
    return {a, b};
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::gsph::Model<Tvec, SPHKernel>::get_ideal_hcp_box(
    Tscal dr, std::pair<Tvec, Tvec> box) -> std::pair<Tvec, Tvec> {
    StackEntry stack_loc{};
    // Use FCC box for now since HCP is similar
    auto [a, b] = generic::setup::generators::get_ideal_fcc_box<Tscal>(
        dr, std::make_tuple(box.first, box.second));
    return {a, b};
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::Model<Tvec, SPHKernel>::add_cube_fcc_3d(
    Tscal dr, std::pair<Tvec, Tvec> _box) {
    StackEntry stack_loc{};

    shammath::CoordRange<Tvec> box = _box;

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    std::string log = "";

    auto make_sliced = [&]() {
        std::vector<Tvec> vec_lst;
        generic::setup::generators::add_particles_fcc(
            dr,
            std::make_tuple(box.lower, box.upper),
            [&](Tvec r) {
                return box.contain_pos(r);
            },
            [&](Tvec r, Tscal h) {
                vec_lst.push_back(r);
            });

        std::vector<std::vector<Tvec>> sliced_buf;

        u32 sz_buf = sched.crit_patch_split * 4;

        std::vector<Tvec> cur_buf;
        for (u32 i = 0; i < vec_lst.size(); i++) {
            cur_buf.push_back(vec_lst[i]);

            if (cur_buf.size() > sz_buf) {
                sliced_buf.push_back(std::exchange(cur_buf, std::vector<Tvec>{}));
            }
        }

        if (cur_buf.size() > 0) {
            sliced_buf.push_back(std::exchange(cur_buf, std::vector<Tvec>{}));
        }

        return sliced_buf;
    };

    std::vector<std::vector<Tvec>> sliced_buf = make_sliced();

    for (std::vector<Tvec> to_ins : sliced_buf) {

        sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
            PatchCoordTransform<Tvec> ptransf
                = sched.get_sim_box().template get_patch_transform<Tvec>();

            shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

            std::vector<Tvec> vec_acc;
            for (Tvec r : to_ins) {
                if (patch_coord.contain_pos(r)) {
                    vec_acc.push_back(r);
                }
            }

            if (vec_acc.size() == 0) {
                return;
            }

            log += shambase::format(
                "\n  rank = {}  patch id={}, add N={} particles, coords = {} {}",
                shamcomm::world_rank(),
                p.id_patch,
                vec_acc.size(),
                patch_coord.lower,
                patch_coord.upper);

            PatchDataLayer tmp(sched.get_layout_ptr());
            tmp.resize(vec_acc.size());
            tmp.fields_raz();

            {
                u32 len = vec_acc.size();
                PatchDataField<Tvec> &f
                    = tmp.template get_field<Tvec>(sched.pdl().template get_field_idx<Tvec>("xyz"));
                sycl::buffer<Tvec> buf(vec_acc.data(), len);
                f.override(buf, len);
            }

            {
                PatchDataField<Tscal> &f = tmp.template get_field<Tscal>(
                    sched.pdl().template get_field_idx<Tscal>("hpart"));
                using Kernel = SPHKernel<Tscal>;
                f.override(Kernel::hfactd * dr);
            }

            pdat.insert_elements(tmp);
        });

        sched.check_patchdata_locality_corectness();
        sched.scheduler_step(true, true);
    }

    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_load_value([&](Patch p) {
        return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });

    shamlog_debug_ln("setup", log);
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::Model<Tvec, SPHKernel>::add_cube_hcp_3d(
    Tscal dr, std::pair<Tvec, Tvec> _box) {
    StackEntry stack_loc{};

    shammath::CoordRange<Tvec> box = _box;

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    std::string log = "";

    auto make_sliced = [&]() {
        std::vector<Tvec> vec_lst;
        // Use FCC for now since HCP is similar
        generic::setup::generators::add_particles_fcc(
            dr,
            std::make_tuple(box.lower, box.upper),
            [&](Tvec r) {
                return box.contain_pos(r);
            },
            [&](Tvec r, Tscal h) {
                vec_lst.push_back(r);
            });

        std::vector<std::vector<Tvec>> sliced_buf;

        u32 sz_buf = sched.crit_patch_split * 4;

        std::vector<Tvec> cur_buf;
        for (u32 i = 0; i < vec_lst.size(); i++) {
            cur_buf.push_back(vec_lst[i]);

            if (cur_buf.size() > sz_buf) {
                sliced_buf.push_back(std::exchange(cur_buf, std::vector<Tvec>{}));
            }
        }

        if (cur_buf.size() > 0) {
            sliced_buf.push_back(std::exchange(cur_buf, std::vector<Tvec>{}));
        }

        return sliced_buf;
    };

    std::vector<std::vector<Tvec>> sliced_buf = make_sliced();

    for (std::vector<Tvec> to_ins : sliced_buf) {

        sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
            PatchCoordTransform<Tvec> ptransf
                = sched.get_sim_box().template get_patch_transform<Tvec>();

            shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

            std::vector<Tvec> vec_acc;
            for (Tvec r : to_ins) {
                if (patch_coord.contain_pos(r)) {
                    vec_acc.push_back(r);
                }
            }

            if (vec_acc.size() == 0) {
                return;
            }

            log += shambase::format(
                "\n  rank = {}  patch id={}, add N={} particles, coords = {} {}",
                shamcomm::world_rank(),
                p.id_patch,
                vec_acc.size(),
                patch_coord.lower,
                patch_coord.upper);

            PatchDataLayer tmp(sched.get_layout_ptr());
            tmp.resize(vec_acc.size());
            tmp.fields_raz();

            {
                u32 len = vec_acc.size();
                PatchDataField<Tvec> &f
                    = tmp.template get_field<Tvec>(sched.pdl().template get_field_idx<Tvec>("xyz"));
                sycl::buffer<Tvec> buf(vec_acc.data(), len);
                f.override(buf, len);
            }

            {
                PatchDataField<Tscal> &f = tmp.template get_field<Tscal>(
                    sched.pdl().template get_field_idx<Tscal>("hpart"));
                using Kernel = SPHKernel<Tscal>;
                f.override(Kernel::hfactd * dr);
            }

            pdat.insert_elements(tmp);
        });

        sched.check_patchdata_locality_corectness();
        sched.scheduler_step(true, true);
    }

    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_load_value([&](Patch p) {
        return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });

    shamlog_debug_ln("setup", log);
}

template<class Tvec, template<class> class SPHKernel>
u64 shammodels::gsph::Model<Tvec, SPHKernel>::create_wall_particles(
    u32 num_layers, u32 wall_flags) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;

    if (wall_flags == 0) {
        logger::warn_ln("GSPH", "create_wall_particles called with wall_flags=0 - skipping");
        return 0;
    }

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    // Get domain bounds
    auto bounding_box = sched.get_sim_box().template get_bounding_box<Tvec>();
    Tvec box_min      = std::get<0>(bounding_box);
    Tvec box_max      = std::get<1>(bounding_box);

    // Get field indices
    const u32 ixyz       = sched.pdl().template get_field_idx<Tvec>("xyz");
    const u32 ivxyz      = sched.pdl().template get_field_idx<Tvec>("vxyz");
    const u32 ihpart     = sched.pdl().template get_field_idx<Tscal>("hpart");
    const u32 iwall_flag = sched.pdl().template get_field_idx<u32>("wall_flag");

    // Check if uint field exists (for adiabatic EOS)
    bool has_uint = solver.solver_config.has_field_uint();
    u32 iuint     = 0;
    if (has_uint) {
        iuint = sched.pdl().template get_field_idx<Tscal>("uint");
    }

    // Collect all particles that need to be mirrored
    // Structure: position, velocity, hpart, uint, source wall bit
    struct WallParticleData {
        Tvec pos;
        Tvec vel;
        Tscal h;
        Tscal u;
    };

    std::vector<WallParticleData> wall_particles;

    // Estimate wall depth based on typical smoothing length
    // Use kernel radius * h * num_layers
    Tscal typical_h = Tscal{0};
    u32 part_cnt    = 0;

    // Get average h from first patch
    sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
        if (pdat.get_obj_cnt() > 0 && part_cnt == 0) {
            auto h_mirror
                = pdat.get_field<Tscal>(ihpart).get_buf().template mirror_to<sham::host>();
            for (u32 i = 0; i < std::min(u32(100), pdat.get_obj_cnt()); i++) {
                typical_h += h_mirror[i];
                part_cnt++;
            }
        }
    });

    if (part_cnt > 0) {
        typical_h /= part_cnt;
    } else {
        logger::warn_ln("GSPH", "No particles found to determine wall depth");
        return 0;
    }

    // Wall depth = kernel radius * h * num_layers
    Tscal wall_depth = Kernel::Rkern * typical_h * num_layers;

    logger::info_ln(
        "GSPH",
        shambase::format(
            "Creating wall particles: num_layers={}, wall_flags=0x{:02x}, wall_depth={}",
            num_layers,
            wall_flags,
            wall_depth));

    // For each patch, find boundary particles and mirror them
    sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
        u32 n = pdat.get_obj_cnt();
        if (n == 0)
            return;

        auto xyz_mirror  = pdat.get_field<Tvec>(ixyz).get_buf().template mirror_to<sham::host>();
        auto vxyz_mirror = pdat.get_field<Tvec>(ivxyz).get_buf().template mirror_to<sham::host>();
        auto h_mirror    = pdat.get_field<Tscal>(ihpart).get_buf().template mirror_to<sham::host>();

        std::vector<Tscal> u_vals(n, Tscal{0});
        if (has_uint) {
            auto u_mirror = pdat.get_field<Tscal>(iuint).get_buf().template mirror_to<sham::host>();
            for (u32 i = 0; i < n; i++) {
                u_vals[i] = u_mirror[i];
            }
        }

        for (u32 i = 0; i < n; i++) {
            Tvec pos = xyz_mirror[i];
            Tvec vel = vxyz_mirror[i];
            Tscal h  = h_mirror[i];
            Tscal u  = u_vals[i];

            // Check each wall and mirror if needed
            // Bit 0: -x wall, Bit 1: +x wall
            // Bit 2: -y wall, Bit 3: +y wall
            // Bit 4: -z wall, Bit 5: +z wall

            // -x wall (bit 0)
            if ((wall_flags & 0x01) && (pos[0] - box_min[0]) < wall_depth) {
                WallParticleData wp;
                wp.pos    = pos;
                wp.pos[0] = 2 * box_min[0] - pos[0]; // Mirror across x_min
                wp.vel    = vel;
                wp.vel[0] = -vel[0]; // Reflect x velocity for wall
                wp.h      = h;
                wp.u      = u;
                wall_particles.push_back(wp);
            }

            // +x wall (bit 1)
            if ((wall_flags & 0x02) && (box_max[0] - pos[0]) < wall_depth) {
                WallParticleData wp;
                wp.pos    = pos;
                wp.pos[0] = 2 * box_max[0] - pos[0]; // Mirror across x_max
                wp.vel    = vel;
                wp.vel[0] = -vel[0]; // Reflect x velocity for wall
                wp.h      = h;
                wp.u      = u;
                wall_particles.push_back(wp);
            }

            // -y wall (bit 2)
            if ((wall_flags & 0x04) && (pos[1] - box_min[1]) < wall_depth) {
                WallParticleData wp;
                wp.pos    = pos;
                wp.pos[1] = 2 * box_min[1] - pos[1];
                wp.vel    = vel;
                wp.vel[1] = -vel[1];
                wp.h      = h;
                wp.u      = u;
                wall_particles.push_back(wp);
            }

            // +y wall (bit 3)
            if ((wall_flags & 0x08) && (box_max[1] - pos[1]) < wall_depth) {
                WallParticleData wp;
                wp.pos    = pos;
                wp.pos[1] = 2 * box_max[1] - pos[1];
                wp.vel    = vel;
                wp.vel[1] = -vel[1];
                wp.h      = h;
                wp.u      = u;
                wall_particles.push_back(wp);
            }

            // -z wall (bit 4)
            if ((wall_flags & 0x10) && (pos[2] - box_min[2]) < wall_depth) {
                WallParticleData wp;
                wp.pos    = pos;
                wp.pos[2] = 2 * box_min[2] - pos[2];
                wp.vel    = vel;
                wp.vel[2] = -vel[2];
                wp.h      = h;
                wp.u      = u;
                wall_particles.push_back(wp);
            }

            // +z wall (bit 5)
            if ((wall_flags & 0x20) && (box_max[2] - pos[2]) < wall_depth) {
                WallParticleData wp;
                wp.pos    = pos;
                wp.pos[2] = 2 * box_max[2] - pos[2];
                wp.vel    = vel;
                wp.vel[2] = -vel[2];
                wp.h      = h;
                wp.u      = u;
                wall_particles.push_back(wp);
            }
        }
    });

    u64 total_wall = wall_particles.size();
    logger::info_ln("GSPH", "Found ", total_wall, " wall particles to create on this rank");

    if (total_wall == 0) {
        return shamalgs::collective::allreduce_sum(total_wall);
    }

    // Expand simulation box to include wall particles
    Tvec new_box_min = box_min;
    Tvec new_box_max = box_max;
    for (const auto &wp : wall_particles) {
        for (u32 d = 0; d < dim; d++) {
            new_box_min[d] = std::min(new_box_min[d], wp.pos[d] - typical_h);
            new_box_max[d] = std::max(new_box_max[d], wp.pos[d] + typical_h);
        }
    }

    // Update domain bounds to include wall particles
    sched.get_sim_box().template set_bounding_box<Tvec>({new_box_min, new_box_max});

    // Now add wall particles to the scheduler - batch insert per patch
    sched.for_each_local_patchdata([&](const Patch p, PatchDataLayer &pdat) {
        PatchCoordTransform<Tvec> ptransf
            = sched.get_sim_box().template get_patch_transform<Tvec>();
        shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

        // Collect wall particles belonging to this patch
        std::vector<WallParticleData> patch_wall_parts;
        for (const auto &wp : wall_particles) {
            if (patch_coord.contain_pos(wp.pos)) {
                patch_wall_parts.push_back(wp);
            }
        }

        if (patch_wall_parts.empty()) {
            return;
        }

        u32 n_add = patch_wall_parts.size();

        // Create batch of particles
        PatchDataLayer tmp(sched.get_layout_ptr());
        tmp.resize(n_add);
        tmp.fields_raz();

        // Set positions
        {
            auto acc = tmp.get_field<Tvec>(ixyz).get_buf().template mirror_to<sham::host>();
            for (u32 i = 0; i < n_add; i++) {
                acc[i] = patch_wall_parts[i].pos;
            }
        }

        // Set velocities
        {
            auto acc = tmp.get_field<Tvec>(ivxyz).get_buf().template mirror_to<sham::host>();
            for (u32 i = 0; i < n_add; i++) {
                acc[i] = patch_wall_parts[i].vel;
            }
        }

        // Set smoothing lengths
        {
            auto acc = tmp.get_field<Tscal>(ihpart).get_buf().template mirror_to<sham::host>();
            for (u32 i = 0; i < n_add; i++) {
                acc[i] = patch_wall_parts[i].h;
            }
        }

        // Set wall flags
        {
            auto acc = tmp.get_field<u32>(iwall_flag).get_buf().template mirror_to<sham::host>();
            for (u32 i = 0; i < n_add; i++) {
                acc[i] = 1; // Mark as wall particle
            }
        }

        // Set internal energy if adiabatic
        if (has_uint) {
            auto acc = tmp.get_field<Tscal>(iuint).get_buf().template mirror_to<sham::host>();
            for (u32 i = 0; i < n_add; i++) {
                acc[i] = patch_wall_parts[i].u;
            }
        }

        pdat.insert_elements(tmp);
    });

    // Update scheduler
    sched.check_patchdata_locality_corectness();
    sched.scheduler_step(true, true);
    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_load_value([&](Patch p) {
        return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });

    u64 global_total = shamalgs::collective::allreduce_sum(total_wall);
    logger::info_ln("GSPH", "Created ", global_total, " wall particles total");

    return global_total;
}

// Explicit template instantiations for all supported kernel types
template class shammodels::gsph::Model<f64_3, shammath::M4>;
template class shammodels::gsph::Model<f64_3, shammath::M6>;
template class shammodels::gsph::Model<f64_3, shammath::M8>;
template class shammodels::gsph::Model<f64_3, shammath::C2>;
template class shammodels::gsph::Model<f64_3, shammath::C4>;
template class shammodels::gsph::Model<f64_3, shammath::C6>;

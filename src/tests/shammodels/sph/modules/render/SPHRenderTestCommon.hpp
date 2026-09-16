// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SPHRenderTestCommon.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Shared helpers for SPH render node unit tests.
 *
 */

#include "shamcomm/worldInfo.hpp"
#include "shammath/AABB.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/solvergraph/DeviceBufferEdge.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace sph_render_test {

    using Tvec   = f64_3;
    using Tscal  = f64;
    using Kernel = shammath::M4<Tscal>;

    constexpr u32 npart                    = 100;
    constexpr u32 nx                       = 100;
    constexpr u32 ny                       = 100;
    constexpr u32 n_pixels                 = nx * ny;
    constexpr u64 seed                     = 0xC0FFEE;
    constexpr Tscal h_const                = 0.15;
    constexpr Tscal f_const                = 1.0;
    constexpr Tscal gpart_mass_val         = 1.0e-3;
    constexpr u32 tree_reduction_level_val = 3;
    constexpr Tscal img_lo                 = -1.0;
    constexpr Tscal img_hi                 = 1.0;
    constexpr Tscal part_lo                = -0.5;
    constexpr Tscal part_hi                = 0.5;
    constexpr Tscal tol                    = 1e-10;

    struct ParticleDataset {
        std::vector<Tvec> xyz;
        std::vector<Tscal> h;
        std::vector<Tscal> field;
    };

    inline ParticleDataset make_global_dataset() {
        ParticleDataset ds;
        ds.xyz.resize(npart);
        ds.h.resize(npart, h_const);
        ds.field.resize(npart, f_const);

        std::mt19937 eng(seed);
        std::uniform_real_distribution<Tscal> dist(part_lo, part_hi);
        for (u32 i = 0; i < npart; ++i) {
            ds.xyz[i] = Tvec{dist(eng), dist(eng), dist(eng)};
        }
        return ds;
    }

    inline Tscal pixel_coord(u32 i, u32 n) {
        return img_lo + (img_hi - img_lo) * (Tscal(i) + Tscal(0.5)) / Tscal(n);
    }

    inline std::vector<Tvec> make_slice_points() {
        std::vector<Tvec> pts(n_pixels);
        for (u32 iy = 0; iy < ny; ++iy) {
            for (u32 ix = 0; ix < nx; ++ix) {
                pts[iy * nx + ix] = Tvec{pixel_coord(ix, nx), pixel_coord(iy, ny), Tscal(0)};
            }
        }
        return pts;
    }

    inline std::vector<shammath::Ray<Tvec>> make_column_rays() {
        std::vector<shammath::Ray<Tvec>> rays;
        rays.reserve(n_pixels);
        Tvec dir{0, 0, 1};
        for (u32 iy = 0; iy < ny; ++iy) {
            for (u32 ix = 0; ix < nx; ++ix) {
                Tvec origin{pixel_coord(ix, nx), pixel_coord(iy, ny), Tscal(-1)};
                rays.emplace_back(origin, dir);
            }
        }
        return rays;
    }

    inline std::vector<shammath::RingRay<Tvec>> make_azymuthal_ring_rays() {
        std::vector<shammath::RingRay<Tvec>> rays;
        rays.reserve(n_pixels);
        Tvec e_x{1, 0, 0};
        Tvec e_y{0, 0, 1};
        for (u32 iy = 0; iy < ny; ++iy) {
            for (u32 ix = 0; ix < nx; ++ix) {
                Tscal radius = pixel_coord(ix, nx); // in [-1,1], covers outside particle box
                Tscal z      = pixel_coord(iy, ny);
                // center on z-axis; radius may be negative, use abs
                Tscal r = std::abs(radius);
                Tvec center{0, 0, z};
                rays.emplace_back(center, r, e_x, e_y);
            }
        }
        return rays;
    }

    struct LocalFields {
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> positions;
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> h_part;
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> field_data;
        u64 patch_id             = 0;
        bool has_local_particles = false;
    };

    inline LocalFields make_round_robin_fields(const ParticleDataset &global) {
        LocalFields loc;
        loc.part_counts = shamrock::solvergraph::Indexes<u32>::make_shared("part_counts", "N");
        loc.positions   = std::make_shared<shamrock::solvergraph::Field<Tvec>>(1, "xyz", "r");
        loc.h_part      = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "h", "h");
        loc.field_data  = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "f", "f");

        i32 wrank    = shamcomm::world_rank();
        i32 wsize    = shamcomm::world_size();
        loc.patch_id = static_cast<u64>(wrank);

        std::vector<Tvec> local_xyz;
        std::vector<Tscal> local_h;
        std::vector<Tscal> local_f;
        local_xyz.reserve(npart / static_cast<u32>(wsize) + 1);
        local_h.reserve(local_xyz.capacity());
        local_f.reserve(local_xyz.capacity());

        for (u32 i = 0; i < npart; ++i) {
            if (static_cast<i32>(i % static_cast<u32>(wsize)) != wrank) {
                continue;
            }
            local_xyz.push_back(global.xyz[i]);
            local_h.push_back(global.h[i]);
            local_f.push_back(global.field[i]);
        }

        if (local_xyz.empty()) {
            loc.has_local_particles = false;
            return loc;
        }

        loc.has_local_particles = true;
        u32 local_n             = static_cast<u32>(local_xyz.size());
        loc.part_counts->indexes.add_obj(loc.patch_id, u32{local_n});

        loc.positions->ensure_sizes(loc.part_counts->indexes);
        loc.h_part->ensure_sizes(loc.part_counts->indexes);
        loc.field_data->ensure_sizes(loc.part_counts->indexes);

        loc.positions->get_buf(loc.patch_id).copy_from_stdvec(local_xyz);
        loc.h_part->get_buf(loc.patch_id).copy_from_stdvec(local_h);
        loc.field_data->get_buf(loc.patch_id).copy_from_stdvec(local_f);

        return loc;
    }

    inline std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> make_gpart_mass() {
        auto e  = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("gpart_mass", "m");
        e->data = gpart_mass_val;
        return e;
    }

    inline std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> make_tree_reduction_level() {
        auto e  = shamrock::solvergraph::IDataEdge<u32>::make_shared("tree_reduction_level", "l");
        e->data = tree_reduction_level_val;
        return e;
    }

    template<class T>
    inline std::shared_ptr<shamrock::solvergraph::DeviceBufferEdge<T>> make_query_edge(
        const std::string &name, const std::vector<T> &host) {
        auto e = std::make_shared<shamrock::solvergraph::DeviceBufferEdge<T>>(name, name);
        e->value.resize(host.size());
        e->value.copy_from_stdvec(host);
        return e;
    }

    inline std::shared_ptr<shamrock::solvergraph::DeviceBufferEdge<Tscal>> make_output_edge() {
        return std::make_shared<shamrock::solvergraph::DeviceBufferEdge<Tscal>>(
            "interpolated_field", "f");
    }

    inline std::vector<Tscal> reference_slice(
        const ParticleDataset &ds, const std::vector<Tvec> &points) {
        std::vector<Tscal> out(points.size(), 0);
        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

        for (u32 ip = 0; ip < points.size(); ++ip) {
            Tscal acc = 0;
            for (u32 ib = 0; ib < ds.xyz.size(); ++ib) {
                Tvec dr    = points[ip] - ds.xyz[ib];
                Tscal rab2 = sycl::dot(dr, dr);
                Tscal h_b  = ds.h[ib];
                if (rab2 > h_b * h_b * Rker2) {
                    continue;
                }
                Tscal rab   = sycl::sqrt(rab2);
                Tscal rho_b = shamrock::sph::rho_h(gpart_mass_val, h_b, Kernel::hfactd);
                acc += gpart_mass_val * ds.field[ib] * Kernel::W_3d(rab, h_b) / rho_b;
            }
            out[ip] = acc;
        }
        return out;
    }

    inline std::vector<Tscal> reference_column(
        const ParticleDataset &ds, const std::vector<shammath::Ray<Tvec>> &rays) {
        std::vector<Tscal> out(rays.size(), 0);
        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

        for (u32 ir = 0; ir < rays.size(); ++ir) {
            Tscal acc               = 0;
            shammath::Ray<Tvec> ray = rays[ir];
            for (u32 ib = 0; ib < ds.xyz.size(); ++ib) {
                Tvec dr = ray.origin - ds.xyz[ib];
                dr -= ray.direction * sycl::dot(dr, ray.direction);
                Tscal rab2 = sycl::dot(dr, dr);
                Tscal h_b  = ds.h[ib];
                if (rab2 > h_b * h_b * Rker2) {
                    continue;
                }
                Tscal rab   = sycl::sqrt(rab2);
                Tscal rho_b = shamrock::sph::rho_h(gpart_mass_val, h_b, Kernel::hfactd);
                acc += gpart_mass_val * ds.field[ib] * Kernel::Y_3d(rab, h_b, 4) / rho_b;
            }
            out[ir] = acc;
        }
        return out;
    }

    inline std::vector<Tscal> reference_azymuthal(
        const ParticleDataset &ds, const std::vector<shammath::RingRay<Tvec>> &ring_rays) {
        std::vector<Tscal> out(ring_rays.size(), 0);
        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

        for (u32 ir = 0; ir < ring_rays.size(); ++ir) {
            Tscal acc                        = 0;
            shammath::RingRay<Tvec> ring_ray = ring_rays[ir];
            Tvec ez                          = sycl::cross(ring_ray.e_x, ring_ray.e_y);
            for (u32 ib = 0; ib < ds.xyz.size(); ++ib) {
                Tvec r_center = ring_ray.center - ds.xyz[ib];

                Tscal z_val   = sycl::dot(r_center, ez);
                Tscal x_val   = sycl::dot(r_center, ring_ray.e_x);
                Tscal y_val   = sycl::dot(r_center, ring_ray.e_y);
                Tscal r_val   = sycl::sqrt(x_val * x_val + y_val * y_val);
                Tscal delta_r = r_val - ring_ray.radius;

                Tscal rab2_ring = z_val * z_val + delta_r * delta_r;
                Tscal h_b       = ds.h[ib];
                if (rab2_ring > h_b * h_b * Rker2) {
                    continue;
                }
                Tscal rab   = sycl::sqrt(rab2_ring);
                Tscal rho_b = shamrock::sph::rho_h(gpart_mass_val, h_b, Kernel::hfactd);
                // Match production: curvature not accounted for
                acc += gpart_mass_val * ds.field[ib] * Kernel::Y_3d(rab, h_b, 4) / rho_b;
            }
            out[ir] = acc;
        }
        return out;
    }

    inline bool almost_equal_vec(
        const std::vector<Tscal> &a, const std::vector<Tscal> &b, Tscal prec = tol) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > prec) {
                return false;
            }
        }
        return true;
    }

} // namespace sph_render_test

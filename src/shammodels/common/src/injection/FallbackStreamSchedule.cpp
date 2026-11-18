// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FallbackStreamSchedule.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/constants.hpp"
#include "shambase/exception.hpp"
#include "sham/format/format.hpp"
#include "shammodels/common/injection/FallbackStreamSchedule.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

    template<class Tvec>
    std::array<Tvec, 2> orthogonal_basis(Tvec dir) {
        // @@@ maybe move that to shammath at some point
        Tvec n = dir / sycl::length(dir);

        Tvec axis = {0, 0, 1};
        if (std::fabs(n.x()) <= std::fabs(n.y()) && std::fabs(n.x()) <= std::fabs(n.z())) {
            axis = {1, 0, 0};
        } else if (std::fabs(n.y()) <= std::fabs(n.z())) {
            axis = {0, 1, 0};
        }

        Tvec e1 = sycl::cross(n, axis);
        e1      = e1 / sycl::length(e1);
        Tvec e2 = sycl::cross(n, e1);
        return {e1, e2};
    }

} // namespace

namespace shammodels::common::injection {

    template<class Tvec>
    FallbackStreamSchedule<Tvec>::FallbackStreamSchedule(
        const FallbackStream<Tvec> &stream, Tscal part_mass, Tscal hfact)
        : stream(stream), part_mass(part_mass), hfact(hfact),
          basis(orthogonal_basis(stream.inject_vel)), t_last_birth(stream.rate.t_start()) {

        stream.check();

        if (!(part_mass > 0) || !std::isfinite(part_mass)) {
            shambase::throw_with_loc<std::invalid_argument>(sham::format(
                "the fallback stream needs a positive particle mass, got {}", part_mass));
        }
        if (!(hfact > 0) || !std::isfinite(hfact)) {
            shambase::throw_with_loc<std::invalid_argument>(
                sham::format("the fallback stream needs a positive hfact, got {}", hfact));
        }
    }

    template<class Tvec>
    bool FallbackStreamSchedule<Tvec>::generate_next_wall() {
        if (stopped) {
            return false;
        }

        Tscal mdot = stream.rate.mdot(t_last_birth);
        if (!(mdot > 0)) {
            stopped = true;
            return false;
        }

        Tscal speed  = sycl::length(stream.inject_vel);
        Tscal radius = stream.stream_radius;

        // spacing of an isotropic lattice at the density of the stream, rho = mdot / (pi R^2 v)
        Tscal target_spacing = std::cbrt(
            part_mass * shambase::constants::pi<Tscal> * radius * radius * speed / mdot);
        Tscal lattice_n_real = 2 * radius / target_spacing;

        if (!(lattice_n_real < max_lattice_n)) {
            shambase::throw_with_loc<std::runtime_error>(sham::format(
                "the fallback stream would need {} particles along its diameter (max {}), the "
                "particle mass is too small for this stream",
                lattice_n_real,
                max_lattice_n));
        }

        u32 lattice_n  = static_cast<u32>(std::max<long long>(1, std::llround(lattice_n_real)));
        u32 part_count = static_cast<u32>(get_lattice(lattice_n).size());

        u64 part_count_after = generated_part_count + part_count;
        if (part_count_after - 1 > part_id_max_index) {
            shambase::throw_with_loc<std::runtime_error>(sham::format(
                "the fallback stream injected more than {} particles, the maximum that the "
                "particle ids can index",
                part_id_max_index));
        }

        Tscal t_birth = stream.rate.time_at_injected_mass(part_mass * Tscal(part_count_after));
        if (!std::isfinite(t_birth)) {
            // the fallback stops before supplying the mass of this wall
            stopped = true;
            return false;
        }

        // smoothing length of the volume per particle, spacing^2 * distance to the previous wall
        Tscal spacing   = 2 * radius / lattice_n;
        Tscal wall_dist = speed * (t_birth - t_last_birth);
        Tscal hpart
            = (wall_dist > 0) ? hfact * std::cbrt(spacing * spacing * wall_dist) : hfact * spacing;

        unborn_walls.push_back(Wall{t_birth, hpart, lattice_n, part_count, generated_part_count});

        generated_part_count = part_count_after;
        t_last_birth         = t_birth;
        return true;
    }

    template<class Tvec>
    auto FallbackStreamSchedule<Tvec>::pop_born_walls(Tscal t) -> std::vector<Wall> {
        std::vector<Wall> born;
        while (!unborn_walls.empty() || generate_next_wall()) {
            const Wall &next = unborn_walls.front();
            if (next.t_birth > t) {
                break;
            }
            born_part_count += next.part_count;
            born.push_back(next);
            unborn_walls.pop_front();
        }
        return born;
    }

    template<class Tvec>
    auto FallbackStreamSchedule<Tvec>::peek_born_walls(Tscal t) -> std::vector<Wall> {
        std::vector<Wall> born;
        for (size_t i = 0; i < unborn_walls.size() || generate_next_wall(); i++) {
            if (unborn_walls[i].t_birth > t) {
                break;
            }
            born.push_back(unborn_walls[i]);
        }
        return born;
    }

    template<class Tvec>
    auto FallbackStreamSchedule<Tvec>::peek_unborn_walls(u32 n) -> std::vector<Wall> {
        while (unborn_walls.size() < n && generate_next_wall()) {
        }
        size_t count = std::min<size_t>(n, unborn_walls.size());
        return std::vector<Wall>(unborn_walls.begin(), unborn_walls.begin() + count);
    }

    template<class Tvec>
    auto FallbackStreamSchedule<Tvec>::get_lattice(u32 lattice_n) -> const std::vector<Site> & {
        auto it = lattices.find(lattice_n);
        if (it != lattices.end()) {
            return it->second;
        }

        // cell centered square lattice of the square [-R, R]^2, cropped to the disc of radius R
        Tscal radius  = stream.stream_radius;
        Tscal spacing = 2 * radius / lattice_n;

        std::vector<Site> sites;
        for (u32 a = 0; a < lattice_n; a++) {
            for (u32 b = 0; b < lattice_n; b++) {
                Tscal x = (a + Tscal(0.5)) * spacing - radius;
                Tscal y = (b + Tscal(0.5)) * spacing - radius;
                if (x * x + y * y < radius * radius) {
                    sites.push_back({x, y});
                }
            }
        }

        return lattices.emplace(lattice_n, std::move(sites)).first->second;
    }

    template class FallbackStreamSchedule<f64_3>;

} // namespace shammodels::common::injection

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ParticleInjectionTests.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Unit tests for the fallback stream injection: rate, wall schedule and config json
 */

#include "shambase/numeric_limits.hpp"
#include "shammodels/common/injection/FallbackStreamSchedule.hpp"
#include "shammodels/common/injection/ParticleInjectionConfig.hpp"
#include "shamtest/shamtest.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

    using namespace shammodels::common::injection;

    using Tvec  = f64_3;
    using Tscal = f64;
    using Wall  = StreamWall<Tscal>;

    constexpr Tscal part_mass = 1e-4;
    constexpr Tscal hfact     = 1.2;

    /// rate starting at t_start = t_disrupt + t_mb = 1
    FallbackRate<Tscal> make_rate(Tscal index, Tscal t_end) {
        return FallbackRate<Tscal>{2.0, 1.5, -0.5, index, t_end};
    }

    FallbackStream<Tvec> make_stream(Tscal index) {
        return FallbackStream<Tvec>{
            {1.0, -2.0, 0.5},
            {0.3, 4.0, -1.0},
            0.2,
            0.01,
            make_rate(index, shambase::get_infty<Tscal>()),
            4};
    }

    Tscal rel_diff(Tscal a, Tscal b) { return std::abs(a - b) / std::abs(b); }

    bool same_walls(const std::vector<Wall> &a, const std::vector<Wall> &b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i].t_birth != b[i].t_birth || a[i].hpart != b[i].hpart
                || a[i].lattice_n != b[i].lattice_n || a[i].part_count != b[i].part_count
                || a[i].first_part != b[i].first_part) {
                return false;
            }
        }
        return true;
    }

    void test_fallback_rate() {
        for (Tscal index : {-5. / 3., -1., 0., 0.5}) {
            for (Tscal t_end : {shambase::get_infty<Tscal>(), 7.0}) {
                FallbackRate<Tscal> rate = make_rate(index, t_end);

                REQUIRE_EQUAL(rate.mdot(0.99), 0.);
                REQUIRE_FLOAT_EQUAL(rate.mdot(rate.t_start()), rate.mdot_peak, 1e-14);
                REQUIRE_EQUAL(rate.injected_mass(rate.t_start()), 0.);

                // the injected mass is the integral of the rate (midpoint rule)
                Tscal t1    = 3.0;
                u32 n       = 100000;
                Tscal dt    = (t1 - rate.t_start()) / n;
                Tscal integ = 0;
                for (u32 i = 0; i < n; i++) {
                    integ += rate.mdot(rate.t_start() + (i + 0.5) * dt) * dt;
                }
                REQUIRE(rel_diff(rate.injected_mass(t1), integ) < 1e-8);

                // time_at_injected_mass inverts injected_mass
                for (Tscal t : {1.001, 1.5, 2.5, 6.9}) {
                    REQUIRE(rel_diff(rate.time_at_injected_mass(rate.injected_mass(t)), t) < 1e-12);
                }

                // the injection stops at t_end, and never supplies more than the total mass
                if (std::isfinite(t_end)) {
                    REQUIRE_EQUAL(rate.mdot(t_end), 0.);
                    REQUIRE_EQUAL(rate.injected_mass(t_end + 1), rate.injected_mass(t_end));
                }
                if (std::isfinite(rate.total_mass())) {
                    REQUIRE(!std::isfinite(rate.time_at_injected_mass(rate.total_mass() * 1.001)));
                }
            }
        }
    }

    void test_fallback_rate_total_mass() {
        // flat dM/dE fallback: total mass mdot_peak t_mb / (2/3)
        FallbackRate<Tscal> rate = make_rate(-5. / 3., shambase::get_infty<Tscal>());
        REQUIRE_FLOAT_EQUAL(rate.total_mass(), 2.0 * 1.5 * 1.5, 1e-12);
        REQUIRE(!std::isfinite(make_rate(0., shambase::get_infty<Tscal>()).total_mass()));
    }

    void test_schedule_step_independence() {
        FallbackStream<Tvec> stream = make_stream(-5. / 3.);

        // all the walls in a single step
        FallbackStreamSchedule<Tvec> coarse(stream, part_mass, hfact);
        std::vector<Wall> walls_coarse = coarse.pop_born_walls(3.0);

        // the same time range in many uneven steps, peeking at the boundary walls in between
        FallbackStreamSchedule<Tvec> fine(stream, part_mass, hfact);
        std::vector<Wall> walls_fine;
        for (u32 i = 1; i <= 1000; i++) {
            Tscal t = 1.0 + 2.0 * std::pow(i / 1000., 1.3);
            for (const Wall &w : fine.pop_born_walls(t)) {
                walls_fine.push_back(w);
            }
            fine.peek_unborn_walls(4);
        }

        REQUIRE(walls_coarse.size() > 10);
        REQUIRE(same_walls(walls_coarse, walls_fine));
        REQUIRE(same_walls(coarse.peek_unborn_walls(4), fine.peek_unborn_walls(4)));
    }

    void test_schedule_mass_and_ids() {
        for (Tscal index : {-5. / 3., 0.}) {
            FallbackStream<Tvec> stream = make_stream(index);
            FallbackStreamSchedule<Tvec> schedule(stream, part_mass, hfact);

            u64 expected_first = 0;
            Tscal t_prev       = stream.rate.t_start();
            bool contiguous    = true;
            bool ordered       = true;

            for (Tscal t : {1.2, 1.7, 2.9, 4.0}) {
                for (const Wall &w : schedule.pop_born_walls(t)) {
                    contiguous = contiguous && (w.first_part == expected_first);
                    ordered    = ordered && (w.t_birth >= t_prev) && (w.t_birth <= t);
                    expected_first += w.part_count;
                    t_prev = w.t_birth;
                }
                REQUIRE_EQUAL(schedule.get_born_part_count(), expected_first);

                // the born walls hold the injected mass, minus the next wall still accumulating
                Tscal mass_born        = part_mass * expected_first;
                Tscal mass_injected    = stream.rate.injected_mass(t);
                std::vector<Wall> next = schedule.peek_unborn_walls(1);
                REQUIRE_EQUAL(next.size(), 1);
                REQUIRE(next[0].t_birth > t);
                REQUIRE(mass_born <= mass_injected * (1 + 1e-12));
                REQUIRE(
                    mass_injected
                    < part_mass * (expected_first + next[0].part_count) * (1 + 1e-12));
            }

            REQUIRE(contiguous);
            REQUIRE(ordered);
        }
    }

    void test_schedule_isotropic_lattice() {
        // constant rate: the walls are spaced like the particles within a wall
        FallbackStream<Tvec> stream = make_stream(0.);
        FallbackStreamSchedule<Tvec> schedule(stream, part_mass, hfact);

        std::vector<Wall> walls = schedule.pop_born_walls(2.0);
        REQUIRE(walls.size() > 10);

        Tscal speed = sycl::length(stream.inject_vel);
        for (size_t i = 1; i < walls.size(); i++) {
            Tscal lattice_spacing = 2 * stream.stream_radius / walls[i].lattice_n;
            Tscal wall_spacing    = speed * (walls[i].t_birth - walls[i - 1].t_birth);
            REQUIRE(rel_diff(wall_spacing, lattice_spacing) < 0.25);
            REQUIRE(rel_diff(walls[i].hpart, hfact * lattice_spacing) < 0.25);
        }
    }

    void test_schedule_lattice() {
        FallbackStream<Tvec> stream = make_stream(-5. / 3.);
        FallbackStreamSchedule<Tvec> schedule(stream, part_mass, hfact);

        for (u32 n : {1u, 2u, 7u, 20u}) {
            const auto &sites = schedule.get_lattice(n);
            REQUIRE(sites.size() > 0);

            Tscal sum_x  = 0;
            Tscal sum_y  = 0;
            bool in_disc = true;
            for (const auto &s : sites) {
                sum_x += s[0];
                sum_y += s[1];
                in_disc = in_disc && (s[0] * s[0] + s[1] * s[1] < 0.2 * 0.2);
            }
            REQUIRE(in_disc);
            REQUIRE_FLOAT_EQUAL(sum_x, 0., 1e-12);
            REQUIRE_FLOAT_EQUAL(sum_y, 0., 1e-12);
        }

        // the cross-section basis is orthonormal and orthogonal to the stream
        const auto &[e1, e2] = schedule.get_cross_section_basis();
        Tvec dir             = stream.inject_vel / sycl::length(stream.inject_vel);
        REQUIRE_FLOAT_EQUAL(sycl::dot(e1, e1), 1., 1e-14);
        REQUIRE_FLOAT_EQUAL(sycl::dot(e2, e2), 1., 1e-14);
        REQUIRE_FLOAT_EQUAL(sycl::dot(e1, e2), 0., 1e-14);
        REQUIRE_FLOAT_EQUAL(sycl::dot(e1, dir), 0., 1e-14);
        REQUIRE_FLOAT_EQUAL(sycl::dot(e2, dir), 0., 1e-14);
    }

    void test_config_json_roundtrip() {
        ParticleInjectionConfig<Tvec> config;

        FallbackStream<Tvec> stream_inf = make_stream(-5. / 3.);
        FallbackStream<Tvec> stream_end = make_stream(0.);
        stream_end.rate.t_end           = 5.0;
        config.add_fallback_stream(stream_inf);
        config.add_fallback_stream(stream_end);

        nlohmann::json j = config;
        ParticleInjectionConfig<Tvec> out
            = nlohmann::json::parse(j.dump()).get<ParticleInjectionConfig<Tvec>>();

        REQUIRE_EQUAL(out.injector_list.size(), 2);
        REQUIRE_EQUAL(nlohmann::json(out).dump(), j.dump());

        const auto &s0 = std::get<FallbackStream<Tvec>>(out.injector_list[0]);
        const auto &s1 = std::get<FallbackStream<Tvec>>(out.injector_list[1]);
        REQUIRE(!std::isfinite(s0.rate.t_end));
        REQUIRE_EQUAL(s1.rate.t_end, 5.0);
        REQUIRE_EQUAL(s0.rate.index, -5. / 3.);
        REQUIRE_EQUAL(s0.n_boundary_walls, 4);
    }

    void test_config_check() {
        ParticleInjectionConfig<Tvec> config;

        FallbackStream<Tvec> bad_radius = make_stream(0.);
        bad_radius.stream_radius        = -1;
        REQUIRE_EXCEPTION_THROW(config.add_fallback_stream(bad_radius), std::invalid_argument);

        FallbackStream<Tvec> bad_vel = make_stream(0.);
        bad_vel.inject_vel           = {0, 0, 0};
        REQUIRE_EXCEPTION_THROW(config.add_fallback_stream(bad_vel), std::invalid_argument);

        FallbackStream<Tvec> bad_end = make_stream(0.);
        bad_end.rate.t_end           = bad_end.rate.t_start();
        REQUIRE_EXCEPTION_THROW(config.add_fallback_stream(bad_end), std::invalid_argument);

        REQUIRE(!config.has_injectors());

        // a stream rejected by check_config is not kept in the list
        config.injector_list.assign(max_injector_count, make_stream(0.));
        REQUIRE_EXCEPTION_THROW(config.add_fallback_stream(make_stream(0.)), std::invalid_argument);
        REQUIRE_EQUAL(config.injector_list.size(), max_injector_count);
    }

} // namespace

NEW_TEST(Unittest, "shammodels/common/injection/fallback_rate", 1) {
    test_fallback_rate();
    test_fallback_rate_total_mass();
}

NEW_TEST(Unittest, "shammodels/common/injection/fallback_stream_schedule", 1) {
    test_schedule_step_independence();
    test_schedule_mass_and_ids();
    test_schedule_isotropic_lattice();
    test_schedule_lattice();
}

NEW_TEST(Unittest, "shammodels/common/injection/config", 1) {
    test_config_json_roundtrip();
    test_config_check();
}

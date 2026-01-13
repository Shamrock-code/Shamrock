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
 * @file orbits.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/math.hpp"
#include "shammath/matrix.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"

namespace shamphys {

    /**
     * @brief Convert a binary system from Keplerian to Cartesian coordinates.
     *
     * This function calculates the positions and velocities of two objects in a binary system
     * using their masses, semi-major axis, eccentricity, and true anomaly.
     *
     * @param m1 Mass of the first object.
     * @param m2 Mass of the second object.
     * @param a Semi-major axis of the system.
     * @param e Eccentricity of the system.
     * @param nu True anomaly of the system (in radians).
     * @param G Gravitational constant.
     * @return A tuple containing four elements: the positions of the two objects and the
     * velocities of the two objects.
     */
    inline auto get_binary_pair(double m1, double m2, double a, double e, double nu, double G) {

        double M = m1 + m2;
        // double mu = m1 * m2 / M;

        double r = a * (1 - e * e) / (1 + e * sycl::cos(nu));

        double x_orb = r * sycl::cos(nu);
        double y_orb = r * sycl::sin(nu);

        double h      = sycl::sqrt(G * M * a * (1 - e * e));
        double vx_orb = -G * M / h * sycl::sin(nu);
        double vy_orb = G * M / h * (e + sycl::cos(nu));

        f64_3 r_orb = {x_orb, y_orb, 0};
        f64_3 v_orb = {vx_orb, vy_orb, 0};

        f64_3 r1 = -m2 / M * r_orb;
        f64_3 r2 = m1 / M * r_orb;

        f64_3 v1 = -m2 / M * v_orb;
        f64_3 v2 = m1 / M * v_orb;

        return std::make_tuple(r1, r2, v1, v2);
    }

    /// Shortcut for get_binary_pair with G from UnitSystem
    auto get_binary_pair(
        double m1,
        double m2,
        double a,
        double e,
        double nu,
        const shamunits::UnitSystem<double> usys = shamunits::UnitSystem<double>{}) {

        double G = shamunits::Constants{usys}.G();

        return get_binary_pair(m1, m2, a, e, nu, G);
    }

    /**
     * @brief Create a rotation matrix for a 3D rotation given Euler angles (roll, pitch, yaw).
     * @param roll Rotation about the X-axis (in radians)
     * @param pitch Rotation about the Y-axis (in radians)
     * @param yaw Rotation about the Z-axis (in radians)
     * @return 3x3 rotation matrix
     */
    static f64_3x3 rotation_matrix(double roll, double pitch, double yaw) {

        // clang-format off
        // Rotation matrix around X-axis (roll)
        f64_3x3 Rx = {1, 0, 0,
                           0, sycl::cos(roll), -sycl::sin(roll),
                           0, sycl::sin(roll), sycl::cos(roll)};

        // Rotation matrix around Y-axis (pitch)
        f64_3x3 Ry = {sycl::cos(pitch), 0, sycl::sin(pitch),
                           0, 1, 0,
                           -sycl::sin(pitch), 0, sycl::cos(pitch)};

        // Rotation matrix around Z-axis (yaw)
        f64_3x3 Rz = {sycl::cos(yaw), -sycl::sin(yaw), 0,
                           sycl::sin(yaw), sycl::cos(yaw), 0,
                           0, 0, 1};
        // clang-format on

        f64_3x3 Ryx = {};
        f64_3x3 R   = {};

        // Combine the rotations (R = Rz * Ry * Rx)
        shammath::mat_prod(Ry.get_mdspan(), Rx.get_mdspan(), Ryx.get_mdspan());
        shammath::mat_prod(Rz.get_mdspan(), Ryx.get_mdspan(), R.get_mdspan());

        return R;
    }

    /**
     * @brief Rotate a 3D point using Euler angles.
     * @param point 3D point as a f64_3
     * @param roll Rotation about the X-axis (in radians)
     * @param pitch Rotation about the Y-axis (in radians)
     * @param yaw Rotation about the Z-axis (in radians)
     * @return Rotated point as a f64_3
     */
    static f64_3 rotate_point(const f64_3 &point, double roll, double pitch, double yaw) {
        shammath::mat<f64, 3, 1> r = {point.x(), point.y(), point.z()};

        // Get the rotation matrix
        f64_3x3 R = rotation_matrix(roll, pitch, yaw);

        // Perform the rotation by multiplying the point with the rotation matrix
        shammath::mat<f64, 3, 1> rotated_point = {};
        shammath::mat_prod(R.get_mdspan(), r.get_mdspan(), rotated_point.get_mdspan());

        return {rotated_point.data[0], rotated_point.data[1], rotated_point.data[2]};
    }

    /**
     * @brief Rotate a binary orbit by Euler angles and return the positions and velocities of the
     * two objects.
     * @param m1 Mass of the first object.
     * @param m2 Mass of the second object.
     * @param a Semi-major axis of the system.
     * @param e Eccentricity of the system.
     * @param nu True anomaly of the system (in radians).
     * @param G Gravitational constant.
     * @param roll Rotation about the X-axis (in radians).
     * @param pitch Rotation about the Y-axis (in radians).
     * @param yaw Rotation about the Z-axis (in radians).
     * @return A tuple containing four elements: the rotated positions of the two objects and the
     * rotated velocities of the two objects.
     */
    auto get_binary_rotated(
        double m1,
        double m2,
        double a,
        double e,
        double nu,
        double G,
        double roll,
        double pitch,
        double yaw) {

        auto [r1, r2, v1, v2] = get_binary_pair(m1, m2, a, e, nu, G);

        r1 = rotate_point({r1.x(), r1.y(), r1.z()}, roll, pitch, yaw);
        r2 = rotate_point({r2.x(), r2.y(), r2.z()}, roll, pitch, yaw);
        v1 = rotate_point({v1.x(), v1.y(), v1.z()}, roll, pitch, yaw);
        v2 = rotate_point({v2.x(), v2.y(), v2.z()}, roll, pitch, yaw);

        return std::make_tuple(r1, r2, v1, v2);
    }

    /// Shortcut for get_binary_rotated with G from UnitSystem
    auto get_binary_rotated(
        double m1,
        double m2,
        double a,
        double e,
        double nu,
        const shamunits::UnitSystem<double> usys,
        double roll,
        double pitch,
        double yaw) {
        return get_binary_rotated(
            m1, m2, a, e, nu, shamunits::Constants{usys}.G(), roll, pitch, yaw);
    }

} // namespace shamphys

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Camera3dTests.cpp
 * @author Yann Bernard (yann.bernard@univ-grenoble-alpes.fr)
 * @brief Unit tests for shammath::Camera3d.
 *
 */

#include "shambase/constants.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shammath/Camera3d.hpp"
#include "shammath/matrix.hpp"
#include "shamtest/shamtest.hpp"
#include <cmath>
#include <stdexcept>

namespace {

    using Camera = shammath::Camera3d<f64_3>;

    constexpr f64 prec = 1e-12;
    constexpr f64 pi   = shambase::constants::pi<f64>;

    /// Camera looking from an oblique direction, with a non square image
    Camera make_oblique_camera() {
        return Camera({0.5, -1.0, 5.0}, {1.0, 1.0, -1.0}, {0.0, 1.0, 0.0}, 7, 5, pi / 4, 0.1, 10.0);
    }

    /// Apply a 4x4 matrix to the homogeneous point (p, 1)
    f64_4 mat_apply(const f64_4x4 &m, f64_3 p) {
        auto row = [&](int i) {
            return m(i, 0) * p.x() + m(i, 1) * p.y() + m(i, 2) * p.z() + m(i, 3);
        };
        return {row(0), row(1), row(2), row(3)};
    }

    f64_4x4 mat_mul(const f64_4x4 &a, const f64_4x4 &b) {
        f64_4x4 ret;
        shammath::mat_prod(a.get_mdspan(), b.get_mdspan(), ret.get_mdspan());
        return ret;
    }

    f64 pixel_ndc(u32 i, u32 n) { return 2.0 * (f64(i) + 0.5) / f64(n) - 1.0; }

} // namespace

NEW_TEST(Unittest, "shammath/Camera3d:basis", 1) {

    { // Test case 1: OpenGL default camera (looking toward -z, y up)
        Camera cam({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}, 4, 4, pi / 2, 0.1, 10.0);

        f64_3 expected_dir{0.0, 0.0, -1.0};
        f64_3 expected_up{0.0, 1.0, 0.0};
        f64_3 expected_right{1.0, 0.0, 0.0};

        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(
            "dir", cam.get_dir(), expected_dir, prec, sycl::length);
        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED("up", cam.get_up(), expected_up, prec, sycl::length);
        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(
            "right", cam.get_right(), expected_right, prec, sycl::length);
    }

    { // Test case 2: oblique camera, the basis is orthonormal and right-handed
        Camera cam = make_oblique_camera();

        f64_3 dir   = cam.get_dir();
        f64_3 up    = cam.get_up();
        f64_3 right = cam.get_right();

        REQUIRE_FLOAT_EQUAL(sycl::length(dir), 1.0, prec);
        REQUIRE_FLOAT_EQUAL(sycl::length(up), 1.0, prec);
        REQUIRE_FLOAT_EQUAL(sycl::length(right), 1.0, prec);

        REQUIRE_FLOAT_EQUAL(sycl::dot(dir, up), 0.0, prec);
        REQUIRE_FLOAT_EQUAL(sycl::dot(dir, right), 0.0, prec);
        REQUIRE_FLOAT_EQUAL(sycl::dot(up, right), 0.0, prec);

        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(
            "right-handed", sycl::cross(right, up), -dir, prec, sycl::length);

        f64_3 expected_dir = f64_3{1.0, 1.0, -1.0} / sycl::sqrt(3.0);
        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED("dir", dir, expected_dir, prec, sycl::length);

        // the image up stays on the side of the requested up vector
        REQUIRE(up.y() > 0);
    }
}

NEW_TEST(Unittest, "shammath/Camera3d:get_pixel_ray", 1) {

    { // Test case 1: the central pixel of an odd sized image looks along dir
        Camera cam = make_oblique_camera();

        shammath::Ray<f64_3> ray = cam.get_pixel_ray(3, 2);

        f64_3 expected_origin = cam.get_pos() + cam.get_dir() * cam.get_znear();

        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(
            "direction", ray.direction, cam.get_dir(), prec, sycl::length);
        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(
            "origin", ray.origin, expected_origin, prec, sycl::length);
        REQUIRE_EQUAL(ray.t_min, 0.0);
    }

    { // Test case 2: every ray starts on the near plane and goes through its pixel center
        Camera cam = make_oblique_camera();

        f64 tan_half = std::tan(cam.get_fovy() / 2);

        for (u32 iy = 0; iy < cam.get_ny(); iy++) {
            for (u32 ix = 0; ix < cam.get_nx(); ix++) {
                shammath::Ray<f64_3> ray = cam.get_pixel_ray(ix, iy);

                f64_3 d   = ray.direction;
                f64 d_fwd = sycl::dot(d, cam.get_dir());

                REQUIRE(d_fwd > 0);
                REQUIRE_EQUAL(ray.t_min, 0.0);
                REQUIRE_FLOAT_EQUAL(sycl::length(d), 1.0, prec);
                REQUIRE_FLOAT_EQUAL(
                    sycl::dot(ray.origin - cam.get_pos(), cam.get_dir()), cam.get_znear(), prec);
                REQUIRE_FLOAT_EQUAL(
                    sycl::dot(d, cam.get_right()) / d_fwd,
                    pixel_ndc(ix, cam.get_nx()) * tan_half * cam.get_aspect(),
                    prec);
                REQUIRE_FLOAT_EQUAL(
                    sycl::dot(d, cam.get_up()) / d_fwd,
                    pixel_ndc(iy, cam.get_ny()) * tan_half,
                    prec);
            }
        }
    }

    { // Test case 3: pixel (0, 0) is the bottom left one
        Camera cam({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}, 4, 4, pi / 2, 0.1, 10.0);

        shammath::Ray<f64_3> ray = cam.get_pixel_ray(0, 0);

        REQUIRE(ray.direction.x() < 0);
        REQUIRE(ray.direction.y() < 0);
    }
}

NEW_TEST(Unittest, "shammath/Camera3d:get_rays", 1) {

    Camera cam = make_oblique_camera();

    std::vector<shammath::Ray<f64_3>> rays = cam.get_rays();

    REQUIRE_EQUAL(rays.size(), size_t(cam.get_nx() * cam.get_ny()));

    for (u32 iy = 0; iy < cam.get_ny(); iy++) {
        for (u32 ix = 0; ix < cam.get_nx(); ix++) {
            shammath::Ray<f64_3> expected = cam.get_pixel_ray(ix, iy);
            shammath::Ray<f64_3> ray      = rays[iy * cam.get_nx() + ix];

            REQUIRE(sham::equals(ray.origin, expected.origin));
            REQUIRE(sham::equals(ray.direction, expected.direction));
            REQUIRE_EQUAL(ray.t_min, expected.t_min);
        }
    }
}

NEW_TEST(Unittest, "shammath/Camera3d:matrices", 1) {

    f64_4x4 identity = shammath::mat_identity<f64, 4>();

    { // Test case 1: the OpenGL default camera has an identity view matrix
        Camera cam({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}, 4, 4, pi / 2, 0.1, 10.0);

        REQUIRE(cam.get_view_matrix().equal_at_precision(identity, prec));
    }

    { // Test case 2: projection matrix matches gluPerspective
        f64 znear = 0.5;
        f64 zfar  = 20.0;
        Camera cam({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}, 8, 4, pi / 2, znear, zfar);

        // fovy = 90 deg -> cot(fovy / 2) = 1, aspect = 2
        f64_4x4 expected{};
        expected(0, 0) = 0.5;
        expected(1, 1) = 1.0;
        expected(2, 2) = (zfar + znear) / (znear - zfar);
        expected(2, 3) = 2 * zfar * znear / (znear - zfar);
        expected(3, 2) = -1.0;

        REQUIRE(cam.get_projection_matrix().equal_at_precision(expected, prec));
    }

    { // Test case 3: inverses
        Camera cam = make_oblique_camera();

        REQUIRE_NAMED(
            "view * view_inv = I",
            mat_mul(cam.get_view_matrix(), cam.get_view_matrix_inv())
                .equal_at_precision(identity, prec));
        REQUIRE_NAMED(
            "proj * proj_inv = I",
            mat_mul(cam.get_projection_matrix(), cam.get_projection_matrix_inv())
                .equal_at_precision(identity, prec));
        REQUIRE_NAMED(
            "transform * transform_inv = I",
            mat_mul(cam.get_camera_transform(), cam.get_camera_transform_inv())
                .equal_at_precision(identity, 1e-10));
    }

    { // Test case 4: the rays are consistent with the camera transform
        Camera cam = make_oblique_camera();

        f64_4x4 transform = cam.get_camera_transform();

        for (u32 iy = 0; iy < cam.get_ny(); iy++) {
            for (u32 ix = 0; ix < cam.get_nx(); ix++) {
                shammath::Ray<f64_3> ray = cam.get_pixel_ray(ix, iy);

                f64_3 expected_near{pixel_ndc(ix, cam.get_nx()), pixel_ndc(iy, cam.get_ny()), -1};

                // the ray origin lies on the near plane in clip space
                f64_4 clip_near = mat_apply(transform, ray.origin);
                f64_3 ndc_near = f64_3{clip_near.x(), clip_near.y(), clip_near.z()} / clip_near.w();
                REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(
                    "near", ndc_near, expected_near, 1e-10, sycl::length);

                // a point further along the ray projects on the same pixel
                f64_4 clip_far = mat_apply(transform, ray.origin + ray.direction * 3.0);
                f64 ndc_far_x  = clip_far.x() / clip_far.w();
                f64 ndc_far_y  = clip_far.y() / clip_far.w();
                f64 ndc_far_z  = clip_far.z() / clip_far.w();
                REQUIRE_FLOAT_EQUAL(ndc_far_x, expected_near.x(), 1e-10);
                REQUIRE_FLOAT_EQUAL(ndc_far_y, expected_near.y(), 1e-10);
                REQUIRE(ndc_far_z > -1 && ndc_far_z < 1);
            }
        }
    }
}

NEW_TEST(Unittest, "shammath/Camera3d:from_aspect", 1) {

    { // Test case 1: ny is rounded, the aspect ratio is kept as given
        f64 aspect = 16.0 / 9.0;
        Camera cam = Camera::from_aspect(
            {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}, 100, aspect, pi / 3, 0.1, 10.0);

        REQUIRE_EQUAL(cam.get_nx(), 100u);
        REQUIRE_EQUAL(cam.get_ny(), 56u);
        REQUIRE_EQUAL(cam.get_aspect(), aspect);
    }

    { // Test case 2: at least one pixel vertically
        Camera cam = Camera::from_aspect(
            {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}, 1, 10.0, pi / 3, 0.1, 10.0);

        REQUIRE_EQUAL(cam.get_ny(), 1u);
    }

    { // Test case 3: the default constructor uses nx / ny
        Camera cam({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}, 30, 20, pi / 3, 0.1, 10.0);

        REQUIRE_FLOAT_EQUAL(cam.get_aspect(), 1.5, prec);
    }
}

NEW_TEST(Unittest, "shammath/Camera3d:look_at", 1) {

    { // Test case 1: same camera as giving dir = target - pos
        f64_3 pos{0.5, -1.0, 5.0};
        f64_3 target{2.0, 3.0, -1.0};
        f64_3 up{0.0, 1.0, 0.0};

        Camera cam      = Camera::look_at(pos, target, up, 7, 5, pi / 4, 0.1, 10.0);
        Camera expected = Camera(pos, target - pos, up, 7, 5, pi / 4, 0.1, 10.0);

        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(
            "dir", cam.get_dir(), expected.get_dir(), prec, sycl::length);
        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(
            "up", cam.get_up(), expected.get_up(), prec, sycl::length);
        REQUIRE(
            cam.get_camera_transform().equal_at_precision(expected.get_camera_transform(), prec));
    }

    { // Test case 2: the target projects on the center of the image
        f64_3 target{2.0, 3.0, -1.0};
        Camera cam
            = Camera::look_at({0.5, -1.0, 5.0}, target, {0.0, 1.0, 0.0}, 7, 5, pi / 4, 0.1, 10.0);

        f64_4 clip = mat_apply(cam.get_camera_transform(), target);

        REQUIRE_FLOAT_EQUAL(clip.x() / clip.w(), 0.0, prec);
        REQUIRE_FLOAT_EQUAL(clip.y() / clip.w(), 0.0, prec);
    }
}

NEW_TEST(Unittest, "shammath/Camera3d:invalid_args", 1) {

    f64_3 pos{0.0, 0.0, 0.0};
    f64_3 dir{0.0, 0.0, -1.0};
    f64_3 up{0.0, 1.0, 0.0};

    REQUIRE_EXCEPTION_THROW(Camera(pos, dir, up, 0, 4, pi / 2, 0.1, 10.0), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(Camera(pos, dir, up, 4, 0, pi / 2, 0.1, 10.0), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(Camera(pos, dir, up, 4, 4, 0.0, 0.1, 10.0), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(Camera(pos, dir, up, 4, 4, pi, 0.1, 10.0), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(Camera(pos, dir, up, 4, 4, pi / 2, 0.0, 10.0), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(Camera(pos, dir, up, 4, 4, pi / 2, 1.0, 0.5), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(
        Camera(pos, {0.0, 0.0, 0.0}, up, 4, 4, pi / 2, 0.1, 10.0), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(
        Camera(pos, dir, {0.0, 0.0, 2.0}, 4, 4, pi / 2, 0.1, 10.0), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(
        Camera::from_aspect(pos, dir, up, 4, 0.0, pi / 2, 0.1, 10.0), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(
        Camera::look_at(pos, pos, up, 4, 4, pi / 2, 0.1, 10.0), std::invalid_argument);
}

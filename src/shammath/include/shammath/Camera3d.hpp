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
 * @file Camera3d.hpp
 * @author Yann Bernard (yann.bernard@univ-grenoble-alpes.fr)
 * @brief Perspective camera generating one ray per pixel, following the OpenGL conventions.
 *
 */

#include "shambase/constants.hpp"
#include "shambase/exception.hpp"
#include "sham/format/format.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shammath/AABB.hpp"
#include "shammath/matrix.hpp"
#include <cmath>
#include <vector>

namespace shammath {

    /**
     * @brief Perspective camera following the OpenGL conventions (gluLookAt + gluPerspective)
     *
     * The camera sits at ``pos`` and looks along ``dir``, ``up`` gives the vertical direction
     * of the image. In camera space the camera looks toward -z, with x to the right and y up.
     *
     * Pixel ``(ix, iy)`` covers ``ix`` in ``[0, nx)`` from left to right and ``iy`` in
     * ``[0, ny)`` from bottom to top. Its ray goes through the pixel center, starts on the near
     * plane (at depth ``znear``) and extends to infinity (``Ray::t_min = 0``), so that nothing
     * behind the image plane is rendered. ``zfar`` is only used by the projection matrix.
     *
     * @tparam Tvec 3D vector type
     */
    template<class Tvec>
    class Camera3d {
        public:
        using Tscal = shambase::VecComponent<Tvec>;

        /**
         * @brief Construct a camera from the image size in pixels
         *
         * The aspect ratio is ``nx / ny``.
         *
         * @param pos Position of the camera
         * @param dir Viewing direction (does not need to be normalized)
         * @param up Up direction (does not need to be normalized nor orthogonal to ``dir``)
         * @param nx Number of pixels along the horizontal axis
         * @param ny Number of pixels along the vertical axis
         * @param fovy Vertical field of view in radians, in ``(0, pi)``
         * @param znear Distance to the near plane (> 0)
         * @param zfar Distance to the far plane (> znear)
         */
        Camera3d(Tvec pos, Tvec dir, Tvec up, u32 nx, u32 ny, Tscal fovy, Tscal znear, Tscal zfar)
            : Camera3d(pos, dir, up, nx, ny, Tscal(nx) / Tscal(ny), fovy, znear, zfar) {}

        /**
         * @brief Construct a camera from the horizontal resolution and the aspect ratio
         *
         * ``ny`` is set to ``round(nx / aspect)`` (at least 1), the aspect ratio is kept as given.
         *
         * @param aspect Aspect ratio of the image (width / height)
         * @see Camera3d::Camera3d for the other parameters
         */
        static Camera3d from_aspect(
            Tvec pos,
            Tvec dir,
            Tvec up,
            u32 nx,
            Tscal aspect,
            Tscal fovy,
            Tscal znear,
            Tscal zfar) {
            if (!(aspect > 0)) {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    sham::format("Camera3d: aspect must be > 0, got aspect = {}", aspect));
            }
            u32 ny = static_cast<u32>(sycl::max(Tscal(1), std::round(Tscal(nx) / aspect)));
            return Camera3d(pos, dir, up, nx, ny, aspect, fovy, znear, zfar);
        }

        /**
         * @brief Construct a camera looking at a target point (gluLookAt convention)
         *
         * Equivalent to the main constructor with ``dir = target - pos``.
         *
         * @param target Point the camera looks at (must differ from ``pos``)
         * @see Camera3d::Camera3d for the other parameters
         */
        static Camera3d look_at(
            Tvec pos, Tvec target, Tvec up, u32 nx, u32 ny, Tscal fovy, Tscal znear, Tscal zfar) {
            if (sham::equals(pos, target)) {
                throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                    "Camera3d: target must differ from pos, got pos = target = {}", pos));
            }
            return Camera3d(pos, target - pos, up, nx, ny, fovy, znear, zfar);
        }

        inline Tvec get_pos() const { return pos; }
        /// Normalized viewing direction
        inline Tvec get_dir() const { return dir; }
        /// Normalized up direction, orthogonal to ``dir``
        inline Tvec get_up() const { return up; }
        /// Normalized right direction ``dir x up``
        inline Tvec get_right() const { return right; }
        inline u32 get_nx() const { return nx; }
        inline u32 get_ny() const { return ny; }
        inline Tscal get_aspect() const { return aspect; }
        inline Tscal get_fovy() const { return fovy; }
        inline Tscal get_znear() const { return znear; }
        inline Tscal get_zfar() const { return zfar; }

        /**
         * @brief Ray going through the center of pixel ``(ix, iy)``
         *
         * The ray starts on the near plane and extends to infinity (``t_min = 0``).
         */
        inline Ray<Tvec> get_pixel_ray(u32 ix, u32 iy) const {
            Tscal tan_half = sycl::tan(fovy / 2);
            Tscal ndc_x    = Tscal(2) * (Tscal(ix) + Tscal(0.5)) / Tscal(nx) - Tscal(1);
            Tscal ndc_y    = Tscal(2) * (Tscal(iy) + Tscal(0.5)) / Tscal(ny) - Tscal(1);

            // camera space direction (ndc_x * tan_half * aspect, ndc_y * tan_half, -1)
            Tvec d = dir + right * (ndc_x * tan_half * aspect) + up * (ndc_y * tan_half);

            // the component of d along dir is 1, so this point lies on the near plane
            return Ray<Tvec>(pos + d * znear, d, Tscal(0));
        }

        /// Rays of all the pixels, pixel ``(ix, iy)`` being at index ``iy * nx + ix``
        inline std::vector<Ray<Tvec>> get_rays() const {
            std::vector<Ray<Tvec>> rays;
            rays.reserve(size_t(nx) * size_t(ny));
            for (u32 iy = 0; iy < ny; iy++) {
                for (u32 ix = 0; ix < nx; ix++) {
                    rays.push_back(get_pixel_ray(ix, iy));
                }
            }
            return rays;
        }

        /// View matrix (world -> camera space), equivalent to gluLookAt
        inline mat<Tscal, 4, 4> get_view_matrix() const {
            mat<Tscal, 4, 4> view{};

            view(0, 0) = right.x();
            view(0, 1) = right.y();
            view(0, 2) = right.z();
            view(0, 3) = -sycl::dot(right, pos);

            view(1, 0) = up.x();
            view(1, 1) = up.y();
            view(1, 2) = up.z();
            view(1, 3) = -sycl::dot(up, pos);

            view(2, 0) = -dir.x();
            view(2, 1) = -dir.y();
            view(2, 2) = -dir.z();
            view(2, 3) = sycl::dot(dir, pos);

            view(3, 3) = 1;

            return view;
        }

        /// Inverse of the view matrix (camera -> world space)
        inline mat<Tscal, 4, 4> get_view_matrix_inv() const {
            mat<Tscal, 4, 4> view_inv{};

            view_inv(0, 0) = right.x();
            view_inv(1, 0) = right.y();
            view_inv(2, 0) = right.z();

            view_inv(0, 1) = up.x();
            view_inv(1, 1) = up.y();
            view_inv(2, 1) = up.z();

            view_inv(0, 2) = -dir.x();
            view_inv(1, 2) = -dir.y();
            view_inv(2, 2) = -dir.z();

            view_inv(0, 3) = pos.x();
            view_inv(1, 3) = pos.y();
            view_inv(2, 3) = pos.z();

            view_inv(3, 3) = 1;

            return view_inv;
        }

        /// Projection matrix (camera -> clip space), equivalent to gluPerspective
        inline mat<Tscal, 4, 4> get_projection_matrix() const {
            mat<Tscal, 4, 4> proj{};

            Tscal f = Tscal(1) / sycl::tan(fovy / 2);

            proj(0, 0) = f / aspect;
            proj(1, 1) = f;
            proj(2, 2) = (zfar + znear) / (znear - zfar);
            proj(2, 3) = Tscal(2) * zfar * znear / (znear - zfar);
            proj(3, 2) = -1;

            return proj;
        }

        /// Inverse of the projection matrix (clip -> camera space)
        inline mat<Tscal, 4, 4> get_projection_matrix_inv() const {
            mat<Tscal, 4, 4> proj = get_projection_matrix();
            mat<Tscal, 4, 4> proj_inv{};

            proj_inv(0, 0) = Tscal(1) / proj(0, 0);
            proj_inv(1, 1) = Tscal(1) / proj(1, 1);
            proj_inv(2, 3) = -1;
            proj_inv(3, 2) = Tscal(1) / proj(2, 3);
            proj_inv(3, 3) = proj(2, 2) / proj(2, 3);

            return proj_inv;
        }

        /// Full camera transform (world -> clip space), ``projection * view``
        inline mat<Tscal, 4, 4> get_camera_transform() const {
            mat<Tscal, 4, 4> ret;
            mat<Tscal, 4, 4> proj = get_projection_matrix();
            mat<Tscal, 4, 4> view = get_view_matrix();
            mat_prod(proj.get_mdspan(), view.get_mdspan(), ret.get_mdspan());
            return ret;
        }

        /// Inverse of the full camera transform (clip -> world space)
        inline mat<Tscal, 4, 4> get_camera_transform_inv() const {
            mat<Tscal, 4, 4> ret;
            mat<Tscal, 4, 4> view_inv = get_view_matrix_inv();
            mat<Tscal, 4, 4> proj_inv = get_projection_matrix_inv();
            mat_prod(view_inv.get_mdspan(), proj_inv.get_mdspan(), ret.get_mdspan());
            return ret;
        }

        private:
        Tvec pos;
        Tvec dir;
        Tvec up;
        Tvec right;

        u32 nx;
        u32 ny;

        Tscal aspect;
        Tscal fovy;
        Tscal znear;
        Tscal zfar;

        Camera3d(
            Tvec pos,
            Tvec dir,
            Tvec up,
            u32 nx,
            u32 ny,
            Tscal aspect,
            Tscal fovy,
            Tscal znear,
            Tscal zfar)
            : pos(pos), nx(nx), ny(ny), aspect(aspect), fovy(fovy), znear(znear), zfar(zfar) {

            auto throw_invalid = [&](const std::string &reason) {
                throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                    "Camera3d: {}\n"
                    "  args :\n"
                    "    pos    = {}\n"
                    "    dir    = {}\n"
                    "    up     = {}\n"
                    "    nx     = {}\n"
                    "    ny     = {}\n"
                    "    aspect = {}\n"
                    "    fovy   = {}\n"
                    "    znear  = {}\n"
                    "    zfar   = {}",
                    reason,
                    pos,
                    dir,
                    up,
                    nx,
                    ny,
                    aspect,
                    fovy,
                    znear,
                    zfar));
            };

            if (nx == 0 || ny == 0) {
                throw_invalid("nx and ny must be > 0");
            }
            if (!(aspect > 0)) {
                throw_invalid("aspect must be > 0");
            }
            if (!(fovy > 0 && fovy < shambase::constants::pi<Tscal>) ) {
                throw_invalid("fovy must be in (0, pi)");
            }
            if (!(znear > 0 && zfar > znear)) {
                throw_invalid("znear and zfar must satisfy 0 < znear < zfar");
            }

            Tscal len_dir = sycl::length(dir);
            if (!(len_dir > 0)) {
                throw_invalid("dir must be non zero");
            }
            this->dir = dir / len_dir;

            Tvec r      = sycl::cross(this->dir, up);
            Tscal len_r = sycl::length(r);
            if (!(len_r > 0)) {
                throw_invalid("up must be non zero and not colinear with dir");
            }
            this->right = r / len_r;
            this->up    = sycl::cross(this->right, this->dir);
        }
    };

} // namespace shammath

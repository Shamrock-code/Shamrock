// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file crystalLattice_smap_sphere.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/constants.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/CoordRange.hpp"
#include "shammath/DiscontinuousIterator.hpp"
#include "shammath/LatticeError.hpp"
#include "shammath/integrator.hpp"
#include <shambackends/sycl.hpp>
#include <array>
#include <functional>
#include <utility>
#include <vector>

namespace shammath {

    /**
     * @brief utility for generating stretched spherical HCP crystal lattices given a density
     * profile
     *
     * @tparam Tvec position vector type
     */
    template<class Tvec>
    class LatticeHCP_smap_sphere {
        using Tscal = shambase::VecComponent<Tvec>;

        public:
        static constexpr u32 dim       = 3;
        static constexpr Tscal pi      = shambase::constants::pi<Tscal>;
        static constexpr Tscal tol     = 1e-9;
        static constexpr u32 maxits    = 200;
        static constexpr u32 maxits_nr = 20;

        Tscal dr;
        std::array<i32, dim> coord_min;
        std::array<i32, dim> coord_max;
        Tscal rmin;
        Tscal rmax;
        Tvec center;
        Tscal integral_profile;
        std::function<Tscal(Tscal)> rhoprofile;
        std::function<Tscal(Tscal)> S;
        std::function<Tscal(Tscal)> rhodS;
        std::function<Tscal(Tvec)> a_from_pos;
        std::function<Tvec(Tscal, Tvec)> a_to_pos;
        Tscal step;

        static_assert(
            dim == shambase::VectorProperties<Tvec>::dimension,
            "this lattice exists only in dim 3");

        LatticeHCP_smap_sphere(
            Tscal dr,
            std::array<i32, dim> coord_min,
            std::array<i32, dim> coord_max,
            std::function<Tscal(Tscal)> rhoprofile,
            std::function<Tscal(Tscal)> S,
            std::function<Tscal(Tvec)> a_from_pos,
            std::function<Tvec(Tscal, Tvec)> a_to_pos,
            Tscal integral_profile,
            Tscal rmin,
            Tscal rmax,
            Tvec center,
            Tscal step)
            : dr(dr), coord_min(coord_min), coord_max(coord_max), rhoprofile(rhoprofile), S(S),
              rhodS([&S, &rhoprofile](Tscal r) {
                  return rhoprofile(r) * S(r);
              }),
              a_from_pos(a_from_pos), a_to_pos(a_to_pos), integral_profile(integral_profile),
              rmin(rmin), rmax(rmax), center(center), step(step) {}

        static Tscal stretchindiv(
            Tscal a,
            std::function<Tscal(Tscal)> rhoprofile,
            std::function<Tscal(Tscal)> S,
            Tscal integral_profile,
            Tscal rmin,
            Tscal rmax,
            Tvec center,
            Tscal step) {

            auto rhodS = [&rhoprofile, &S](Tscal a) -> Tscal {
                return S(a) * rhoprofile(a);
            };
            u32 its       = 0;
            Tscal initpos = a;
            Tscal newr    = a;
            Tscal prevr   = a;

            Tscal initrelatpos = (sycl::pown(initpos, 3) - sycl::pown(rmin, 3))
                                 / (sycl::pown(rmax, 3) - sycl::pown(rmin, 3));
            Tscal newrelatpos
                = shammath::integ_riemann_sum(rmin, newr, step, rhodS) / integral_profile;
            Tscal func       = newrelatpos - initrelatpos;
            Tscal xminbisect = rmin;
            Tscal xmaxbisect = rmax;
            Tscal dfunc      = 0;
            Tscal dx         = 0;
            bool bisect      = false;
            while ((sham::abs(func) > tol) && (its < maxits)) {
                its++;
                if (bisect) {
                    if (func > 0.) {
                        xmaxbisect = newr;
                    } else {
                        xminbisect = newr;
                    }
                    newr = 0.5 * (xminbisect + xmaxbisect);
                } else {
                    dfunc = rhodS(newr) / integral_profile;
                    dx    = func / dfunc;
                    newr  = newr - dx;
                }

                if (sham::abs(newr) < 0.8 * sham::abs(prevr)) {
                    newr = 0.8 * prevr;
                } else if (sham::abs(newr) > 1.2 * sham::abs(prevr)) {
                    newr = 1.2 * prevr;
                }

                // NR iteration
                if ((newr > rmax) || (newr < rmin) || (its > maxits_nr)) {
                    bisect = true;
                    newr   = 0.5 * (xminbisect + xmaxbisect);
                }
                newrelatpos
                    = shammath::integ_riemann_sum(rmin, newr, step, rhodS) / integral_profile;
                func  = newrelatpos - initrelatpos;
                prevr = newr;
            }
            return newr;
        }

        /**
         * @brief generate a smap_sphere HCP lattice centered on (0,0,0)
         *
         * @param dr
         * @param coord
         * @param rhoprofile
         * @param rmin
         * @param rmax
         * @return constexpr Tvec
         */
        static inline constexpr Tvec generator(
            Tscal dr,
            std::array<i32, dim> coord,
            std::function<Tscal(Tscal)> rhoprofile,
            std::function<Tscal(Tscal)> S,
            std::function<Tscal(Tvec)> a_from_pos,
            std::function<Tvec(Tscal, Tvec)> a_to_pos,
            Tscal integral_profile,
            Tscal rmin,
            Tscal rmax,
            Tvec center,
            Tscal step) noexcept {

            i32 i = coord[0];
            i32 j = coord[1];
            i32 k = coord[2];

            Tvec r
                = {2 * i + (sycl::abs(j + k) % 2),
                   sycl::sqrt(3.) * (j + (1. / 3.) * (sycl::abs(k) % 2)),
                   2 * sycl::sqrt(6.) * k / 3};

            Tvec pos     = dr * r;
            Tscal r_norm = sycl::length(pos);

            if (r_norm >= rmax) { // kick the particle (only when coord="r")!
                return (r * 10 * rmax / r_norm) - center;
            }
            if (i == 0 && j == 0 && k == 0) {
                return center * (-1.);
            }

            Tscal a = a_from_pos(pos);
            // a_from_pos =  sycl::length e.g
            // or a_from_pos = pos_a.get("x")
            Tscal new_a
                = stretchindiv(a, rhoprofile, S, integral_profile, rmin, rmax, center, step);
            Tvec new_pos = a_to_pos(new_a, pos);
            // pos_a * (new_a / a_from_pos(pos_a)) dans le cas sphérique
            // {new_a, pos_a.get("y"), pos_a.get("z")} dans le cas x
            // j'imagine :
            // def  a_to_pos(new_a, pos_a):
            //      new_pos_a[0] = a_to_x(new_a, pos_a)
            //      new_pos_a[1] = a_to_y(new_a, pos_a)
            //      new_pos_a[2] = a_to_z(new_a, pos_a)
            // avec a_to_x(pos_a) = pos_a[0]
            //   ou a_to_x(pos_a) = pos_a[0]*newanew_a,

            return new_pos - center;
        };

        /**
         * @brief check if the given lattice coordinates bounds can make a periodic box
         *
         * @param coord_min integer triplet for the minimal coordinates on the lattice
         * @param coord_max integer triplet for the maximal coordinates on the lattice
         * @return true
         * @return false
         */

        constexpr static bool can_make_periodic_box(
            std::array<i32, dim> coord_min, std::array<i32, dim> coord_max) {
            if (coord_max[0] - coord_min[0] < 2) {
                return false;
            }

            if ((coord_max[1] + coord_min[1]) % 2 != 0) {
                return false;
            }

            if ((coord_max[2] + coord_min[2]) % 2 != 0) {
                return false;
            }

            return true;
        }

        /**
         * @brief Get the periodic box corresponding to integer lattice coordinates
         * this function will throw if the coordinates asked cannot make a periodic lattice
         *
         * @param dr the particle spacing in the lattice
         * @param coord_min integer triplet for the minimal coordinates on the lattice
         * @param coord_max integer triplet for the maximal coordinates on the lattice
         * @return constexpr CoordRange<Tvec> the periodic box bounds
         */
        static inline constexpr CoordRange<Tvec> get_periodic_box(
            Tscal dr, std::array<i32, dim> coord_min, std::array<i32, dim> coord_max) {
            Tscal xmin, xmax, ymin, ymax, zmin, zmax;

            xmin = 2 * coord_min[0];
            xmax = 2 * coord_max[0];

            ymin = sycl::sqrt(3.) * coord_min[1];
            ymax = sycl::sqrt(3.) * coord_max[1];

            zmin = 2 * sycl::sqrt(6.) * coord_min[2] / 3;
            zmax = 2 * sycl::sqrt(6.) * coord_max[2] / 3;

            if (!can_make_periodic_box(coord_min, coord_max)) {
                throw LatticeError(
                    "x axis count should be greater than 1\n"
                    "y axis count should be even\n"
                    "z axis count should be even");
            }

            return {Tvec{xmin, ymin, zmin} * dr, Tvec{xmax, ymax, zmax} * dr};
        }

        static inline constexpr std::pair<std::array<i32, dim>, std::array<i32, dim>>
        get_box_index_bounds(Tscal dr, Tvec box_min, Tvec box_max) {

            Tvec coord_min;
            Tvec coord_max;

            coord_min[0] = box_min[0] / 2.;
            coord_max[0] = box_max[0] / 2.;

            coord_min[1] = box_min[1] / sycl::sqrt(3.);
            coord_max[1] = box_max[1] / sycl::sqrt(3.);

            coord_min[2] = box_min[2] / (2 * sycl::sqrt(6.) / 3);
            coord_max[2] = box_max[2] / (2 * sycl::sqrt(6.) / 3);

            coord_min /= dr;
            coord_max /= dr;

            std::array<i32, 3> ret_coord_min
                = {i32(coord_min.x()) - 1, i32(coord_min.y()) - 1, i32(coord_min.z()) - 1};
            std::array<i32, 3> ret_coord_max
                = {i32(coord_max.x()) + 1, i32(coord_max.y()) + 1, i32(coord_max.z()) + 1};

            return {ret_coord_min, ret_coord_max};
        }

        /**
         * @brief get the nearest integer triplets bound that gives a periodic box
         *
         * @param coord_min integer triplet for the minimal coordinates on the lattice
         * @param coord_max integer triplet for the maximal coordinates on the lattice
         * @return constexpr std::pair<std::array<i32, dim>, std::array<i32, dim>> the new bounds
         */
        static inline constexpr std::pair<std::array<i32, dim>, std::array<i32, dim>>
        nearest_periodic_box_indices(
            std::array<i32, dim> coord_min, std::array<i32, dim> coord_max) {
            std::array<i32, dim> ret_coord_min;
            std::array<i32, dim> ret_coord_max;

            ret_coord_min[0] = coord_min[0];
            ret_coord_min[1] = coord_min[1];
            ret_coord_min[2] = coord_min[2];

            ret_coord_max[0] = coord_max[0];
            ret_coord_max[1] = coord_max[1];
            ret_coord_max[2] = coord_max[2];

            if (coord_max[0] - coord_min[0] < 2) {
                ret_coord_max[0]++;
            }

            if ((coord_max[1] + coord_min[1]) % 2 != 0) {
                ret_coord_max[1]++;
            }

            if ((coord_max[2] + coord_min[2]) % 2 != 0) {
                ret_coord_max[2]++;
            }

            return {ret_coord_min, ret_coord_max};
        }

        /**
         * @brief Iterator utility to generate the lattice
         *
         */

        class Iterator {

            LatticeHCP_smap_sphere &parent;
            std::array<i32, dim> coord_min;
            std::array<size_t, dim> coord_delta;
            size_t current_idx;
            size_t max_coord;

            bool done = false;

            public:
            Iterator(LatticeHCP_smap_sphere &p)
                : parent(p), current_idx(0), coord_delta({
                                                 size_t(p.coord_max[0] - p.coord_min[0]),
                                                 size_t(p.coord_max[1] - p.coord_min[1]),
                                                 size_t(p.coord_max[2] - p.coord_min[2]),
                                             }) {

                // must check for all axis otherwise we loop forever
                for (int ax = 0; ax < dim; ax++) {
                    if (parent.coord_min[ax] == parent.coord_max[ax]) {
                        done = true;
                    }
                }

                max_coord = coord_delta[0] * coord_delta[1] * coord_delta[2];
            }

            inline bool is_done() { return done; }

            inline Tvec next() {

                std::array<i32, 3> current = {
                    coord_min[0] + i32(current_idx % coord_delta[0]),
                    coord_min[1] + i32((current_idx / coord_delta[0]) % coord_delta[1]),
                    coord_min[2] + i32((current_idx / (coord_delta[0] * coord_delta[1]))),
                };

                Tvec ret = generator(
                    parent.dr,
                    current,
                    parent.rhoprofile,
                    parent.S,
                    parent.a_from_pos,
                    parent.a_to_pos,
                    parent.integral_profile,
                    parent.rmin,
                    parent.rmax,
                    parent.center,
                    parent.step);

                if (!done) {
                    current_idx++;
                }
                if (current_idx >= max_coord) {
                    done = true;
                }

                return ret;
            }

            inline std::vector<Tvec> next_n(u64 nmax) {
                std::vector<Tvec> ret{};
                for (u64 i = 0; i < nmax; i++) {
                    if (done) {
                        break;
                    }

                    ret.push_back(next());
                }
                shamlog_debug_ln("Discontinuous iterator", "next_n final idx", current_idx);
                return ret;
            }

            inline void skip(u64 n) {
                if (!done) {
                    current_idx += n;
                }
                if (current_idx >= max_coord) {
                    done = true;
                }
                shamlog_debug_ln("Discontinuous iterator", "skip final idx", current_idx);
            }
        };

        /**
         * @brief Iterator utility to generate the lattice
         *
         */
        class IteratorDiscontinuous {

            LatticeHCP_smap_sphere &parent;

            std::array<std::vector<size_t>, dim> remapped_indices;

            std::array<size_t, dim> coord_delta;
            size_t max_coord;

            bool done = false;

            public:
            size_t current_idx;
            IteratorDiscontinuous(LatticeHCP_smap_sphere &p)
                : parent(p), current_idx(0), coord_delta({
                                                 size_t(p.coord_max[0] - p.coord_min[0]),
                                                 size_t(p.coord_max[1] - p.coord_min[1]),
                                                 size_t(p.coord_max[2] - p.coord_min[2]),
                                             }) {

                // must check for all axis otherwise we loop forever
                for (int ax = 0; ax < dim; ax++) {
                    if (parent.coord_min[ax] == parent.coord_max[ax]) {
                        done = true;
                    }
                }

                max_coord = coord_delta[0] * coord_delta[1] * coord_delta[2];

                for (int ax = 0; ax < dim; ax++) {
                    DiscontinuousIterator<i32> it(parent.coord_min[ax], parent.coord_max[ax]);
                    while (!it.is_done()) {
                        remapped_indices[ax].push_back(it.next());
                    }
                }
            }

            inline bool is_done() { return done; }

            inline Tvec next() {

                // std::array<i32, 3> current = {
                //     coord_min[0] + i32(current_idx / (coord_delta[1] * coord_delta[2])),
                //     coord_min[1] + i32((current_idx / coord_delta[2]) % coord_delta[1]),
                //     coord_min[2] + i32(current_idx % coord_delta[2]),
                // };

                std::array<i32, 3> current = {
                    i32(current_idx % coord_delta[0]),
                    i32((current_idx / coord_delta[0]) % coord_delta[1]),
                    i32((current_idx / (coord_delta[0] * coord_delta[1]))),
                };

                current[0] = remapped_indices[0][current[0]];
                current[1] = remapped_indices[1][current[1]];
                current[2] = remapped_indices[2][current[2]];

                // logger::raw_ln(current, current_idx, max_coord);

                Tvec ret = generator(
                    parent.dr,
                    current,
                    parent.rhoprofile,
                    parent.S,
                    parent.a_from_pos,
                    parent.a_to_pos,
                    parent.integral_profile,
                    parent.rmin,
                    parent.rmax,
                    parent.center,
                    parent.step);

                if (!done) {
                    current_idx++;
                }
                if (current_idx >= max_coord) {
                    done = true;
                }

                return ret;
            }

            inline std::vector<Tvec> next_n(u64 nmax) {
                std::vector<Tvec> ret{};
                for (u64 i = 0; i < nmax; i++) {
                    if (done) {
                        break;
                    }

                    ret.push_back(next());
                }
                shamlog_debug_ln("Discontinuous iterator", "next_n final idx", current_idx);
                return ret;
            }

            inline void skip(u64 n) {
                if (!done) {
                    current_idx += n;
                }
                if (current_idx >= max_coord) {
                    done = true;
                }
                shamlog_debug_ln("Discontinuous iterator", "skip final idx", current_idx);
            }
        };

        IteratorDiscontinuous get_IteratorDiscontinuous() { return IteratorDiscontinuous(*this); }
        Iterator get_Iterator() { return Iterator(*this); }
    };

} // namespace shammath

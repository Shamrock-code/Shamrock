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
 * @file crystalLattice.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/CoordRange.hpp"
#include "shammath/DiscontinuousIterator.hpp"
#include <array>
#include <functional>
#include <utility>
#include <vector>

namespace shammath {

    class LatticeError : public std::exception {
        public:
        explicit LatticeError(const char *message) : msg_(message) {}

        explicit LatticeError(const std::string &message) : msg_(message) {}

        ~LatticeError() noexcept override = default;

        [[nodiscard]] const char *what() const noexcept override { return msg_.c_str(); }

        protected:
        std::string msg_;
    };

    /**
     * @brief utility for generating HCP crystal lattices
     *
     * @tparam Tvec position vector type
     */
    template<class Tvec>
    class LatticeHCP {
        public:
        static constexpr u32 dim = 3;

        static_assert(
            dim == shambase::VectorProperties<Tvec>::dimension, "this lattice exist only in dim 3");

        using Tscal = shambase::VecComponent<Tvec>;

        /**
         * @brief generate a HCP lattice centered on (0,0,0)
         *
         * @param dr
         * @param coord
         * @return constexpr Tvec
         */
        static inline constexpr Tvec generator(Tscal dr, std::array<i32, dim> coord) noexcept {

            i32 i = coord[0];
            i32 j = coord[1];
            i32 k = coord[2];

            Tvec r_a
                = {2 * i + (sycl::abs(j + k) % 2),
                   sycl::sqrt(3.) * (j + (1. / 3.) * (sycl::abs(k) % 2)),
                   2 * sycl::sqrt(6.) * k / 3};

            return dr * r_a;
        }

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
            Tscal dr;
            std::array<i32, dim> coord_min;

            std::array<size_t, dim> coord_delta;
            size_t current_idx;
            size_t max_coord;

            bool done = false;

            public:
            Iterator(Tscal dr, std::array<i32, dim> coord_min, std::array<i32, dim> coord_max)
                : dr(dr), coord_min(coord_min), current_idx(0),
                  coord_delta({
                      size_t(coord_max[0] - coord_min[0]),
                      size_t(coord_max[1] - coord_min[1]),
                      size_t(coord_max[2] - coord_min[2]),
                  }) {

                // must check for all axis otherwise we loop forever
                for (int ax = 0; ax < dim; ax++) {
                    if (coord_min[ax] == coord_max[ax]) {
                        done = true;
                    }
                }

                max_coord = coord_delta[0] * coord_delta[1] * coord_delta[2];
            }

            inline bool is_done() { return done; }

            inline Tvec next() {

                // std::array<i32, 3> current = {
                //     coord_min[0] + i32(current_idx / (coord_delta[1] * coord_delta[2])),
                //     coord_min[1] + i32((current_idx / coord_delta[2]) % coord_delta[1]),
                //     coord_min[2] + i32(current_idx % coord_delta[2]),
                // };

                std::array<i32, 3> current = {
                    coord_min[0] + i32(current_idx % coord_delta[0]),
                    coord_min[1] + i32((current_idx / coord_delta[0]) % coord_delta[1]),
                    coord_min[2] + i32((current_idx / (coord_delta[0] * coord_delta[1]))),
                };

                // logger::raw_ln(current, current_idx, max_coord);

                Tvec ret = generator(dr, current);

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
            Tscal dr;

            std::array<std::vector<size_t>, dim> remapped_indices;

            std::array<size_t, dim> coord_delta;
            size_t max_coord;

            bool done = false;

            public:
            size_t current_idx;
            IteratorDiscontinuous(
                Tscal dr, std::array<i32, dim> coord_min, std::array<i32, dim> coord_max)
                : dr(dr), current_idx(0), coord_delta({
                                              size_t(coord_max[0] - coord_min[0]),
                                              size_t(coord_max[1] - coord_min[1]),
                                              size_t(coord_max[2] - coord_min[2]),
                                          }) {

                // must check for all axis otherwise we loop forever
                for (int ax = 0; ax < dim; ax++) {
                    if (coord_min[ax] == coord_max[ax]) {
                        done = true;
                    }
                }

                max_coord = coord_delta[0] * coord_delta[1] * coord_delta[2];

                for (int ax = 0; ax < dim; ax++) {
                    DiscontinuousIterator<i32> it(coord_min[ax], coord_max[ax]);
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

                Tvec ret = generator(dr, current);

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
    };

} // namespace shammath

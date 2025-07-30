// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

//%Impl status : Good

#pragma once

/**
 * @file generators.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/constants.hpp"
#include "shamalgs/random.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"

namespace generic::setup::generators {

    template<class flt>
    inline sycl::vec<flt, 3> get_box_dim(flt r_particle, u32 xcnt, u32 ycnt, u32 zcnt) {

        using vec3 = sycl::vec<flt, 3>;

        u32 im = xcnt;
        u32 jm = ycnt;
        u32 km = zcnt;

        auto get_pos = [&](u32 i, u32 j, u32 k) -> vec3 {
            vec3 r_a
                = {2 * i + ((j + k) % 2),
                   sycl::sqrt(3.) * (j + (1. / 3.) * (k % 2)),
                   2 * sycl::sqrt(6.) * k / 3};

            r_a *= r_particle;

            return r_a;
        };

        return get_pos(im, jm, km);
    }

    template<class flt>
    inline std::tuple<sycl::vec<flt, 3>, sycl::vec<flt, 3>>
    get_ideal_fcc_box(flt r_particle, std::tuple<sycl::vec<flt, 3>, sycl::vec<flt, 3>> box) {

        using vec3 = sycl::vec<flt, 3>;

        vec3 box_min = std::get<0>(box);
        vec3 box_max = std::get<1>(box);

        vec3 box_dim = box_max - box_min;

        vec3 iboc_dim = (box_dim / vec3({2, sycl::sqrt(3.), 2 * sycl::sqrt(6.) / 3})) / r_particle;

        u32 i = iboc_dim.x();
        u32 j = iboc_dim.y();
        u32 k = iboc_dim.z();

        // std::cout << "get_ideal_box_idim :" << i << " " << j << " " << k << std::endl;

        i -= i % 2;
        j -= j % 2;
        k -= k % 2;

        vec3 m1 = get_box_dim(r_particle, i, j, k);

        return {box_min, box_min + m1};
    }

    template<class flt, class Tpred_select, class Tpred_pusher>
    inline void add_particles_fcc(
        flt r_particle,
        std::tuple<sycl::vec<flt, 3>, sycl::vec<flt, 3>> box,
        Tpred_select &&selector,
        Tpred_pusher &&part_pusher) {

        using vec3 = sycl::vec<flt, 3>;

        vec3 box_min = std::get<0>(box);
        vec3 box_max = std::get<1>(box);

        vec3 box_dim = box_max - box_min;

        vec3 iboc_dim = (box_dim / vec3({2, sycl::sqrt(3.), 2 * sycl::sqrt(6.) / 3})) / r_particle;

        // std::cout << "part box size : (" << iboc_dim.x() << ", " << iboc_dim.y() << ", " <<
        // iboc_dim.z() << ")" << std::endl;
        u32 ix = std::ceil(iboc_dim.x());
        u32 iy = std::ceil(iboc_dim.y());
        u32 iz = std::ceil(iboc_dim.z());

        if (shamcomm::world_rank() == 0)
            logger::info_ln("SPH", "Add fcc lattice size : (", ix, iy, iz, ")");
        // std::cout << "part box size : (" << ix << ", " << iy << ", " << iz << ")" << std::endl;

        if ((iy % 2) != 0 && (iz % 2) != 0) {
            std::cout << "Warning : particle count is odd on axis y or z -> this may lead to "
                         "periodicity issues";
        }

        for (u32 i = 0; i < ix; i++) {
            for (u32 j = 0; j < iy; j++) {
                for (u32 k = 0; k < iz; k++) {

                    vec3 r_a
                        = {2 * i + ((j + k) % 2),
                           sycl::sqrt(3.) * (j + (1. / 3.) * (k % 2)),
                           2 * sycl::sqrt(6.) * k / 3};

                    r_a *= r_particle;
                    r_a += box_min;

                    if (selector(r_a))
                        part_pusher(r_a, r_particle);
                }
            }
        }
    }

    template<class Tscal>
    struct DiscOutput {
        sycl::vec<Tscal, 3> pos;
        sycl::vec<Tscal, 3> velocity;
        Tscal cs;
        Tscal rho;
    };

    /**
     * @brief
     *
     * @tparam flt
     * @param Npart
     * @param r_in
     * @param r_out
     * @param sigma_profile
     * @param cs_profile
     * @param rot_profile
     * @param pusher
     */
    template<class flt>
    inline void add_disc2(
        u32 Npart,
        flt r_in,
        flt r_out,
        std::function<flt(flt)> sigma_profile,
        std::function<flt(flt)> cs_profile,
        std::function<flt(flt)> rot_profile,
        std::function<void(DiscOutput<flt>)> pusher) {
        constexpr flt _2pi = 2 * shambase::constants::pi<flt>;

        auto f_func = [&](flt r) {
            return r * sigma_profile(r);
        };

        flt fmax = f_func(r_out);

        std::mt19937 eng(0x111);

        auto find_r = [&]() {
            while (true) {
                flt u2 = shamalgs::random::mock_value<flt>(eng, 0, fmax);
                flt r  = shamalgs::random::mock_value<flt>(eng, r_in, r_out);
                if (u2 < f_func(r)) {
                    return r;
                }
            }
        };

        // eq 298 phantom paper & appendix A.7

        for (u32 i = 0; i < Npart; i++) {

            flt theta = shamalgs::random::mock_value<flt>(eng, 0, _2pi);
            flt Gauss = shamalgs::random::mock_gaussian<flt>(eng);

            flt r = find_r();

            flt vk    = rot_profile(r);
            flt cs    = cs_profile(r);
            flt sigma = sigma_profile(r);

            flt H_r = cs / vk;
            flt H   = H_r * r;

            flt z = H * Gauss;

            auto pos = sycl::vec<flt, 3>{r * sycl::cos(theta), z, r * sycl::sin(theta)};

            auto etheta = sycl::vec<flt, 3>{-pos.z(), 0, pos.x()};
            etheta /= sycl::length(etheta);

            auto vel = vk * etheta;

            flt rho = 0.1 * (sigma / (H * shambase::constants::pi2_sqrt<flt>) )
                      * sycl::exp(-z * z / (2 * H * H));

            DiscOutput<flt> out{pos, vel, cs, rho};

            pusher(out);
        }
    }

    /**
     * @brief
     *
     * @tparam flt
     * @tparam Tpred_pusher
     * @param Npart number of particles
     * @param p radial power law surface density (default = 1)  sigma prop r^-p
     * @param rho_0 rho_0 volumic density (at r = 1)
     * @param m mass part
     * @param r_in inner cuttof
     * @param r_out outer cuttof
     * @param q T prop r^-q
     */
    template<class flt, class Tpred_pusher>
    inline void add_disc(
        u32 Npart,
        flt p,
        flt rho_0,
        flt m,
        flt r_in,
        flt r_out,
        flt q,
        Tpred_pusher &&part_pusher) {
        flt _2pi = 2 * M_PI;

        flt K = _2pi * rho_0 / m;
        flt c = 2 - p;

        flt y = K * (r_out - r_in) / c;

        std::mt19937 eng(0x111);

        for (u32 i = 0; i < Npart; i++) {

            flt r_1 = shamalgs::random::mock_value<flt>(eng, 0, y);

            flt r = sycl::pow(sycl::pow(r_in, c) + c * r_1 / K, 1 / c);

            flt theta = shamalgs::random::mock_value<flt>(eng, 0, _2pi);

            flt u = shamalgs::random::mock_gaussian<flt>(eng);

            flt H = 0.1 * sycl::pow(r, (flt) (3. / 2. - q / 2));

            part_pusher(
                sycl::vec<flt, 3>({r * sycl::cos(theta), u * H, r * sycl::sin(theta)}), 0.1);
        }
    }

} // namespace generic::setup::generators

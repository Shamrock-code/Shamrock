// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeGravWave.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Compute the gravitational wave quadrupole. Based on Toscani et. al. 2021.
 *
 */

#include "shambase/constants.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/matrix_exponential.hpp"
#include "shammodels/common/modules/ComputeGravWave.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::common::modules {

    namespace {

        // 3x3 mdspan view over a fixed-size array
        template<class Tscal>
        using Mat3 = std::mdspan<Tscal, std::extents<std::size_t, 3, 3>>;

    } // namespace

    template<class Tvec>
    void ComputeGravWave<Tvec>::_impl_evaluate_internal() {

        __shamrock_stack_entry();

        auto edges = get_edges();

        edges.spans_positions.check_sizes(edges.sizes.indexes);
        edges.spans_velocities.check_sizes(edges.sizes.indexes);
        edges.spans_accelerations.check_sizes(edges.sizes.indexes);
        edges.spans_masses.check_sizes(edges.sizes.indexes);
        edges.spans_accel_ext.check_sizes(edges.sizes.indexes);

        const Tvec x0         = edges.central_pos.data;
        const Tvec v0         = edges.central_vel.data;
        const Tvec a0         = edges.central_acc.data;
        const Tscal fac       = edges.gw_prefactor.data;
        const Tscal theta_deg = edges.theta_gw.data;
        const Tscal phi_deg   = edges.phi_gw.data;

        constexpr Tscal pi = shambase::constants::pi<Tscal>;

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        // thought you could pass a DeviceBuffer to distributed_data_kernel_call through Multiref ?
        // guess NOT motherfcker
        // sham::DeviceBuffer<Tscal> ddq0{npart, dev_sched};

        auto make_ddq = [&]() {
            return edges.sizes.indexes.template map<sham::DeviceBuffer<Tscal>>(
                [&](u64 id, const u32 &n) {
                    return sham::DeviceBuffer<Tscal>{n, dev_sched};
                });
        };

        shambase::DistributedData<sham::DeviceBuffer<Tscal>> ddq0 = make_ddq();
        shambase::DistributedData<sham::DeviceBuffer<Tscal>> ddq1 = make_ddq();
        shambase::DistributedData<sham::DeviceBuffer<Tscal>> ddq2 = make_ddq();
        shambase::DistributedData<sham::DeviceBuffer<Tscal>> ddq3 = make_ddq();
        shambase::DistributedData<sham::DeviceBuffer<Tscal>> ddq4 = make_ddq();
        shambase::DistributedData<sham::DeviceBuffer<Tscal>> ddq5 = make_ddq();

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{
                edges.spans_positions.get_spans(),
                edges.spans_velocities.get_spans(),
                edges.spans_accelerations.get_spans(),
                edges.spans_masses.get_spans(),
                edges.spans_accel_ext.get_spans()},
            sham::DDMultiRef{ddq0, ddq1, ddq2, ddq3, ddq4, ddq5},
            edges.sizes.indexes,
            [x0, v0, a0](
                u32 gid,
                const Tvec *xyz,
                const Tvec *vxyz,
                const Tvec *axyz,
                const Tscal *mass,
                const Tvec *axyz_ext,
                Tscal *ddq0,
                Tscal *ddq1,
                Tscal *ddq2,
                Tscal *ddq3,
                Tscal *ddq4,
                Tscal *ddq5) {
                const Tscal m = mass[gid];

                const Tscal x  = xyz[gid][0] - x0[0];
                const Tscal y  = xyz[gid][1] - x0[1];
                const Tscal z  = xyz[gid][2] - x0[2];
                const Tscal vx = vxyz[gid][0] - v0[0];
                const Tscal vy = vxyz[gid][1] - v0[1];
                const Tscal vz = vxyz[gid][2] - v0[2];

                // normally axyz should have axyzext added to it already
                const Tscal ax = axyz[gid][0] - a0[0];
                const Tscal ay = axyz[gid][1] - a0[1];
                const Tscal az = axyz[gid][2] - a0[2];

                ddq0[gid] = m * (Tscal(2.) * vx * vx + x * ax + x * ax);
                ddq1[gid] = m * (Tscal(2.) * vx * vy + x * ay + y * ax);
                ddq2[gid] = m * (Tscal(2.) * vx * vz + x * az + z * ax);
                ddq3[gid] = m * (Tscal(2.) * vy * vy + y * ay + y * ay);
                ddq4[gid] = m * (Tscal(2.) * vy * vz + y * az + z * ay);
                ddq5[gid] = m * (Tscal(2.) * vz * vz + z * az + z * az);
            });

        // Tscal ddq0_per_rank = shamalgs::primitives::sum(dev_sched, ddq0, 0, npart);

        Tscal ddq0_per_rank = 0, ddq1_per_rank = 0, ddq2_per_rank = 0, ddq3_per_rank = 0,
              ddq4_per_rank = 0, ddq5_per_rank = 0;

        edges.sizes.indexes.for_each([&](u64 id, const u32 &n) {
            ddq0_per_rank += shamalgs::primitives::sum(dev_sched, ddq0.get(id), 0, n);
            ddq1_per_rank += shamalgs::primitives::sum(dev_sched, ddq1.get(id), 0, n);
            ddq2_per_rank += shamalgs::primitives::sum(dev_sched, ddq2.get(id), 0, n);
            ddq3_per_rank += shamalgs::primitives::sum(dev_sched, ddq3.get(id), 0, n);
            ddq4_per_rank += shamalgs::primitives::sum(dev_sched, ddq4.get(id), 0, n);
            ddq5_per_rank += shamalgs::primitives::sum(dev_sched, ddq5.get(id), 0, n);
        });

        edges.ddq.data[0] = shamalgs::collective::allreduce_sum(ddq0_per_rank);
        edges.ddq.data[1] = shamalgs::collective::allreduce_sum(ddq1_per_rank);
        edges.ddq.data[2] = shamalgs::collective::allreduce_sum(ddq2_per_rank);
        edges.ddq.data[3] = shamalgs::collective::allreduce_sum(ddq3_per_rank);
        edges.ddq.data[4] = shamalgs::collective::allreduce_sum(ddq4_per_rank);
        edges.ddq.data[5] = shamalgs::collective::allreduce_sum(ddq5_per_rank);

        std::array<Tscal, 9> Q_arr{};
        Mat3<Tscal> Q(Q_arr.data());
        Q(0, 0) = edges.ddq.data[0];
        Q(0, 1) = Q(1, 0) = edges.ddq.data[1];
        Q(0, 2) = Q(2, 0) = edges.ddq.data[2];
        Q(1, 1)           = edges.ddq.data[3];
        Q(1, 2) = Q(2, 1) = edges.ddq.data[4];
        Q(2, 2)           = edges.ddq.data[5];

        std::array<Tscal, 9> ddq_xy_arr{};
        Mat3<Tscal> ddq_xy(ddq_xy_arr.data());

        const bool rotate = std::abs(theta_deg) > static_cast<Tscal>(1e-30);
        if (rotate) {
            const Tscal lam = theta_deg * pi / static_cast<Tscal>(180);
            const Tscal c   = std::cos(lam);
            const Tscal s   = std::sin(lam);

            std::array<Tscal, 9> R_arr
                = {c, Tscal(0), s, Tscal(0), Tscal(1), Tscal(0), -s, Tscal(0), c};
            Mat3<Tscal> R(R_arr.data());

            std::array<Tscal, 9> inter_arr{};
            Mat3<Tscal> inter(inter_arr.data());
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    Tscal sum = Tscal(0);
                    for (std::size_t k = 0; k < 3; ++k) {
                        sum += Q(i, k) * R(k, j);
                    }
                    inter(i, j) = sum;
                }
            }

            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    Tscal sum = Tscal(0);
                    for (std::size_t k = 0; k < 3; ++k) {
                        sum += R(k, i) * inter(k, j);
                    }
                    ddq_xy(i, j) = sum;
                }
            }
        } else {
            for (std::size_t i = 0; i < 9; ++i) {
                ddq_xy_arr[i] = Q_arr[i];
            }
        }

        // h+ / hx
        const Tscal phi     = phi_deg * pi / static_cast<Tscal>(180);
        const Tscal sinphi  = std::sin(phi);
        const Tscal cosphi  = std::cos(phi);
        const Tscal sinphi2 = sinphi * sinphi;
        const Tscal cosphi2 = cosphi * cosphi;
        const Tscal sin2phi = std::sin(Tscal(2) * phi);
        const Tscal cos2phi = std::cos(Tscal(2) * phi);

        std::array<Tscal, 4> hx_out{};
        std::array<Tscal, 4> hp_out{};

        for (u32 i = 0; i < 4; ++i) {
            const Tscal eta     = static_cast<Tscal>(i) * pi / static_cast<Tscal>(6);
            const Tscal sineta  = std::sin(eta);
            const Tscal coseta  = std::cos(eta);
            const Tscal sineta2 = sineta * sineta;
            const Tscal coseta2 = coseta * coseta;
            const Tscal sin2eta = std::sin(Tscal(2) * eta);

            hp_out[i] = fac
                        * (ddq_xy(0, 0) * (cosphi2 - sinphi2 * coseta2)
                           + ddq_xy(1, 1) * (sinphi2 - cosphi2 * coseta2) - ddq_xy(2, 2) * sineta2
                           - ddq_xy(0, 1) * sin2phi * (Tscal(1) + coseta2)
                           + ddq_xy(0, 2) * sinphi * sin2eta + ddq_xy(1, 2) * cosphi * sin2eta);

            hx_out[i] = Tscal(2) * fac
                        * (Tscal(0.5) * (ddq_xy(0, 0) - ddq_xy(1, 1)) * sin2phi * coseta
                           + ddq_xy(0, 1) * cos2phi * coseta - ddq_xy(0, 2) * cosphi * sineta
                           + ddq_xy(1, 2) * sinphi * sineta);
        }

        edges.ddq_xy.data = ddq_xy_arr;
        edges.hx.data     = hx_out;
        edges.hp.data     = hp_out;
    }

    template<class Tvec>
    inline std::string ComputeGravWave<Tvec>::_impl_get_tex() const {
        std::string tex = R"tex(
                Gravitational-wave strain from the quadrupole formula
                (Toscani et al. 2021)

                \begin{align}
                r_i      &= {spans_positions}_i - {central_pos}_i\\
                v_i      &= {spans_velocities}_i - {central_vel}_i\\
                a_i      &= {spans_accelerations}_i - {central_acc}_i + {spans_accel_ext}_i\\
                \ddot Q_{ij} &= \sum_p {spans_masses}_p
                                \left(2 v_i v_j + r_i a_j + r_j a_i\right)\\
                R        &= \begin{pmatrix}
                              \cos\lambda & 0 & \sin\lambda\\
                              0           & 1 & 0\\
                             -\sin\lambda & 0 & \cos\lambda
                            \end{pmatrix},\quad
                            \lambda = {theta_gw}\,\frac{\pi}{180}\\
                \ddot Q^{xy} &= R^{T}\,\ddot Q\,R\\
                h_+(\eta,\phi) &= {gw_prefactor}\,
                    \Big[\ddot Q^{xy}_{11}(\cos^2\phi - \sin^2\phi\cos^2\eta)
                        + \ddot Q^{xy}_{22}(\sin^2\phi - \cos^2\phi\cos^2\eta)
                        - \ddot Q^{xy}_{33}\sin^2\eta\\
                    &\qquad - \ddot Q^{xy}_{12}\sin 2\phi\,(1 + \cos^2\eta)
                        + \ddot Q^{xy}_{13}\sin\phi\,\sin 2\eta
                        + \ddot Q^{xy}_{23}\cos\phi\,\sin 2\eta\Big]\\
                h_\times(\eta,\phi) &= 2\,{gw_prefactor}\,
                    \Big[\tfrac12(\ddot Q^{xy}_{11} - \ddot Q^{xy}_{22})\sin 2\phi\,\cos\eta\\
                    &\qquad + \ddot Q^{xy}_{12}\cos 2\phi\,\cos\eta
                        - \ddot Q^{xy}_{13}\cos\phi\,\sin\eta
                        + \ddot Q^{xy}_{23}\sin\phi\,\sin\eta\Big]
                \end{align}

                Evaluated at $\eta = 0,\ \pi/6,\ \pi/3,\ \pi/2$
                with $\phi = {phi_gw}^\circ$, giving {hx} and {hp}.
            )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

    template class shammodels::common::modules::ComputeGravWave<f64_3>;

} // namespace shammodels::common::modules

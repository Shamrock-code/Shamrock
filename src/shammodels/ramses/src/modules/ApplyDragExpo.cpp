// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ApplyDragExpo.cpp
 * @author Anass Serhani (anass.serhani@cnrs.fr) --no git blame--
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammath/matrix_exponential.hpp"
#include "shammodels/ramses/modules/ApplyDragExpo.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    /// Expo drag update of a single cell, given 5 scratch matrices of size (ndust + 1)^2
    template<class Tvec>
    struct ExpoDragCell {
        using Tscal = shambase::VecComponent<Tvec>;
        using mdspan_t
            = std::mdspan<Tscal, std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>;

        Tscal dt;
        u32 ndust;
        u32 friction_control;
        size_t mat_size;

        const Tscal *rho_next;
        const Tvec *rhov_next;
        const Tscal *rhoe_next;
        const Tscal *rho_d_next;
        const Tvec *rhov_d_next;
        const Tscal *alphas;

        Tscal *rho;
        Tvec *rhov;
        Tscal *rhoe;
        Tscal *rho_d;
        Tvec *rhov_d;

        inline void operator()(
            u32 id_a, Tscal *ptr_A, Tscal *ptr_B, Tscal *ptr_F, Tscal *ptr_I, Tscal *ptr_Id) const {

            // sparse jacobian matrix
            auto get_jacobian = [&](u32 id, mdspan_t &jacobian) {
                shammath::mat_set_nul<Tscal>(jacobian);
                // fill first row
                for (auto j = 1; j < jacobian.extent(1); j++)
                    jacobian(0, j) = alphas[j - 1];
                // fil first column
                for (auto i = 1; i < jacobian.extent(0); i++) {
                    jacobian(i, 0)
                        = alphas[i - 1] * (rho_d_next[id * ndust + (i - 1)] / rho_next[id]);
                    jacobian(0, 0) -= jacobian(i, 0);
                }
                // fill diagonal from (i,j)=(1,1)
                for (auto i = 1; i < jacobian.extent(0); i++)
                    jacobian(i, i) = -alphas[i - 1];
                // the rest of the buffer is set to zero
            };

            Tscal mu = 0;
            for (auto i = 0; i < ndust; i++) {
                mu += (1 + (rho_d_next[id_a * ndust + i] / rho_next[id_a])) * alphas[i];
            }
            mu *= (-dt / (ndust + 1));

            // create mdspan(s)
            mdspan_t mdspan_A(ptr_A, mat_size, mat_size);
            mdspan_t mdspan_B(ptr_B, mat_size, mat_size);
            mdspan_t mdspan_F(ptr_F, mat_size, mat_size);
            mdspan_t mdspan_I(ptr_I, mat_size, mat_size);
            mdspan_t mdspan_Id(ptr_Id, mat_size, mat_size);

            get_jacobian(id_a, mdspan_A);

            // pre-processing step
            shammath::mat_set_identity<Tscal>(mdspan_Id);
            shammath::mat_axpy_beta<Tscal, Tscal>(-mu, mdspan_Id, dt, mdspan_A);

            // compute matrix exponential
            const i32 K_exp = 9;
            shammath::mat_exp<Tscal, Tscal>(
                K_exp, mdspan_A, mdspan_F, mdspan_B, mdspan_I, mdspan_Id, ndust + 1);

            // post-processing step
            shammath::mat_mul_scalar<Tscal>(mdspan_A, sycl::exp(mu));

            // use the matrix exponential to for to updates momemtum
            Tvec r = {0., 0., 0.}, dd = {0., 0., 0.};
            r += mdspan_A(0, 0) * rhov_next[id_a];

            for (auto j = 1; j < ndust + 1; j++) {
                r += mdspan_A(0, j) * rhov_d_next[id_a * ndust + (j - 1)];
            }

            dd = r - rhov_next[id_a];

            Tscal dissipation = 0, drag_work = 0;

            // compute work of drag terms
            Tscal inv_rho = 1.0 / (rho_next[id_a]);

            Tvec v_bf = inv_rho * rhov_next[id_a];
            Tvec v_af = inv_rho * r;

            drag_work = 0.5
                        * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
                           + dd[2] * (v_bf[2] + v_af[2]));

            // save gas momentum back
            rhov[id_a] = r;
            rho[id_a]  = rho_next[id_a];

            for (auto d_id = 1; d_id <= ndust; d_id++) {
                r *= 0;
                r += mdspan_A(d_id, 0) * rhov_next[id_a];

                for (auto j = 1; j <= ndust; j++) {

                    r += mdspan_A(d_id, j) * rhov_d_next[id_a * ndust + (j - 1)];
                }

                dd = r - rhov_d_next[id_a * ndust + (d_id - 1)];

                inv_rho = 1.0 / (rho_d_next[id_a * ndust + (d_id - 1)]);

                v_bf = inv_rho * rhov_d_next[id_a * ndust + (d_id - 1)];

                v_af = inv_rho * r;

                // compute dissipaation by id-th dust
                dissipation += 0.5
                               * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
                                  + dd[2] * (v_bf[2] + v_af[2]));

                // save dust momentum back
                rhov_d[id_a * ndust + (d_id - 1)] = r;
                rho_d[id_a * ndust + (d_id - 1)]  = rho_d_next[id_a * ndust + (d_id - 1)];
            }

            // updates energy
            rhoe[id_a] = rhoe_next[id_a] + (1 - friction_control) * drag_work
                         - friction_control * dissipation;
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void ApplyDragExpo<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho_next.check_sizes(edges.sizes.indexes);
        edges.spans_rhov_next.check_sizes(edges.sizes.indexes);
        edges.spans_rhoe_next.check_sizes(edges.sizes.indexes);
        edges.spans_rho_d_next.check_sizes(edges.sizes.indexes);
        edges.spans_rhov_d_next.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_rhov.check_sizes(edges.sizes.indexes);
        edges.spans_rhoe.check_sizes(edges.sizes.indexes);
        edges.spans_rho_d.check_sizes(edges.sizes.indexes);
        edges.spans_rhov_d.check_sizes(edges.sizes.indexes);

        Tscal dt             = edges.dt.value;
        u32 ndust            = this->ndust;
        u32 friction_control = (enable_frictional_heating == false) ? 1 : 0;

        auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
        sham::DeviceQueue &q = dev_sched->get_queue();

        sham::DeviceBuffer<Tscal> alphas_buf(ndust, dev_sched);
        alphas_buf.copy_from_stdvec(alphas);

        size_t mat_size         = ndust + 1;
        size_t mat_size_squared = mat_size * mat_size;
        size_t group_size
            = (q.get_device_prop().local_mem_size) / (5 * mat_size_squared * sizeof(Tscal));
        size_t loc_acc_size = mat_size_squared * group_size;

        size_t loc_mem_size = 5 * sizeof(Tscal) * loc_acc_size;

        bool use_global_mem = group_size < 8;

        if (!use_global_mem && loc_mem_size > q.get_device_prop().local_mem_size) {
            shambase::throw_with_loc<std::runtime_error>(sham::format(
                "not enough local memory for expo drag integrator:\n"
                "loc_mem_size: {} > max_local_mem: {}\n"
                "loc_acc_size: {}\n"
                "group_size: {}\n"
                "ndust: {}\n",
                loc_mem_size,
                q.get_device_prop().local_mem_size,
                loc_acc_size,
                group_size,
                ndust));
        }

        edges.sizes.indexes.for_each([&](u64 id_patch, u32 block_count) {
            u32 cell_count = block_count * block_size;

            auto in = sham::MultiRef{
                edges.spans_rho_next.get_spans().get(id_patch),
                edges.spans_rhov_next.get_spans().get(id_patch),
                edges.spans_rhoe_next.get_spans().get(id_patch),
                edges.spans_rho_d_next.get_spans().get(id_patch),
                edges.spans_rhov_d_next.get_spans().get(id_patch),
                alphas_buf};

            auto make_cell = [dt, ndust, friction_control, mat_size](
                                 const Tscal *rho_next,
                                 const Tvec *rhov_next,
                                 const Tscal *rhoe_next,
                                 const Tscal *rho_d_next,
                                 const Tvec *rhov_d_next,
                                 const Tscal *alphas,
                                 Tscal *rho,
                                 Tvec *rhov,
                                 Tscal *rhoe,
                                 Tscal *rho_d,
                                 Tvec *rhov_d) {
                return ExpoDragCell<Tvec>{
                    dt,
                    ndust,
                    friction_control,
                    mat_size,
                    rho_next,
                    rhov_next,
                    rhoe_next,
                    rho_d_next,
                    rhov_d_next,
                    alphas,
                    rho,
                    rhov,
                    rhoe,
                    rho_d,
                    rhov_d};
            };

            if (use_global_mem) {
                sham::DeviceBuffer<Tscal> scratch_expo(
                    5 * mat_size_squared * cell_count, dev_sched);

                sham::kernel_call(
                    q,
                    in,
                    sham::MultiRef{
                        edges.spans_rho.get_spans().get(id_patch),
                        edges.spans_rhov.get_spans().get(id_patch),
                        edges.spans_rhoe.get_spans().get(id_patch),
                        edges.spans_rho_d.get_spans().get(id_patch),
                        edges.spans_rhov_d.get_spans().get(id_patch),
                        scratch_expo},
                    cell_count,
                    [mat_size_squared, make_cell](
                        u32 id_a,
                        const Tscal *rho_next,
                        const Tvec *rhov_next,
                        const Tscal *rhoe_next,
                        const Tscal *rho_d_next,
                        const Tvec *rhov_d_next,
                        const Tscal *alphas,
                        Tscal *rho,
                        Tvec *rhov,
                        Tscal *rhoe,
                        Tscal *rho_d,
                        Tvec *rhov_d,
                        Tscal *exp_scratch_ptr_base) {
                        auto cell = make_cell(
                            rho_next,
                            rhov_next,
                            rhoe_next,
                            rho_d_next,
                            rhov_d_next,
                            alphas,
                            rho,
                            rhov,
                            rhoe,
                            rho_d,
                            rhov_d);

                        Tscal *ptr_base = exp_scratch_ptr_base + (id_a * 5 * mat_size_squared);

                        cell(
                            id_a,
                            ptr_base,
                            ptr_base + mat_size_squared,
                            ptr_base + 2 * mat_size_squared,
                            ptr_base + 3 * mat_size_squared,
                            ptr_base + 4 * mat_size_squared);
                    });

            } else {

                sham::kernel_call_hndl(
                    q,
                    in,
                    sham::MultiRef{
                        edges.spans_rho.get_spans().get(id_patch),
                        edges.spans_rhov.get_spans().get(id_patch),
                        edges.spans_rhoe.get_spans().get(id_patch),
                        edges.spans_rho_d.get_spans().get(id_patch),
                        edges.spans_rhov_d.get_spans().get(id_patch)},
                    cell_count,
                    [mat_size_squared, loc_acc_size, group_size, make_cell](
                        u32 n,
                        const Tscal *rho_next,
                        const Tvec *rhov_next,
                        const Tscal *rhoe_next,
                        const Tscal *rho_d_next,
                        const Tvec *rhov_d_next,
                        const Tscal *alphas,
                        Tscal *rho,
                        Tvec *rhov,
                        Tscal *rhoe,
                        Tscal *rho_d,
                        Tvec *rhov_d) {
                        auto cell = make_cell(
                            rho_next,
                            rhov_next,
                            rhoe_next,
                            rho_d_next,
                            rhov_d_next,
                            alphas,
                            rho,
                            rhov,
                            rhoe,
                            rho_d,
                            rhov_d);

                        return [=](sycl::handler &cgh) {
                            // local/shared memory alloc for each work-item
                            sycl::local_accessor<Tscal> local_A(loc_acc_size, cgh);
                            sycl::local_accessor<Tscal> local_B(loc_acc_size, cgh);
                            sycl::local_accessor<Tscal> local_F(loc_acc_size, cgh);
                            sycl::local_accessor<Tscal> local_I(loc_acc_size, cgh);
                            sycl::local_accessor<Tscal> local_Id(loc_acc_size, cgh);

                            cgh.parallel_for(
                                shambase::make_range(n, group_size), [=](sycl::nd_item<1> id) {
                                    u32 loc_id = id.get_local_id();
                                    u32 id_a   = id.get_global_id();
                                    if (id_a >= n)
                                        return;

                                    cell(
                                        id_a,
                                        &(local_A[0]) + mat_size_squared * loc_id,
                                        &(local_B[0]) + mat_size_squared * loc_id,
                                        &(local_F[0]) + mat_size_squared * loc_id,
                                        &(local_I[0]) + mat_size_squared * loc_id,
                                        &(local_Id[0]) + mat_size_squared * loc_id);
                                });
                        };
                    });
            }
        });
    }

    template<class Tvec>
    std::string ApplyDragExpo<Tvec>::_impl_get_tex() const {
        std::string tex = R"tex(
            Gas-dust drag, matrix exponential integrator

            Update ${spans_rho}$, ${spans_rhov}$, ${spans_rhoe}$, ${spans_rho_d}$, ${spans_rhov_d}$
            from the state without drag ${spans_rho_next}$, ${spans_rhov_next}$,
            ${spans_rhoe_next}$, ${spans_rho_d_next}$, ${spans_rhov_d_next}$ over a time step
            ${dt}$.
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ApplyDragExpo<f64_3>;

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeGradients.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief MUSCL reconstruction gradient computation implementation
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/math.hpp"
#include "shammodels/gsph/modules/ComputeGradients.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/TreeTraversalCache.hpp"

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void ComputeGradients<Tvec, SPHKernel>::compute() {
        StackEntry stack_loc{};

        if (!solver_config.requires_gradients()) {
            return;
        }

        using namespace shamrock;
        using namespace shamrock::patch;

        const Tscal pmass = solver_config.gpart_mass;
        const Tscal gamma = solver_config.get_eos_gamma();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        PatchDataLayerLayout &pdl = scheduler().pdl();
        const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
        const bool has_uint       = solver_config.has_field_uint();
        const u32 iuint           = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;

        shamrock::solvergraph::Field<Tvec> &grad_density_field
            = shambase::get_check_ref(storage.grad_density);
        shamrock::solvergraph::Field<Tvec> &grad_pressure_field
            = shambase::get_check_ref(storage.grad_pressure);
        shamrock::solvergraph::Field<Tvec> &grad_vx_field
            = shambase::get_check_ref(storage.grad_vx);
        shamrock::solvergraph::Field<Tvec> &grad_vy_field
            = shambase::get_check_ref(storage.grad_vy);
        shamrock::solvergraph::Field<Tvec> &grad_vz_field
            = shambase::get_check_ref(storage.grad_vz);

        shamrock::solvergraph::Field<Tscal> &density_field
            = shambase::get_check_ref(storage.density);

        shambase::DistributedData<u32> &counts
            = shambase::get_check_ref(storage.part_counts).indexes;

        grad_density_field.ensure_sizes(counts);
        grad_pressure_field.ensure_sizes(counts);
        grad_vx_field.ensure_sizes(counts);
        grad_vy_field.ensure_sizes(counts);
        grad_vz_field.ensure_sizes(counts);

        auto &merged_xyzh = storage.merged_xyzh.get();
        auto &neigh_cache = storage.neigh_cache->neigh_cache;

        static constexpr Tscal Rkern = Kernel::Rkern;

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &mfield = merged_xyzh.get(p.id_patch);
            auto &pcache = neigh_cache.get(p.id_patch);

            auto &buf_xyz   = mfield.template get_field_buf_ref<Tvec>(0);
            auto &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);
            auto &buf_vxyz  = pdat.get_field_buf_ref<Tvec>(ivxyz);

            auto &dens_field   = density_field.get_field(p.id_patch);
            auto &grad_d_field = grad_density_field.get_field(p.id_patch);
            auto &grad_p_field = grad_pressure_field.get_field(p.id_patch);
            auto &grad_vx_buf  = grad_vx_field.get_field(p.id_patch);
            auto &grad_vy_buf  = grad_vy_field.get_field(p.id_patch);
            auto &grad_vz_buf  = grad_vz_field.get_field(p.id_patch);

            sham::DeviceQueue &q = dev_sched->get_queue();
            sham::EventList depends_list;

            auto ploop_ptrs  = pcache.get_read_access(depends_list);
            auto xyz_acc     = buf_xyz.get_read_access(depends_list);
            auto h_acc       = buf_hpart.get_read_access(depends_list);
            auto v_acc       = buf_vxyz.get_read_access(depends_list);
            auto dens_acc    = dens_field.get_buf().get_read_access(depends_list);
            auto grad_d_acc  = grad_d_field.get_buf().get_write_access(depends_list);
            auto grad_p_acc  = grad_p_field.get_buf().get_write_access(depends_list);
            auto grad_vx_acc = grad_vx_buf.get_buf().get_write_access(depends_list);
            auto grad_vy_acc = grad_vy_buf.get_buf().get_write_access(depends_list);
            auto grad_vz_acc = grad_vz_buf.get_buf().get_write_access(depends_list);

            const Tscal *uint_ptr = nullptr;
            if (has_uint) {
                uint_ptr = pdat.get_field_buf_ref<Tscal>(iuint).get_read_access(depends_list);
            }

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                shambase::parallel_for(cgh, cnt, "gsph_compute_gradients", [=](u64 gid) {
                    u32 id_a = (u32) gid;

                    Tvec xyz_a  = xyz_acc[id_a];
                    Tscal h_a   = h_acc[id_a];
                    Tvec v_a    = v_acc[id_a];
                    Tscal rho_a = sycl::max(dens_acc[id_a], Tscal(1e-30));
                    Tscal dint  = h_a * h_a * Rkern * Rkern;

                    Tscal u_a = Tscal(0);
                    if (uint_ptr != nullptr) {
                        u_a = uint_ptr[id_a];
                    }

                    Tvec grad_d  = {0, 0, 0};
                    Tvec grad_u  = {0, 0, 0};
                    Tvec grad_vx = {0, 0, 0};
                    Tvec grad_vy = {0, 0, 0};
                    Tvec grad_vz = {0, 0, 0};

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        Tvec dr    = xyz_a - xyz_acc[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);

                        if (rab2 > dint || id_a == id_b) {
                            return;
                        }

                        Tscal rab = sycl::sqrt(rab2);

                        Tscal dWdr = Kernel::dW_3d(rab, h_a);
                        Tvec gradW = dr * (dWdr * sham::inv_sat_positive(rab));

                        grad_d += gradW * pmass;

                        Tscal u_b = (uint_ptr != nullptr) ? uint_ptr[id_b] : Tscal(0);
                        grad_u += gradW * (pmass * (u_b - u_a));

                        Tvec v_b = v_acc[id_b];
                        grad_vx += gradW * (pmass * (v_b[0] - v_a[0]));
                        grad_vy += gradW * (pmass * (v_b[1] - v_a[1]));
                        grad_vz += gradW * (pmass * (v_b[2] - v_a[2]));
                    });

                    grad_d_acc[id_a] = grad_d;

                    Tvec grad_p      = (grad_d * u_a + grad_u) * (gamma - Tscal(1));
                    grad_p_acc[id_a] = grad_p;

                    Tscal rho_inv     = sham::inv_sat_positive(rho_a);
                    grad_vx_acc[id_a] = grad_vx * rho_inv;
                    grad_vy_acc[id_a] = grad_vy * rho_inv;
                    grad_vz_acc[id_a] = grad_vz * rho_inv;
                });
            });

            pcache.complete_event_state({e});
            buf_xyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            dens_field.get_buf().complete_event_state(e);
            grad_d_field.get_buf().complete_event_state(e);
            grad_p_field.get_buf().complete_event_state(e);
            grad_vx_buf.get_buf().complete_event_state(e);
            grad_vy_buf.get_buf().complete_event_state(e);
            grad_vz_buf.get_buf().complete_event_state(e);
            if (has_uint) {
                pdat.get_field_buf_ref<Tscal>(iuint).complete_event_state(e);
            }
        });
    }

    // Explicit instantiations
    template class ComputeGradients<f64_3, shammath::M4>;
    template class ComputeGradients<f64_3, shammath::M6>;
    template class ComputeGradients<f64_3, shammath::M8>;
    template class ComputeGradients<f64_3, shammath::C2>;
    template class ComputeGradients<f64_3, shammath::C4>;
    template class ComputeGradients<f64_3, shammath::C6>;
    template class ComputeGradients<f64_3, shammath::TGauss3>;

} // namespace shammodels::gsph::modules

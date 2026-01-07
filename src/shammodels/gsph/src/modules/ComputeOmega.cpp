// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeOmega.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Omega and density computation implementation
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/gsph/modules/ComputeOmega.hpp"
#include "shammodels/gsph/modules/IterateSmoothingLengthVolume.hpp"
#include "shammodels/sph/modules/IterateSmoothingLengthDensity.hpp"
#include "shammodels/sph/modules/LoopSmoothingLengthIter.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/TreeTraversalCache.hpp"

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    bool ComputeOmega<Tvec, SPHKernel>::compute() {
        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;

        const Tscal pmass = solver_config.gpart_mass;

        if (shamcomm::world_rank() == 0) {
            if (pmass <= Tscal(0) || pmass < Tscal(1e-100) || !std::isfinite(pmass)) {
                logger::warn_ln("GSPH", "Invalid particle mass in compute_omega: pmass =", pmass);
            }
        }

        shamrock::solvergraph::Field<Tscal> &omega_field = shambase::get_check_ref(storage.omega);
        shamrock::solvergraph::Field<Tscal> &density_field
            = shambase::get_check_ref(storage.density);

        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes
            = std::make_shared<shamrock::solvergraph::Indexes<u32>>("sizes", "N");
        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            sizes->indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
        });

        omega_field.ensure_sizes(sizes->indexes);
        density_field.ensure_sizes(sizes->indexes);

        PatchDataLayerLayout &pdl = scheduler().pdl();
        const u32 ihpart          = pdl.template get_field_idx<Tscal>("hpart");

        auto &merged_xyzh = storage.merged_xyzh.get();

        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> pos_merged
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("pos", "r");
        shamrock::solvergraph::DDPatchDataFieldRef<Tvec> pos_refs = {};

        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hold
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("h_old", "h^{old}");
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> hold_refs = {};

        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hnew
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("h_new", "h^{new}");
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> hnew_refs = {};

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &mfield = merged_xyzh.get(p.id_patch);

            pos_refs.add_obj(p.id_patch, std::ref(mfield.template get_field<Tvec>(0)));
            hold_refs.add_obj(p.id_patch, std::ref(mfield.template get_field<Tscal>(1)));
            hnew_refs.add_obj(p.id_patch, std::ref(pdat.template get_field<Tscal>(ihpart)));
        });

        pos_merged->set_refs(pos_refs);
        hold->set_refs(hold_refs);
        hnew->set_refs(hnew_refs);

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &mfield           = merged_xyzh.get(p.id_patch);
            auto &buf_hpart_merged = mfield.template get_field_buf_ref<Tscal>(1);
            auto &buf_hpart_local  = pdat.template get_field_buf_ref<Tscal>(ihpart);

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{buf_hpart_merged},
                sham::MultiRef{buf_hpart_local},
                cnt,
                [](u32 i, const Tscal *h_old, Tscal *h_new) {
                    h_new[i] = h_old[i];
                });
        });

        shamrock::SchedulerUtility utility(scheduler());
        ComputeField<Tscal> _epsilon_h = utility.make_compute_field<Tscal>("epsilon_h", 1);

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &eps_buf = _epsilon_h.get_buf_check(p.id_patch);

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{},
                sham::MultiRef{eps_buf},
                cnt,
                [](u32 i, Tscal *eps) {
                    eps[i] = Tscal(1);
                });
        });

        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> eps_h
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("eps_h", "\\epsilon_h");
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> eps_h_refs = {};
        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &field = _epsilon_h.get_field(p.id_patch);
            eps_h_refs.add_obj(p.id_patch, std::ref(field));
        });
        eps_h->set_refs(eps_h_refs);

        // Standard density-based h iteration (Newtonian mode)
        // SR mode uses its own volume-based implementation
        auto std_iter = std::make_shared<sph::modules::IterateSmoothingLengthDensity<Tvec, Kernel>>(
            solver_config.gpart_mass,
            solver_config.htol_up_coarse_cycle,
            solver_config.htol_up_fine_cycle,
            solver_config.c_smooth);
        std_iter->set_edges(sizes, storage.neigh_cache, pos_merged, hold, hnew, eps_h);
        std::shared_ptr<shamrock::solvergraph::INode> smth_h_iter = std_iter;

        std::shared_ptr<shamrock::solvergraph::ScalarEdge<bool>> is_converged
            = std::make_shared<shamrock::solvergraph::ScalarEdge<bool>>(
                "is_converged", "converged");

        shammodels::sph::modules::LoopSmoothingLengthIter<Tvec> loop_smth_h_iter(
            smth_h_iter, solver_config.epsilon_h, solver_config.h_iter_per_subcycles, false);
        loop_smth_h_iter.set_edges(eps_h, is_converged);

        loop_smth_h_iter.evaluate();

        bool needs_cache_rebuild = false;
        if (!is_converged->value) {
            Tscal local_max_eps  = shamrock::solvergraph::get_rank_max(*eps_h);
            Tscal global_max_eps = shamalgs::collective::allreduce_max(local_max_eps);

            u64 cnt_unconverged = 0;
            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                auto res
                    = _epsilon_h.get_field(p.id_patch).get_ids_buf_where([](auto access, u32 id) {
                          return access[id] < Tscal(0);
                      });
                cnt_unconverged += std::get<1>(res);
            });
            u64 global_cnt_unconverged = shamalgs::collective::allreduce_sum(cnt_unconverged);

            if (global_cnt_unconverged > 0) {
                needs_cache_rebuild = true;
                if (shamcomm::world_rank() == 0) {
                    logger::info_ln(
                        "GSPH",
                        "Smoothing length iteration: ",
                        global_cnt_unconverged,
                        " particles need cache rebuild (h grew beyond tolerance)");
                }
            } else if (shamcomm::world_rank() == 0) {
                logger::warn_ln(
                    "GSPH",
                    "Smoothing length iteration did not converge, max eps =",
                    global_max_eps);
            }
        }

        static constexpr Tscal Rkern = Kernel::Rkern;

        auto &neigh_cache = storage.neigh_cache->neigh_cache;

        const bool has_pmass = solver_config.has_field_pmass();
        const u32 ipmass     = has_pmass ? pdl.template get_field_idx<Tscal>("pmass") : 0;

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &mfield = merged_xyzh.get(p.id_patch);
            auto &pcache = neigh_cache.get(p.id_patch);

            auto &buf_xyz   = mfield.template get_field_buf_ref<Tvec>(0);
            auto &buf_hpart = pdat.template get_field_buf_ref<Tscal>(ihpart);

            auto &dens_field = density_field.get_field(p.id_patch);
            auto &omeg_field = omega_field.get_field(p.id_patch);

            sham::DeviceQueue &q = dev_sched->get_queue();
            sham::EventList depends_list;

            auto ploop_ptrs  = pcache.get_read_access(depends_list);
            auto xyz_acc     = buf_xyz.get_read_access(depends_list);
            auto h_acc       = buf_hpart.get_read_access(depends_list);
            auto density_acc = dens_field.get_buf().get_write_access(depends_list);
            auto omega_acc   = omeg_field.get_buf().get_write_access(depends_list);

            const Tscal *pmass_acc                   = nullptr;
            sham::DeviceBuffer<Tscal> *buf_pmass_ptr = nullptr;
            if (has_pmass) {
                buf_pmass_ptr = &pdat.template get_field_buf_ref<Tscal>(ipmass);
                pmass_acc     = buf_pmass_ptr->get_read_access(depends_list);
            }

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                shambase::parallel_for(cgh, cnt, "gsph_compute_density_omega", [=](u64 gid) {
                    u32 id_a = (u32) gid;

                    Tvec xyz_a = xyz_acc[id_a];
                    Tscal h_a  = h_acc[id_a];
                    Tscal dint = h_a * h_a * Rkern * Rkern;

                    Tscal rho_sum = Tscal(0);
                    Tscal sumdWdh = Tscal(0);

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        Tvec dr    = xyz_a - xyz_acc[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);

                        if (rab2 > dint) {
                            return;
                        }

                        Tscal rab = sycl::sqrt(rab2);

                        if (has_pmass) {
                            rho_sum += Kernel::W_3d(rab, h_a);
                            sumdWdh += Kernel::dhW_3d(rab, h_a);
                        } else {
                            rho_sum += pmass * Kernel::W_3d(rab, h_a);
                            sumdWdh += pmass * Kernel::dhW_3d(rab, h_a);
                        }
                    });

                    if (has_pmass) {
                        Tscal nu_a = pmass_acc[id_a];
                        rho_sum *= nu_a;
                        sumdWdh *= nu_a;
                    }

                    density_acc[id_a] = sycl::max(rho_sum, Tscal(1e-30));

                    Tscal omega_val = Tscal(1);
                    if (rho_sum > Tscal(1e-30)) {
                        omega_val = Tscal(1) + h_a / (Tscal(dim) * rho_sum) * sumdWdh;
                        omega_val = sycl::clamp(omega_val, Tscal(0.5), Tscal(2.0));
                    }
                    omega_acc[id_a] = omega_val;
                });
            });

            pcache.complete_event_state({e});
            buf_xyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            dens_field.get_buf().complete_event_state(e);
            omeg_field.get_buf().complete_event_state(e);
            if (has_pmass && buf_pmass_ptr) {
                buf_pmass_ptr->complete_event_state(e);
            }
        });

        return !needs_cache_rebuild;
    }

    // Explicit instantiations
    template class ComputeOmega<f64_3, shammath::M4>;
    template class ComputeOmega<f64_3, shammath::M6>;
    template class ComputeOmega<f64_3, shammath::M8>;
    template class ComputeOmega<f64_3, shammath::C2>;
    template class ComputeOmega<f64_3, shammath::C4>;
    template class ComputeOmega<f64_3, shammath::C6>;
    template class ComputeOmega<f64_3, shammath::TGauss3>;

} // namespace shammodels::gsph::modules

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DiffOperator.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/StorageComponent.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/DiffOperator.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <functional>

namespace {

    class NeighCache : public shamrock::solvergraph::IDataEdgeNamed {
        public:
        NeighCache(
            std::string name,
            std::string texsymbol,
            shambase::StorageComponent<shamrock::tree::ObjectCacheHandler> &neigh_cache_handle)
            : IDataEdgeNamed(name, texsymbol), neigh_cache_handle(neigh_cache_handle) {}

        shambase::StorageComponent<shamrock::tree::ObjectCacheHandler> &neigh_cache_handle;

        inline virtual void free_alloc() { neigh_cache_handle.get().reset(); }
    };

#if false
    template<class Tvec>
    struct KernelDivv {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &buf_xyz,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_rhov,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rhoe,

            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_vel,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_P,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size,
            Tscal gamma) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_rho, spans_rhov, spans_rhoe},
                sham::DDMultiRef{spans_vel, spans_P},
                cell_counts,
                [gamma](
                    u32 i,
                    const Tscal *__restrict rho,
                    const Tvec *__restrict rhov,
                    const Tscal *__restrict rhoe,
                    Tvec *__restrict vel,
                    Tscal *__restrict P) {
                    auto conststate = shammath::ConsState<Tvec>{rho[i], rhoe[i], rhov[i]};

                    auto prim_state = shammath::cons_to_prim(conststate, gamma);

                    vel[i] = prim_state.vel;
                    P[i]   = prim_state.press;
                });
        }
    };
#endif

} // namespace

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::DiffOperators<Tvec, SPHKernel>::update_divv() {

    StackEntry stack_loc{};
    logger::debug_ln("SPH", "Updating divv");

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    auto &merged_xyzh = storage.merged_xyzh.get();

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    const u32 idivv = pdl.get_field_idx<Tscal>("divv");

    shamrock::solvergraph::Indexes<u32> part_counts("part_count", "N_{\\rm part}");
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        part_counts.indexes.add_obj(cur_p.id_patch, pdat.get_obj_cnt());
    });

    shamrock::solvergraph::Indexes<u32> part_counts_with_ghosts(
        "part_count_with_ghost", "N_{\\rm part, with ghost}");
    part_counts_with_ghosts.indexes
        = storage.merged_patchdata_ghost.get().template map<u32>([&](u64 id, auto &mpdat) {
              return mpdat.total_elements;
          });

    NeighCache ncache("neigh_cache", "neigh cache", storage.neighbors_cache);

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> refs_xyz
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("xyz", "\\mathbf{r}");
    refs_xyz->set_refs(storage.merged_patchdata_ghost.get()
                           .template map<std::reference_wrapper<PatchDataField<Tvec>>>(
                               [&](u64 id, MergedPatchData &mpdat) {
                                   return std::ref(merged_xyzh.get(id).field_pos);
                               }));

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> refs_vxyz
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("vxyz", "\\mathbf{v}");
    refs_vxyz->set_refs(storage.merged_patchdata_ghost.get()
                            .template map<std::reference_wrapper<PatchDataField<Tvec>>>(
                                [&](u64 id, MergedPatchData &mpdat) {
                                    return std::ref(mpdat.pdat.get_field<Tvec>(ivxyz_interf));
                                }));

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> refs_hpart
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("hpart", "\\mathbf{v}");
    refs_hpart->set_refs(storage.merged_patchdata_ghost.get()
                             .template map<std::reference_wrapper<PatchDataField<Tscal>>>(
                                 [&](u64 id, MergedPatchData &mpdat) {
                                     return std::ref(mpdat.pdat.get_field<Tscal>(ihpart_interf));
                                 }));

    refs_xyz->check_sizes(part_counts.indexes);
    refs_vxyz->check_sizes(part_counts.indexes);

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;

        PatchDataFieldSpanPointer<Tvec> buf_xyz    = refs_xyz->get_spans().get(cur_p.id_patch);
        PatchDataFieldSpanPointer<Tvec> buf_vxyz   = refs_vxyz->get_spans().get(cur_p.id_patch);
        PatchDataFieldSpanPointer<Tscal> buf_hpart = refs_hpart->get_spans().get(cur_p.id_patch);
        // sham::DeviceBuffer<Tvec> &buf_xyz    =
        // merged_xyzh.get(cur_p.id_patch).field_pos.get_buf(); sham::DeviceBuffer<Tvec> &buf_vxyz
        // = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        // sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sham::DeviceBuffer<Tscal> &buf_divv  = pdat.get_field_buf_ref<Tscal>(idivv);

        tree::ObjectCache &pcache = ncache.neigh_cache_handle.get().get_cache(cur_p.id_patch);

        u32 npart = part_counts.indexes.get(cur_p.id_patch);

        {
            NamedStackEntry tmppp{"compute divv"};

            sham::EventList depends_list;

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::kernel_call(
                q,
                sham::MultiRef{buf_xyz, buf_vxyz, buf_hpart, buf_omega, pcache},
                sham::MultiRef{buf_divv},
                npart,
                [pmass = gpart_mass](
                    u32 id_a,
                    const Tvec *__restrict xyz,
                    const Tvec *__restrict vxyz,
                    const Tscal *__restrict hpart,
                    const Tscal *__restrict omega,
                    auto ploop_ptrs,
                    Tscal *__restrict divv) {
                    tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                    constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                    using namespace shamrock::sph;

                    Tvec sum_axyz  = {0, 0, 0};
                    Tscal sum_du_a = 0;
                    Tscal h_a      = hpart[id_a];
                    Tvec xyz_a     = xyz[id_a];
                    Tvec vxyz_a    = vxyz[id_a];
                    Tscal omega_a  = omega[id_a];

                    Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);
                    // Tscal rho_a_sq  = rho_a * rho_a;
                    // Tscal rho_a_inv = 1. / rho_a;
                    Tscal inv_rho_omega_a = 1. / (omega_a * rho_a);

                    Tscal sum_nabla_v = 0;

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        // compute only omega_a
                        Tvec dr    = xyz_a - xyz[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);
                        Tscal h_b  = hpart[id_b];

                        if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                            return;
                        }

                        Tscal rab   = sycl::sqrt(rab2);
                        Tvec vxyz_b = vxyz[id_b];
                        Tvec v_ab   = vxyz_a - vxyz_b;

                        Tvec r_ab_unit = dr / rab;

                        if (rab < 1e-9) {
                            r_ab_unit = {0, 0, 0};
                        }

                        Tvec dWab_a = Kernel::dW_3d(rab, h_a) * r_ab_unit;

                        sum_nabla_v += pmass * sycl::dot(v_ab, dWab_a);
                    });

                    divv[id_a] = -inv_rho_omega_a * sum_nabla_v;
                });
        }
    });
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::DiffOperators<Tvec, SPHKernel>::update_curlv() {

    StackEntry stack_loc{};
    logger::debug_ln("SPH", "Updating curlv");

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    auto &merged_xyzh = storage.merged_xyzh.get();

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    const u32 icurlv = pdl.get_field_idx<Tvec>("curlv");
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;

        sham::DeviceBuffer<Tvec> &buf_xyz    = merged_xyzh.get(cur_p.id_patch).field_pos.get_buf();
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sham::DeviceBuffer<Tvec> &buf_curlv  = pdat.get_field_buf_ref<Tvec>(icurlv);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        {
            NamedStackEntry tmppp{"compute curlv"};

            sham::EventList depends_list;
            auto xyz        = buf_xyz.get_read_access(depends_list);
            auto vxyz       = buf_vxyz.get_read_access(depends_list);
            auto hpart      = buf_hpart.get_read_access(depends_list);
            auto omega      = buf_omega.get_read_access(depends_list);
            auto curlv      = buf_curlv.get_write_access(depends_list);
            auto ploop_ptrs = pcache.get_read_access(depends_list);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                const Tscal pmass = gpart_mass;

                tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute curlv", [=](i32 id_a) {
                    using namespace shamrock::sph;

                    Tvec sum_axyz  = {0, 0, 0};
                    Tscal sum_du_a = 0;
                    Tscal h_a      = hpart[id_a];
                    Tvec xyz_a     = xyz[id_a];
                    Tvec vxyz_a    = vxyz[id_a];
                    Tscal omega_a  = omega[id_a];

                    Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);
                    // Tscal rho_a_sq  = rho_a * rho_a;
                    // Tscal rho_a_inv = 1. / rho_a;
                    Tscal inv_rho_omega_a = 1. / (omega_a * rho_a);

                    Tvec sum_nabla_cross_v{};

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        // compute only omega_a
                        Tvec dr    = xyz_a - xyz[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);
                        Tscal h_b  = hpart[id_b];

                        if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                            return;
                        }

                        Tscal rab   = sycl::sqrt(rab2);
                        Tvec vxyz_b = vxyz[id_b];
                        Tvec v_ab   = vxyz_a - vxyz_b;

                        Tvec r_ab_unit = dr / rab;

                        if (rab < 1e-9) {
                            r_ab_unit = {0, 0, 0};
                        }

                        Tvec dWab_a = Kernel::dW_3d(rab, h_a) * r_ab_unit;

                        sum_nabla_cross_v += pmass * sycl::cross(v_ab, dWab_a);
                    });

                    curlv[id_a] = -inv_rho_omega_a * sum_nabla_cross_v;
                });
            });

            buf_xyz.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            buf_omega.complete_event_state(e);
            buf_curlv.complete_event_state(e);

            sham::EventList resulting_events;
            resulting_events.add_event(e);
            pcache.complete_event_state(resulting_events);
        }
    });
}

using namespace shammath;
template class shammodels::sph::modules::DiffOperators<f64_3, M4>;
template class shammodels::sph::modules::DiffOperators<f64_3, M6>;
template class shammodels::sph::modules::DiffOperators<f64_3, M8>;

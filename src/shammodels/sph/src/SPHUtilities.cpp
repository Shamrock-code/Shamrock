// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHUtilities.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/SPHUtilities.hpp"
#include "shammath/sphkernels.hpp"
#include "shamtree/TreeTraversal.hpp"

using namespace shamrock::sph;

namespace shammodels::sph {

    template<class vec, class SPHKernel>
    void SPHUtilities<vec, SPHKernel>::iterate_smoothing_length_cache(

        sham::DeviceBuffer<vec> &merged_r,
        sham::DeviceBuffer<flt> &hnew,
        sham::DeviceBuffer<flt> &hold,
        sham::DeviceBuffer<flt> &eps_h,
        sycl::range<1> update_range,
        shamrock::tree::ObjectCache &neigh_cache,

        flt gpart_mass,
        flt h_evol_max,
        flt h_evol_iter_max

    ) {

        sham::DeviceQueue &queue = shamsys::instance::get_compute_scheduler().get_queue();

        sham::EventList depends_list;

        auto r          = merged_r.get_read_access(depends_list);
        auto h_old      = hold.get_read_access(depends_list);
        auto ploop_ptrs = neigh_cache.get_read_access(depends_list);
        auto eps        = eps_h.get_write_access(depends_list);
        auto h_new      = hnew.get_write_access(depends_list);

        auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
            // tree::ObjectIterator particle_looper(tree,cgh);

            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            // sycl::accessor omega {omega_h, cgh, sycl::write_only, sycl::no_init};

            const flt part_mass          = gpart_mass;
            const flt h_max_tot_max_evol = h_evol_max;
            const flt h_max_evol_p       = h_evol_iter_max;
            const flt h_max_evol_m       = 1 / h_evol_iter_max;

            shambase::parallel_for(cgh, update_range.size(), "iter h", [=](u32 id_a) {
                if (eps[id_a] > 1e-6) {

                    vec xyz_a = r[id_a]; // could be recovered from lambda

                    flt h_a  = h_new[id_a];
                    flt dint = h_a * h_a * Rkern * Rkern;

                    vec inter_box_a_min = xyz_a - h_a * Rkern;
                    vec inter_box_a_max = xyz_a + h_a * Rkern;

                    flt rho_sum = 0;
                    flt sumdWdh = 0;

                    // particle_looper.rtree_for([&](u32, vec bmin,vec bmax) -> bool {
                    //     return
                    //     shammath::domain_are_connected(bmin,bmax,inter_box_a_min,inter_box_a_max);
                    // },[&](u32 id_b){
                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        vec dr   = xyz_a - r[id_b];
                        flt rab2 = sycl::dot(dr, dr);

                        if (rab2 > dint) {
                            return;
                        }

                        flt rab = sycl::sqrt(rab2);

                        rho_sum += part_mass * SPHKernel::W_3d(rab, h_a);
                        sumdWdh += part_mass * SPHKernel::dhW_3d(rab, h_a);
                    });

                    using namespace shamrock::sph;

                    flt rho_ha = rho_h(part_mass, h_a, SPHKernel::hfactd);
                    flt new_h  = newtown_iterate_new_h(rho_ha, rho_sum, sumdWdh, h_a);

                    if (new_h < h_a * h_max_evol_m)
                        new_h = h_max_evol_m * h_a;
                    if (new_h > h_a * h_max_evol_p)
                        new_h = h_max_evol_p * h_a;

                    flt ha_0 = h_old[id_a];

                    if (new_h < ha_0 * h_max_tot_max_evol) {
                        h_new[id_a] = new_h;
                        eps[id_a]   = sycl::fabs(new_h - h_a) / ha_0;
                    } else {
                        h_new[id_a] = ha_0 * h_max_tot_max_evol;
                        eps[id_a]   = -1;
                    }
                }
            });
        });

        merged_r.complete_event_state(e);
        eps_h.complete_event_state(e);
        hnew.complete_event_state(e);
        hold.complete_event_state(e);

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        neigh_cache.complete_event_state(resulting_events);
    }

    template<class vec, class SPHKernel, class u_morton>
    void SPHTreeUtilities<vec, SPHKernel, u_morton>::iterate_smoothing_length_tree(
        sycl::buffer<vec> &merged_r,
        sycl::buffer<flt> &hnew,
        sycl::buffer<flt> &hold,
        sycl::buffer<flt> &eps_h,
        sycl::range<1> update_range,
        RadixTree<u_morton, vec> &tree,

        flt gpart_mass,
        flt h_evol_max,
        flt h_evol_iter_max) {

        shamsys::instance::get_compute_queue()
            .submit([&](sycl::handler &cgh) {
                shamrock::tree::ObjectIterator particle_looper(tree, cgh);
                // shamrock::tree::ObjectCacheIterator particle_looper(neigh_cache, cgh);

                sycl::accessor eps{eps_h, cgh, sycl::read_write};
                sycl::accessor r{merged_r, cgh, sycl::read_only};
                sycl::accessor h_new{hnew, cgh, sycl::read_write};
                sycl::accessor h_old{hold, cgh, sycl::read_only};
                // sycl::accessor omega {omega_h, cgh, sycl::write_only, sycl::no_init};

                const flt part_mass          = gpart_mass;
                const flt h_max_tot_max_evol = h_evol_max;
                const flt h_max_evol_p       = h_evol_iter_max;
                const flt h_max_evol_m       = 1 / h_evol_iter_max;

                shambase::parallel_for(cgh, update_range.size(), "iter h", [=](u32 id_a) {
                    if (eps[id_a] > 1e-6) {

                        vec xyz_a = r[id_a]; // could be recovered from lambda

                        flt h_a  = h_new[id_a];
                        flt dint = h_a * h_a * Rkern * Rkern;

                        vec inter_box_a_min = xyz_a - h_a * Rkern;
                        vec inter_box_a_max = xyz_a + h_a * Rkern;

                        flt rho_sum = 0;
                        flt sumdWdh = 0;

                        particle_looper.rtree_for(
                            [&](u32, vec bmin, vec bmax) -> bool {
                                return shammath::domain_are_connected(
                                    bmin, bmax, inter_box_a_min, inter_box_a_max);
                            },
                            [&](u32 id_b) {
                                // particle_looper.for_each_object(id_a, [&](u32 id_b) {
                                vec dr   = xyz_a - r[id_b];
                                flt rab2 = sycl::dot(dr, dr);

                                if (rab2 > dint) {
                                    return;
                                }

                                flt rab = sycl::sqrt(rab2);

                                rho_sum += part_mass * SPHKernel::W_3d(rab, h_a);
                                sumdWdh += part_mass * SPHKernel::dhW_3d(rab, h_a);
                            });

                        using namespace shamrock::sph;

                        flt rho_ha = rho_h(part_mass, h_a, SPHKernel::hfactd);
                        flt new_h  = newtown_iterate_new_h(rho_ha, rho_sum, sumdWdh, h_a);

                        if (new_h < h_a * h_max_evol_m)
                            new_h = h_max_evol_m * h_a;
                        if (new_h > h_a * h_max_evol_p)
                            new_h = h_max_evol_p * h_a;

                        flt ha_0 = h_old[id_a];

                        if (new_h < ha_0 * h_max_tot_max_evol) {
                            h_new[id_a] = new_h;
                            eps[id_a]   = sycl::fabs(new_h - h_a) / ha_0;
                        } else {
                            h_new[id_a] = ha_0 * h_max_tot_max_evol;
                            eps[id_a]   = -1;
                        }
                    }
                });
            })
            .wait();
    }

    template<class vec, class SPHKernel>
    void SPHUtilities<vec, SPHKernel>::compute_omega(
        sham::DeviceBuffer<vec> &merged_r,
        sham::DeviceBuffer<flt> &h_part,
        sham::DeviceBuffer<flt> &omega_h,
        sycl::range<1> part_range,
        shamrock::tree::ObjectCache &neigh_cache,
        flt gpart_mass) {
        using namespace shamrock::tree;

        sham::DeviceQueue &queue = shamsys::instance::get_compute_scheduler().get_queue();

        sham::EventList depends_list;

        auto r          = merged_r.get_read_access(depends_list);
        auto hpart      = h_part.get_read_access(depends_list);
        auto omega      = omega_h.get_write_access(depends_list);
        auto ploop_ptrs = neigh_cache.get_read_access(depends_list);

        auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
            // tree::ObjectIterator particle_looper(tree,cgh);

            ObjectCacheIterator particle_looper(ploop_ptrs);

            const flt part_mass = gpart_mass;

            shambase::parallel_for(cgh, part_range.size(), "compute omega", [=](u32 id_a) {
                vec xyz_a = r[id_a]; // could be recovered from lambda

                flt h_a  = hpart[id_a];
                flt dint = h_a * h_a * Rkern * Rkern;

                // vec inter_box_a_min = xyz_a - h_a * Rkern;
                // vec inter_box_a_max = xyz_a + h_a * Rkern;

                flt rho_sum        = 0;
                flt part_omega_sum = 0;

                // particle_looper.rtree_for([&](u32, vec bmin,vec bmax) -> bool {
                //     return
                //     shammath::domain_are_connected(bmin,bmax,inter_box_a_min,inter_box_a_max);
                // },[&](u32 id_b){
                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    vec dr   = xyz_a - r[id_b];
                    flt rab2 = sycl::dot(dr, dr);

                    if (rab2 > dint) {
                        return;
                    }

                    flt rab = sycl::sqrt(rab2);

                    rho_sum += part_mass * SPHKernel::W_3d(rab, h_a);
                    part_omega_sum += part_mass * SPHKernel::dhW_3d(rab, h_a);
                });

                using namespace shamrock::sph;

                flt rho_ha  = rho_h(part_mass, h_a, SPHKernel::hfactd);
                flt omega_a = 1 + (h_a / (3 * rho_ha)) * part_omega_sum;
                omega[id_a] = omega_a;

                // logger::raw(shambase::format("pmass {}, rho_a {}, omega_a {}\n",
                // part_mass,rho_ha, omega_a));
            });
        });

        merged_r.complete_event_state(e);
        h_part.complete_event_state(e);
        omega_h.complete_event_state(e);

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        neigh_cache.complete_event_state(resulting_events);
    }

    template class SPHUtilities<f64_3, shammath::M4<f64>>;
    template class SPHUtilities<f64_3, shammath::M6<f64>>;
    template class SPHUtilities<f64_3, shammath::M8<f64>>;

    template class SPHUtilities<f64_3, shammath::C2<f64>>;
    template class SPHUtilities<f64_3, shammath::C4<f64>>;
    template class SPHUtilities<f64_3, shammath::C6<f64>>;

    template class SPHTreeUtilities<f64_3, shammath::M4<f64>, u32>;
    template class SPHTreeUtilities<f64_3, shammath::M6<f64>, u64>;
    template class SPHTreeUtilities<f64_3, shammath::M8<f64>, u64>;

    template class SPHTreeUtilities<f64_3, shammath::C2<f64>, u32>;
    template class SPHTreeUtilities<f64_3, shammath::C4<f64>, u64>;
    template class SPHTreeUtilities<f64_3, shammath::C6<f64>, u64>;

} // namespace shammodels::sph

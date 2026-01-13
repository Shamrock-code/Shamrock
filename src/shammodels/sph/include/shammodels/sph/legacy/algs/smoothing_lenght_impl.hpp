// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

//%Impl status : Good

#pragma once

/**
 * @file smoothing_lenght_impl.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/legacy/patch/base/patchdata.hpp"
// #include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/sph/kernels.hpp"
#include "shamrock/sph/sphpart.hpp"
#include "shamtree/RadixTree.hpp"

namespace impl {

    template<class flt>
    void sycl_init_h_iter_bufs(
        sycl::queue &queue,
        u32 or_element_cnt,

        u32 ihpart,
        shamrock::patch::PatchData &pdat_merge,
        sycl::buffer<flt> &hnew,
        sycl::buffer<flt> &omega,
        sycl::buffer<flt> &eps_h

    );

    template<>
    inline void sycl_init_h_iter_bufs<f32>(
        sycl::queue &queue,
        u32 or_element_cnt,

        u32 ihpart,
        shamrock::patch::PatchData &pdat_merge,
        sycl::buffer<f32> &hnew,
        sycl::buffer<f32> &omega,
        sycl::buffer<f32> &eps_h

    ) {

        sycl::range range_npart{or_element_cnt};

        queue.submit([&](sycl::handler &cgh) {
            auto acc_hpart
                = pdat_merge.get_field<f32>(ihpart).get_buf()->get_access<sycl::access::mode::read>(
                    cgh);
            auto eps = eps_h.get_access<sycl::access::mode::discard_write>(cgh);
            auto h   = hnew.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<class Init_iterate_h_f32>(range_npart, [=](sycl::item<1> item) {
                u32 id_a = (u32) item.get_id(0);

                h[id_a]   = acc_hpart[id_a];
                eps[id_a] = 100;
            });
        });
    }

    template<>
    inline void sycl_init_h_iter_bufs<f64>(
        sycl::queue &queue,
        u32 or_element_cnt,

        u32 ihpart,
        shamrock::patch::PatchData &pdat_merge,
        sycl::buffer<f64> &hnew,
        sycl::buffer<f64> &omega,
        sycl::buffer<f64> &eps_h

    ) {

        sycl::range range_npart{or_element_cnt};

        queue.submit([&](sycl::handler &cgh) {
            auto acc_hpart
                = pdat_merge.get_field<f64>(ihpart).get_buf()->get_access<sycl::access::mode::read>(
                    cgh);
            auto eps = eps_h.get_access<sycl::access::mode::discard_write>(cgh);
            auto h   = hnew.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<class Init_iterate_h_f64>(range_npart, [=](sycl::item<1> item) {
                u32 id_a = (u32) item.get_id(0);

                h[id_a]   = acc_hpart[id_a];
                eps[id_a] = 100;
            });
        });
    }

#if false

    template<class flt>
    [[deprecated]]
    inline void sycl_init_h_iter_bufs(
        sycl::queue & queue,
        u32 or_element_cnt,

        u32 ihpart,
        PatchDataBuffer & pdat_buf_merge,
        sycl::buffer<flt> & hnew,
        sycl::buffer<flt> & omega,
        sycl::buffer<flt> & eps_h

        );

    template<>
    inline void sycl_init_h_iter_bufs<f32>(
        sycl::queue & queue,
        u32 or_element_cnt,

        u32 ihpart,
        PatchDataBuffer & pdat_buf_merge,
        sycl::buffer<f32> & hnew,
        sycl::buffer<f32> & omega,
        sycl::buffer<f32> & eps_h

        ){

        sycl::range range_npart{or_element_cnt};

        queue.submit([&](sycl::handler &cgh) {

            auto acc_hpart = pdat_buf_merge.fields_f32[ihpart]->get_access<sycl::access::mode::read>(cgh);
            auto eps = eps_h.get_access<sycl::access::mode::discard_write>(cgh);
            auto h    = hnew.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<class Init_iterate_h>( range_npart, [=](sycl::item<1> item) {

                u32 id_a = (u32) item.get_id(0);

                h[id_a] = acc_hpart[id_a];
                eps[id_a] = 100;

            });

        });
    }

#endif

    template<class A, class B, class C>
    class Kernel_Iterh;
    template<class A, class B, class C>
    class Kernel_Finalize_omega;

    template<class morton_prec, class Kernel>
    class IntSmoothinglengthCompute {
        public:
        template<class flt>
        static void sycl_h_iter_step(
            sycl::queue &queue,
            u32 or_element_cnt,

            u32 ihpart,
            u32 ixyz,

            flt gpart_mass,
            flt htol_up_tol,
            flt htol_up_iter,

            RadixTree<morton_prec, sycl::vec<flt, 3>> &radix_t,
            RadixTreeField<flt> &int_rad,

            shamrock::patch::PatchData &pdat_merge,
            sycl::buffer<flt> &hnew,
            sycl::buffer<flt> &omega,
            sycl::buffer<flt> &eps_h);

        template<>
        inline static void sycl_h_iter_step<f32>(
            sycl::queue &queue,
            u32 or_element_cnt,

            u32 ihpart,
            u32 ixyz,

            f32 gpart_mass,
            f32 htol_up_tol,
            f32 htol_up_iter,

            RadixTree<morton_prec, f32_3> &radix_t,
            RadixTreeField<f32> &int_rad,

            shamrock::patch::PatchData &pdat_merge,
            sycl::buffer<f32> &hnew,
            sycl::buffer<f32> &omega,
            sycl::buffer<f32> &eps_h) {

            using Rta = walker::Radix_tree_accessor<u32, f32_3>;

            sycl::range range_npart{or_element_cnt};

            queue.submit([&](sycl::handler &cgh) {
                auto h_new = hnew.get_access<sycl::access::mode::read_write>(cgh);
                auto eps   = eps_h.get_access<sycl::access::mode::read_write>(cgh);

                auto acc_hpart = pdat_merge.get_field<f32>(ihpart)
                                     .get_buf()
                                     ->get_access<sycl::access::mode::read>(cgh);
                auto r = pdat_merge.get_field<f32_3>(ixyz)
                             .get_buf()
                             ->get_access<sycl::access::mode::read>(cgh);

                Rta tree_acc(radix_t, cgh);

                auto cell_int_r
                    = int_rad.radix_tree_field_buf->template get_access<sycl::access::mode::read>(
                        cgh);

                const f32 part_mass = gpart_mass;

                const f32 h_max_tot_max_evol = htol_up_tol;
                const f32 h_max_evol_p       = htol_up_iter;
                const f32 h_max_evol_m       = 1 / htol_up_iter;

                cgh.parallel_for<Kernel_Iterh<f32, morton_prec, Kernel>>(
                    range_npart, [=](sycl::item<1> item) {
                        u32 id_a = (u32) item.get_id(0);

                        if (eps[id_a] > 1e-6) {

                            f32_3 xyz_a = r[id_a]; // could be recovered from lambda

                            f32 h_a = h_new[id_a];
                            // f32 h_a2 = h_a*h_a;

                            f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                            f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                            f32 rho_sum = 0;
                            f32 sumdWdh = 0;

                            walker::rtree_for(
                                tree_acc,
                                [&tree_acc,
                                 &xyz_a,
                                 &inter_box_a_min,
                                 &inter_box_a_max,
                                 &cell_int_r](u32 node_id) {
                                    f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                                    f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                                    float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern;

                                    using namespace walker::interaction_crit;

                                    return sph_radix_cell_crit(
                                        xyz_a,
                                        inter_box_a_min,
                                        inter_box_a_max,
                                        cur_pos_min_cell_b,
                                        cur_pos_max_cell_b,
                                        int_r_max_cell);
                                },
                                [&r, &xyz_a, &h_a, &rho_sum, &part_mass, &sumdWdh](u32 id_b) {
                                    // f32_3 dr = xyz_a - r[id_b];
                                    f32 rab = sycl::distance(xyz_a, r[id_b]);

                                    if (rab > h_a * Kernel::Rkern) {
                                        return;
                                    }

                                    // f32 rab = sycl::sqrt(rab2);

                                    rho_sum += part_mass * Kernel::W(rab, h_a);
                                    sumdWdh += part_mass * Kernel::dhW(rab, h_a);
                                },
                                [](u32 node_id) {});

                            using namespace shamrock::sph;

                            f32 rho_ha = rho_h(part_mass, h_a, Kernel::hfactd);
                            f32 new_h  = newton_iterate_new_h(rho_ha, rho_sum, sumdWdh, h_a);

                            if (new_h < h_a * h_max_evol_m)
                                new_h = h_max_evol_m * h_a;
                            if (new_h > h_a * h_max_evol_p)
                                new_h = h_max_evol_p * h_a;

                            f32 ha_0 = acc_hpart[id_a];

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
        }

        template<>
        inline static void sycl_h_iter_step<f64>(
            sycl::queue &queue,
            u32 or_element_cnt,

            u32 ihpart,
            u32 ixyz,

            f64 gpart_mass,
            f64 htol_up_tol,
            f64 htol_up_iter,

            RadixTree<morton_prec, f64_3> &radix_t,
            RadixTreeField<f64> &int_rad,

            shamrock::patch::PatchData &pdat_merge,
            sycl::buffer<f64> &hnew,
            sycl::buffer<f64> &omega,
            sycl::buffer<f64> &eps_h) {

            using Rta = walker::Radix_tree_accessor<u32, f64_3>;

            sycl::range range_npart{or_element_cnt};

            queue.submit([&](sycl::handler &cgh) {
                auto h_new = hnew.get_access<sycl::access::mode::read_write>(cgh);
                auto eps   = eps_h.get_access<sycl::access::mode::read_write>(cgh);

                auto acc_hpart = pdat_merge.get_field<f64>(ihpart)
                                     .get_buf()
                                     ->get_access<sycl::access::mode::read>(cgh);
                auto r = pdat_merge.get_field<f64_3>(ixyz)
                             .get_buf()
                             ->get_access<sycl::access::mode::read>(cgh);

                Rta tree_acc(radix_t, cgh);

                auto cell_int_r
                    = int_rad.radix_tree_field_buf->template get_access<sycl::access::mode::read>(
                        cgh);

                const f64 part_mass = gpart_mass;

                const f64 h_max_tot_max_evol = htol_up_tol;
                const f64 h_max_evol_p       = htol_up_iter;
                const f64 h_max_evol_m       = 1 / htol_up_iter;

                cgh.parallel_for<Kernel_Iterh<f64, morton_prec, Kernel>>(
                    range_npart, [=](sycl::item<1> item) {
                        u32 id_a = (u32) item.get_id(0);

                        if (eps[id_a] > 1e-6) {

                            f64_3 xyz_a = r[id_a]; // could be recovered from lambda

                            f64 h_a = h_new[id_a];
                            // f64 h_a2 = h_a*h_a;

                            f64_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                            f64_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                            f64 rho_sum = 0;
                            f64 sumdWdh = 0;

                            walker::rtree_for(
                                tree_acc,
                                [&tree_acc,
                                 &xyz_a,
                                 &inter_box_a_min,
                                 &inter_box_a_max,
                                 &cell_int_r](u32 node_id) {
                                    f64_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                                    f64_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                                    float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern;

                                    using namespace walker::interaction_crit;

                                    return sph_radix_cell_crit(
                                        xyz_a,
                                        inter_box_a_min,
                                        inter_box_a_max,
                                        cur_pos_min_cell_b,
                                        cur_pos_max_cell_b,
                                        int_r_max_cell);
                                },
                                [&r, &xyz_a, &h_a, &rho_sum, &part_mass, &sumdWdh](u32 id_b) {
                                    // f64_3 dr = xyz_a - r[id_b];
                                    f64 rab = sycl::distance(xyz_a, r[id_b]);

                                    if (rab > h_a * Kernel::Rkern) {
                                        return;
                                    }

                                    // f64 rab = sycl::sqrt(rab2);

                                    rho_sum += part_mass * Kernel::W(rab, h_a);
                                    sumdWdh += part_mass * Kernel::dhW(rab, h_a);
                                },
                                [](u32 node_id) {});

                            using namespace shamrock::sph;

                            f64 rho_ha = rho_h(part_mass, h_a, Kernel::hfactd);

                            f64 f_iter  = rho_sum - rho_ha;
                            f64 df_iter = sumdWdh + 3 * rho_ha / h_a;

                            // f64 omega_a = 1 + (h_a/(3*rho_ha))*sumdWdh;
                            // f64 new_h = h_a - (rho_ha - rho_sum)/((-3*rho_ha/h_a)*omega_a);

                            f64 new_h = h_a - f_iter / df_iter;

                            if (new_h < h_a * h_max_evol_m)
                                new_h = h_max_evol_m * h_a;
                            if (new_h > h_a * h_max_evol_p)
                                new_h = h_max_evol_p * h_a;

                            f64 ha_0 = acc_hpart[id_a];

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
        }

        template<class flt>
        static void sycl_h_iter_omega(
            sycl::queue &queue,
            u32 or_element_cnt,

            u32 ihpart,
            u32 ixyz,

            flt gpart_mass,
            flt htol_up_tol,
            flt htol_up_iter,

            RadixTree<morton_prec, sycl::vec<flt, 3>> &radix_t,
            RadixTreeField<flt> &int_rad,

            shamrock::patch::PatchData &pdat_merge,
            sycl::buffer<flt> &hnew,
            sycl::buffer<flt> &omega,
            sycl::buffer<flt> &eps_h);

        template<>
        inline static void sycl_h_iter_omega<f32>(
            sycl::queue &queue,
            u32 or_element_cnt,

            u32 ihpart,
            u32 ixyz,

            f32 gpart_mass,
            f32 htol_up_tol,
            f32 htol_up_iter,

            RadixTree<morton_prec, f32_3> &radix_t,
            RadixTreeField<f32> &int_rad,

            shamrock::patch::PatchData &pdat_merge,
            sycl::buffer<f32> &hnew,
            sycl::buffer<f32> &omega,
            sycl::buffer<f32> &eps_h) {

            using Rta = walker::Radix_tree_accessor<u32, f32_3>;

            sycl::range range_npart{or_element_cnt};

            queue.submit([&](sycl::handler &cgh) {
                auto h_new = hnew.get_access<sycl::access::mode::read_write>(cgh);
                auto omga  = omega.get_access<sycl::access::mode::discard_write>(cgh);

                auto r = pdat_merge.get_field<f32_3>(ixyz)
                             .get_buf()
                             ->get_access<sycl::access::mode::read>(cgh);

                using Rta = walker::Radix_tree_accessor<u32, f32_3>;
                Rta tree_acc(radix_t, cgh);

                auto cell_int_r
                    = int_rad.radix_tree_field_buf->template get_access<sycl::access::mode::read>(
                        cgh);

                const f32 part_mass = gpart_mass;

                const f32 h_max_tot_max_evol = htol_up_tol;
                const f32 h_max_evol_p       = htol_up_tol;
                const f32 h_max_evol_m       = 1 / htol_up_tol;

                cgh.parallel_for<Kernel_Finalize_omega<f32, morton_prec, Kernel>>(
                    range_npart, [=](sycl::item<1> item) {
                        u32 id_a = (u32) item.get_id(0);

                        f32_3 xyz_a = r[id_a]; // could be recovered from lambda

                        f32 h_a = h_new[id_a];
                        // f32 h_a2 = h_a*h_a;

                        f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                        f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                        f32 rho_sum        = 0;
                        f32 part_omega_sum = 0;

                        walker::rtree_for(
                            tree_acc,
                            [&tree_acc, &xyz_a, &inter_box_a_min, &inter_box_a_max, &cell_int_r](
                                u32 node_id) {
                                f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                                f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                                float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern;

                                using namespace walker::interaction_crit;

                                return sph_radix_cell_crit(
                                    xyz_a,
                                    inter_box_a_min,
                                    inter_box_a_max,
                                    cur_pos_min_cell_b,
                                    cur_pos_max_cell_b,
                                    int_r_max_cell);
                            },
                            [&r, &xyz_a, &h_a, &rho_sum, &part_mass, &part_omega_sum](u32 id_b) {
                                // f32_3 dr = xyz_a - r[id_b];
                                f32 rab = sycl::distance(xyz_a, r[id_b]);

                                if (rab > h_a * Kernel::Rkern)
                                    return;

                                // f32 rab = sycl::sqrt(rab2);

                                rho_sum += part_mass * Kernel::W(rab, h_a);
                                part_omega_sum += part_mass * Kernel::dhW(rab, h_a);
                            },
                            [](u32 node_id) {});

                        using namespace shamrock::sph;
                        f32 rho_ha = rho_h(part_mass, h_a, Kernel::hfactd);
                        omga[id_a] = 1 + (h_a / (3 * rho_ha)) * part_omega_sum;
                    });
            });
        }

        template<>
        inline static void sycl_h_iter_omega<f64>(
            sycl::queue &queue,
            u32 or_element_cnt,

            u32 ihpart,
            u32 ixyz,

            f64 gpart_mass,
            f64 htol_up_tol,
            f64 htol_up_iter,

            RadixTree<morton_prec, f64_3> &radix_t,
            RadixTreeField<f64> &int_rad,

            shamrock::patch::PatchData &pdat_merge,
            sycl::buffer<f64> &hnew,
            sycl::buffer<f64> &omega,
            sycl::buffer<f64> &eps_h) {

            using Rta = walker::Radix_tree_accessor<u32, f64_3>;

            sycl::range range_npart{or_element_cnt};

            queue.submit([&](sycl::handler &cgh) {
                auto h_new = hnew.get_access<sycl::access::mode::read_write>(cgh);
                auto omga  = omega.get_access<sycl::access::mode::discard_write>(cgh);

                auto r = pdat_merge.get_field<f64_3>(ixyz)
                             .get_buf()
                             ->get_access<sycl::access::mode::read>(cgh);

                using Rta = walker::Radix_tree_accessor<u32, f64_3>;
                Rta tree_acc(radix_t, cgh);

                auto cell_int_r
                    = int_rad.radix_tree_field_buf->template get_access<sycl::access::mode::read>(
                        cgh);

                const f64 part_mass = gpart_mass;

                const f64 h_max_tot_max_evol = htol_up_tol;
                const f64 h_max_evol_p       = htol_up_tol;
                const f64 h_max_evol_m       = 1 / htol_up_tol;

                cgh.parallel_for<Kernel_Finalize_omega<f64, morton_prec, Kernel>>(
                    range_npart, [=](sycl::item<1> item) {
                        u32 id_a = (u32) item.get_id(0);

                        f64_3 xyz_a = r[id_a]; // could be recovered from lambda

                        f64 h_a = h_new[id_a];
                        // f64 h_a2 = h_a*h_a;

                        f64_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                        f64_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                        f64 rho_sum        = 0;
                        f64 part_omega_sum = 0;

                        walker::rtree_for(
                            tree_acc,
                            [&tree_acc, &xyz_a, &inter_box_a_min, &inter_box_a_max, &cell_int_r](
                                u32 node_id) {
                                f64_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                                f64_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                                float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern;

                                using namespace walker::interaction_crit;

                                return sph_radix_cell_crit(
                                    xyz_a,
                                    inter_box_a_min,
                                    inter_box_a_max,
                                    cur_pos_min_cell_b,
                                    cur_pos_max_cell_b,
                                    int_r_max_cell);
                            },
                            [&r, &xyz_a, &h_a, &rho_sum, &part_mass, &part_omega_sum](u32 id_b) {
                                // f32_3 dr = xyz_a - r[id_b];
                                f64 rab = sycl::distance(xyz_a, r[id_b]);

                                if (rab > h_a * Kernel::Rkern)
                                    return;

                                // f32 rab = sycl::sqrt(rab2);

                                rho_sum += part_mass * Kernel::W(rab, h_a);
                                part_omega_sum += part_mass * Kernel::dhW(rab, h_a);
                            },
                            [](u32 node_id) {});

                        using namespace shamrock::sph;
                        f64 rho_ha = rho_h(part_mass, h_a, Kernel::hfactd);
                        omga[id_a] = 1 + (h_a / (3 * rho_ha)) * part_omega_sum;
                    });
            });
        }

#if false

        template<class flt>
        [[deprecated]]
        static void sycl_h_iter_step(
            sycl::queue & queue,
            u32 or_element_cnt,

            u32 ihpart,
            u32 ixyz,

            flt gpart_mass,
            flt htol_up_tol,
            flt htol_up_iter,

            Radix_Tree<morton_prec, sycl::vec<flt,3>> & radix_t,

            PatchDataBuffer & pdat_buf_merge,
            sycl::buffer<flt> & hnew,
            sycl::buffer<flt> & omega,
            sycl::buffer<flt> & eps_h
        );

        template<>
        inline static void sycl_h_iter_step<f32>(
            sycl::queue & queue,
            u32 or_element_cnt,

            u32 ihpart,
            u32 ixyz,

            f32 gpart_mass,
            f32 htol_up_tol,
            f32 htol_up_iter,

            Radix_Tree<morton_prec, f32_3> & radix_t,

            PatchDataBuffer & pdat_buf_merge,
            sycl::buffer<f32> & hnew,
            sycl::buffer<f32> & omega,
            sycl::buffer<f32> & eps_h
        ){

            using Rta = walker::Radix_tree_accessor<u32, f32_3>;

            sycl::range range_npart{or_element_cnt};

            queue.submit([&](sycl::handler &cgh) {

                auto h_new = hnew.get_access<sycl::access::mode::read_write>(cgh);
                auto eps = eps_h.get_access<sycl::access::mode::read_write>(cgh);

                auto acc_hpart = pdat_buf_merge.fields_f32.at(ihpart)->get_access<sycl::access::mode::read>(cgh);
                auto r = pdat_buf_merge.fields_f32_3[ixyz]->get_access<sycl::access::mode::read>(cgh);


                Rta tree_acc(radix_t, cgh);



                auto cell_int_r = radix_t.buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

                const f32 part_mass = gpart_mass;

                const f32 h_max_tot_max_evol = htol_up_tol;
                const f32 h_max_evol_p = htol_up_iter;
                const f32 h_max_evol_m = 1/htol_up_iter;

                cgh.parallel_for<Kernel_Iterh<f32, morton_prec, Kernel>>(range_npart, [=](sycl::item<1> item) {
                    u32 id_a = (u32)item.get_id(0);


                    if(eps[id_a] > 1e-6){

                        f32_3 xyz_a = r[id_a]; // could be recovered from lambda

                        f32 h_a = h_new[id_a];
                        //f32 h_a2 = h_a*h_a;

                        f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                        f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                        f32 rho_sum = 0;
                        f32 sumdWdh = 0;

                        walker::rtree_for(
                            tree_acc,
                            [&tree_acc,&xyz_a,&inter_box_a_min,&inter_box_a_max,&cell_int_r](u32 node_id) {
                                f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                                f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                                float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern;

                                using namespace walker::interaction_crit;

                                return sph_radix_cell_crit(xyz_a, inter_box_a_min, inter_box_a_max, cur_pos_min_cell_b,
                                                            cur_pos_max_cell_b, int_r_max_cell);
                            },
                            [&r,&xyz_a,&h_a,&rho_sum,&part_mass,&sumdWdh](u32 id_b) {
                                //f32_3 dr = xyz_a - r[id_b];
                                f32 rab = sycl::distance( xyz_a , r[id_b]);

                                if(rab > h_a*Kernel::Rkern) return;

                                //f32 rab = sycl::sqrt(rab2);

                                rho_sum += part_mass*Kernel::W(rab,h_a);
                                sumdWdh += part_mass*Kernel::dhW(rab,h_a);

                            },
                            [](u32 node_id) {});



                        f32 rho_ha = rho_h(part_mass, h_a);

                        f32 f_iter = rho_sum - rho_ha;
                        f32 df_iter = sumdWdh + 3*rho_ha/h_a;

                        //f32 omega_a = 1 + (h_a/(3*rho_ha))*sumdWdh;
                        //f32 new_h = h_a - (rho_ha - rho_sum)/((-3*rho_ha/h_a)*omega_a);

                        f32 new_h = h_a - f_iter/df_iter;


                        if(new_h < h_a*h_max_evol_m) new_h = h_max_evol_m*h_a;
                        if(new_h > h_a*h_max_evol_p) new_h = h_max_evol_p*h_a;


                        f32 ha_0 = acc_hpart[id_a];


                        if (new_h < ha_0*h_max_tot_max_evol) {
                            h_new[id_a] = new_h;
                            eps[id_a] = sycl::fabs(new_h - h_a)/ha_0;
                        }else{
                            h_new[id_a] = ha_0*h_max_tot_max_evol;
                            eps[id_a] = -1;
                        }
                    }

                });

            });
        }












        template<class flt>
        [[deprecated]]
        static void _sycl_h_iter_omega(
            sycl::queue & queue,
            u32 or_element_cnt,

            u32 ihpart,
            u32 ixyz,

            flt gpart_mass,
            flt htol_up_tol,
            flt htol_up_iter,

            Radix_Tree<morton_prec, sycl::vec<flt,3>> & radix_t,

            PatchDataBuffer & pdat_buf_merge,
            sycl::buffer<flt> & hnew,
            sycl::buffer<flt> & omega,
            sycl::buffer<flt> & eps_h
        );

        template<>
        inline static void _sycl_h_iter_omega<f32>(
            sycl::queue & queue,
            u32 or_element_cnt,

            u32 ihpart,
            u32 ixyz,

            f32 gpart_mass,
            f32 htol_up_tol,
            f32 htol_up_iter,

            Radix_Tree<morton_prec, f32_3> & radix_t,

            PatchDataBuffer & pdat_buf_merge,
            sycl::buffer<f32> & hnew,
            sycl::buffer<f32> & omega,
            sycl::buffer<f32> & eps_h
        ){

            using Rta = walker::Radix_tree_accessor<u32, f32_3>;

            sycl::range range_npart{or_element_cnt};


            queue.submit([&](sycl::handler &cgh) {

                auto h_new = hnew.get_access<sycl::access::mode::read_write>(cgh);
                auto omga = omega.get_access<sycl::access::mode::discard_write>(cgh);

                auto r = pdat_buf_merge.fields_f32_3.at(ixyz)->get_access<sycl::access::mode::read>(cgh);

                using Rta = walker::Radix_tree_accessor<u32, f32_3>;
                Rta tree_acc(radix_t, cgh);



                auto cell_int_r =radix_t.buf_cell_interact_rad->template get_access<sycl::access::mode::read>(cgh);

                const f32 part_mass = gpart_mass;

                const f32 h_max_tot_max_evol = htol_up_tol;
                const f32 h_max_evol_p = htol_up_tol;
                const f32 h_max_evol_m = 1/htol_up_tol;

                cgh.parallel_for<Kernel_Finalize_omega<f32, morton_prec, Kernel>>(range_npart, [=](sycl::item<1> item) {
                    u32 id_a = (u32)item.get_id(0);

                    f32_3 xyz_a = r[id_a]; // could be recovered from lambda

                    f32 h_a = h_new[id_a];
                    //f32 h_a2 = h_a*h_a;

                    f32_3 inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                    f32_3 inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                    f32 rho_sum = 0;
                    f32 part_omega_sum = 0;

                    walker::rtree_for(
                        tree_acc,
                        [&tree_acc,&xyz_a,&inter_box_a_min,&inter_box_a_max,&cell_int_r](u32 node_id) {
                            f32_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                            f32_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];
                            float int_r_max_cell     = cell_int_r[node_id] * Kernel::Rkern;

                            using namespace walker::interaction_crit;

                            return sph_radix_cell_crit(xyz_a, inter_box_a_min, inter_box_a_max, cur_pos_min_cell_b,
                                                        cur_pos_max_cell_b, int_r_max_cell);
                        },
                        [&r,&xyz_a,&h_a,&rho_sum,&part_mass,&part_omega_sum](u32 id_b) {
                            //f32_3 dr = xyz_a - r[id_b];
                            f32 rab = sycl::distance( xyz_a , r[id_b]);

                            if(rab > h_a*Kernel::Rkern) return;

                            //f32 rab = sycl::sqrt(rab2);

                            rho_sum += part_mass*Kernel::W(rab,h_a);
                            part_omega_sum += part_mass * Kernel::dhW(rab,h_a);

                        },
                        [](u32 node_id) {});



                    f32 rho_ha = rho_h(part_mass, h_a);
                    omga[id_a] = 1 + (h_a/(3*rho_ha))*part_omega_sum;


                });

            });
        }

#endif
    };

} // namespace impl

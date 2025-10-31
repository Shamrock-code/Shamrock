// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file fmmTests.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shammath/symtensor_collections.hpp"
#include "shamphys/fmm/GreenFuncGravCartesian.hpp"
#include "shamphys/fmm/grav_moments.hpp"
#include "shamphys/fmm/offset_multipole.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/RadixTree.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

template<class T, u32 order>
class FMM_prec_eval {
    public:
    static T eval_prec_fmm_pot(
        sycl::vec<T, 3> xi, sycl::vec<T, 3> xj, sycl::vec<T, 3> sa, sycl::vec<T, 3> sb) {

        using namespace shammath;

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = SymTensorCollection<f64, 0, order>::num_component;
        sycl::buffer<f64> buf_multipoles{number_elem_multip};

        // compute multipoles
        {
            sycl::host_accessor pos{buf_pos, sycl::read_only};
            sycl::host_accessor multipoles{buf_multipoles, sycl::write_only, sycl::no_init};

            for (u32 j = 0; j < pos_table.size(); j++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = SymTensorCollection<f64, 0, order>::from_vec(bj);

                B_n *= m_j;

                B_n.store(multipoles, 0);
            }
        }

        // compute fmm
        {

            sycl::host_accessor multipoles{buf_multipoles, sycl::read_only};

            f64_3 r_fmm = sb - sa;

            f64_3 a_i = xi - sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = SymTensorCollection<f64, 0, order>::load(multipoles, 0);

            auto D_n = shamphys::GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = shamphys::get_M_mat(D_n, Q_n);

            f64 phi_val = M_k.t0 * a_k.t0;
            if constexpr (order >= 1) {
                phi_val += M_k.t1 * a_k.t1;
            }
            if constexpr (order >= 2) {
                phi_val += M_k.t2 * a_k.t2;
            }
            if constexpr (order >= 3) {
                phi_val += M_k.t3 * a_k.t3;
            }
            if constexpr (order >= 4) {
                phi_val += M_k.t4 * a_k.t4;
            }
            if constexpr (order >= 5) {
                phi_val += M_k.t5 * a_k.t5;
            }

            // printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);

            f64_3 real_r = xi - xj;

            f64 r_sq   = sycl::sqrt(sycl::dot(real_r, real_r));
            f64 phi_th = 1 / r_sq;

            // printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val - phi_th) / phi_th;
        }
    }
    static T eval_prec_fmm_force(
        sycl::vec<T, 3> xi, sycl::vec<T, 3> xj, sycl::vec<T, 3> sa, sycl::vec<T, 3> sb) {

        using namespace shammath;
        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        using moment_types = SymTensorCollection<f64, 0, order - 1>;

        constexpr u32 number_elem_multip = moment_types::num_component;
        sycl::buffer<f64> buf_multipoles{number_elem_multip};

        // compute multipoles
        {
            sycl::host_accessor pos{buf_pos, sycl::read_only};
            sycl::host_accessor multipoles{buf_multipoles, sycl::write_only, sycl::no_init};

            for (u32 j = 0; j < pos_table.size(); j++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = moment_types::from_vec(bj);

                B_n *= m_j;

                B_n.store(multipoles, 0);
            }
        }

        // compute fmm
        {

            sycl::host_accessor multipoles{buf_multipoles, sycl::read_only};

            f64_3 r_fmm = sb - sa;

            f64_3 a_i = xi - sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = moment_types::load(multipoles, 0);

            auto D_n = shamphys::GreenFuncGravCartesian<f64, 1, order>::get_der_tensors(r_fmm);

            auto dM_k = shamphys::get_dM_mat(D_n, Q_n);

            auto tensor_to_sycl = [](SymTensor3d_1<T> a) {
                return sycl::vec<T, 3>{a.v_0, a.v_1, a.v_2};
            };

            auto force_val = tensor_to_sycl(dM_k.t1 * a_k.t0);
            if constexpr (order >= 2) {
                force_val += tensor_to_sycl(dM_k.t2 * a_k.t1);
            }
            if constexpr (order >= 3) {
                force_val += tensor_to_sycl(dM_k.t3 * a_k.t2);
            }
            if constexpr (order >= 4) {
                force_val += tensor_to_sycl(dM_k.t4 * a_k.t3);
            }
            if constexpr (order >= 5) {
                force_val += tensor_to_sycl(dM_k.t5 * a_k.t4);
            }

            // printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);

            f64_3 real_r = xi - xj;

            f64 r_n    = sycl::sqrt(sycl::dot(real_r, real_r));
            f64_3 f_th = real_r / (r_n * r_n * r_n);

            f64_3 delta = force_val - f_th;

            return sycl::distance(force_val, f_th) / sycl::length(f_th);
        }
    }
};

TestStart(ValidationTest, "models/generic/fmm/precision", fmm_prec, 1) {

    std::mt19937 eng(0x1111);

    std::uniform_real_distribution<f64> distf64(-1, 1);

    f64 avg_spa = 1e-2;
    std::uniform_real_distribution<f64> distf64_red(-avg_spa, avg_spa);

    struct Entry {
        f64 angle;
        f64 result_pot_5;
        f64 result_pot_4;
        f64 result_pot_3;
        f64 result_pot_2;
        f64 result_pot_1;
        f64 result_pot_0;
        f64 result_force_5;
        f64 result_force_4;
        f64 result_force_3;
        f64 result_force_2;
        f64 result_force_1;
    };
    std::vector<Entry> vec_result;

    for (u32 i = 0; i < 1e4; i++) {

        f64_3 s_a = f64_3{distf64(eng), distf64(eng), distf64(eng)};
        f64_3 s_b = f64_3{distf64(eng), distf64(eng), distf64(eng)};

        f64_3 x_i = s_a + f64_3{distf64_red(eng), distf64_red(eng), distf64_red(eng)};
        f64_3 x_j = s_b + f64_3{distf64_red(eng), distf64_red(eng), distf64_red(eng)};

        auto dist_func = [](f64_3 a, f64_3 b) {
            f64_3 d = a - b;

            f64_3 dabs = sycl::fabs(d);

            return sycl::max(sycl::max(dabs.x(), dabs.y()), dabs.z());
        };

        f64 angle = 2 * (dist_func(x_i, s_a) + dist_func(x_j, s_b)) / dist_func(s_a, s_b);

        vec_result.push_back(
            Entry{
                angle,
                FMM_prec_eval<f64, 5>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b),
                FMM_prec_eval<f64, 4>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b),
                FMM_prec_eval<f64, 3>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b),
                FMM_prec_eval<f64, 2>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b),
                FMM_prec_eval<f64, 1>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b),
                FMM_prec_eval<f64, 0>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b),
                FMM_prec_eval<f64, 5>::eval_prec_fmm_force(x_i, x_j, s_a, s_b),
                FMM_prec_eval<f64, 4>::eval_prec_fmm_force(x_i, x_j, s_a, s_b),
                FMM_prec_eval<f64, 3>::eval_prec_fmm_force(x_i, x_j, s_a, s_b),
                FMM_prec_eval<f64, 2>::eval_prec_fmm_force(x_i, x_j, s_a, s_b),
                FMM_prec_eval<f64, 1>::eval_prec_fmm_force(x_i, x_j, s_a, s_b)});

        if (i % 10000 == 0) {
            shamlog_debug_ln("Tests", "i =", i, "\\", 100000);
        }
    }

    std::sort(vec_result.begin(), vec_result.end(), [](const auto &i, const auto &j) {
        return i.angle < j.angle;
    });

    std::vector<f64> vec_angle;
    std::vector<f64> vec_result_pot_5;
    std::vector<f64> vec_result_pot_4;
    std::vector<f64> vec_result_pot_3;
    std::vector<f64> vec_result_pot_2;
    std::vector<f64> vec_result_pot_1;
    std::vector<f64> vec_result_pot_0;
    std::vector<f64> vec_result_force_5;
    std::vector<f64> vec_result_force_4;
    std::vector<f64> vec_result_force_3;
    std::vector<f64> vec_result_force_2;
    std::vector<f64> vec_result_force_1;

    for (auto &f : vec_result) {
        vec_angle.push_back(f.angle);
        vec_result_pot_5.push_back(f.result_pot_5);
        vec_result_pot_4.push_back(f.result_pot_4);
        vec_result_pot_3.push_back(f.result_pot_3);
        vec_result_pot_2.push_back(f.result_pot_2);
        vec_result_pot_1.push_back(f.result_pot_1);
        vec_result_pot_0.push_back(f.result_pot_0);
        vec_result_force_5.push_back(f.result_force_5);
        vec_result_force_4.push_back(f.result_force_4);
        vec_result_force_3.push_back(f.result_force_3);
        vec_result_force_2.push_back(f.result_force_2);
        vec_result_force_1.push_back(f.result_force_1);
    }

    PyScriptHandle hdnl{};

    hdnl.data()["angle"]              = vec_angle;
    hdnl.data()["vec_result_pot_5"]   = vec_result_pot_5;
    hdnl.data()["vec_result_pot_4"]   = vec_result_pot_4;
    hdnl.data()["vec_result_pot_3"]   = vec_result_pot_3;
    hdnl.data()["vec_result_pot_2"]   = vec_result_pot_2;
    hdnl.data()["vec_result_pot_1"]   = vec_result_pot_1;
    hdnl.data()["vec_result_pot_0"]   = vec_result_pot_0;
    hdnl.data()["vec_result_force_5"] = vec_result_force_5;
    hdnl.data()["vec_result_force_4"] = vec_result_force_4;
    hdnl.data()["vec_result_force_3"] = vec_result_force_3;
    hdnl.data()["vec_result_force_2"] = vec_result_force_2;
    hdnl.data()["vec_result_force_1"] = vec_result_force_1;

    hdnl.exec(R"(
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('custom_style.mplstyle')

        fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

        def plot_curve(ax,X,Y,lab):
            global np;

            print("plotting :",lab)

            ratio = 40
            cnt = len(X)//ratio
            X_m = X.reshape((cnt,ratio))
            X_m = np.max(X_m,axis=1)
            Y_m = Y.reshape((cnt,ratio))
            Y_m = np.max(Y_m,axis=1)
            ax.plot(X_m,Y_m, label = lab)

        plot_curve(axs[0],np.array(angle),np.abs(vec_result_pot_5  ), "fmm order = 5")
        plot_curve(axs[0],np.array(angle),np.abs(vec_result_pot_4  ), "fmm order = 4")
        plot_curve(axs[0],np.array(angle),np.abs(vec_result_pot_3  ), "fmm order = 3")
        plot_curve(axs[0],np.array(angle),np.abs(vec_result_pot_2  ), "fmm order = 2")
        plot_curve(axs[0],np.array(angle),np.abs(vec_result_pot_1  ), "fmm order = 1")
        plot_curve(axs[0],np.array(angle),np.abs(vec_result_pot_0  ), "fmm order = 0")
        plot_curve(axs[1],np.array(angle),np.abs(vec_result_force_5), "fmm order = 5")
        plot_curve(axs[1],np.array(angle),np.abs(vec_result_force_4), "fmm order = 4")
        plot_curve(axs[1],np.array(angle),np.abs(vec_result_force_3), "fmm order = 3")
        plot_curve(axs[1],np.array(angle),np.abs(vec_result_force_2), "fmm order = 2")
        plot_curve(axs[1],np.array(angle),np.abs(vec_result_force_1), "fmm order = 1")

        axs[0].set_title('Gravitational potential ($\Phi$)')
        axs[1].set_title('Gravitational force ($\mathbf{f}$)')

        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')

        axs[0].set_ylim(1e-16,1)
        axs[1].set_ylim(1e-16,1)


        axs[1].set_xscale('log')
        axs[1].set_yscale('log')

        axs[0].set_xlabel(r"$\theta$")
        axs[1].set_xlabel(r"$\theta$")

        axs[0].set_ylabel(r"$\vert \Phi_{\rm fmm} - \Phi_{\rm th} \vert /\vert \Phi_{\rm th}\vert$")
        axs[1].set_ylabel(r"$\vert \mathbf{f}_{\rm fmm} - \mathbf{f}_{\rm th} \vert /\vert \mathbf{f}_{\rm th}\vert$")

        axs[0].legend()
        axs[0].grid()
        axs[1].legend()
        axs[1].grid()

        plt.tight_layout()

        plt.savefig("tests/figures/fmm_precision.pdf")

    )");

    TEX_REPORT(R"==(

        \begin{figure}[ht!]
        \center
        \includegraphics[width=0.95\linewidth]{figures/fmm_precision.pdf}
        \caption{FMM precision}
        \end{figure}

    )==")
}

TestStart(Unittest, "models/generic/fmm/multipole_moment_offset", multipole_moment_offset, 1) {
    using namespace shammath;
    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<f64> distf64(-1, 1);

    f64_3 s_bp = f64_3{distf64(eng), distf64(eng), distf64(eng)};
    f64_3 s_b  = f64_3{distf64(eng), distf64(eng), distf64(eng)};

    // f64_3 s_bp = f64_3{1, 0, 0};
    // f64_3 s_b  = f64_3{0, 0, 0};

    auto B_nB  = SymTensorCollection<f64, 0, 5>::zeros();
    auto B_nBp = SymTensorCollection<f64, 0, 5>::zeros();

    for (u32 i = 0; i < 100; i++) {

        f64_3 x_1 = f64_3{distf64(eng), distf64(eng), distf64(eng)};

        f64_3 bj  = x_1 - s_b;
        f64_3 bpj = x_1 - s_bp;

        auto tB_nB = SymTensorCollection<f64, 0, 5>::from_vec(bj);

        auto tB_nBp = SymTensorCollection<f64, 0, 5>::from_vec(bpj);

        B_nB.t0 += tB_nB.t0;
        B_nB.t1 += tB_nB.t1;
        B_nB.t2 += tB_nB.t2;
        B_nB.t3 += tB_nB.t3;
        B_nB.t4 += tB_nB.t4;
        B_nB.t5 += tB_nB.t5;

        B_nBp.t0 += tB_nBp.t0;
        B_nBp.t1 += tB_nBp.t1;
        B_nBp.t2 += tB_nBp.t2;
        B_nBp.t3 += tB_nBp.t3;
        B_nBp.t4 += tB_nBp.t4;
        B_nBp.t5 += tB_nBp.t5;
    }

    auto d = s_b - s_bp;

    SymTensorCollection<f64, 0, 5> d_ = SymTensorCollection<f64, 0, 5>::from_vec(d);

    auto B_nb_offseted = shamphys::offset_multipole_delta(B_nB, d);

    auto diff_0 = B_nBp.t0 - B_nb_offseted.t0;
    auto diff_1 = B_nBp.t1 - B_nb_offseted.t1;
    auto diff_2 = B_nBp.t2 - B_nb_offseted.t2;
    auto diff_3 = B_nBp.t3 - B_nb_offseted.t3;
    auto diff_4 = B_nBp.t4 - B_nb_offseted.t4;
    auto diff_5 = B_nBp.t5 - B_nb_offseted.t5;

    f64 diff = (diff_0 * diff_0) + (diff_1 * diff_1) + (diff_2 * diff_2) + (diff_3 * diff_3)
               + (diff_4 * diff_4) + (diff_5 * diff_5);

    REQUIRE_FLOAT_EQUAL_NAMED("multipole offset valid", diff, 0, 1e-13);

    printf("order 0 %f\n", B_nB.t0);

    printf("diff %e\n", diff);

    printf("dvec   %f %f %f\n", d.x(), d.y(), d.z());

    printf("np     %f %f %f\n", B_nBp.t1.v_0, B_nBp.t1.v_1, B_nBp.t1.v_2);
    printf("B      %f %f %f\n", B_nB.t1.v_0, B_nB.t1.v_1, B_nB.t1.v_2);
    printf("offset %f %f %f\n", B_nb_offseted.t1.v_0, B_nb_offseted.t1.v_1, B_nb_offseted.t1.v_2);

    printf("------ order 2 ---------\n");
    printf(
        "B      %f %f %f %f %f %f\n",
        B_nB.t2.v_00,
        B_nB.t2.v_01,
        B_nB.t2.v_02,
        B_nB.t2.v_11,
        B_nB.t2.v_12,
        B_nB.t2.v_22);
    printf(
        "np     %f %f %f %f %f %f\n",
        B_nBp.t2.v_00,
        B_nBp.t2.v_01,
        B_nBp.t2.v_02,
        B_nBp.t2.v_11,
        B_nBp.t2.v_12,
        B_nBp.t2.v_22);
    printf(
        "offset %f %f %f %f %f %f\n",
        B_nb_offseted.t2.v_00,
        B_nb_offseted.t2.v_01,
        B_nb_offseted.t2.v_02,
        B_nb_offseted.t2.v_11,
        B_nb_offseted.t2.v_12,
        B_nb_offseted.t2.v_22);
    printf(
        "d      %f %f %f %f %f %f\n",
        d_.t2.v_00,
        d_.t2.v_01,
        d_.t2.v_02,
        d_.t2.v_11,
        d_.t2.v_12,
        d_.t2.v_22);

    auto DG = shamphys::GreenFuncGravCartesian<f64, 0, 5>::get_der_tensors(s_b - f64_3{5, 1, 2});

    printf(
        "%e %e %e %e %e %e\n",
        (diff_0 * diff_0),
        (diff_1 * diff_1),
        (diff_2 * diff_2),
        (diff_3 * diff_3),
        (diff_4 * diff_4),
        (diff_5 * diff_5));
}

template<class flt, u32 fmm_order>
void are_u32_u64_morton_same(u32 npart, u32 reduc_level) {
    using vec = sycl::vec<flt, 3>;
}

template<class flt, class morton_mode, u32 fmm_order>
struct Result_nompi_fmm_testing {
    using vec = sycl::vec<flt, 3>;

    f64 time;
    f64 prec;

    f64 leaf_cnt;
    f64 reject_cnt;

    std::unique_ptr<sycl::buffer<vec>> &pos_buf;
    std::unique_ptr<sycl::buffer<vec>> force_buf;

    RadixTree<morton_mode, vec> rtree;

    std::unique_ptr<sycl::buffer<flt>> grav_multipoles;
};

template<class flt, class morton_mode, u32 fmm_order>
Result_nompi_fmm_testing<flt, morton_mode, fmm_order> nompi_fmm_testing(
    std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> &pos_part, u32 reduc_level, flt open_crit) {
    using namespace shammath;
    using vec = sycl::vec<flt, 3>;

    u32 npart = pos_part->size();

    const flt pre_open_crit_sq = open_crit * open_crit;

    auto buf_force = std::make_unique<sycl::buffer<vec>>(npart);
    {
        sycl::host_accessor<vec> force{*buf_force};

        for (u32 i = 0; i < npart; i++) {
            force[i] = vec{0, 0, 0};
        }
    }

    shamsys::instance::get_compute_queue().wait();
    shambase::Timer timer;
    timer.start();

    RadixTree<morton_mode, vec> rtree = RadixTree<morton_mode, vec>(
        shamsys::instance::get_compute_queue(),
        {vec{-1, -1, -1}, vec{1, 1, 1}},
        pos_part,
        npart,
        reduc_level);

    //{
    //    auto acc_lid = sycl::host_accessor {*rtree.buf_lchild_id};
    //    auto acc_rid = sycl::host_accessor {*rtree.buf_rchild_id};
    //    auto acc_lflag = sycl::host_accessor {*rtree.buf_lchild_flag};
    //    auto acc_rflag = sycl::host_accessor {*rtree.buf_rchild_flag};
    //    for(u32 i = 0 ; i < rtree.tree_internal_count ; i++){
    //        logger::raw("(",acc_lid[i],acc_rid[i],u32{acc_lflag[i]},u32{acc_rflag[i]},") ");
    //    }
    //}

    rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
    rtree.convert_bounding_box(shamsys::instance::get_compute_queue());

    u32 num_component_multipoles_fmm
        = (rtree.tree_struct.internal_cell_count + rtree.tree_reduced_morton_codes.tree_leaf_count)
          * SymTensorCollection<flt, 0, fmm_order>::num_component;
    shamlog_debug_ln(
        "RTreeFMM", "allocating", num_component_multipoles_fmm, "component for multipoles");
    auto grav_multipoles = std::make_unique<sycl::buffer<flt>>(num_component_multipoles_fmm);

    shamlog_debug_ln(
        "RTreeFMM",
        "computing leaf moments (",
        rtree.tree_reduced_morton_codes.tree_leaf_count,
        ")");
    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
        u32 offset_leaf = rtree.tree_struct.internal_cell_count;

        auto xyz               = sycl::accessor{*pos_part, cgh, sycl::read_only};
        auto cell_particle_ids = sycl::accessor{
            *rtree.tree_reduced_morton_codes.buf_reduc_index_map, cgh, sycl::read_only};
        auto particle_index_map
            = sycl::accessor{*rtree.tree_morton_codes.buf_particle_index_map, cgh, sycl::read_only};
        auto cell_max
            = sycl::accessor{*rtree.tree_cell_ranges.buf_pos_max_cell_flt, cgh, sycl::read_only};
        auto cell_min
            = sycl::accessor{*rtree.tree_cell_ranges.buf_pos_min_cell_flt, cgh, sycl::read_only};
        auto multipoles = sycl::accessor{*grav_multipoles, cgh, sycl::write_only, sycl::no_init};

        sycl::range<1> range_leaf_cell{rtree.tree_reduced_morton_codes.tree_leaf_count};

        cgh.parallel_for(range_leaf_cell, [=](sycl::item<1> item) {
            u32 gid = (u32) item.get_id(0);

            u32 min_ids = cell_particle_ids[gid];
            u32 max_ids = cell_particle_ids[gid + 1];

            vec cell_pmax = cell_max[offset_leaf + gid];
            vec cell_pmin = cell_min[offset_leaf + gid];

            vec s_b = (cell_pmax + cell_pmin) / 2;

            auto B_n = SymTensorCollection<flt, 0, fmm_order>::zeros();

            for (u32 id_s = min_ids; id_s < max_ids; id_s++) {
                u32 idx_j = particle_index_map[id_s];
                vec bj    = xyz[idx_j] - s_b;

                auto tB_n = SymTensorCollection<flt, 0, fmm_order>::from_vec(bj);

                constexpr flt m_j = 1;

                tB_n *= m_j;
                B_n += tB_n;
            }

            B_n.store(
                multipoles,
                (gid + offset_leaf) * SymTensorCollection<flt, 0, fmm_order>::num_component);
        });
    });

    auto buf_is_computed = std::make_unique<sycl::buffer<u8>>(
        (rtree.tree_struct.internal_cell_count + rtree.tree_reduced_morton_codes.tree_leaf_count));

    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
        auto is_computed = sycl::accessor{*buf_is_computed, cgh, sycl::write_only, sycl::no_init};
        sycl::range<1> range_internal_count{
            rtree.tree_struct.internal_cell_count
            + rtree.tree_reduced_morton_codes.tree_leaf_count};

        u32 int_cnt = rtree.tree_struct.internal_cell_count;

        cgh.parallel_for(range_internal_count, [=](sycl::item<1> item) {
            is_computed[item] = item.get_linear_id() >= int_cnt;
        });
    });

    shamlog_debug_ln("RTreeFMM", "iterating moment cascade");
    for (u32 iter = 0; iter < rtree.tree_depth; iter++) {

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            u32 leaf_offset = rtree.tree_struct.internal_cell_count;

            auto cell_max = sycl::accessor{
                *rtree.tree_cell_ranges.buf_pos_max_cell_flt, cgh, sycl::read_only};
            auto cell_min = sycl::accessor{
                *rtree.tree_cell_ranges.buf_pos_min_cell_flt, cgh, sycl::read_only};
            auto multipoles  = sycl::accessor{*grav_multipoles, cgh, sycl::read_write};
            auto is_computed = sycl::accessor{*buf_is_computed, cgh, sycl::read_write};

            sycl::range<1> range_internal_count{rtree.tree_struct.internal_cell_count};

            auto rchild_id = sycl::accessor{*rtree.tree_struct.buf_rchild_id, cgh, sycl::read_only};
            auto lchild_id = sycl::accessor{*rtree.tree_struct.buf_lchild_id, cgh, sycl::read_only};
            auto rchild_flag
                = sycl::accessor{*rtree.tree_struct.buf_rchild_flag, cgh, sycl::read_only};
            auto lchild_flag
                = sycl::accessor{*rtree.tree_struct.buf_lchild_flag, cgh, sycl::read_only};

            cgh.parallel_for(range_internal_count, [=](sycl::item<1> item) {
                u32 cid = item.get_linear_id();

                u32 lid = lchild_id[cid] + leaf_offset * lchild_flag[cid];
                u32 rid = rchild_id[cid] + leaf_offset * rchild_flag[cid];

                bool should_compute = (!is_computed[cid]) && (is_computed[lid] && is_computed[rid]);

                if (should_compute) {

                    vec cell_pmax = cell_max[cid];
                    vec cell_pmin = cell_min[cid];

                    vec sbp = (cell_pmax + cell_pmin) / 2;

                    auto B_n = SymTensorCollection<flt, 0, fmm_order>::zeros();

                    auto add_multipole_offset = [&](u32 s_cid) {
                        vec s_cell_pmax = cell_max[s_cid];
                        vec s_cell_pmin = cell_min[s_cid];

                        vec sb = (s_cell_pmax + s_cell_pmin) / 2;

                        auto d = sb - sbp;

                        auto B_ns = SymTensorCollection<flt, 0, fmm_order>::load(
                            multipoles,
                            s_cid * SymTensorCollection<flt, 0, fmm_order>::num_component);

                        auto B_ns_offseted = shamphys::offset_multipole_delta(B_ns, d);

                        B_n += B_ns_offseted;
                    };

                    add_multipole_offset(lid);
                    add_multipole_offset(rid);

                    is_computed[cid] = true;
                    B_n.store(
                        multipoles, cid * SymTensorCollection<flt, 0, fmm_order>::num_component);
                }
            });
        });
    }

    shamlog_debug_ln("RTreeFMM", "computing cell infos");
    std::unique_ptr<sycl::buffer<vec>> cell_centers = std::make_unique<sycl::buffer<vec>>(
        rtree.tree_struct.internal_cell_count + rtree.tree_reduced_morton_codes.tree_leaf_count);
    std::unique_ptr<sycl::buffer<flt>> cell_length = std::make_unique<sycl::buffer<flt>>(
        rtree.tree_struct.internal_cell_count + rtree.tree_reduced_morton_codes.tree_leaf_count);

    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
        sycl::range<1> range_tree = sycl::range<1>{
            rtree.tree_reduced_morton_codes.tree_leaf_count
            + rtree.tree_struct.internal_cell_count};

        auto pos_min_cell
            = sycl::accessor{*rtree.tree_cell_ranges.buf_pos_min_cell_flt, cgh, sycl::read_only};
        auto pos_max_cell
            = sycl::accessor{*rtree.tree_cell_ranges.buf_pos_max_cell_flt, cgh, sycl::read_only};

        auto c_centers = sycl::accessor{*cell_centers, cgh, sycl::write_only, sycl::no_init};
        auto c_length  = sycl::accessor{*cell_length, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
            vec cur_pos_min_cell_a = pos_min_cell[item];
            vec cur_pos_max_cell_a = pos_max_cell[item];

            vec sa = (cur_pos_min_cell_a + cur_pos_max_cell_a) / 2;

            vec dc_a = (cur_pos_max_cell_a - cur_pos_min_cell_a);

            flt l_cell_a = sycl::max(sycl::max(dc_a.x(), dc_a.y()), dc_a.z());

            c_centers[item] = sa;
            c_length[item]  = l_cell_a;
        });
    });

#if false
    rtree.for_each_leaf(shamsys::instance::get_compute_queue(), [&](sycl::handler &cgh,auto && par_for){

        //user accessors
        auto c_centers = sycl::accessor{*cell_centers,cgh,sycl::read_only};
        auto c_length = sycl::accessor{*cell_length,cgh,sycl::read_only};

        auto xyz = sycl::accessor {*pos_part, cgh,sycl::read_only};
        auto fxyz = sycl::accessor {*buf_force, cgh,sycl::read_write};

        auto multipoles = sycl::accessor {*grav_multipoles, cgh,sycl::read_only};


        auto pos_min_cell = rtree.buf_pos_min_cell_flt-> template get_access<sycl::access::mode::read>(cgh);
        auto pos_max_cell = rtree.buf_pos_max_cell_flt-> template get_access<sycl::access::mode::read>(cgh);


        par_for([=](const u32 & id_cell_a, auto && walk_loop,auto && obj_iterator){


            //user funcs
            vec cur_pos_min_cell_a = pos_min_cell[id_cell_a];
            vec cur_pos_max_cell_a = pos_max_cell[id_cell_a];

            vec sa = c_centers[id_cell_a];
            flt l_cell_a = c_length[id_cell_a];

            auto dM_k = SymTensorCollection<flt, 1, fmm_order+1>::zeros();

            constexpr flt open_crit = 0.1;
            constexpr flt open_crit_sq = open_crit*open_crit;


            walk_loop(id_cell_a,
                [&](const u32 & id_cell_b, auto && walk_logic){

                    //user defs for the cell pair a-b (current_node_id) return interact cd
                    vec cur_pos_min_cell_b = pos_min_cell[id_cell_b];
                    vec cur_pos_max_cell_b = pos_max_cell[id_cell_b];

                    vec sb = c_centers[id_cell_b];
                    vec r_fmm = sb-sa;
                    flt l_cell_b = c_length[id_cell_b];

                    flt opening_angle_sq = (l_cell_a + l_cell_b)*(l_cell_a + l_cell_b)/sycl::dot(r_fmm,r_fmm);

                    using namespace walker::interaction_crit;

                    const bool cells_interact = sph_cell_cell_crit(
                        cur_pos_min_cell_a,
                        cur_pos_max_cell_a,
                        cur_pos_min_cell_b,
                        cur_pos_max_cell_b,
                        0,
                        0) || (opening_angle_sq > open_crit_sq);

                    walk_logic(cells_interact,
                        [&](){
                            //func_leaf_found

                            vec sb = c_centers[id_cell_b];
                            vec r_fmm = sb-sa;
                            flt l_cell_b = c_length[id_cell_b];

                            flt opening_angle_sq = (l_cell_a + l_cell_b)*(l_cell_a + l_cell_b)/sycl::dot(r_fmm,r_fmm);

                            if(opening_angle_sq < open_crit_sq){
                                //this is useless this lambda is already executed only if cd above true
                                auto Q_n = SymTensorCollection<flt, 0, fmm_order>::load(multipoles,id_cell_b*SymTensorCollection<flt,0,fmm_order>::num_component);
                                auto D_n = GreenFuncGravCartesian<flt, 1, fmm_order+1>::get_der_tensors(r_fmm);

                                dM_k += get_dM_mat(D_n,Q_n);
                            }else{


                                obj_iterator( id_cell_a, [&](u32 id_a){

                                    vec x_i = xyz[id_a];
                                    vec sum_fi{0,0,0};

                                    obj_iterator( id_cell_b, [&](u32 id_b){

                                        if(id_a != id_b){
                                            vec x_j = xyz[id_b];

                                            vec real_r = x_i-x_j;

                                            flt inv_r_n = sycl::rsqrt(sycl::dot(real_r,real_r));
                                            sum_fi += real_r*(inv_r_n*inv_r_n*inv_r_n);
                                        }

                                    });

                                    fxyz[id_a] += sum_fi;

                                });
                            }


                        },
                        [&](){
                            //func_node_rejected
                            vec sb = c_centers[id_cell_b];
                            vec r_fmm = sb-sa;

                            auto Q_n = SymTensorCollection<flt, 0, fmm_order>::load(multipoles,id_cell_b*SymTensorCollection<flt,0,fmm_order>::num_component);
                            auto D_n = GreenFuncGravCartesian<flt, 1, fmm_order+1>::get_der_tensors(r_fmm);

                            dM_k += get_dM_mat(D_n,Q_n);
                        }
                    );

                }
            );


            obj_iterator(  id_cell_a, [&](u32 id_a){

                auto ai = SymTensorCollection<flt, 0, fmm_order>::from_vec(xyz[id_a]);

                auto tensor_to_sycl = [](SymTensor3d_1<flt> a){
                    return vec{a.v_0,a.v_1,a.v_2};
                };

                vec tmp {0,0,0};

                tmp += tensor_to_sycl(dM_k.t1*ai.t0);
                tmp += tensor_to_sycl(dM_k.t2*ai.t1);
                tmp += tensor_to_sycl(dM_k.t3*ai.t2);
                if constexpr (fmm_order >= 3) { tmp += tensor_to_sycl(dM_k.t4*ai.t3); }
                if constexpr (fmm_order >= 4) { tmp += tensor_to_sycl(dM_k.t5*ai.t4); }
                fxyz[id_a]  += tmp;

                //auto dphi_0 = tensor_to_sycl(dM_k.t1*ai.t0);
                //auto dphi_1 = tensor_to_sycl(dM_k.t2*ai.t1);
                //auto dphi_2 = tensor_to_sycl(dM_k.t3*ai.t2);
                //auto dphi_3 = tensor_to_sycl(dM_k.t4*ai.t3);
                //auto dphi_4 = tensor_to_sycl(dM_k.t5*ai.t4);
                //fxyz[id_a]  += dphi_0+ dphi_1+ dphi_2+ dphi_3+ dphi_4;

            });

        });

    });
#endif

    // #if false

    shamlog_debug_ln("RTreeFMM", "walking");
    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
        using Rta = walker::Radix_tree_accessor<morton_mode, vec>;
        Rta tree_acc(rtree, cgh);

        auto c_centers = sycl::accessor{*cell_centers, cgh, sycl::read_only};
        auto c_length  = sycl::accessor{*cell_length, cgh, sycl::read_only};

        sycl::range<1> range_leaf = sycl::range<1>{rtree.tree_reduced_morton_codes.tree_leaf_count};

        u32 leaf_offset = rtree.tree_struct.internal_cell_count;

        auto xyz  = sycl::accessor{*pos_part, cgh, sycl::read_only};
        auto fxyz = sycl::accessor{*buf_force, cgh, sycl::read_write};

        auto multipoles = sycl::accessor{*grav_multipoles, cgh, sycl::read_only};

        const auto open_crit_sq = pre_open_crit_sq;

        cgh.parallel_for(range_leaf, [=](sycl::item<1> item) {
            u32 id_cell_a = (u32) item.get_id(0) + leaf_offset;

            vec cur_pos_min_cell_a = tree_acc.pos_min_cell[id_cell_a];
            vec cur_pos_max_cell_a = tree_acc.pos_max_cell[id_cell_a];

            vec sa       = c_centers[id_cell_a];
            flt l_cell_a = c_length[id_cell_a];

            auto dM_k = SymTensorCollection<flt, 1, fmm_order + 1>::zeros();

            walker::rtree_for_cell(
                tree_acc,
                [&tree_acc,
                 &cur_pos_min_cell_a,
                 &cur_pos_max_cell_a,
                 &sa,
                 &l_cell_a,
                 &c_centers,
                 &c_length,
                 &open_crit_sq](u32 id_cell_b) {
                    vec cur_pos_min_cell_b = tree_acc.pos_min_cell[id_cell_b];
                    vec cur_pos_max_cell_b = tree_acc.pos_max_cell[id_cell_b];

                    vec sb       = c_centers[id_cell_b];
                    vec r_fmm    = sb - sa;
                    flt l_cell_b = c_length[id_cell_b];

                    flt opening_angle_sq
                        = (l_cell_a + l_cell_b) * (l_cell_a + l_cell_b) / sycl::dot(r_fmm, r_fmm);

                    using namespace walker::interaction_crit;

                    return sph_cell_cell_crit(
                               cur_pos_min_cell_a,
                               cur_pos_max_cell_a,
                               cur_pos_min_cell_b,
                               cur_pos_max_cell_b,
                               0,
                               0)
                           || (opening_angle_sq > open_crit_sq);
                },
                [&](u32 node_b) {
                    // vec sb = c_centers[node_b];
                    // vec r_fmm = sb-sa;
                    // flt l_cell_b = c_length[node_b];
                    //
                    // flt opening_angle_sq = (l_cell_a + l_cell_b)*(l_cell_a +
                    // l_cell_b)/sycl::dot(r_fmm,r_fmm);
                    //
                    // if(opening_angle_sq < open_crit_sq){
                    //    //this is useless this lambda is already executed only if cd above true
                    //    auto Q_n = SymTensorCollection<flt, 0,
                    //    fmm_order>::load(multipoles,node_b*SymTensorCollection<flt,0,fmm_order>::num_component);
                    //    auto D_n = GreenFuncGravCartesian<flt, 1,
                    //    fmm_order+1>::get_der_tensors(r_fmm);
                    //
                    //    dM_k += get_dM_mat(D_n,Q_n);
                    //}else{

                    walker::iter_object_in_cell(tree_acc, id_cell_a, [&](u32 id_a) {
                        vec x_i = xyz[id_a];
                        vec sum_fi{0, 0, 0};

                        walker::iter_object_in_cell(tree_acc, node_b, [&](u32 id_b) {
                            if (id_a != id_b) {
                                vec x_j = xyz[id_b];

                                vec real_r = x_i - x_j;

                                flt inv_r_n = sycl::rsqrt(sycl::dot(real_r, real_r));
                                sum_fi += real_r * (inv_r_n * inv_r_n * inv_r_n);
                            }
                        });

                        fxyz[id_a] += sum_fi;
                    });
                    //}
                },
                [&](u32 node_b) {
                    vec sb    = c_centers[node_b];
                    vec r_fmm = sb - sa;

                    auto Q_n = SymTensorCollection<flt, 0, fmm_order>::load(
                        multipoles, node_b * SymTensorCollection<flt, 0, fmm_order>::num_component);
                    auto D_n
                        = shamphys::GreenFuncGravCartesian<flt, 1, fmm_order + 1>::get_der_tensors(
                            r_fmm);

                    dM_k += shamphys::get_dM_mat(D_n, Q_n);
                });

            walker::iter_object_in_cell(tree_acc, id_cell_a, [&](u32 id_a) {
                auto ai = SymTensorCollection<flt, 0, fmm_order>::from_vec(xyz[id_a] - sa);

                auto tensor_to_sycl = [](SymTensor3d_1<flt> a) {
                    return vec{a.v_0, a.v_1, a.v_2};
                };

                vec tmp{0, 0, 0};

                tmp += tensor_to_sycl(dM_k.t1 * ai.t0);
                tmp += tensor_to_sycl(dM_k.t2 * ai.t1);
                tmp += tensor_to_sycl(dM_k.t3 * ai.t2);
                if constexpr (fmm_order >= 3) {
                    tmp += tensor_to_sycl(dM_k.t4 * ai.t3);
                }
                if constexpr (fmm_order >= 4) {
                    tmp += tensor_to_sycl(dM_k.t5 * ai.t4);
                }
                fxyz[id_a] += tmp;

                // auto dphi_0 = tensor_to_sycl(dM_k.t1*ai.t0);
                // auto dphi_1 = tensor_to_sycl(dM_k.t2*ai.t1);
                // auto dphi_2 = tensor_to_sycl(dM_k.t3*ai.t2);
                // auto dphi_3 = tensor_to_sycl(dM_k.t4*ai.t3);
                // auto dphi_4 = tensor_to_sycl(dM_k.t5*ai.t4);
                // fxyz[id_a]  += dphi_0+ dphi_1+ dphi_2+ dphi_3+ dphi_4;
            });
        });
    });
    // #endif

    shamsys::instance::get_compute_queue().wait();
    timer.end();

    f64 r_f, r_r;

    {

        auto c_centers = sycl::host_accessor{*cell_centers, sycl::read_only};
        auto c_length  = sycl::host_accessor{*cell_length, sycl::read_only};

        auto pos_min_cell = sycl::host_accessor{*rtree.tree_cell_ranges.buf_pos_min_cell_flt};
        auto pos_max_cell = sycl::host_accessor{*rtree.tree_cell_ranges.buf_pos_max_cell_flt};

        auto sample = [&](u32 id) {
            auto [found_leafs, rejected_nodes] = rtree.get_walk_res_set([&](u32 cell_b) -> bool {
                u32 cell_a = rtree.tree_struct.internal_cell_count + id;

                vec cur_pos_min_cell_a = pos_min_cell[cell_a];
                vec cur_pos_max_cell_a = pos_max_cell[cell_a];

                vec sa       = c_centers[cell_a];
                flt l_cell_a = c_length[cell_a];

                // user defs for the cell pair a-b (current_node_id) return interact cd
                vec cur_pos_min_cell_b = pos_min_cell[cell_b];
                vec cur_pos_max_cell_b = pos_max_cell[cell_b];

                vec sb       = c_centers[cell_b];
                vec r_fmm    = sb - sa;
                flt l_cell_b = c_length[cell_b];

                flt opening_angle_sq
                    = (l_cell_a + l_cell_b) * (l_cell_a + l_cell_b) / sycl::dot(r_fmm, r_fmm);

                using namespace walker::interaction_crit;

                const bool cells_interact = sph_cell_cell_crit(
                                                cur_pos_min_cell_a,
                                                cur_pos_max_cell_a,
                                                cur_pos_min_cell_b,
                                                cur_pos_max_cell_b,
                                                0,
                                                0)
                                            || (opening_angle_sq > pre_open_crit_sq);

                return cells_interact;
            });

            logger::raw_ln(
                "leaf",
                id,
                ": found -> leafs ",
                found_leafs.size(),
                " reject :",
                rejected_nodes.size());

            return std::pair<u32, u32>{found_leafs.size(), rejected_nodes.size()};
        };

        auto [r1_f, r1_r] = sample(0);
        auto [re_f, re_r] = sample(rtree.tree_reduced_morton_codes.tree_leaf_count - 1);
        auto [rm_f, rm_r] = sample(rtree.tree_reduced_morton_codes.tree_leaf_count / 2);

        r_f = (r1_f + re_f + rm_f) / 3.;
        r_r = (r1_r + re_r + rm_r) / 3.;
    }

    flt prec = 0;
    if (npart <= 1e4) {

        sycl::host_accessor<vec> pos{*pos_part};
        sycl::host_accessor<vec> force{*buf_force};

        flt err_sum = 0;
        flt err_max = 0;

        for (u32 i = 0; i < npart; i++) {
            vec sum_fi{0, 0, 0};

            vec x_i = pos[i];

            for (u32 j = 0; j < npart; j++) {

                if (i != j) {

                    vec x_j = pos[j];

                    vec real_r = x_i - x_j;

                    flt r_n = sycl::sqrt(sycl::dot(real_r, real_r));
                    sum_fi += real_r / (r_n * r_n * r_n);
                }
            }

            flt err = sycl::distance(force[i], sum_fi) / sycl::length(sum_fi);

            if (i < 10) {
                logger::raw_ln(
                    "local relative error : ",
                    shambase::format_printf(
                        "%e (%e %e %e) (%e %e %e)",
                        err,
                        force[i].x(),
                        force[i].y(),
                        force[i].z(),
                        sum_fi.x(),
                        sum_fi.y(),
                        sum_fi.z()));
            }
            err_sum += err;

            err_max = sycl::max(err_max, err);
        }

        logger::raw_ln(
            "global relative error :",
            shambase::format_printf("avg = %e max = %e", err_sum / npart, err_max));

        prec = err_sum / npart;
    }

    return Result_nompi_fmm_testing<flt, morton_mode, fmm_order>{
        timer.nanosec,
        prec,
        r_f,
        r_r,
        pos_part,
        std::move(buf_force),
        std::move(rtree),
        std::move(grav_multipoles)};
}

template<class flt>
std::unique_ptr<sycl::buffer<sycl::vec<flt, 3>>> pos_partgen_distrib(u32 npart) {

    using vec = sycl::vec<flt, 3>;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<flt> distf(-1, 1);

    auto pos_part = std::make_unique<sycl::buffer<vec>>(npart);

    {
        sycl::host_accessor<vec> pos{*pos_part};

        for (u32 i = 0; i < npart; i++) {
            pos[i] = vec{distf(eng), distf(eng), distf(eng)};
        }
    }

    return std::move(pos_part);
}

TestStart(ValidationTest, "models/generic/fmm/fmm_1_gpu_prec", fmm_1_gpu_prec, 1) {

    constexpr u32 reduc_level = 5;
    constexpr f64 open_crit   = 0.3;

    if (false) {
        auto pos = pos_partgen_distrib<f32>(1e4);
        auto res = nompi_fmm_testing<f32, u32, 4>(pos, reduc_level, open_crit);
        REQUIRE_FLOAT_EQUAL_NAMED("fmm_f32_u32_order4", res.prec, 0, 1e-5);
    }

    if (false) {
        auto pos = pos_partgen_distrib<f32>(1e4);
        auto res = nompi_fmm_testing<f32, u64, 4>(pos, reduc_level, open_crit);
        REQUIRE_FLOAT_EQUAL_NAMED("fmm_f32_u64_order4", res.prec, 0, 1e-5);
    }

    if (false) {
        auto pos = pos_partgen_distrib<f64>(1e4);
        auto res = nompi_fmm_testing<f64, u32, 4>(pos, reduc_level, open_crit);
        REQUIRE_FLOAT_EQUAL_NAMED("fmm_f64_u32_order4", res.prec, 0, 1e-5);
    }

    {
        auto pos = pos_partgen_distrib<f64>(1e4);
        auto res = nompi_fmm_testing<f64, u64, 4>(pos, reduc_level, open_crit);
        REQUIRE_FLOAT_EQUAL_NAMED("fmm_f64_u64_order4", res.prec, 0, 1e-5);
    }

    // compare u32 / u64

    if (false) {
        using flt = f32;
        using vec = sycl::vec<flt, 3>;

        vec min{-1, -1, -1};
        vec max{1, 1, 1};

        vec diff = max - min;

        std::vector<vec> pos_part_tmp{
            min, min + diff * vec{1. / 64., 0, 0}, max, max - diff * vec{1. / 64., 0, 0}};

        std::unique_ptr<sycl::buffer<vec>> pos_part
            = std::make_unique<sycl::buffer<vec>>(pos_part_tmp.data(), pos_part_tmp.size());

        auto res_u32 = nompi_fmm_testing<flt, u32, 4>(pos_part, 0, 0.1);
        auto res_u64 = nompi_fmm_testing<flt, u64, 4>(pos_part, 0, 0.1);

        logger::raw_ln("u32 :", res_u32.prec, "u64 :", res_u64.prec);

        {
            auto acc_multip_u32 = sycl::host_accessor{*res_u32.grav_multipoles};
            auto acc_multip_u64 = sycl::host_accessor{*res_u64.grav_multipoles};

            logger::raw_ln(
                "multipoles -> u32 len :",
                res_u32.grav_multipoles->size(),
                "u64 len :",
                res_u64.grav_multipoles->size());

            for (u32 i = 0; i < res_u32.grav_multipoles->size() && i < 100; i++) {
                auto v_u32 = acc_multip_u32[i];
                auto v_u64 = acc_multip_u64[i];

                if (v_u32 != v_u64) {
                    logger::raw_ln("i=", i, "->", "u32 :", v_u32, "u64 :", v_u64);
                }
            }
        }

        if (false) {
            auto acc_u32_lid   = sycl::host_accessor{*res_u32.rtree.tree_struct.buf_lchild_id};
            auto acc_u32_rid   = sycl::host_accessor{*res_u32.rtree.tree_struct.buf_rchild_id};
            auto acc_u32_lflag = sycl::host_accessor{*res_u32.rtree.tree_struct.buf_lchild_flag};
            auto acc_u32_rflag = sycl::host_accessor{*res_u32.rtree.tree_struct.buf_rchild_flag};

            auto acc_u64_lid   = sycl::host_accessor{*res_u64.rtree.tree_struct.buf_lchild_id};
            auto acc_u64_rid   = sycl::host_accessor{*res_u64.rtree.tree_struct.buf_rchild_id};
            auto acc_u64_lflag = sycl::host_accessor{*res_u64.rtree.tree_struct.buf_lchild_flag};
            auto acc_u64_rflag = sycl::host_accessor{*res_u64.rtree.tree_struct.buf_rchild_flag};

            for (u32 i = 0; i < res_u64.rtree.tree_struct.internal_cell_count; i++) {
                bool same = true;
                same      = same && (acc_u32_lid[i] == acc_u64_lid[i]);
                same      = same && (acc_u32_rid[i] == acc_u64_rid[i]);
                same      = same && (acc_u32_lflag[i] == acc_u64_lflag[i]);
                same      = same && (acc_u32_rflag[i] == acc_u64_rflag[i]);

                if (!same) {

                    logger::raw_ln(
                        "i=",
                        i,
                        "-> diff "

                        ,
                        " ",
                        acc_u32_lid[i],
                        "|",
                        acc_u64_lid[i],
                        " ",
                        " ",
                        acc_u32_rid[i],
                        "|",
                        acc_u64_rid[i],
                        " ",
                        " ",
                        u32{acc_u32_lflag[i]},
                        "|",
                        u32{acc_u64_lflag[i]},
                        " ",
                        " ",
                        u32{acc_u32_rflag[i]},
                        "|",
                        u32{acc_u64_rflag[i]},
                        " ");
                }
            }

            auto acc_min_u32
                = sycl::host_accessor{*res_u32.rtree.tree_cell_ranges.buf_pos_min_cell};
            auto acc_max_u32
                = sycl::host_accessor{*res_u32.rtree.tree_cell_ranges.buf_pos_max_cell};

            auto acc_min_u64
                = sycl::host_accessor{*res_u64.rtree.tree_cell_ranges.buf_pos_min_cell};
            auto acc_max_u64
                = sycl::host_accessor{*res_u64.rtree.tree_cell_ranges.buf_pos_max_cell};

            for (u32 i = 0; i < res_u64.rtree.tree_struct.internal_cell_count
                                    + res_u64.rtree.tree_reduced_morton_codes.tree_leaf_count;
                 i++) {
                bool same = true;
                // same = same && test_sycl_eq(acc_min_u32[i] , acc_min_u64[i]);
                // same = same && test_sycl_eq(acc_min_u64[i] , acc_max_u64[i]);
                same = false;

                if (!same) {

                    logger::raw_ln(
                        "i=",
                        i,
                        "-> diff ",
                        " ",
                        acc_min_u32[i],
                        acc_min_u64[i],
                        "  ",
                        acc_max_u32[i],
                        acc_max_u64[i]);
                }
            }
        }
    }
}

#if false

Bench_start("fmm shit", "multipole_compute", fmm_perf_multipole, 1){


    std::vector<f64_3> pos_table = {{1,1,1}};

    sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

    constexpr u32 order = 5;

    constexpr u32 number_elem_multip = SymTensorCollection<f64,0,order>::num_component;
    sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


    //compute multipoles
    auto l = [&]{
        sycl::host_accessor<f64_3> pos {buf_pos};
        sycl::host_accessor<f64> multipoles {buf_multipoles};

        f64_3 sa = {0,0,0};
        f64_3 sb = {0,0,0};

        for (u32 j = 0; j < pos_table.size(); j ++) {

            f64_3 xj = pos[j];

            f64_3 bj = xj - sb;

            auto B_n = SymTensorCollection<f64,0,order>::from_vec(bj);



        }

    };

    TimeitFor(100,l())

}


Bench_start("fmm shit", "multipole_offset", fmm_perf_offset, 1){


    std::vector<f64_3> pos_table = {{1,1,1}};

    sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

    constexpr u32 order = 5;

    constexpr u32 number_elem_multip = SymTensorCollection<f64,0,order>::num_component;
    sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


    //compute multipoles
    auto l = [&]{
        sycl::host_accessor<f64_3> pos {buf_pos};
        sycl::host_accessor<f64> multipoles {buf_multipoles};

        f64_3 sa = {0,0,0};
        f64_3 sb = {0,0,0};

        for (u32 j = 0; j < pos_table.size(); j ++) {

            f64_3 xj = pos[j];

            f64_3 bj = xj - sb;

            auto B_n = SymTensorCollection<f64,0,order>::from_vec(bj);



        }

    };

    l();



    sycl::buffer<f64> buf_multipoles2 = sycl::buffer<f64>(number_elem_multip);


    auto l2 = [&]{
        sycl::host_accessor<f64> multipoles_old {buf_multipoles};
        sycl::host_accessor<f64> multipoles_new {buf_multipoles2};

        f64_3 d_vec = {1,2,3};

        for (u32 j = 0; j < pos_table.size(); j ++) {

            auto B_n = SymTensorCollection<f64,0,order>::load(multipoles_old,0);

            offset_multipole(B_n,d_vec).store(multipoles_new, 0);

        }

    };


    TimeitFor(1000,l2())

}

#endif

constexpr auto list_npart_test = {
    1.14975700e+02, 1.35304777e+02, 1.59228279e+02, 1.87381742e+02, 2.20513074e+02, 2.59502421e+02,
    3.05385551e+02, 3.59381366e+02, 4.22924287e+02, 4.97702356e+02, 5.85702082e+02, 6.89261210e+02,
    8.11130831e+02, 9.54548457e+02, 1.12332403e+03, 1.32194115e+03, 1.55567614e+03, 1.83073828e+03,
    2.15443469e+03, 2.53536449e+03, 2.98364724e+03, 3.51119173e+03, 4.13201240e+03, 4.86260158e+03,
    5.72236766e+03, 6.73415066e+03, 7.92482898e+03, 9.32603347e+03, 1.09749877e+04, 1.29154967e+04,
    1.51991108e+04, 1.78864953e+04, 2.10490414e+04, 2.47707636e+04, 2.91505306e+04, 3.43046929e+04,
    4.03701726e+04, 4.75081016e+04, 5.59081018e+04, 6.57933225e+04, 7.74263683e+04, 9.11162756e+04,
    1.07226722e+05, 1.26185688e+05, 1.48496826e+05, 1.74752840e+05, 2.05651231e+05, 2.42012826e+05,
    2.84803587e+05, 3.35160265e+05, 3.94420606e+05, 4.64158883e+05, 5.46227722e+05, 6.42807312e+05,
    7.56463328e+05, 8.90215085e+05, 1.04761575e+06
    // 1.23284674e+06, 1.45082878e+06, 1.70735265e+06, 2.00923300e+06,
    // 2.36448941e+06, 2.78255940e+06, 3.27454916e+06, 3.85352859e+06,
    // 4.53487851e+06, 5.33669923e+06, 6.28029144e+06, 7.39072203e+06,
    // 8.69749003e+06, 1.02353102e+07
};

template<class flt, class morton_mode, u32 fmm_order>
class Walk_kernel;

template<class flt, class morton_mode, u32 fmm_order>
class Cascade_multip;

template<class flt, class morton_mode, u32 fmm_order>
void run_test_no_mpi_fmm(std::string dset_name) {

    auto &dset = shamtest::test_data().new_dataset(dset_name);

    std::vector<f64> Npart;
    std::vector<f64> time_red0_full;
    std::vector<f64> time_red1_full;
    std::vector<f64> time_red2_full;
    std::vector<f64> time_red3_full;
    std::vector<f64> time_red4_full;
    std::vector<f64> time_red5_full;
    std::vector<f64> time_red6_full;
    std::vector<f64> time_red7_full;
    std::vector<f64> time_red8_full;
    std::vector<f64> red0_leaf_found;
    std::vector<f64> red1_leaf_found;
    std::vector<f64> red2_leaf_found;
    std::vector<f64> red3_leaf_found;
    std::vector<f64> red4_leaf_found;
    std::vector<f64> red5_leaf_found;
    std::vector<f64> red6_leaf_found;
    std::vector<f64> red7_leaf_found;
    std::vector<f64> red8_leaf_found;
    std::vector<f64> red0_leaf_rej;
    std::vector<f64> red1_leaf_rej;
    std::vector<f64> red2_leaf_rej;
    std::vector<f64> red3_leaf_rej;
    std::vector<f64> red4_leaf_rej;
    std::vector<f64> red5_leaf_rej;
    std::vector<f64> red6_leaf_rej;
    std::vector<f64> red7_leaf_rej;
    std::vector<f64> red8_leaf_rej;

    auto get_max_part = [&]() {
        f64 gsz = shamsys::instance::get_compute_queue()
                      .get_device()
                      .get_info<sycl::info::device::global_mem_size>();
        gsz            = 1024 * 1024 * 1024 * 1;
        f64 part_per_g = 2500000;

        return (gsz / (1024. * 1024. * 1024.)) * part_per_g / 100.;
    };

    f64 Nmax = get_max_part();
    shamlog_debug_ln("Benchmark FMM", "Nmax =", Nmax);

    for (f64 cnt = 1000; cnt <= Nmax; cnt *= 1.5) {
        shamlog_debug_ln("Benchmark FMM", "cnt =", cnt);

        auto pos_part = pos_partgen_distrib<flt>(u32(cnt));
        Npart.push_back(u32(cnt));

        {
            auto res_red0 = nompi_fmm_testing<flt, morton_mode, fmm_order>(pos_part, 0, 0.5);
            time_red0_full.push_back(res_red0.time);
            red0_leaf_found.push_back(res_red0.leaf_cnt);
            red0_leaf_rej.push_back(res_red0.reject_cnt);
        }
        {
            auto res_red1 = nompi_fmm_testing<flt, morton_mode, fmm_order>(pos_part, 1, 0.5);
            time_red1_full.push_back(res_red1.time);
            red1_leaf_found.push_back(res_red1.leaf_cnt);
            red1_leaf_rej.push_back(res_red1.reject_cnt);
        }
        {
            auto res_red2 = nompi_fmm_testing<flt, morton_mode, fmm_order>(pos_part, 2, 0.5);
            time_red2_full.push_back(res_red2.time);
            red2_leaf_found.push_back(res_red2.leaf_cnt);
            red2_leaf_rej.push_back(res_red2.reject_cnt);
        }
        {
            auto res_red3 = nompi_fmm_testing<flt, morton_mode, fmm_order>(pos_part, 3, 0.5);
            time_red3_full.push_back(res_red3.time);
            red3_leaf_found.push_back(res_red3.leaf_cnt);
            red3_leaf_rej.push_back(res_red3.reject_cnt);
        }
        {
            auto res_red4 = nompi_fmm_testing<flt, morton_mode, fmm_order>(pos_part, 4, 0.5);
            time_red4_full.push_back(res_red4.time);
            red4_leaf_found.push_back(res_red4.leaf_cnt);
            red4_leaf_rej.push_back(res_red4.reject_cnt);
        }
        {
            auto res_red5 = nompi_fmm_testing<flt, morton_mode, fmm_order>(pos_part, 5, 0.5);
            time_red5_full.push_back(res_red5.time);
            red5_leaf_found.push_back(res_red5.leaf_cnt);
            red5_leaf_rej.push_back(res_red5.reject_cnt);
        }
        {
            auto res_red6 = nompi_fmm_testing<flt, morton_mode, fmm_order>(pos_part, 6, 0.5);
            time_red6_full.push_back(res_red6.time);
            red6_leaf_found.push_back(res_red6.leaf_cnt);
            red6_leaf_rej.push_back(res_red6.reject_cnt);
        }
        {
            auto res_red7 = nompi_fmm_testing<flt, morton_mode, fmm_order>(pos_part, 7, 0.5);
            time_red7_full.push_back(res_red7.time);
            red7_leaf_found.push_back(res_red7.leaf_cnt);
            red7_leaf_rej.push_back(res_red7.reject_cnt);
        }
        {
            auto res_red8 = nompi_fmm_testing<flt, morton_mode, fmm_order>(pos_part, 8, 0.5);
            time_red8_full.push_back(res_red8.time);
            red8_leaf_found.push_back(res_red8.leaf_cnt);
            red8_leaf_rej.push_back(res_red8.reject_cnt);
        }
    }

    dset.add_data("Npart", Npart);
    dset.add_data("time_red0_full", time_red0_full);
    dset.add_data("time_red1_full", time_red1_full);
    dset.add_data("time_red2_full", time_red2_full);
    dset.add_data("time_red3_full", time_red3_full);
    dset.add_data("time_red4_full", time_red4_full);
    dset.add_data("time_red5_full", time_red5_full);
    dset.add_data("time_red6_full", time_red6_full);
    dset.add_data("time_red7_full", time_red7_full);
    dset.add_data("time_red8_full", time_red8_full);
    dset.add_data("red0_leaf_found", red0_leaf_found);
    dset.add_data("red1_leaf_found", red1_leaf_found);
    dset.add_data("red2_leaf_found", red2_leaf_found);
    dset.add_data("red3_leaf_found", red3_leaf_found);
    dset.add_data("red4_leaf_found", red4_leaf_found);
    dset.add_data("red5_leaf_found", red5_leaf_found);
    dset.add_data("red6_leaf_found", red6_leaf_found);
    dset.add_data("red7_leaf_found", red7_leaf_found);
    dset.add_data("red8_leaf_found", red8_leaf_found);
    dset.add_data("red0_leaf_rej", red0_leaf_rej);
    dset.add_data("red1_leaf_rej", red1_leaf_rej);
    dset.add_data("red2_leaf_rej", red2_leaf_rej);
    dset.add_data("red3_leaf_rej", red3_leaf_rej);
    dset.add_data("red4_leaf_rej", red4_leaf_rej);
    dset.add_data("red5_leaf_rej", red5_leaf_rej);
    dset.add_data("red6_leaf_rej", red6_leaf_rej);
    dset.add_data("red7_leaf_rej", red7_leaf_rej);
    dset.add_data("red8_leaf_rej", red8_leaf_rej);
}

TestStart(Benchmark, "fmm_no_mpi performance", fmm_no_mpi, 1) {
    run_test_no_mpi_fmm<f32, u32, 3>("case f32,u32, order = 3");

    run_test_no_mpi_fmm<f32, u32, 4>("case f32,u32, order = 4");

    run_test_no_mpi_fmm<f32, u64, 3>("case f32,u64, order = 3");

    run_test_no_mpi_fmm<f32, u64, 4>("case f32,u64, order = 4");

    run_test_no_mpi_fmm<f64, u64, 3>("case f64,u64, order = 3");

    run_test_no_mpi_fmm<f64, u64, 4>("case f64,u64, order = 4");
}

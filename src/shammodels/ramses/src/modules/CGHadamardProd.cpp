// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CGHadamardProd.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/CGHadamardProd.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class T>
    struct KernelHadamardProd {

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_phi_p,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_phi_Ap,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>>
                &spans_phi_hadamard_prod,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_phi_p, spans_phi_Ap},
                sham::DDMultiRef{spans_phi_hadamard_prod},
                cell_counts,
                [block_size](
                    u32 i,
                    const T *__restrict phi_p,
                    const T *__restrict phi_Ap,
                    T *__restrict hadamard_prod) {
                    hadamard_prod[i] = phi_p[i] * phi_Ap[i];
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class T>
    void NodeCGHadamardProd<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_phi_p.check_sizes(edges.sizes.indexes);
        edges.spans_phi_Ap.check_sizes(edges.sizes.indexes);

        edges.spans_phi_hadamard_prod.ensure_sizes(edges.sizes.indexes);

        KernelHadamardProd<T>::kernel(
            edges.spans_phi_p.get_spans(),
            edges.spans_phi_Ap.get_spans(),
            edges.spans_phi_hadamard_prod.get_spans(),
            edges.sizes.indexes,
            block_size);
    }

    template<class T>
    std::string NodeCGHadamardProd<T>::_impl_get_tex() {

        auto block_count       = get_ro_edge_base(0).get_tex_symbol();
        auto phi_p             = get_ro_edge_base(1).get_tex_symbol();
        auto phi_Ap            = get_ro_edge_base(2).get_tex_symbol();
        auto phi_hadamard_prod = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Compute Hadamard product
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeCGHadamardProd<f64>;

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeAYPXTwoVectors.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/NodeAYPXTwoVectors.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class T>
    struct KernelAYPXTwoVectors {

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_x,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_y,
            const T alpha,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      // logger::raw_ln("i = [ ", id , "] = ",cell_count, "\n");
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_x},
                sham::DDMultiRef{spans_y},
                cell_counts,
                [alpha](u32 i, const T *__restrict x, T *__restrict y) {
                    auto tmp = y[i];
                    y[i]     = alpha * y[i] + x[i];

                    // if( (i >= 145) && (i <= 196 ))
                    // if(i % 25 == 0)
                    // {
                    //     logger::raw_ln("id_a = [ ", i, " ] : ", y[i], " --- ",x[i], " *** ", tmp,
                    //     "  ","\n");
                    // }
                    // logger::raw_ln("i = [ ", i , "] = ",y[i], "\n");
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class T>
    void NodeAYPXTwoVectors<T>::_impl_evaluate_internal() {
        auto edges = get_edges();
        //   logger::raw_ln("AYPX:[x, y] \t", &edges.spans_x, "-", &edges.spans_y,"\n");
        edges.spans_x.check_sizes(edges.sizes.indexes);
        edges.spans_y.ensure_sizes(edges.sizes.indexes);

        KernelAYPXTwoVectors<T>::kernel(
            edges.spans_x.get_spans(),
            edges.spans_y.get_spans(),
            edges.alpha.value,
            edges.sizes.indexes,
            block_size);
    }

    template<class T>
    std::string NodeAYPXTwoVectors<T>::_impl_get_tex() {

        std::string tex = R"tex(
            AYPX(2 vectors)
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeAYPXTwoVectors<f64>;

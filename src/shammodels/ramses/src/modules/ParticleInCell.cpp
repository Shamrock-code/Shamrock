// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ParticleInCell.cpp
 * @author
 * @brief
 */

#include "shammodels/ramses/modules/ParticleInCell.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/riemann.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    struct KernelPIC {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rho,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rho_pic,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_rho},
                sham::DDMultiRef{spans_rho_pic},
                cell_counts,
                [](
                    u32 i,
                    const Tscal *__restrict rho,
                    Tscal *__restrict rho_pic) {

                    /*

                    PIC KERNELS IN PROGRESS

                    */
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodePIC<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_rho_pic.ensure_sizes(edges.sizes.indexes);

        KernelPIC<Tvec>::kernel(
            edges.spans_rho.get_spans(),
            edges.spans_rho_pic.get_spans(),
            edges.sizes.indexes,
            block_size);
    }

    template<class Tvec>
    std::string NodePIC<Tvec>::_impl_get_tex() const {
        auto block_count = get_ro_edge_base(0).get_tex_symbol();
        auto rho         = get_ro_edge_base(1).get_tex_symbol();
        auto rho_pic     = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            // TODO: Add TeX description here
        )tex";

        shambase::replace_all(tex, "{rho_pic}", rho_pic);
        shambase::replace_all(tex, "{rho}", rho);
        shambase::replace_all(tex, "{block_count}", block_count);
        shambase::replace_all(tex, "{block_size}", shambase::format("{}", block_size));

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodePIC<f64_3>;
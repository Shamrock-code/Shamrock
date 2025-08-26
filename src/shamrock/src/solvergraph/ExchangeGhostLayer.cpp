// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExchangeGhostLayer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of ghost layer data exchange for distributed hydrodynamics simulations
 *
 * This file implements the ExchangeGhostLayer class methods, providing the concrete
 * functionality for exchanging ghost layer data between distributed computational
 * domains using sparse communication and serialization mechanisms.
 */

#include "shamrock/solvergraph/ExchangeGhostLayer.hpp"
#include "shamalgs/collective/distributedDataComm.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"

void shamrock::solvergraph::ExchangeGhostLayer::_impl_evaluate_internal() {
    auto edges = get_edges();

    // outputs
    auto &ghost_layer                                         = edges.ghost_layer;
    const shamrock::solvergraph::ScalarsEdge<u32> &rank_owner = edges.rank_owner;

    shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> recv_dat;

    shamalgs::collective::serialize_sparse_comm<shamrock::patch::PatchDataLayer>(
        shamsys::instance::get_compute_scheduler_ptr(),
        std::move(ghost_layer.patchdatas),
        recv_dat,
        [&](u64 id) {
            return rank_owner.values.get(id);
        },
        [](shamrock::patch::PatchDataLayer &pdat) {
            shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
            ser.allocate(pdat.serialize_buf_byte_size());
            pdat.serialize_buf(ser);
            return ser.finalize();
        },
        [&](sham::DeviceBuffer<u8> &&buf) {
            // exchange the buffer held by the distrib data and give it to the serializer
            shamalgs::SerializeHelper ser(
                shamsys::instance::get_compute_scheduler_ptr(),
                std::forward<sham::DeviceBuffer<u8>>(buf));
            return shamrock::patch::PatchDataLayer::deserialize_buf(ser, ghost_layer_layout);
        });

    ghost_layer.patchdatas = std::move(recv_dat);
}

std::string shamrock::solvergraph::ExchangeGhostLayer::_impl_get_tex() {
    auto rank_owner  = get_ro_edge_base(0).get_tex_symbol();
    auto ghost_layer = get_rw_edge_base(0).get_tex_symbol();

    std::string tex = R"tex(
        Exchange ghost layer data between distributed processes

        \begin{align}
        {ghost_layer}_{i \rightarrow \underline{j}} = \text{Sparse comm}({ghost_layer}_{\underline{i} \rightarrow j}) \\
        \text{where } {rank_owner}_{\underline{j}} = \text{MPI world rank} \\
        \text{and } {rank_owner}_{\underline{i}} = \text{MPI world rank} \\
        \text{and } i \in [0, N_{\rm patch}] \\
        \text{and } j \in [0, N_{\rm patch}] \\
        \end{align},
        underlined indices denotes one that currently owned by the local process.
    )tex";

    shambase::replace_all(tex, "{rank_owner}", rank_owner);
    shambase::replace_all(tex, "{ghost_layer}", ghost_layer);

    return tex;
}

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FindGhostLayerCandidates.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/assert.hpp"
#include "shambase/exception.hpp"
#include "shammath/AABB.hpp"
#include "shammath/paving_function.hpp"
#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include <stdexcept>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FindGhostLayerCandidates<Tvec, TgridVec>::
    _impl_evaluate_internal() {
    auto edges = get_edges();

    // inputs
    auto &sim_box     = edges.sim_box.value;
    auto &patch_tree  = edges.patch_tree.get_patch_tree();
    auto &patch_boxes = edges.patch_boxes;

    using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

    // outputs
    auto &ghost_layers_candidates = edges.ghost_layers_candidates.values;

    auto paving = get_paving(mode, sim_box);

    using namespace shamrock::patch;

    // for each repetitions
    for_each_paving_tile(mode, [&](i32 xoff, i32 yoff, i32 zoff) {
        // for all local patches
        patch_boxes.values.for_each([&](u64 id, const shammath::AABB<TgridVec> &patch_box) {
            // f(patch)
            auto patch_box_mapped = paving.f_aabb(patch_box, xoff, yoff, zoff);

            patch_tree.host_for_each_leafs(
                [&](u64 tree_id, PtNode n) {
                    shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};

                    // f(patch) V box =! empty (a surface is not an empty set btw)
                    // <=> is ghost layer != empty
                    return tree_cell.get_intersect(patch_box_mapped).is_not_empty();
                },
                [&](u64 id_found, PtNode n) {
                    // skip self intersection (but not if we are through a boundary)
                    if ((id_found == id) && (xoff == 0) && (yoff == 0) && (zoff == 0)) {
                        return;
                    }

                    // we have an ghost layer between
                    // patch `id` and patch `id_found` for this offset
                    // so we store that
                    ghost_layers_candidates.add_obj(
                        id, id_found, GhostLayerCandidateInfos{xoff, yoff, zoff});
                });
        });
    });
}

template<class Tvec, class TgridVec>
std::string
shammodels::basegodunov::modules::FindGhostLayerCandidates<Tvec, TgridVec>::_impl_get_tex() {
    auto sim_box                 = get_ro_edge_base(0).get_tex_symbol();
    auto patch_tree              = get_ro_edge_base(1).get_tex_symbol();
    auto patch_boxes             = get_ro_edge_base(2).get_tex_symbol();
    auto ghost_layers_candidates = get_rw_edge_base(0).get_tex_symbol();

    std::string tex = R"tex(
        Find Ghost Layer Candidates

        \begin{algorithm}[H]
        \caption{Find Ghost Layer Candidates Algorithm}
        \SetAlgoLined
        \SetKwInOut{Input}{Input}
        \SetKwInOut{Output}{Output}
        \Input{Simulation box, patch tree, patch boxes}
        \Output{Ghost layer candidates}
        \BlankLine
        \For{each paving tile offset $(x_{\rm off}, y_{\rm off}, z_{\rm off})$}{
            \uIf{periodic in $x$}{$x_{\rm off} \in \{-1, 0, 1\}$}
            \Else{$x_{\rm off} = 0$}
            \uIf{periodic in $y$}{$y_{\rm off} \in \{-1, 0, 1\}$}
            \Else{$y_{\rm off} = 0$}
            \uIf{periodic in $z$}{$z_{\rm off} \in \{-1, 0, 1\}$}
            \Else{$z_{\rm off} = 0$}
            \BlankLine
            \For{each local patch $P_i$}{
                $B_i \leftarrow$ patch box of $P_i$\;
                $B_i^{\rm mapped} \leftarrow f(B_i, x_{\rm off}, y_{\rm off}, z_{\rm off})$\;
                \BlankLine
                \For{each tree node $T_j$}{
                    $B_j \leftarrow$ tree node box\;
                    \If{$B_i^{\rm mapped} \cap B_j \neq \emptyset$ \textbf{and} $(i \neq j$ \textbf{or} $(x_{\rm off}, y_{\rm off}, z_{\rm off}) \neq (0,0,0))$}{
                        Add ghost layer candidate: $(P_i, P_j, x_{\rm off}, y_{\rm off}, z_{\rm off})$\;
                    }
                }
            }
        }
        \end{algorithm}

        \textbf{Note:} $f(B, x, y, z)$ is the paving function that maps box $B$ by offset $(x, y, z)$
    )tex";

    shambase::replace_all(tex, "{sim_box}", sim_box);
    shambase::replace_all(tex, "{patch_tree}", patch_tree);
    shambase::replace_all(tex, "{patch_boxes}", patch_boxes);
    shambase::replace_all(tex, "{ghost_layers_candidates}", ghost_layers_candidates);

    return tex;
}

template class shammodels::basegodunov::modules::FindGhostLayerCandidates<f64_3, i64_3>;

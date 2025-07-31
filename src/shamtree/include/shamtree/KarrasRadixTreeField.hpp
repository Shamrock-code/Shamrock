// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file KarrasRadixTreeField.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/CellIterator.hpp"
#include "shamtree/KarrasRadixTree.hpp"
#include <functional>
#include <utility>

namespace shamtree {

    /**
     * @class KarrasRadixTreeField
     * @brief A data structure representing a Karras Radix Tree Field.
     *
     * This class encapsulates the structure of a Karras Radix Tree Field, which is used for
     * efficiently handling hierarchical data based on Morton codes. It manages buffers for left and
     * right child identifiers and flags, as well as end ranges.
     */
    template<class T>
    class KarrasRadixTreeField;
} // namespace shamtree

template<class T>
class shamtree::KarrasRadixTreeField {

    public:
    /// Get internal cell count
    inline u32 get_total_cell_count() { return buf_field.get_size(); }

    sham::DeviceBuffer<T> buf_field; ///< left child id (size = internal_count)

    /// CTOR
    KarrasRadixTreeField(sham::DeviceBuffer<T> &&buf_field) : buf_field(std::move(buf_field)) {}

    static inline KarrasRadixTreeField make_empty(sham::DeviceScheduler_ptr dev_sched) {
        return KarrasRadixTreeField{sham::DeviceBuffer<T>(0, dev_sched)};
    }
};

namespace shamtree {

    template<class T>
    KarrasRadixTreeField<T> new_empty_karras_radix_tree_field() {
        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        return KarrasRadixTreeField<T>(sham::DeviceBuffer<T>(0, dev_sched));
    }

    template<class T>
    KarrasRadixTreeField<T> prepare_karras_radix_tree_field(
        const KarrasRadixTree &tree, KarrasRadixTreeField<T> &&recycled_tree_field) {

        KarrasRadixTreeField<T> ret = std::forward<KarrasRadixTreeField<T>>(recycled_tree_field);

        ret.buf_field.resize(tree.get_total_cell_count());

        return ret;
    }

    template<class T, class Fct>
    void propagate_field_up(
        KarrasRadixTreeField<T> &tree_field, const KarrasRadixTree &tree, Fct fct_combine) {

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        u32 int_cell_count = tree.get_internal_cell_count();

        if (int_cell_count == 0) {
            return;
        }

        auto step = [&]() {
            auto traverser = tree.get_structure_traverser();

            sham::kernel_call(
                q,
                sham::MultiRef{traverser},
                sham::MultiRef{tree_field.buf_field},
                int_cell_count,
                [=](u32 gid, auto tree_traverser, T *__restrict tree_field) {
                    u32 left_child  = tree_traverser.get_left_child(gid);
                    u32 right_child = tree_traverser.get_right_child(gid);

                    T fieldl = tree_field[left_child];
                    T fieldr = tree_field[right_child];

                    T field_val = fct_combine(fieldl, fieldr);

                    tree_field[gid] = field_val;
                });
        };

        for (u32 i = 0; i < tree.tree_depth; i++) {
            step();
        }
    }

    template<class T, class Fct>
    KarrasRadixTreeField<T> compute_tree_field(
        const KarrasRadixTree &tree,
        KarrasRadixTreeField<T> &&recycled_tree_field,
        const std::function<void(KarrasRadixTreeField<T> &, u32)> &fct_fill_leaf,
        Fct fct_combine) {

        auto tree_field = prepare_karras_radix_tree_field(
            tree, std::forward<KarrasRadixTreeField<T>>(recycled_tree_field));

        fct_fill_leaf(tree_field, tree.get_internal_cell_count());

        propagate_field_up(tree_field, tree, std::forward<Fct>(fct_combine));

        return tree_field;
    }

    template<class T>
    KarrasRadixTreeField<T> compute_tree_field_max_field(
        const KarrasRadixTree &tree,
        const CellIterator &cell_it,
        KarrasRadixTreeField<T> &&recycled_tree_field,
        sham::DeviceBuffer<T> &field) {

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        auto fill_leafs = [&](KarrasRadixTreeField<T> &tree_field, u32 leaf_offset) {
            sham::kernel_call(
                q,
                sham::MultiRef{field, cell_it},
                sham::MultiRef{tree_field.buf_field},
                tree.get_leaf_count(),
                [leaf_offset](u32 i, const T *field, auto cell_iter, T *comp_field) {
                    // Init with the min value of the type
                    T field_val = shambase::VectorProperties<T>::get_min();

                    cell_iter.for_each_in_cell(i, [&](u32 obj_id) {
                        field_val = sham::max(field_val, field[obj_id]);
                    });

                    comp_field[leaf_offset + i] = field_val;
                });
        };

        return compute_tree_field<T>(
            tree,
            std::forward<KarrasRadixTreeField<T>>(recycled_tree_field),
            fill_leafs,
            [](T a, T b) {
                return sham::max(a, b);
            });
    }

} // namespace shamtree

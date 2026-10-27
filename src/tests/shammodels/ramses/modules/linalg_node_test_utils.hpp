// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file linalg_node_test_utils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Helpers to build the edges of the ramses linear algebra node tests by hand.
 *
 * The fields span several patches of different block counts (with sparse patch ids). The values
 * are small dyadic rationals (k / 8 with |k| <= 64), so that the elementwise operations tested
 * with dyadic coefficients are exact in floating point, whatever the evaluation order or FMA
 * contraction, and can be compared bitwise to a host-side reference.
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <memory>
#include <utility>
#include <vector>

namespace shammodels::ramses::test_utils {

    /// (patch id, block count) of the patches used by the tests
    inline const std::vector<std::pair<u64, u32>> patch_block_counts = {{0, 3}, {5, 11}, {7, 1}};

    /// block sizes the nodes are tested with (1 cell per block and an AMR block of 2^3 cells)
    inline const std::vector<u32> block_sizes = {1, 8};

    /// Build the `sizes` edge (block counts per patch)
    inline std::shared_ptr<shamrock::solvergraph::Indexes<u32>> make_sizes(
        const std::vector<std::pair<u64, u32>> &block_counts = patch_block_counts) {
        auto sizes = std::make_shared<shamrock::solvergraph::Indexes<u32>>("sizes", "N");
        for (auto [id_patch, block_count] : block_counts) {
            sizes->indexes.add_obj(id_patch, block_count);
        }
        return sizes;
    }

    /// Generate `len` dyadic values k / 8 with k in [-64, 64]
    template<class T>
    inline std::vector<T> dyadic_values(u64 seed, u32 len) {
        using Tscal        = shambase::VecComponent<T>;
        constexpr u32 nvar = shambase::VectorProperties<T>::dimension;

        std::vector<i32> ints = shamalgs::primitives::mock_vector<i32>(seed, len * nvar, -64, 64);

        std::vector<T> ret(len);
        for (u32 i = 0; i < len; i++) {
            if constexpr (nvar == 1) {
                ret[i] = Tscal(ints[i]) / 8;
            } else {
                for (u32 c = 0; c < nvar; c++) {
                    ret[i][c] = Tscal(ints[i * nvar + c]) / 8;
                }
            }
        }
        return ret;
    }

    /// Build a field with `block_size` cells per block, filled with `values_gen(seed, len)` per
    /// patch (the seed is offset by the patch id)
    template<class T, class Gen>
    inline std::shared_ptr<shamrock::solvergraph::Field<T>> make_field(
        const shamrock::solvergraph::Indexes<u32> &sizes,
        u32 block_size,
        u64 seed,
        Gen &&values_gen) {
        auto field = std::make_shared<shamrock::solvergraph::Field<T>>(block_size, "field", "f");
        field->ensure_sizes(sizes.indexes);
        sizes.indexes.for_each_ordered([&](u64 id_patch, const u32 &block_count) {
            field->get_buf(id_patch).copy_from_stdvec(
                values_gen(seed + id_patch, block_count * block_size));
        });
        return field;
    }

    /// Build a field filled with dyadic values (see dyadic_values)
    template<class T>
    inline std::shared_ptr<shamrock::solvergraph::Field<T>> make_dyadic_field(
        const shamrock::solvergraph::Indexes<u32> &sizes, u32 block_size, u64 seed) {
        return make_field<T>(sizes, block_size, seed, [](u64 s, u32 len) {
            return dyadic_values<T>(s, len);
        });
    }

    /// Build a scalar edge holding `value`
    template<class T>
    inline std::shared_ptr<shamrock::solvergraph::ScalarEdge<T>> make_scalar(T value) {
        auto edge   = std::make_shared<shamrock::solvergraph::ScalarEdge<T>>("scalar", "a");
        edge->value = value;
        return edge;
    }

    /// Copy the content of a patch of a field to the host
    template<class T>
    inline std::vector<T> to_host(shamrock::solvergraph::Field<T> &field, u64 id_patch) {
        return field.get_buf(id_patch).copy_to_stdvec();
    }

} // namespace shammodels::ramses::test_utils

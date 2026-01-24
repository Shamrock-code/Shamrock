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
 * @file InvariantParallelGenerator.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shambase/aliases_int.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shamalgs/collective/reduction.hpp"
#include <random>

namespace shamalgs::collective {

    /// A parralel generator that will spit the same sequence regardless of the number of ranks.
    template<class Engine = std::mt19937_64>
    class InvariantParallelGenerator {
        Engine eng_global;
        u64 nval_max;
        u64 nval_current;
        bool done;

        void skip(u64 n) {
            u64 remaining_n = nval_max - nval_current;
            u64 to_skip     = std::min(remaining_n, n);

            eng_global.discard(to_skip);
            nval_current += to_skip;
            if (nval_current == nval_max) {
                done = true;
            }
        }

        std::vector<u64> next_n_sequential(u64 val_count) {

            if (is_done()) {
                return {};
            }

            u64 to_generate = std::min(val_count, nval_max - nval_current);

            std::vector<u64> ret(to_generate);
            for (u64 i = 0; i < to_generate; i++) {
                ret[i] = eng_global();
            }

            nval_current += to_generate;
            if (nval_current == nval_max) {
                done = true;
            }
            return ret;
        }

        std::vector<u64> next_n_parallel(u64 val_count) {

            if (is_done()) {
                return {};
            }

            auto gen_info = shamalgs::collective::fetch_view(val_count);

            // here i keep the temp variable for clarity
            u64 skip_start = gen_info.head_offset;
            u64 gen_cnt    = val_count;
            u64 skip_end   = gen_info.total_byte_count - val_count - gen_info.head_offset;

            shamlog_debug_ln(
                "InvariantParallelGenerator",
                "generate : ",
                skip_start,
                gen_cnt,
                skip_end,
                "total",
                skip_start + gen_cnt + skip_end);

            skip(skip_start);
            std::vector<u64> ret = next_n_sequential(gen_cnt);
            skip(skip_end);
            return ret;
        }

        public:
        InvariantParallelGenerator(Engine eng, u64 nval_max)
            : eng_global(eng), nval_max(nval_max), nval_current(0), done(false) {
            if (nval_max == 0) {
                done = true;
            }
        }

        InvariantParallelGenerator(u64 seed, u64 nval_max)
            : InvariantParallelGenerator(Engine(seed), nval_max) {}

        /**
         * @brief Generate the next `val_count` values
         *
         * @param val_count the number of values to generate
         * @param sequential if true, the values are generated sequentially on all ranks, otherwise
         * in parallel on all ranks
         * @return std::vector<u64> the generated values (max size is val_count) actual size depends
         * on the number of remaining values
         *
         * If `sequential` is true, the values are generated sequentially on all ranks.
         * If `sequential` is false, the values are generated in parallel on all ranks.
         * @note allgatherv(next_n(n, false)) == next_n(n,true)
         *
         * @note The values are generated in a way that is invariant to the number of ranks.
         */
        std::vector<u64> next_n(u64 val_count, bool sequential = false) {
            if (sequential) {
                u64 sum_ranks = collective::allreduce_sum<u64>(val_count);
                return next_n_sequential(sum_ranks);
            } else {
                return next_n_parallel(val_count);
            }
        }

        /// quite explicit isn't it ?
        bool is_done() { return done; }

        /// check if all ranks have the same generator state
        bool all_ranks_are_in_sync() {
            Engine duplicated_eng = eng_global;
            u64 check_val         = duplicated_eng();

            std::vector<u64> collected_data{};
            shamalgs::collective::vector_allgatherv({check_val}, collected_data, MPI_COMM_WORLD);

            for (u64 val : collected_data) {
                if (val != check_val) {
                    return false;
                }
            }
            return true;
        }
    };
} // namespace shamalgs::collective

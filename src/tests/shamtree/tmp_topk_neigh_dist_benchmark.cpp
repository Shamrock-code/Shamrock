// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file tmp_topk_neigh_dist_benchmark.cpp
 * @brief Temporary benchmark of a top-k search on the neighbor distances in a GPU kernel.
 *
 * For each particle, find the distance of its k-th closest neighbor (k = 500) by looping on a
 * mock neighbor cache with shamrock::tree::ObjectCacheIterator, or infinity if it has less than
 * k neighbors. Three kernels are compared:
 * - heap: a per thread max-heap of the k smallest squared distances (private array of k values);
 * - radix select: a bitwise binary search on the bits of the squared distance, one pass on the
 *   neighbors per bit with a single counter (no array);
 * - count bisection: a bisection on the squared distance driven by the neighbor count, stopping
 *   as soon as the count hits k or k - 1 (a few scalars, no array);
 * - count bisection + guess: same as count bisection, with a cube law guess (count ~ r^3) instead
 *   of halving, falling back to halving when the guess is outside of the bracket;
 * - count bisection + guess (group): same as count bisection + guess, but each particle is
 *   handled by a work-group whose items split the neighbor loop, the count / min / max of each
 *   pass being reduced over the group;
 * - count guess + top-K: the bracket on the squared distance is narrowed with a cube law guess
 *   (count ~ r^3) falling back to bisection, until it holds at most K candidates, which are then
 *   collected in a small sorted array of K values.
 */

#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <algorithm>
#include <exception>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

    /// rank of the neighbor distance to find (1-indexed)
    constexpr u32 topk_k = 500;

    /// simple integer hash used to generate the mock neighbor indexes
    inline u64 hash_u64(u64 x) {
        x ^= x >> 33U;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33U;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33U;
        return x;
    }

    /// k-th smallest neighbor distance with a per thread max-heap of the k smallest r^2
    void topk_heap(
        sham::DeviceQueue &q,
        shamrock::tree::ObjectCache &pcache,
        sham::DeviceBuffer<f64_3> &pos,
        sham::DeviceBuffer<f64> &result,
        u32 npart) {
        sham::kernel_call(
            q,
            sham::MultiRef{pcache, pos},
            sham::MultiRef{result},
            npart,
            [](u32 id_a, auto ploop_ptrs, const f64_3 *__restrict r, f64 *__restrict out) {
                if (ploop_ptrs.cnt_neigh[id_a] < topk_k) {
                    out[id_a] = std::numeric_limits<f64>::infinity();
                    return;
                }

                shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                f64_3 xyz_a = r[id_a];
                f64 heap[topk_k];
                u32 n = 0;

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    f64_3 dr = xyz_a - r[id_b];
                    f64 r2   = sycl::dot(dr, dr);

                    if (n < topk_k) {
                        // sift up
                        u32 i   = n++;
                        heap[i] = r2;
                        while (i > 0) {
                            u32 p = (i - 1) / 2;
                            if (heap[p] >= heap[i]) {
                                break;
                            }
                            f64 tmp = heap[p];
                            heap[p] = heap[i];
                            heap[i] = tmp;
                            i       = p;
                        }
                    } else if (r2 < heap[0]) {
                        // replace the largest and sift down
                        heap[0] = r2;
                        u32 i   = 0;
                        while (true) {
                            u32 l   = 2 * i + 1;
                            u32 rr  = l + 1;
                            u32 big = i;
                            if (l < topk_k && heap[l] > heap[big]) {
                                big = l;
                            }
                            if (rr < topk_k && heap[rr] > heap[big]) {
                                big = rr;
                            }
                            if (big == i) {
                                break;
                            }
                            f64 tmp   = heap[big];
                            heap[big] = heap[i];
                            heap[i]   = tmp;
                            i         = big;
                        }
                    }
                });

                out[id_a] = sycl::sqrt(heap[0]);
            });
    }

    /// k-th smallest neighbor distance with a bitwise binary search on the bits of r^2
    /// (positive doubles sort like their bits as u64), one pass on the neighbors per bit
    void topk_radix_select(
        sham::DeviceQueue &q,
        shamrock::tree::ObjectCache &pcache,
        sham::DeviceBuffer<f64_3> &pos,
        sham::DeviceBuffer<f64> &result,
        u32 npart) {
        sham::kernel_call(
            q,
            sham::MultiRef{pcache, pos},
            sham::MultiRef{result},
            npart,
            [](u32 id_a, auto ploop_ptrs, const f64_3 *__restrict r, f64 *__restrict out) {
                if (ploop_ptrs.cnt_neigh[id_a] < topk_k) {
                    out[id_a] = std::numeric_limits<f64>::infinity();
                    return;
                }

                shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                f64_3 xyz_a = r[id_a];

                // largest key with less than k keys strictly below it = k-th smallest key
                // (the sign bit of r^2 is always 0, start below it)
                u64 res = 0;
                for (i32 bit = 62; bit >= 0; bit--) {
                    u64 cand = res | (u64(1) << u32(bit));
                    u32 less = 0;
                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        f64_3 dr = xyz_a - r[id_b];
                        u64 key  = sycl::bit_cast<u64>(sycl::dot(dr, dr));
                        less += (key < cand) ? 1 : 0;
                    });
                    if (less < topk_k) {
                        res = cand;
                    }
                }

                out[id_a] = sycl::sqrt(sycl::bit_cast<f64>(res));
            });
    }

    /// k-th smallest neighbor distance with a bisection on r^2 driven by the neighbor count:
    /// each pass counts c = #(r^2 <= mid) and keeps the largest r^2 <= mid and the smallest
    /// r^2 > mid, so that c == k or c == k - 1 gives the k-th value directly
    void topk_count_bisection(
        sham::DeviceQueue &q,
        shamrock::tree::ObjectCache &pcache,
        sham::DeviceBuffer<f64_3> &pos,
        sham::DeviceBuffer<f64> &result,
        u32 npart) {
        sham::kernel_call(
            q,
            sham::MultiRef{pcache, pos},
            sham::MultiRef{result},
            npart,
            [](u32 id_a, auto ploop_ptrs, const f64_3 *__restrict r, f64 *__restrict out) {
                if (ploop_ptrs.cnt_neigh[id_a] < topk_k) {
                    out[id_a] = std::numeric_limits<f64>::infinity();
                    return;
                }

                shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                f64_3 xyz_a = r[id_a];

                // invariant: #(r^2 <= lo) < k <= #(r^2 <= hi)
                f64 lo = -1;
                f64 hi = 0;
                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    f64_3 dr = xyz_a - r[id_b];
                    hi       = sycl::fmax(hi, sycl::dot(dr, dr));
                });

                f64 res = hi;
                while (true) {
                    f64 mid = 0.5 * (lo + hi);
                    if (!(lo < mid && mid < hi)) {
                        // no value strictly between lo and hi: the k-th is hi
                        res = hi;
                        break;
                    }

                    u32 c         = 0;
                    f64 max_below = -1;
                    f64 min_above = std::numeric_limits<f64>::infinity();
                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        f64_3 dr = xyz_a - r[id_b];
                        f64 r2   = sycl::dot(dr, dr);
                        if (r2 <= mid) {
                            c++;
                            max_below = sycl::fmax(max_below, r2);
                        } else {
                            min_above = sycl::fmin(min_above, r2);
                        }
                    });

                    if (c == topk_k) {
                        res = max_below;
                        break;
                    }
                    if (c == topk_k - 1) {
                        res = min_above;
                        break;
                    }

                    if (c > topk_k) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }

                out[id_a] = sycl::sqrt(res);
            });
    }

    /// same as topk_count_bisection, but the next trial r^2 is a cube law guess (count ~ r^3, so
    /// r^2 ~ count^(2/3)) instead of the middle of the bracket, falling back to the middle when the
    /// guess is outside of it
    void topk_count_bisection_guess(
        sham::DeviceQueue &q,
        shamrock::tree::ObjectCache &pcache,
        sham::DeviceBuffer<f64_3> &pos,
        sham::DeviceBuffer<f64> &result,
        u32 npart) {
        sham::kernel_call(
            q,
            sham::MultiRef{pcache, pos},
            sham::MultiRef{result},
            npart,
            [](u32 id_a, auto ploop_ptrs, const f64_3 *__restrict r, f64 *__restrict out) {
                u32 n_tot = ploop_ptrs.cnt_neigh[id_a];
                if (n_tot < topk_k) {
                    out[id_a] = std::numeric_limits<f64>::infinity();
                    return;
                }

                shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                f64_3 xyz_a = r[id_a];

                // invariant: #(r^2 <= lo) < k <= #(r^2 <= hi)
                f64 lo = -1;
                f64 hi = 0;
                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    f64_3 dr = xyz_a - r[id_b];
                    hi       = sycl::fmax(hi, sycl::dot(dr, dr));
                });

                f64 guess = hi * sycl::pow(f64(topk_k) / f64(n_tot), 2. / 3.);

                f64 res = hi;
                while (true) {
                    f64 mid = (lo < guess && guess < hi) ? guess : 0.5 * (lo + hi);
                    if (!(lo < mid && mid < hi)) {
                        // no value strictly between lo and hi: the k-th is hi
                        res = hi;
                        break;
                    }

                    u32 c         = 0;
                    f64 max_below = -1;
                    f64 min_above = std::numeric_limits<f64>::infinity();
                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        f64_3 dr = xyz_a - r[id_b];
                        f64 r2   = sycl::dot(dr, dr);
                        if (r2 <= mid) {
                            c++;
                            max_below = sycl::fmax(max_below, r2);
                        } else {
                            min_above = sycl::fmin(min_above, r2);
                        }
                    });

                    if (c == topk_k) {
                        res = max_below;
                        break;
                    }
                    if (c == topk_k - 1) {
                        res = min_above;
                        break;
                    }

                    if (c > topk_k) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }

                    // cube law guess for the next trial
                    guess = mid * sycl::pow(f64(topk_k) / f64(sycl::max(c, 1U)), 2. / 3.);
                }

                out[id_a] = sycl::sqrt(res);
            });
    }

    /// same as topk_count_bisection_guess, but one work-group of group_size items per particle:
    /// the items split the neighbor loop (coalesced reads of the neighbor indexes) and the count,
    /// max and min of each pass are reduced over the group, so every item follows the same
    /// bisection path. ObjectCacheIterator only loops serially, so the cache is read directly.
    /// The reductions are done in local memory: several sycl::reduce_over_group in a row in the
    /// same kernel are buggy on AdaptiveCpp (common scratch).
    template<u32 group_size>
    void topk_count_bisection_guess_group(
        sham::DeviceQueue &q,
        shamrock::tree::ObjectCache &pcache,
        sham::DeviceBuffer<f64_3> &pos,
        sham::DeviceBuffer<f64> &result,
        u32 npart) {
        sham::kernel_call_hndl(
            q,
            sham::MultiRef{pcache, pos},
            sham::MultiRef{result},
            npart * group_size,
            [](u32 nthreads, auto ploop_ptrs, const f64_3 *__restrict r, f64 *__restrict out) {
                static_assert(
                    (group_size & (group_size - 1)) == 0, "group_size must be a power of 2");

                return [=](sycl::handler &cgh) {
                    sycl::local_accessor<u32, 1> scratch_u32{group_size, cgh};
                    sycl::local_accessor<f64, 1> scratch_f64{group_size, cgh};

                    cgh.parallel_for(
                        sycl::nd_range<1>{nthreads, group_size}, [=](sycl::nd_item<1> item) {
                            u32 id_a = item.get_group_linear_id();
                            u32 lid  = item.get_local_id(0);

                            // tree reduction over the group in local memory, the last barrier
                            // ensures that every item has read the result before the scratch
                            // is reused by the next reduction
                            auto group_reduce = [&](auto &scratch, auto v, auto op) {
                                scratch[lid] = v;
                                item.barrier(sycl::access::fence_space::local_space);
                                for (u32 stride = group_size / 2; stride > 0; stride /= 2) {
                                    if (lid < stride) {
                                        scratch[lid] = op(scratch[lid], scratch[lid + stride]);
                                    }
                                    item.barrier(sycl::access::fence_space::local_space);
                                }
                                auto res = scratch[0];
                                item.barrier(sycl::access::fence_space::local_space);
                                return res;
                            };
                            auto op_sum = [](u32 a, u32 b) {
                                return a + b;
                            };
                            auto op_max = [](f64 a, f64 b) {
                                return sycl::fmax(a, b);
                            };
                            auto op_min = [](f64 a, f64 b) {
                                return sycl::fmin(a, b);
                            };

                            u32 n_tot = ploop_ptrs.cnt_neigh[id_a];
                            if (n_tot < topk_k) {
                                if (lid == 0) {
                                    out[id_a] = std::numeric_limits<f64>::infinity();
                                }
                                return;
                            }

                            const u32 *neigh
                                = ploop_ptrs.index_neigh_map + ploop_ptrs.scanned_cnt[id_a];

                            f64_3 xyz_a = r[id_a];

                            // invariant: #(r^2 <= lo) < k <= #(r^2 <= hi)
                            f64 lo     = -1;
                            f64 hi_loc = 0;
                            for (u32 j = lid; j < n_tot; j += group_size) {
                                f64_3 dr = xyz_a - r[neigh[j]];
                                hi_loc   = sycl::fmax(hi_loc, sycl::dot(dr, dr));
                            }
                            f64 hi = group_reduce(scratch_f64, hi_loc, op_max);

                            f64 guess = hi * sycl::pow(f64(topk_k) / f64(n_tot), 2. / 3.);

                            f64 res = hi;
                            while (true) {
                                f64 mid = (lo < guess && guess < hi) ? guess : 0.5 * (lo + hi);
                                if (!(lo < mid && mid < hi)) {
                                    // no value strictly between lo and hi: the k-th is hi
                                    res = hi;
                                    break;
                                }

                                u32 c_loc         = 0;
                                f64 max_below_loc = -1;
                                f64 min_above_loc = std::numeric_limits<f64>::infinity();
                                for (u32 j = lid; j < n_tot; j += group_size) {
                                    f64_3 dr = xyz_a - r[neigh[j]];
                                    f64 r2   = sycl::dot(dr, dr);
                                    if (r2 <= mid) {
                                        c_loc++;
                                        max_below_loc = sycl::fmax(max_below_loc, r2);
                                    } else {
                                        min_above_loc = sycl::fmin(min_above_loc, r2);
                                    }
                                }
                                u32 c         = group_reduce(scratch_u32, c_loc, op_sum);
                                f64 max_below = group_reduce(scratch_f64, max_below_loc, op_max);
                                f64 min_above = group_reduce(scratch_f64, min_above_loc, op_min);

                                if (c == topk_k) {
                                    res = max_below;
                                    break;
                                }
                                if (c == topk_k - 1) {
                                    res = min_above;
                                    break;
                                }

                                if (c > topk_k) {
                                    hi = mid;
                                } else {
                                    lo = mid;
                                }

                                // cube law guess for the next trial
                                guess
                                    = mid * sycl::pow(f64(topk_k) / f64(sycl::max(c, 1U)), 2. / 3.);
                            }

                            if (lid == 0) {
                                out[id_a] = sycl::sqrt(res);
                            }
                        });
                };
            });
    }

    /// k-th smallest neighbor distance by narrowing a bracket [lo2, hi2) on r^2 with a cube law
    /// guess (count ~ r^3, so r^2 ~ count^(2/3)), falling back to bisection, until it holds at most
    /// cand_max candidates, then collecting them in a small sorted array
    template<u32 cand_max>
    void topk_count_guess(
        sham::DeviceQueue &q,
        shamrock::tree::ObjectCache &pcache,
        sham::DeviceBuffer<f64_3> &pos,
        sham::DeviceBuffer<f64> &result,
        u32 npart) {
        sham::kernel_call(
            q,
            sham::MultiRef{pcache, pos},
            sham::MultiRef{result},
            npart,
            [](u32 id_a, auto ploop_ptrs, const f64_3 *__restrict r, f64 *__restrict out) {
                u32 n_tot = ploop_ptrs.cnt_neigh[id_a];
                if (n_tot < topk_k) {
                    out[id_a] = std::numeric_limits<f64>::infinity();
                    return;
                }

                shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                f64_3 xyz_a = r[id_a];

                f64 rmax2 = 0;
                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    f64_3 dr = xyz_a - r[id_b];
                    rmax2    = sycl::fmax(rmax2, sycl::dot(dr, dr));
                });

                // invariant: c_lo = #(r^2 < lo2) < k <= c_hi = #(r^2 < hi2)
                f64 lo2  = 0;
                f64 hi2  = sycl::nextafter(rmax2, std::numeric_limits<f64>::infinity());
                u32 c_lo = 0;
                u32 c_hi = n_tot;

                f64 r2 = rmax2 * sycl::pow(f64(topk_k) / f64(n_tot), 2. / 3.);

                bool all_equal = false;
                while (c_hi - c_lo > cand_max) {
                    if (!(lo2 < r2 && r2 < hi2)) {
                        r2 = 0.5 * (lo2 + hi2);
                    }
                    if (!(lo2 < r2 && r2 < hi2)) {
                        // nothing strictly between lo2 and hi2: all candidates equal lo2
                        all_equal = true;
                        break;
                    }

                    u32 c = 0;
                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        f64_3 dr = xyz_a - r[id_b];
                        c += (sycl::dot(dr, dr) < r2) ? 1 : 0;
                    });

                    if (c < topk_k) {
                        lo2  = r2;
                        c_lo = c;
                    } else {
                        hi2  = r2;
                        c_hi = c;
                    }

                    // cube law guess, the next loop falls back to bisection if it is outside
                    r2 = r2 * sycl::pow(f64(topk_k) / f64(sycl::max(c, 1U)), 2. / 3.);
                }

                if (all_equal) {
                    out[id_a] = sycl::sqrt(lo2);
                    return;
                }

                // collect the (at most cand_max) candidates in [lo2, hi2), sorted
                f64 cand[cand_max];
                u32 n_cand = 0;
                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    f64_3 dr = xyz_a - r[id_b];
                    f64 d2   = sycl::dot(dr, dr);
                    if (d2 >= lo2 && d2 < hi2) {
                        u32 i = n_cand++;
                        while (i > 0 && cand[i - 1] > d2) {
                            cand[i] = cand[i - 1];
                            i--;
                        }
                        cand[i] = d2;
                    }
                });

                out[id_a] = sycl::sqrt(cand[topk_k - c_lo - 1]);
            });
    }

    struct NeighCase {
        std::string name;
        /// fill the neighbor count of each particle
        void (*gen_counts)(std::mt19937 &eng, std::vector<u32> &counts);
    };

    void counts_60_exact(std::mt19937 &eng, std::vector<u32> &counts) {
        std::fill(counts.begin(), counts.end(), 60);
    }

    void counts_60_avg_outliers(std::mt19937 &eng, std::vector<u32> &counts) {
        std::uniform_int_distribution<u32> base(40, 80);
        std::uniform_real_distribution<f64> u(0, 1);
        for (auto &c : counts) {
            c = (u(eng) < 1e-4) ? 10000 : base(eng);
        }
    }

    void counts_150_outliers(std::mt19937 &eng, std::vector<u32> &counts) {
        std::uniform_real_distribution<f64> u(0, 1);
        for (auto &c : counts) {
            c = (u(eng) < 1e-4) ? 10000 : 150;
        }
    }

    void counts_150_half_thousands(std::mt19937 &eng, std::vector<u32> &counts) {
        std::uniform_int_distribution<u32> thousands(1000, 3000);
        std::uniform_real_distribution<f64> u(0, 1);
        for (auto &c : counts) {
            c = (u(eng) < 0.5) ? thousands(eng) : 150;
        }
    }

} // namespace

NEW_TEST(Benchmark, "shamtree/tmp/topk_neigh_dist:benchmark", 1) {

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    auto &q        = dev_sched->get_queue();

    auto &dev        = shambase::get_check_ref(shambase::get_check_ref(dev_sched->ctx).device);
    usize mem_budget = usize(0.8 * f64(dev.prop.global_mem_size));

    std::vector<NeighCase> cases{
        {"60 neigh exactly", counts_60_exact},
        {"60 neigh avg + 1e-4 outliers with 10k", counts_60_avg_outliers},
        {"150 neigh + 1e-4 outliers with 10k", counts_150_outliers},
        {"150 neigh + half with 1000-3000", counts_150_half_thousands},
    };

    std::vector<u32> part_counts{100000, 1000000, 3000000};

    shamcomm::logs::raw_ln(
        sham::format(
            "top-k neighbor distance benchmark, k = {}, device: {}", topk_k, dev.prop.name));

    for (auto &c : cases) {

        std::vector<f64> npart_done, t_heap, t_radix, t_bisect, t_bisect_guess, t_group32,
            t_group128, t_guess;

        for (u32 npart : part_counts) {

            std::mt19937 eng(0x111);

            std::vector<u32> counts(npart);
            c.gen_counts(eng, counts);

            u64 sum_neigh = 0;
            for (u32 cnt : counts) {
                sum_neigh += cnt;
            }

            usize mem_needed = sum_neigh * sizeof(u32)
                               + usize(npart) * (2 * sizeof(u32) + sizeof(f64_3) + 2 * sizeof(f64));

            if (sum_neigh >= std::numeric_limits<u32>::max() || mem_needed > mem_budget) {
                shamcomm::logs::raw_ln(
                    sham::format(
                        "  [{}] N = {:.0e} skipped: {} neighbors, needs {} (budget {})",
                        c.name,
                        f64(npart),
                        sum_neigh,
                        shambase::readable_sizeof(mem_needed),
                        shambase::readable_sizeof(mem_budget)));
                continue;
            }

            // the device can be shared with other processes, so the memory actually available
            // is unknown: skip the case if an allocation fails
            try {

                // random particle positions in the unit cube
                std::uniform_real_distribution<f64> u(0, 1);
                std::vector<f64_3> pos_vec(npart);
                for (auto &p : pos_vec) {
                    p = {u(eng), u(eng), u(eng)};
                }

                sham::DeviceBuffer<f64_3> pos(npart, dev_sched);
                pos.copy_from_stdvec(pos_vec);

                // mock neighbor cache with random neighbor indexes
                sham::DeviceBuffer<u32> neigh_count(npart, dev_sched);
                neigh_count.copy_from_stdvec(counts);

                shamrock::tree::ObjectCache pcache
                    = shamrock::tree::prepare_object_cache(std::move(neigh_count), npart);

                sham::kernel_call(
                    q,
                    sham::MultiRef{pcache.cnt_neigh, pcache.scanned_cnt},
                    sham::MultiRef{pcache.index_neigh_map},
                    npart,
                    [npart](
                        u32 id_a,
                        const u32 *__restrict cnt,
                        const u32 *__restrict offset,
                        u32 *__restrict neigh) {
                        u32 start = offset[id_a];
                        for (u32 j = 0; j < cnt[id_a]; j++) {
                            neigh[start + j]
                                = u32(hash_u64((u64(id_a) << 32U) | u64(j)) % u64(npart));
                        }
                    });

                sham::DeviceBuffer<f64> res_heap(npart, dev_sched);
                sham::DeviceBuffer<f64> res_radix(npart, dev_sched);
                sham::DeviceBuffer<f64> res_bisect(npart, dev_sched);
                sham::DeviceBuffer<f64> res_bisect_guess(npart, dev_sched);
                sham::DeviceBuffer<f64> res_group32(npart, dev_sched);
                sham::DeviceBuffer<f64> res_group128(npart, dev_sched);
                sham::DeviceBuffer<f64> res_guess(npart, dev_sched);

                // warmup (JIT compilation) + correctness check
                topk_heap(q, pcache, pos, res_heap, npart);
                topk_radix_select(q, pcache, pos, res_radix, npart);
                topk_count_bisection(q, pcache, pos, res_bisect, npart);
                topk_count_bisection_guess(q, pcache, pos, res_bisect_guess, npart);
                topk_count_bisection_guess_group<32>(q, pcache, pos, res_group32, npart);
                topk_count_bisection_guess_group<128>(q, pcache, pos, res_group128, npart);
                topk_count_guess<32>(q, pcache, pos, res_guess, npart);
                q.q.wait();

                std::vector<f64> h_heap         = res_heap.copy_to_stdvec();
                std::vector<f64> h_radix        = res_radix.copy_to_stdvec();
                std::vector<f64> h_bisect       = res_bisect.copy_to_stdvec();
                std::vector<f64> h_bisect_guess = res_bisect_guess.copy_to_stdvec();
                std::vector<f64> h_group32      = res_group32.copy_to_stdvec();
                std::vector<f64> h_group128     = res_group128.copy_to_stdvec();
                std::vector<f64> h_guess        = res_guess.copy_to_stdvec();

                u32 mismatch_radix        = 0;
                u32 mismatch_bisect       = 0;
                u32 mismatch_bisect_guess = 0;
                u32 mismatch_group32      = 0;
                u32 mismatch_group128     = 0;
                u32 mismatch_guess        = 0;
                for (u32 i = 0; i < npart; i++) {
                    mismatch_radix += (h_heap[i] == h_radix[i]) ? 0 : 1;
                    mismatch_bisect += (h_heap[i] == h_bisect[i]) ? 0 : 1;
                    mismatch_bisect_guess += (h_heap[i] == h_bisect_guess[i]) ? 0 : 1;
                    mismatch_group32 += (h_heap[i] == h_group32[i]) ? 0 : 1;
                    mismatch_group128 += (h_heap[i] == h_group128[i]) ? 0 : 1;
                    mismatch_guess += (h_heap[i] == h_guess[i]) ? 0 : 1;
                }
                REQUIRE_EQUAL_NAMED(
                    "heap and radix select agree (" + c.name + ")", mismatch_radix, 0_u32);
                REQUIRE_EQUAL_NAMED(
                    "heap and count bisection agree (" + c.name + ")", mismatch_bisect, 0_u32);
                REQUIRE_EQUAL_NAMED(
                    "heap and count bisection + guess agree (" + c.name + ")",
                    mismatch_bisect_guess,
                    0_u32);
                REQUIRE_EQUAL_NAMED(
                    "heap and count bisection + guess (group 32) agree (" + c.name + ")",
                    mismatch_group32,
                    0_u32);
                REQUIRE_EQUAL_NAMED(
                    "heap and count bisection + guess (group 128) agree (" + c.name + ")",
                    mismatch_group128,
                    0_u32);
                REQUIRE_EQUAL_NAMED(
                    "heap and count guess + top-K agree (" + c.name + ")", mismatch_guess, 0_u32);

                // reference on the host for a few particles with at least k neighbors
                auto host_cache = pcache.copy_to_host();
                u32 checked     = 0;
                for (u32 i = 0; i < npart && checked < 8; i++) {
                    if (counts[i] < topk_k) {
                        continue;
                    }
                    std::vector<f64> r2;
                    for (u32 j = 0; j < counts[i]; j++) {
                        f64_3 dr
                            = pos_vec[i]
                              - pos_vec[host_cache.index_neigh_map[host_cache.scanned_cnt[i] + j]];
                        r2.push_back(sycl::dot(dr, dr));
                    }
                    std::nth_element(r2.begin(), r2.begin() + (topk_k - 1), r2.end());
                    REQUIRE_EQUAL_NAMED(
                        "host reference (" + c.name + ")", h_heap[i], sycl::sqrt(r2[topk_k - 1]));
                    checked++;
                }

                f64 th = shambase::timeit(
                    [&]() {
                        topk_heap(q, pcache, pos, res_heap, npart);
                        q.q.wait();
                    },
                    3);
                f64 tr = shambase::timeit(
                    [&]() {
                        topk_radix_select(q, pcache, pos, res_radix, npart);
                        q.q.wait();
                    },
                    3);
                f64 tb = shambase::timeit(
                    [&]() {
                        topk_count_bisection(q, pcache, pos, res_bisect, npart);
                        q.q.wait();
                    },
                    3);
                f64 tbg = shambase::timeit(
                    [&]() {
                        topk_count_bisection_guess(q, pcache, pos, res_bisect_guess, npart);
                        q.q.wait();
                    },
                    3);
                f64 tg32 = shambase::timeit(
                    [&]() {
                        topk_count_bisection_guess_group<32>(q, pcache, pos, res_group32, npart);
                        q.q.wait();
                    },
                    3);
                f64 tg128 = shambase::timeit(
                    [&]() {
                        topk_count_bisection_guess_group<128>(q, pcache, pos, res_group128, npart);
                        q.q.wait();
                    },
                    3);
                f64 tg = shambase::timeit(
                    [&]() {
                        topk_count_guess<32>(q, pcache, pos, res_guess, npart);
                        q.q.wait();
                    },
                    3);

                shamcomm::logs::raw_ln(
                    sham::format(
                        "  [{}] N = {:.0e}, {} neighbors ({:.1f} avg): heap {:.4e} s, radix select "
                        "{:.4e} s, count bisection {:.4e} s, count bisection + guess {:.4e} s, "
                        "count bisection + guess (group 32) {:.4e} s, count bisection + guess "
                        "(group 128) {:.4e} s, count guess + top-32 {:.4e} s",
                        c.name,
                        f64(npart),
                        sum_neigh,
                        f64(sum_neigh) / f64(npart),
                        th,
                        tr,
                        tb,
                        tbg,
                        tg32,
                        tg128,
                        tg));

                npart_done.push_back(npart);
                t_heap.push_back(th);
                t_radix.push_back(tr);
                t_bisect.push_back(tb);
                t_bisect_guess.push_back(tbg);
                t_group32.push_back(tg32);
                t_group128.push_back(tg128);
                t_guess.push_back(tg);

            } catch (const std::exception &e) {
                shamcomm::logs::raw_ln(
                    sham::format(
                        "  [{}] N = {:.0e} skipped: {} neighbors ({}), allocation failed",
                        c.name,
                        f64(npart),
                        sum_neigh,
                        shambase::readable_sizeof(mem_needed)));
            }
        }

        auto &dset = shamtest::test_data().new_dataset(c.name);
        dset.add_data("Npart", npart_done);
        dset.add_data("time heap", t_heap);
        dset.add_data("time radix select", t_radix);
        dset.add_data("time count bisection", t_bisect);
        dset.add_data("time count bisection + guess", t_bisect_guess);
        dset.add_data("time count bisection + guess (group 32)", t_group32);
        dset.add_data("time count bisection + guess (group 128)", t_group128);
        dset.add_data("time count guess + top-32", t_guess);
    }
}

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <vector>

NEW_TEST(Unittest, "shambase/DistributedData::add_obj", 1) {
    using namespace shambase;

    {
        DistributedData<int> data{};
        auto it = data.add_obj(1, 42);
        REQUIRE(it->first == 1);
        REQUIRE(it->second == 42);
        REQUIRE(data.get_element_count() == 1);
    }

    {
        DistributedData<int> data{};
        data.add_obj(1, 42);
        REQUIRE_EXCEPTION_THROW(data.add_obj(1, 43), std::runtime_error);
        REQUIRE(data.get_element_count() == 1);
    }
}

namespace {

    /// ids of dd, filtered by pred, in the iteration order of dd
    template<class T, class Pred>
    std::vector<u64> ids_in_order(const shambase::DistributedData<T> &dd, Pred &&pred) {
        std::vector<u64> ret;
        dd.for_each([&](u64 id, const T &) {
            if (pred(id)) {
                ret.push_back(id);
            }
        });
        return ret;
    }

    struct DiffResult {
        std::vector<u64> missing, matching, extra;
    };

    template<class T1, class T2>
    DiffResult run_dd_diff(
        const shambase::DistributedData<T1> &dd, const shambase::DistributedData<T2> &ref) {
        DiffResult r;
        shambase::on_distributeddata_diff(
            dd,
            ref,
            [&](u64 id) {
                r.missing.push_back(id);
            },
            [&](u64 id) {
                r.matching.push_back(id);
            },
            [&](u64 id) {
                r.extra.push_back(id);
            });
        return r;
    }

    template<class T1>
    DiffResult run_ids_diff(const shambase::DistributedData<T1> &dd, const std::vector<u64> &ref) {
        DiffResult r;
        shambase::on_distributeddata_ids_diff(
            dd,
            ref,
            [&](u64 id) {
                r.missing.push_back(id);
            },
            [&](u64 id) {
                r.matching.push_back(id);
            },
            [&](u64 id) {
                r.extra.push_back(id);
            });
        return r;
    }

} // namespace

NEW_TEST(Unittest, "shambase/DistributedData::on_distributeddata_diff", 1) {
    using namespace shambase;

    // ids above 2^32 check that no u32 truncation happens
    const u64 big = (u64(1) << 33) + 1;

    DistributedData<int> dd{};
    for (u64 id : {u64(5), big, u64(1), u64(3)}) {
        dd.add_obj(id, int(id));
    }

    DistributedData<u32> ref{};
    for (u64 id : {u64(4), u64(3), u64(0), big, big + (u64(1) << 32)}) {
        ref.add_obj(id, u32(id));
    }

    auto in_dd = [&](u64 id) {
        return dd.has_key(id);
    };
    auto in_ref = [&](u64 id) {
        return ref.has_key(id);
    };

    DiffResult r = run_dd_diff(dd, ref);

    // missing and matching follow the iteration order of the reference, extra the one of dd
    REQUIRE(r.missing == ids_in_order(ref, [&](u64 id) {
                return !in_dd(id);
            }));
    REQUIRE(r.matching == ids_in_order(ref, in_dd));
    REQUIRE(r.extra == ids_in_order(dd, [&](u64 id) {
                return !in_ref(id);
            }));

    REQUIRE_EQUAL(r.missing.size(), size_t(3));
    REQUIRE_EQUAL(r.matching.size(), size_t(2));
    REQUIRE_EQUAL(r.extra.size(), size_t(2));

    // ensure_sizes pattern: callbacks mutate dd, which then holds exactly the reference ids
    on_distributeddata_diff(
        dd,
        ref,
        [&](u64 id) {
            dd.add_obj(id, -1);
        },
        [&](u64 id) {
            dd.get(id) = int(ref.get(id));
        },
        [&](u64 id) {
            dd.erase(id);
        });

    REQUIRE_EQUAL(dd.get_element_count(), ref.get_element_count());
    ref.for_each([&](u64 id, const u32 &) {
        REQUIRE(dd.has_key(id));
    });

    DiffResult r2 = run_dd_diff(dd, ref);
    REQUIRE(r2.missing.empty());
    REQUIRE(r2.extra.empty());
    REQUIRE(r2.matching == ref.get_ids());

    // empty cases
    DistributedData<int> empty{};
    DiffResult r3 = run_dd_diff(empty, ref);
    REQUIRE(r3.missing == ref.get_ids());
    REQUIRE(r3.matching.empty());
    REQUIRE(r3.extra.empty());

    DiffResult r4 = run_dd_diff(ref, empty);
    REQUIRE(r4.missing.empty());
    REQUIRE(r4.matching.empty());
    REQUIRE(r4.extra == ref.get_ids());
}

NEW_TEST(Unittest, "shambase/DistributedData::on_distributeddata_ids_diff", 1) {
    using namespace shambase;

    const u64 big = (u64(1) << 40) + 7;

    DistributedData<int> dd{};
    for (u64 id : {u64(2), u64(9), big}) {
        dd.add_obj(id, 0);
    }

    // ref ids in arbitrary order, with a repeated id
    std::vector<u64> ref_ids = {big, 7, 2, big, 11};

    DiffResult r = run_ids_diff(dd, ref_ids);

    // missing and matching follow ref_ids order, a repeated id matches only once
    REQUIRE(r.matching == (std::vector<u64>{big, 2}));
    REQUIRE(r.missing == (std::vector<u64>{7, big, 11}));
    REQUIRE(r.extra == (std::vector<u64>{9}));
}

NEW_TEST(Unittest, "shambase/DistributedData::reserve", 1) {
    using namespace shambase;

    DistributedData<int> dd{};
    dd.add_obj(3, 30);
    dd.reserve(1000);
    REQUIRE_EQUAL(dd.get_element_count(), u64(1));
    REQUIRE_EQUAL(dd.get(3), 30);

    for (u64 id = 0; id < 100; id++) {
        if (id != 3) {
            dd.add_obj(id, int(id) * 10);
        }
    }
    REQUIRE_EQUAL(dd.get_element_count(), u64(100));
    for (u64 id = 0; id < 100; id++) {
        REQUIRE_EQUAL(dd.get(id), int(id) * 10);
    }

    auto mapped = dd.map<u64>([](u64 id, int &v) {
        return u64(v) + id;
    });
    REQUIRE_EQUAL(mapped.get_element_count(), u64(100));
    std::vector<u64> mapped_ids = mapped.get_ids();
    std::vector<u64> dd_ids     = dd.get_ids();
    std::sort(mapped_ids.begin(), mapped_ids.end());
    std::sort(dd_ids.begin(), dd_ids.end());
    REQUIRE(mapped_ids == dd_ids);
}

namespace {

    /// ids of dd in the order visited by the non-const for_each_ordered
    template<class T>
    std::vector<u64> ordered_ids(shambase::DistributedData<T> &dd) {
        std::vector<u64> ret;
        dd.for_each_ordered([&](u64 id, T &) {
            ret.push_back(id);
        });
        return ret;
    }

    /// ids of dd in the order visited by the const for_each_ordered
    template<class T>
    std::vector<u64> ordered_ids_const(const shambase::DistributedData<T> &dd) {
        std::vector<u64> ret;
        dd.for_each_ordered([&](u64 id, const T &) {
            ret.push_back(id);
        });
        return ret;
    }

} // namespace

NEW_TEST(Unittest, "shambase/DistributedData::for_each_ordered", 1) {
    using namespace shambase;

    DistributedData<int> dd{};
    for (u64 id : {u64(5), u64(1), u64(9), u64(3), u64(1) << 40}) {
        dd.add_obj(id, int(id % 1000));
    }

    std::vector<u64> expected = {1, 3, 5, 9, u64(1) << 40};
    REQUIRE(ordered_ids(dd) == expected);
    REQUIRE(ordered_ids_const(dd) == expected);

    // the visited object is the one stored under that id
    dd.for_each_ordered([&](u64 id, int &v) {
        REQUIRE_EQUAL(v, int(id % 1000));
    });

    // a copy built through map() may iterate in a different order with for_each, but not with
    // for_each_ordered
    auto mapped = dd.map<int>([](u64, int &v) {
        return v * 2;
    });
    REQUIRE(ordered_ids(mapped) == expected);

    DistributedData<int> empty{};
    REQUIRE(ordered_ids(empty).empty());
}

NEW_TEST(Unittest, "shambase/DistributedData::for_each_ordered(cache invalidation)", 1) {
    using namespace shambase;

    DistributedData<int> dd{};
    for (u64 id : {5, 1, 9}) {
        dd.add_obj(id, 0);
    }
    REQUIRE(ordered_ids(dd) == (std::vector<u64>{1, 5, 9})); // fills the cache

    dd.add_obj(2, 0);
    REQUIRE(ordered_ids(dd) == (std::vector<u64>{1, 2, 5, 9}));

    dd.erase(5);
    REQUIRE(ordered_ids(dd) == (std::vector<u64>{1, 2, 9}));

    dd.get_native().emplace(0, 0);
    REQUIRE(ordered_ids(dd) == (std::vector<u64>{0, 1, 2, 9}));

    // reserve does not change the key set, the order stays the same
    dd.reserve(1000);
    REQUIRE(ordered_ids(dd) == (std::vector<u64>{0, 1, 2, 9}));

    // a copy carries a valid cache, and modifying it does not affect the original
    DistributedData<int> copy = dd;
    copy.erase(0);
    REQUIRE(ordered_ids(copy) == (std::vector<u64>{1, 2, 9}));
    REQUIRE(ordered_ids(dd) == (std::vector<u64>{0, 1, 2, 9}));

    dd.reset();
    REQUIRE(ordered_ids(dd).empty());
    dd.add_obj(4, 0);
    REQUIRE(ordered_ids(dd) == (std::vector<u64>{4}));
}

NEW_TEST(Unittest, "shambase/DistributedData::for_each_ordered(modified during iteration)", 1) {
    using namespace shambase;

    DistributedData<int> dd{};
    for (u64 id : {3, 1, 2}) {
        dd.add_obj(id, 0);
    }
    REQUIRE(ordered_ids(dd) == (std::vector<u64>{1, 2, 3})); // fills the cache

    // inserting while iterating invalidates the cache, the running loop keeps visiting the ids
    // it started with, and the next call sees the new key
    std::vector<u64> visited;
    dd.for_each_ordered([&](u64 id, int &) {
        visited.push_back(id);
        if (id == 1) {
            for (u64 new_id = 100; new_id < 200; new_id++) { // enough to force a rehash
                dd.add_obj(new_id, 0);
            }
        }
    });
    REQUIRE(visited == (std::vector<u64>{1, 2, 3}));

    std::vector<u64> after = ordered_ids(dd);
    REQUIRE_EQUAL(after.size(), size_t(103));
    REQUIRE(std::is_sorted(after.begin(), after.end()));
}

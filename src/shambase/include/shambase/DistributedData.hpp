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
 * @file DistributedData.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/exception.hpp"
#include "shambase/print.hpp"
#include "shambase/sets.hpp"
#include "shambase/string.hpp"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace shambase {

    /**
     * @brief Represents a collection of objects distributed across patches identified by a u64 id.
     *
     * This class provides methods for managing the distributed data collection,
     * including adding, removing, finding, and iterating over elements.
     * It also supports mapping the collection to a new type using a user-defined mapping function.
     *
     * @tparam T The type of the object in the collection.
     */
    template<class T>
    class DistributedData {

        std::unordered_map<u64, T> data; ///< The underlying collection.

        using iterator = typename std::unordered_map<u64, T>::iterator; ///< Iterator type.

        /// Cache of the ids sorted in increasing order, null when it must be rebuilt.
        /// Held through a shared_ptr so that an iteration in progress keeps its own copy alive
        /// if the collection is modified (and the cache invalidated) during the iteration.
        mutable std::shared_ptr<const std::vector<u64>> sorted_ids_cache;

        /// Invalidate the sorted ids cache, to be called on any potential change of the key set.
        inline void invalidate_sorted_ids() { sorted_ids_cache.reset(); }

        /// Returns the ids sorted in increasing order, rebuilding the cache if needed.
        inline std::shared_ptr<const std::vector<u64>> get_sorted_ids() const {
            if (!sorted_ids_cache) {
                std::vector<u64> ids = get_ids();
                std::sort(ids.begin(), ids.end());
                sorted_ids_cache = std::make_shared<const std::vector<u64>>(std::move(ids));
            }
            return sorted_ids_cache;
        }

        public:
        /**
         * @brief Returns the underlying collection.
         *
         * @return The underlying collection.
         *
         * @remark The sorted ids cache is invalidated by this call, but not by later
         * modifications made through the returned reference while it is held.
         */
        inline std::unordered_map<u64, T> &get_native() {
            invalidate_sorted_ids();
            return data;
        }

        /**
         * @brief Adds a new object to the collection.
         *
         * @param id The id of the patch the object belongs to.
         * @param obj The object to add.
         *
         * @return An iterator pointing to the inserted object.
         *
         * @throw If the key already exist.
         */
        inline iterator add_obj(u64 id, T &&obj) {

            invalidate_sorted_ids();
            std::pair<iterator, bool> ret = data.emplace(id, std::forward<T>(obj));

            if (!ret.second) {
                throw make_except_with_loc<std::runtime_error>("the key already exist");
            }

            return ret.first;
        }

        /**
         * @brief Reserve storage for at least `count` objects.
         *
         * Sets the number of buckets of the underlying hash map so that `count` objects can be
         * inserted without further rehashing. The content of the collection is unchanged, but
         * the rehash may change the order in which for_each visits the objects (still
         * deterministic). for_each_ordered is not affected.
         *
         * @param count The number of objects the collection is expected to hold.
         */
        inline void reserve(u64 count) { data.reserve(count); }

        /**
         * @brief Removes an object from the collection.
         *
         * @param id The id of the patch the object belongs to.
         */
        inline void erase(u64 id) {
            invalidate_sorted_ids();
            data.erase(id);
        }

        /**
         * @brief Applies a function to each object in the collection.
         *
         * @param f A function taking two arguments: the id of the patch the object belongs to,
         * and a non-const reference to the object itself.
         *
         * @remark The iteration order is deterministic but unspecified: do not rely on the
         * objects being visited in ascending id order, nor in insertion order.
         */
        inline void for_each(std::function<void(u64, T &)> &&f) {
            for (auto &[id, obj] : data) {
                f(id, obj);
            }
        }

        /// Same as for_each but for const objects
        inline void for_each(std::function<void(u64, const T &)> &&f) const {
            for (auto &[id, obj] : data) {
                f(id, obj);
            }
        }

        /**
         * @brief Applies a function to each object in the collection, in increasing id order.
         *
         * Unlike for_each, the iteration order does not depend on the insertion history or on the
         * standard library implementation. Use it when the order matters (e.g. concatenation of
         * per-patch data or floating-point reductions that must be reproducible).
         *
         * The sorted ids are cached and only recomputed after the key set may have changed
         * (add_obj, erase, reset, get_native). The cache is not thread safe: do not call this
         * concurrently on the same object, even through a const reference.
         *
         * @param f A function taking two arguments: the id of the patch the object belongs to,
         * and a non-const reference to the object itself.
         */
        inline void for_each_ordered(std::function<void(u64, T &)> &&f) {
            std::shared_ptr<const std::vector<u64>> ids = get_sorted_ids();
            for (u64 id : *ids) {
                f(id, data.at(id));
            }
        }

        /// Same as for_each_ordered but for const objects
        inline void for_each_ordered(std::function<void(u64, const T &)> &&f) const {
            std::shared_ptr<const std::vector<u64>> ids = get_sorted_ids();
            for (u64 id : *ids) {
                f(id, data.at(id));
            }
        }

        /**
         * @brief Returns a vector of all the ids of the objects in the collection.
         *
         * @return A vector of all the ids of the objects in the collection.
         */
        inline std::vector<u64> get_ids() const {
            std::vector<u64> ids;
            ids.reserve(data.size());
            for (const auto &[id, _] : data) {
                ids.push_back(id);
            }
            return ids;
        }

        /**
         * @brief Finds an object in the collection.
         *
         * @param id The id of the patch the object belongs to.
         *
         * @return An iterator pointing to the object if found, or data.end() otherwise.
         */
        inline auto find(u64 id) { return data.find(id); }

        /**
         * @brief Returns an iterator pointing to the end of the collection.
         *
         * @return An iterator pointing to the end of the collection.
         */
        inline auto not_found() { return data.end(); }

        /**
         * @brief Returns a reference to an object in the collection.
         *
         * @param id The id of the patch the object belongs to.
         *
         * @return A reference to the object.
         *
         * @throw If the object is not found.
         */
        inline T &get(u64 id) {
            try {
                return data.at(id);
            } catch (std::out_of_range &) {

                std::vector<u64> id_list;

                for_each([&](u64 id, T &) {
                    id_list.push_back(id);
                });

                throw make_except_with_loc<std::runtime_error>(sham::format(
                    "The querried id {} does not exist, current id list is {}", id, id_list));
            }
        }

        /// Const variant of get
        inline const T &get(u64 id) const {
            try {
                return data.at(id);
            } catch (std::out_of_range &) {

                std::vector<u64> id_list;

                for_each([&](u64 id, const T &) {
                    id_list.push_back(id);
                });

                throw make_except_with_loc<std::runtime_error>(sham::format(
                    "The querried id {} does not exist, current id list is {}", id, id_list));
            }
        }

        /**
         * @brief Checks if an object exists in the collection.
         *
         * @param id The id of the patch the object belongs to.
         *
         * @return True if the object is found, false otherwise.
         */
        inline bool has_key(u64 id) const { return (data.find(id) != data.end()); }

        /**
         * @brief Returns the number of elements in the collection.
         *
         * @return The number of elements in the collection.
         */
        inline u64 get_element_count() const { return data.size(); }

        /**
         * @brief Returns true if the collection is empty.
         *
         * @return True if the collection is empty, false otherwise.
         */
        inline bool is_empty() const { return data.empty(); }

        /**
         * @brief Prints all the objects in the collection to the logger.
         *
         * The format string is passed to fmt::format to format each object, so
         * the syntax is the same as fmt::format. The object is passed as the
         * second argument to fmt::format.
         *
         * Example:
         * \code{.cpp}
         * DistributedData<int> data;
         * data.add_obj(0, 42);
         * data.add_obj(1, 24);
         * data.print_data("{:d}"); // will print "0 -> 42" and "1 -> 24"
         * \endcode
         *
         * @tparam Tf Types of the format string placeholders.
         * @param fmt The format string.
         */
        template<typename... Tf>
        inline void print_data(fmt::format_string<Tf...> fmt) const {
            for_each([&](u64 id_patch, const T &ref) {
                shambase::println(sham::format("{} -> {}", id_patch, sham::format(fmt, ref)));
            });
        }

        /**
         * @brief Apply a function to all objects in the collection and return
         *        a new collection containing the results.
         *
         * The `map()` function applies the given function to each object in the
         * collection and stores the result in a new collection. The function
         * is passed the id of the object and a reference to the object as
         * arguments.
         *
         * Example:
         * \code{.cpp}
         * DistributedData<int> data;
         * data.add_obj(0, 42);
         * data.add_obj(1, 24);
         *
         * // a DistributedData<float> with doubled values from the input
         * auto mapped = data.map<float>([](u64 id, int& val) { return val * 2.; });
         * \endcode
         *
         * @tparam Tmap Type of the objects in the returned collection.
         * @param map_func The function to apply to each object.
         * @return A new collection containing the results of the map function.
         */
        template<class Tmap>
        inline DistributedData<Tmap> map(std::function<Tmap(u64, T &)> map_func) {
            DistributedData<Tmap> ret;
            ret.reserve(data.size());

            for_each([&](u64 id, T &ref) {
                ret.add_obj(id, map_func(id, ref));
            });

            return ret;
        }

        /// Same as map but for const objects
        template<class Tmap>
        inline DistributedData<Tmap> map(std::function<Tmap(u64, const T &)> map_func) const {
            DistributedData<Tmap> ret;
            ret.reserve(data.size());

            for_each([&](u64 id, const T &ref) {
                ret.add_obj(id, map_func(id, ref));
            });

            return ret;
        }

        /**
         * @brief Reset the collection to its initial state
         *
         * All objects in the collection are removed.
         */
        inline void reset() {
            invalidate_sorted_ids();
            // swap with a fresh map rather than clear() to also release the buckets, so that the
            // iteration order after a reset does not depend on what the collection held before
            std::unordered_map<u64, T>{}.swap(data);
        }
    };

    /**
     * @brief Compare two distributed data and apply callbacks based on the difference
     *
     * This is the same as on_distributeddata_diff but where one of the distributed data is
     * represented by a vector of its ids.
     *
     * This function compares the two given distributed data and applies callbacks
     * based on the difference.
     *
     * The callbacks are:
     *   - `func_missing`: called for each id present in the reference but not in `dd`
     *   - `func_match`: called for each id present in both `dd` and the reference
     *   - `func_extra`: called for each id present in `dd` but not in the reference
     *
     * @param dd the distributed data to compare
     * @param ref_ids the reference ids
     * @param func_missing the callback for missing ids
     * @param func_match the callback for matching ids
     * @param func_extra the callback for extra ids
     */
    template<class T1, class FuncMissing, class FuncMatch, class FuncExtra>
    inline void on_distributeddata_ids_diff(
        const shambase::DistributedData<T1> &dd,
        const std::vector<u64> &ref_ids,
        FuncMissing &&func_missing,
        FuncMatch &&func_match,
        FuncExtra &&func_extra) {

        std::vector<u64> missing;
        std::vector<u64> matching;
        std::vector<u64> extra;

        // Only used for membership tests, its iteration order is never observed.
        std::unordered_set<u64> ref_seen;
        ref_seen.reserve(ref_ids.size());

        // missing and matching follow the order of ref_ids, a repeated reference id is
        // matched at most once and reported as missing afterwards
        for (u64 id : ref_ids) {
            bool first_occurrence = ref_seen.insert(id).second;
            if (first_occurrence && dd.has_key(id)) {
                matching.push_back(id);
            } else {
                missing.push_back(id);
            }
        }

        // extra follows the iteration order of dd
        dd.for_each([&](u64 id, const T1 &) {
            if (ref_seen.find(id) == ref_seen.end()) {
                extra.push_back(id);
            }
        });

        for (auto id : missing) {
            func_missing(id);
        }

        for (auto id : matching) {
            func_match(id);
        }

        for (auto id : extra) {
            func_extra(id);
        }
    }

    /**
     * @brief Compare two distributed data and apply callbacks based on the difference
     *
     * This function compares the two given distributed data and applies callbacks
     * based on the difference.
     *
     * The callbacks are:
     *   - `func_missing`: called for each id present in the reference but not in `dd`
     *   - `func_match`: called for each id present in both `dd` and the reference
     *   - `func_extra`: called for each id present in `dd` but not in the reference
     *
     * @param dd the distributed data to compare
     * @param reference the reference distributed data
     * @param func_missing the callback for missing ids
     * @param func_match the callback for matching ids
     * @param func_extra the callback for extra ids
     */
    template<class T1, class T2, class FuncMissing, class FuncMatch, class FuncExtra>
    inline void on_distributeddata_diff(
        const shambase::DistributedData<T1> &dd,
        const shambase::DistributedData<T2> &reference,
        FuncMissing &&func_missing,
        FuncMatch &&func_match,
        FuncExtra &&func_extra) {

        std::vector<u64> missing;
        std::vector<u64> matching;
        std::vector<u64> extra;

        // missing and matching follow the iteration order of reference
        reference.for_each([&](u64 id, const T2 &) {
            if (dd.has_key(id)) {
                matching.push_back(id);
            } else {
                missing.push_back(id);
            }
        });

        // extra follows the iteration order of dd
        dd.for_each([&](u64 id, const T1 &) {
            if (!reference.has_key(id)) {
                extra.push_back(id);
            }
        });

        for (auto id : missing) {
            func_missing(id);
        }

        for (auto id : matching) {
            func_match(id);
        }

        for (auto id : extra) {
            func_extra(id);
        }
    }

} // namespace shambase

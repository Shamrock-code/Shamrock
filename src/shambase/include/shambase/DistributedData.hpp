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
#include <functional>
#include <map>
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

        std::map<u64, T> data; ///< The underlying collection.

        using iterator = typename std::map<u64, T>::iterator; ///< Iterator type.

        public:
        /**
         * @brief Returns the underlying collection.
         *
         * @return The underlying collection.
         */
        inline std::map<u64, T> &get_native() { return data; }

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

            std::pair<iterator, bool> ret = data.emplace(id, std::forward<T>(obj));

            if (!ret.second) {
                throw make_except_with_loc<std::runtime_error>("the key already exist");
            }

            return ret.first;
        }

        /**
         * @brief Removes an object from the collection.
         *
         * @param id The id of the patch the object belongs to.
         */
        inline void erase(u64 id) { data.erase(id); }

        /**
         * @brief Applies a function to each object in the collection.
         *
         * @param f A function taking two arguments: the id of the patch the object belongs to,
         * and a non-const reference to the object itself.
         *
         * @remark The function is applied to each object in the order of insertion.
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

                throw make_except_with_loc<std::runtime_error>(format(
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

                throw make_except_with_loc<std::runtime_error>(format(
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
        inline bool has_key(u64 id) { return (data.find(id) != data.end()); }

        /**
         * @brief Returns the number of elements in the collection.
         *
         * @return The number of elements in the collection.
         */
        inline u64 get_element_count() { return data.size(); }

        /**
         * @brief Returns true if the collection is empty.
         *
         * @return True if the collection is empty, false otherwise.
         */
        inline bool is_empty() { return data.empty(); }

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
                shambase::println(
                    shambase::format("{} -> {}", id_patch, shambase::format(fmt, ref)));
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

            for_each([&](u64 id, T &ref) {
                ret.add_obj(id, map_func(id, ref));
            });

            return ret;
        }

        /// Same as map but for const objects
        template<class Tmap>
        inline DistributedData<Tmap> map(std::function<Tmap(u64, const T &)> map_func) const {
            DistributedData<Tmap> ret;

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
        inline void reset() { data.clear(); }
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
    template<class T1, class FuncMatch, class FuncMissing, class FuncExtra>
    inline void on_distributeddata_ids_diff(
        const shambase::DistributedData<T1> &dd,
        const std::vector<u64> &ref_ids,
        FuncMatch &&func_missing,
        FuncMissing &&func_match,
        FuncExtra &&func_extra) {

        std::vector<u64> dd_ids;

        dd.for_each([&](u32 id, const T1 &data) {
            dd_ids.push_back(id);
        });

        std::vector<u64> missing;
        std::vector<u64> matching;
        std::vector<u64> extra;

        shambase::set_diff(dd_ids, ref_ids, missing, matching, extra);

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
    template<class T1, class T2, class FuncMatch, class FuncMissing, class FuncExtra>
    inline void on_distributeddata_diff(
        const shambase::DistributedData<T1> &dd,
        const shambase::DistributedData<T2> &reference,
        FuncMatch &&func_missing,
        FuncMissing &&func_match,
        FuncExtra &&func_extra) {

        std::vector<u64> ref_ids;

        reference.for_each([&](u32 id, const T2 &data) {
            ref_ids.push_back(id);
        });

        shambase::on_distributeddata_ids_diff(dd, ref_ids, func_missing, func_match, func_extra);
    }

} // namespace shambase

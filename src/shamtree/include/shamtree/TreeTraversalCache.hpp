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
 * @file TreeTraversalCache.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/DistributedData.hpp"
#include "shamalgs/numeric.hpp"
#include "shambackends/sycl.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace shamrock::tree {

    class ObjectCacheHandler {

        shambase::DistributedData<ObjectCache> cache;
        shambase::DistributedData<HostObjectCache> cache_offload;

        std::function<ObjectCache(u64)> generator;
        std::deque<u64> last_device_builds;

        u64 max_device_memsize;
        u64 current_device_memsize = 0;
        u64 current_host_memsize   = 0;

        public:
        ObjectCacheHandler(u64 max_memsize, std::function<ObjectCache(u64)> &&generator)
            : max_device_memsize(max_memsize), generator(generator) {}

        private:
        inline bool offload_exist(u64 id) { return cache_offload.has_key(id); }

        inline bool cache_entry_exist(u64 id) { return cache.has_key(id); }

        inline ObjectCache pop_offload(u64 id) {
            ObjectCache tmp = ObjectCache::build_from_host(cache_offload.get(id));
            cache_offload.erase(id);
            current_host_memsize -= tmp.get_memsize();
            return tmp;
        }

        inline HostObjectCache pop_cache(u64 id) {
            ObjectCache &tmp = cache.get(id);

            HostObjectCache ret = tmp.copy_to_host();

            current_device_memsize -= tmp.get_memsize();
            cache.erase(id);

            return ret;
        }

        inline void push_offload(u64 id, HostObjectCache &&c) {
            current_host_memsize += c.get_memsize();
            cache_offload.add_obj(id, std::forward<HostObjectCache>(c));
        }

        inline void push_cache(u64 id, ObjectCache &&c) {
            current_device_memsize += c.get_memsize();
            cache.add_obj(id, std::forward<ObjectCache>(c));
            last_device_builds.push_back(id);
        }

        inline void offload_entry(u64 id) {
            HostObjectCache tmp = pop_cache(id);
            push_offload(id, std::move(tmp));
        }

        inline ObjectCache load_or_build_cache(u64 id) {
            if (cache_offload.has_key(id)) {
                return pop_offload(id);
            } else {
                return generator(id);
            }
        }

        /**
         * @brief pop oldest entry in the cache
         *
         */
        inline void offload_oldest() {

            bool successfull_pop = false;
            do {

                if (cache.is_empty()) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "the cache is empty no entry can be popped");
                }

                u64 last_id = last_device_builds.front();
                last_device_builds.pop_front();

                if (cache.has_key(last_id)) {

                    offload_entry(last_id);

                    successfull_pop = true;
                    shamlog_debug_ln(
                        "ObjectCacheHandler",
                        "offloaded cache for id =",
                        last_id,
                        "cachesize =",
                        shambase::readable_sizeof(current_device_memsize),
                        "hostsize =",
                        shambase::readable_sizeof(current_host_memsize));
                }

            } while (!successfull_pop);
        }

        /**
         * @brief push a new entry in the cache, and check that enough space is available
         *
         * @param id
         * @param c
         */
        inline void push_new(u64 id, ObjectCache &&c) {
            u64 add_sz = c.get_memsize();

            if (add_sz + current_device_memsize > max_device_memsize) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "their is no space within the imposed limit, try freeing some space in the "
                    "cache, or increase the size limit");
            }

            push_cache(id, std::forward<ObjectCache>(c));
            last_device_builds.push_back(id);
        }

        public:
        /**
         * @brief build a new entry in the cache and free enough
         * entry to make space for it if necessary
         *
         * @param id
         */
        inline void build_cache(u64 id) {

            ObjectCache new_cache = load_or_build_cache(id);

            u64 new_sz = new_cache.get_memsize();

            while (new_sz + current_device_memsize > max_device_memsize) {
                if (cache.is_empty()) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "their no space left to allocate a cache, try with smaller objects, or "
                        "increase the size limit");
                }
                logger::warn_ln(
                    "ObjectCacheHandler",
                    "The cache is too small, or the objects too big, some caches will "
                    "have to be recomputed on the fly");
                offload_oldest();
            }

            push_new(id, std::move(new_cache));
            shamlog_debug_ln(
                "ObjectCacheHandler",
                "built cache for id =",
                id,
                "cachesize =",
                shambase::readable_sizeof(current_device_memsize),
                "hostsize =",
                shambase::readable_sizeof(current_host_memsize));
        }

        /**
         * @brief Get a cache entry and build it if it is not already in the cache
         *
         * @param id
         * @return ObjectCache&
         */
        inline ObjectCache &get_cache(u64 id) {
            if (cache.has_key(id)) {
                return cache.get(id);
            } else {
                build_cache(id);
                return cache.get(id);
            }
        }

        inline void preload(u64 id) { get_cache(id); }

        inline void reset() {
            last_device_builds.clear();
            cache.reset();
            cache_offload.reset();
            current_device_memsize = 0;
            current_host_memsize   = 0;
        }
    };

} // namespace shamrock::tree

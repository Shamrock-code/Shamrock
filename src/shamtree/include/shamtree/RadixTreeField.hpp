// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file RadixTreeField.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamsys/NodeInstance.hpp"

template<class T>
class RadixTreeField {

    RadixTreeField(u32 nvar, u32 cnt)
        : radix_tree_field_buf(std::make_unique<sycl::buffer<T>>(cnt * nvar)), nvar(nvar) {}

    public:
    u32 nvar;
    std::unique_ptr<sycl::buffer<T>> radix_tree_field_buf;

    RadixTreeField() = default;

    RadixTreeField(u32 nvar, std::unique_ptr<sycl::buffer<T>> radix_tree_field_buf)
        : nvar(nvar), radix_tree_field_buf(std::move(radix_tree_field_buf)) {}

    static RadixTreeField<T> make_empty(u32 nvar, u32 cnt) { return RadixTreeField<T>(nvar, cnt); }

    RadixTreeField(RadixTreeField<T> &src, sycl::buffer<u32> &id_extract_field)
        : radix_tree_field_buf(
              std::make_unique<sycl::buffer<T>>(id_extract_field.size() * src.nvar)),
          nvar(src.nvar) {
        // cut new field according to the id map

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor acc_curr{*src.radix_tree_field_buf, cgh, sycl::read_only};
            sycl::accessor acc_other{*radix_tree_field_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_idxs{id_extract_field, cgh, sycl::read_only};

            u32 nvar_loc = nvar;

            cgh.parallel_for(sycl::range<1>{id_extract_field.size()}, [=](sycl::item<1> i) {
                const u32 gid = i.get_linear_id();

                const u32 idx_extr = acc_idxs[gid] * nvar_loc;
                const u32 idx_push = gid * nvar_loc;

                for (u32 a = 0; a < nvar_loc; a++) {
                    acc_other[idx_push + a] = acc_curr[idx_extr + a];
                }
            });
        });
    }
};

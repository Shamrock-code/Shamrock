// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TreeStructure.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamtree/TreeStructure.hpp"
#include "shamalgs/details/memory/memory.hpp"
#include "shamalgs/primitives/equals.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/kernels/karras_alg.hpp"

namespace shamrock::tree {

    template<class u_morton>
    void TreeStructure<u_morton>::build(
        sycl::queue &queue, u32 _internal_cell_count, sycl::buffer<u_morton> &morton_buf) {

        if (!(_internal_cell_count < morton_buf.size())) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "morton buf must be at least with size() greater than internal_cell_count");
        }

        internal_cell_count = _internal_cell_count;

        buf_lchild_id   = std::make_unique<sycl::buffer<u32>>(internal_cell_count);
        buf_rchild_id   = std::make_unique<sycl::buffer<u32>>(internal_cell_count);
        buf_lchild_flag = std::make_unique<sycl::buffer<u8>>(internal_cell_count);
        buf_rchild_flag = std::make_unique<sycl::buffer<u8>>(internal_cell_count);
        buf_endrange    = std::make_unique<sycl::buffer<u32>>(internal_cell_count);

        sycl_karras_alg(
            queue,
            internal_cell_count,
            morton_buf,
            *buf_lchild_id,
            *buf_rchild_id,
            *buf_lchild_flag,
            *buf_rchild_flag,
            *buf_endrange);

        one_cell_mode = false;
    }

    template<class u_morton>
    void TreeStructure<u_morton>::build_one_cell_mode() {
        internal_cell_count = 1;
        buf_lchild_id       = std::make_unique<sycl::buffer<u32>>(internal_cell_count);
        buf_rchild_id       = std::make_unique<sycl::buffer<u32>>(internal_cell_count);
        buf_lchild_flag     = std::make_unique<sycl::buffer<u8>>(internal_cell_count);
        buf_rchild_flag     = std::make_unique<sycl::buffer<u8>>(internal_cell_count);
        buf_endrange        = std::make_unique<sycl::buffer<u32>>(internal_cell_count);

        {
            sycl::host_accessor rchild_id{*buf_rchild_id, sycl::write_only, sycl::no_init};
            sycl::host_accessor lchild_id{*buf_lchild_id, sycl::write_only, sycl::no_init};
            sycl::host_accessor rchild_flag{*buf_rchild_flag, sycl::write_only, sycl::no_init};
            sycl::host_accessor lchild_flag{*buf_lchild_flag, sycl::write_only, sycl::no_init};
            sycl::host_accessor endrange{*buf_endrange, sycl::write_only, sycl::no_init};

            lchild_id[0]   = 0;
            rchild_id[0]   = 1;
            lchild_flag[0] = 1;
            rchild_flag[0] = 1;

            endrange[0] = 1;
        }
        one_cell_mode = true;
    }

    template<class u_morton>
    TreeStructure<u_morton>::TreeStructure(const TreeStructure<u_morton> &other)
        : internal_cell_count(other.internal_cell_count), one_cell_mode(other.one_cell_mode),
          buf_lchild_id(shamalgs::memory::duplicate(
              shamsys::instance::get_compute_queue(),
              other.buf_lchild_id)), // size = internal
          buf_rchild_id(shamalgs::memory::duplicate(
              shamsys::instance::get_compute_queue(),
              other.buf_rchild_id)), // size = internal
          buf_lchild_flag(shamalgs::memory::duplicate(
              shamsys::instance::get_compute_queue(),
              other.buf_lchild_flag)), // size = internal
          buf_rchild_flag(shamalgs::memory::duplicate(
              shamsys::instance::get_compute_queue(),
              other.buf_rchild_flag)), // size = internal
          buf_endrange(shamalgs::memory::duplicate(
              shamsys::instance::get_compute_queue(),
              other.buf_endrange)) // size = internal
    {}

    template<class u_morton>
    bool TreeStructure<u_morton>::operator==(const TreeStructure<u_morton> &rhs) const {
        bool cmp = true;

        cmp = cmp && (internal_cell_count == rhs.internal_cell_count);

        cmp = cmp
              && shamalgs::primitives::equals(
                  shamsys::instance::get_compute_queue(),
                  *buf_lchild_id,
                  *rhs.buf_lchild_id,
                  internal_cell_count);
        cmp = cmp
              && shamalgs::primitives::equals(
                  shamsys::instance::get_compute_queue(),
                  *buf_rchild_id,
                  *rhs.buf_rchild_id,
                  internal_cell_count);
        cmp = cmp
              && shamalgs::primitives::equals(
                  shamsys::instance::get_compute_queue(),
                  *buf_lchild_flag,
                  *rhs.buf_lchild_flag,
                  internal_cell_count);
        cmp = cmp
              && shamalgs::primitives::equals(
                  shamsys::instance::get_compute_queue(),
                  *buf_rchild_flag,
                  *rhs.buf_rchild_flag,
                  internal_cell_count);
        cmp = cmp
              && shamalgs::primitives::equals(
                  shamsys::instance::get_compute_queue(),
                  *buf_endrange,
                  *rhs.buf_endrange,
                  internal_cell_count);
        cmp = cmp && (one_cell_mode == rhs.one_cell_mode);

        return cmp;
    }

    template class TreeStructure<u32>;
    template class TreeStructure<u64>;

} // namespace shamrock::tree

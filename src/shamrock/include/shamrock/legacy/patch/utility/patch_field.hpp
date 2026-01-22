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
 * @file patch_field.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/collective/exchanges.hpp"
#include "shambackends/sycl.hpp"
#include <vector>

template<class type>
class BufferedPField {
    public:
    sycl::buffer<type> buf_local;
    sycl::buffer<type> buf_global;
};

namespace legacy {

    /**
     * @brief Define a field attached to a patch (exemple: FMM multipoles, hmax in SPH)
     *
     * @tparam type type of object to store
     */
    template<class type>
    class PatchField {
        public:
        using T = type;

        std::vector<type> local_nodes_value;

        std::vector<type> global_values;

        inline void build_global(MPI_Datatype &dtype) {
            shamalgs::collective::vector_allgatherv(
                local_nodes_value, dtype, global_values, dtype, MPI_COMM_WORLD);
        }

        inline BufferedPField<type> get_buffers() {
            return BufferedPField<type>{
                sycl::buffer<type>(local_nodes_value.data(), local_nodes_value.size()),
                sycl::buffer<type>(global_values.data(), global_values.size()),
            };
        }
    };

} // namespace legacy

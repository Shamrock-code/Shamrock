// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_aliases.hpp"
#include "shambase/sycl.hpp"

namespace shamalgs {


    class IndexedKeyMultimap {

        public:

        u32 unique_key_count;
        u32 pair_count;

        // [key_id_map[i] , key_id_map[i+1]] gives all the index in a key i
        std::unique_ptr<sycl::buffer<u32>> key_to_val;

        // give the linked object 
        std::unique_ptr<sycl::buffer<u32>> val_id_map;

        

    };


} // namespace shamalgs
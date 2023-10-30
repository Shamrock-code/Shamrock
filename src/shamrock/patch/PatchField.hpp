// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchField.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/DistributedData.hpp"
#include "shambackends/sycl.hpp"
#include <memory>
namespace shamrock::patch {

    template<class T>
    class PatchField{public:

        shambase::DistributedData<T> field_all;

        PatchField(shambase::DistributedData<T> && field_all) : field_all(field_all){}

        T & get(u64 id){
            return field_all.get(id);
        }
    };

    template<class T>
    class PatchtreeField{public:

        std::unique_ptr<sycl::buffer<T>> internal_buf;

        inline void reset(){
            internal_buf.reset();
        }

        inline void allocate(u32 size){
            internal_buf = std::make_unique<sycl::buffer<T>>(size);
        }


    };
}
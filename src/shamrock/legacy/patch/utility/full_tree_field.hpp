// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once
// plan for a tree that can combine both a radix tree field and a patch tree

/**
 * @file full_tree_field.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "patch_field.hpp"

template<class T, class DeviceTree>
class FullTreeField{public:

    using TreeField_t = typename DeviceTree::template RadixTreeField<T>;

    legacy::PatchField<T> patch_field;
    std::unordered_map<u64, std::unique_ptr<TreeField_t>> patch_tree_fields;

    //inline FullTreeField(
    //    PatchField<T> && pf, 
    //    std::unordered_map<u64, std::unique_ptr<TreeField_t>> && ptf) :
    //    patch_field(pf), patch_tree_fields(ptf)
    //{}


    class BufferedFullTreeField{public:
        BufferedPField<T> patch_field;
        std::unordered_map<u64, std::unique_ptr<TreeField_t>> & patch_tree_fields;
    };

    inline BufferedFullTreeField get_buffers(){
        return BufferedFullTreeField{
            patch_field.get_buffers(),
            patch_tree_fields
        };
    }
};
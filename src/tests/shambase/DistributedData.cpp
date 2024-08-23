// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambase/DistributedData::add_obj", distributedDatatests_add_obj, 1) {
    using namespace shambase;

    {
        DistributedData<int> data{};
        auto it = data.add_obj(1, 42);
        _Assert(it->first == 1);
        _Assert(it->second == 42);
        _Assert(data.get_element_count() == 1);
    }

    {
        DistributedData<int> data{};
        data.add_obj(1, 42);
        _Assert_throw(data.add_obj(1, 43), std::runtime_error);
        _Assert(data.get_element_count() == 1);
    }
}

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Test.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "Test.hpp"

#include "shamsys/NodeInstance.hpp"
#include "shambase/exception.hpp"

namespace shamtest::details {

    TestResult current_test{Unittest, "", -1};

    TestResult Test::run() {

        using namespace shamsys::instance;

        if (node_count != -1) {
            if (node_count != world_size) {
                throw shambase::throw_with_loc<std::runtime_error>(
                    "trying to run a test with wrong number of nodes"
                );
            }
        }

        current_test = TestResult{type, name, world_rank};
        
        try{
            test_functor();
        } catch (const std::exception &e) {
            current_test.asserts.assert_add_comment("exception_thrown", false, e.what());
        }
        

        return std::move(current_test);
    }

} // namespace shamtest::details
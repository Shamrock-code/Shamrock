// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyISPHSetupNode.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings for shammodels::sph::modules::ISPHSetupNode
 */

#include "shambindings/pybindaliases.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "sph_pybindings.hpp"
#include <memory>

namespace shammodels::sph::pybindings {

    void add_isph_setup_node(py::module &msph) {
        py::class_<
            shammodels::sph::modules::ISPHSetupNode,
            std::shared_ptr<shammodels::sph::modules::ISPHSetupNode>>(msph, "ISPHSetupNode")
            .def("get_dot", [](std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> &self) {
                return self->get_dot();
            });
    }

} // namespace shammodels::sph::pybindings

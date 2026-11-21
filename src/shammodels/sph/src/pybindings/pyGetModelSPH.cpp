// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyGetModelSPH.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python binding of the get_Model_SPH factory function
 */

#include "shambase/exception.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shammodels/sph/Model.hpp"
#include "sph_pybindings.hpp"
#include <memory>
#include <stdexcept>
#include <variant>

namespace shammodels::sph::pybindings {

    void add_get_model_sph(py::module &m) {
        using namespace shammodels::sph;

        using VariantSPHModelBind = std::variant<
            std::unique_ptr<Model<f64_3, shammath::M4>>,
            std::unique_ptr<Model<f64_3, shammath::M6>>,
            std::unique_ptr<Model<f64_3, shammath::M8>>,
            std::unique_ptr<Model<f64_3, shammath::C2>>,
            std::unique_ptr<Model<f64_3, shammath::C4>>,
            std::unique_ptr<Model<f64_3, shammath::C6>>>;

        m.def(
            "get_Model_SPH",
            [](ShamrockCtx &ctx,
               const std::string &vector_type,
               const std::string &kernel) -> VariantSPHModelBind {
                VariantSPHModelBind ret;

                if (vector_type == "f64_3" && kernel == "M4") {
                    ret = std::make_unique<Model<f64_3, shammath::M4>>(ctx);
                } else if (vector_type == "f64_3" && kernel == "M6") {
                    ret = std::make_unique<Model<f64_3, shammath::M6>>(ctx);
                } else if (vector_type == "f64_3" && kernel == "M8") {
                    ret = std::make_unique<Model<f64_3, shammath::M8>>(ctx);
                } else if (vector_type == "f64_3" && kernel == "C2") {
                    ret = std::make_unique<Model<f64_3, shammath::C2>>(ctx);
                } else if (vector_type == "f64_3" && kernel == "C4") {
                    ret = std::make_unique<Model<f64_3, shammath::C4>>(ctx);
                } else if (vector_type == "f64_3" && kernel == "C6") {
                    ret = std::make_unique<Model<f64_3, shammath::C6>>(ctx);
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown combination of representation and kernel");
                }

                return ret;
            },
            py::kw_only(),
            py::arg("context"),
            py::arg("vector_type"),
            py::arg("sph_kernel"));
    }

} // namespace shammodels::sph::pybindings

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
 * @file ScalarEdgeSerializable.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/pre_main_call.hpp"
#include "shambase/string.hpp"
#include "shambase/type_name_info.hpp"
#include "nlohmann/json_fwd.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/solvergraph/IEdge.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <memory>
#include <stdexcept>

namespace shamrock::solvergraph {

    template<class T>
    class ScalarEdgeSerializable : public ScalarEdge<T>, public JsonSerializable {
        public:
        using ScalarEdge<T>::ScalarEdge;
        using ScalarEdge<T>::value;

        void to_json(nlohmann::json &j) const override {
            j = nlohmann::json{
                {"type", type_name()},
                {"value", value},
                {"label", this->get_label()},
                {"tex_symbol", this->get_raw_tex_symbol()}};
        };

        static ScalarEdgeSerializable<T> from_json(const nlohmann::json &j) {
            std::string type = j.at("type");

            if (type != type_name_static()) {
                throw shambase::make_except_with_loc<std::runtime_error>(shambase::format(
                    "error when deserializing ScalarEdgeSerializable, expected type info "
                    "\"{}\" but got \"{}\"",
                    type_name_static(),
                    type));
            }

            std::string label      = j.at("label").get<std::string>();
            std::string tex_symbol = j.at("tex_symbol").get<std::string>();

            auto tmp  = ScalarEdgeSerializable<T>(label, tex_symbol);
            tmp.value = j.at("value").get<T>();
            return tmp;
        };

        inline static std::string type_name_static() {
            return "ScalarEdgeSerializable<" + shambase::get_type_name<T>() + ">";
        }

        std::string type_name() const override { return type_name_static(); };
    };

} // namespace shamrock::solvergraph

PRE_MAIN_FUNCTION_CALL([&]() {
    using T = shamrock::solvergraph::ScalarEdgeSerializable<f64>;
    shamrock::solvergraph::JsonSerializable_registry::instance().register_type<T>(
        T::type_name_static());
})

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
 * @file IDataEdgeSerializable.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/pre_main_call.hpp"
#include "shambase/type_name_info.hpp"
#include "nlohmann/json_fwd.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/JsonSerializable.hpp"
#include <stdexcept>

namespace shamrock::solvergraph {

    template<class T>
    class IDataEdgeSerializable : public IDataEdge<T>, public JsonSerializable {
        public:
        using IDataEdge<T>::IDataEdge;
        using IDataEdge<T>::data;

        void _impl_to_json(nlohmann::json &j) const override {
            j["data"]       = data;
            j["label"]      = this->get_label();
            j["tex_symbol"] = this->get_raw_tex_symbol();
        };

        static IDataEdgeSerializable<T> from_json(const nlohmann::json &j) {
            std::string label      = j.at("label").get<std::string>();
            std::string tex_symbol = j.at("tex_symbol").get<std::string>();

            auto tmp = IDataEdgeSerializable<T>(label, tex_symbol);
            tmp.data = j.at("data").get<T>();
            return tmp;
        };

        inline static std::string type_name_static() {
            return "IDataEdgeSerializable<" + shambase::get_type_name<T>() + ">";
        }

        std::string type_name() const override { return type_name_static(); };
    };

} // namespace shamrock::solvergraph

PRE_MAIN_FUNCTION_CALL([&]() {
    using T        = shamrock::solvergraph::IDataEdgeSerializable<f64>;
    auto &instance = shamrock::solvergraph::JsonSerializable_registry::instance();
    if (!instance.is_type_registered(T::type_name_static())) {
        instance.register_type<T>(T::type_name_static());
    }
})

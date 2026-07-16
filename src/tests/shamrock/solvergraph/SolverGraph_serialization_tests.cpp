// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "nlohmann/json.hpp"
#include "shamrock/solvergraph/IEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamrock/solvergraph/SolverGraph.hpp"
#include "shamtest/shamtest.hpp"
#include "shambase/type_name_info.hpp"

struct JsonSerializable {
    virtual ~JsonSerializable() {};

    virtual void to_json(nlohmann::json &j)         = 0;
    virtual void from_json(const nlohmann::json &j) = 0;

    virtual std::string type_name() = 0;
};

inline bool json_serializable_edge_constraint(
    const std::shared_ptr<shamrock::solvergraph::IEdge> &edge) {
    // check that the edge can be cross-casted to JsonSerializable
    return bool(std::dynamic_pointer_cast<JsonSerializable>(edge));
};

template<class T>
class ScalarEdgeSerializable : public shamrock::solvergraph::ScalarEdge<T>,
                               public JsonSerializable {
    public:
    using shamrock::solvergraph::ScalarEdge<T>::ScalarEdge;
    using shamrock::solvergraph::ScalarEdge<T>::value;

    virtual void to_json(nlohmann::json &j) {
        j = nlohmann::json{
            {"type", type_name()},
            {"value", value},
            {"label", this->get_label()},
            {"tex_symbol", this->get_raw_tex_symbol()}};
    };

    virtual void from_json(const nlohmann::json &j) {
        std::string type = j.at("type");

        if (type != type_name()) {
            throw shambase::make_except_with_loc<std::runtime_error>(shambase::format(
                "error when deserializing ScalarEdgeSerializable, expected type info "
                "\"{}\" but got \"{}\"",
                type_name(),
                type));
        }

        value = j.at("value").get<T>();
    };

    inline static std::string type_name_static() {
        return "ScalarEdgeSerializable<" + shambase::get_type_name<T>() + ">";
    }

    virtual std::string type_name() { return type_name_static(); };
};

NEW_TEST(Unittest, "shamrock/solvergraph/SolverGraph_json_serialization", 1) {}

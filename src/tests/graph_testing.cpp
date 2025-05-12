// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file fmmTests.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/WithUUID.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/string.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace shamrock::solvergraph;

////////////////////////////////////////////////////////////////////////////////////////////////////
// test api
////////////////////////////////////////////////////////////////////////////////////////////////////

class Field : public IDataEdge {
    std::string name;
    std::string texsymbol;

    public:
    std::vector<float> field_data = {};
    Field(std::string name, std::string texsymbol)
        : name(name), texsymbol(texsymbol), field_data() {}

    inline std::string _impl_get_dot_label() const { return name; };
    inline std::string _impl_get_tex_symbol() const { return texsymbol; };
};

class RhoOp : public INode {
    public:
    void _impl_evaluate_internal() {
        auto &h    = get_ro_edge<Field>(0);
        auto &mass = get_ro_edge<Field>(1);
        auto &rho  = get_rw_edge<Field>(0);

        rho.field_data.resize(h.field_data.size());
        for (int i = 0; i < h.field_data.size(); i++) {
            rho.field_data[i] = mass.field_data[i] / h.field_data[i];
        }
    }

    void _impl_reset_internal() {}

    inline void set_inputs(std::shared_ptr<Field> h, std::shared_ptr<Field> mass) {
        __internal_set_ro_edges({h, mass});
    }
    inline void set_outputs(std::shared_ptr<Field> rho) { __internal_set_rw_edges({rho}); }

    inline std::string _impl_get_label() { return "Compute rho"; }

    inline std::string _impl_get_tex() {
        std::string h    = get_ro_edge<Field>(0).get_tex_symbol() + "_a";
        std::string mass = get_ro_edge<Field>(1).get_tex_symbol() + "_a";
        std::string rho  = get_rw_edge<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + rho + " = \\frac{" + mass + "}{" + h + "}\\]";
    }
};
class FieldLoader : public INode {
    public:
    std::vector<float> mirror;

    FieldLoader(std::vector<float> input) : mirror(input) {}
    FieldLoader(float val, int n) : mirror(n, val) {}

    void _impl_evaluate_internal() {
        auto &field      = get_rw_edge<Field>(0);
        field.field_data = mirror;
    }

    void _impl_reset_internal() {
        mirror.resize(0);
        auto &field      = get_rw_edge<Field>(0);
        field.field_data = {};
    }

    inline void set_inputs() { __internal_set_ro_edges({}); }
    inline void set_outputs(std::shared_ptr<Field> field) { __internal_set_rw_edges({field}); }

    inline std::string _impl_get_label() { return "Loader"; }

    inline std::string _impl_get_tex() {
        std::string field = get_rw_edge<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + field + " \\leftarrow field\\]";
    }
};

class FieldWriter : public INode {
    public:
    std::vector<float> &write_back;

    FieldWriter(std::vector<float> &write_back) : write_back(write_back) {}

    void _impl_evaluate_internal() {
        auto &field = get_ro_edge<Field>(0);
        write_back  = field.field_data;
    }

    void _impl_reset_internal() { write_back = {}; }

    inline void set_inputs(std::shared_ptr<Field> field) { __internal_set_ro_edges({field}); }
    inline void set_outputs() { __internal_set_rw_edges({}); }

    inline std::string _impl_get_label() { return "Writer"; }

    inline std::string _impl_get_tex() {
        std::string field = get_ro_edge<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + field + " \\rightarrow field\\]";
    }
};

class OperationSequence : public INode {
    std::vector<std::shared_ptr<INode>> nodes;

    public:
    OperationSequence(std::vector<std::shared_ptr<INode>> nodes) : nodes(nodes) {
        if (nodes.size() == 0) {
            shambase::throw_with_loc<std::invalid_argument>(
                "OperationSequence must have at least one node");
        }
    }
    void _impl_evaluate_internal() {
        for (auto &node : nodes) {
            node->evaluate();
        }
    }

    void _impl_reset_internal() {
        for (int i = nodes.size() - 1; i >= 0; i--) {
            nodes[i]->reset();
        }
    }

    std::string _impl_get_label() { return "Sequence"; }

    std::string _impl_get_dot_graph_partial() {
        std::stringstream ss;

        ss << "subgraph cluster_" + std::to_string(get_uuid()) + " {\n";
        for (auto &node : nodes) {
            ss << node->get_dot_graph_partial();
        }

        for (int i = 0; i < nodes.size() - 1; i++) {
            ss << nodes[i]->get_dot_graph_node_end() << " -> "
               << nodes[i + 1]->get_dot_graph_node_start() << " [weight=3];\n";
        }

        ss << shambase::format("label = \"{}\";\n", _impl_get_label());
        ss << "}\n";

        return ss.str();
    };

    inline virtual std::string _impl_get_dot_graph_node_start() {
        return nodes[0]->get_dot_graph_node_start();
    }
    inline virtual std::string _impl_get_dot_graph_node_end() {
        return nodes[nodes.size() - 1]->get_dot_graph_node_end();
    }

    std::string _impl_get_tex() {
        std::stringstream ss;
        for (auto &node : nodes) {
            ss << node->get_tex_partial() << "\n";
        }
        return ss.str();
    }
};

class Looper : public INode {
    std::shared_ptr<INode> to_loop;
    int n;

    public:
    Looper(std::shared_ptr<INode> to_loop, int n) : to_loop(to_loop), n(n) {}

    void _impl_evaluate_internal() {
        for (int i = 0; i < n; i++) {
            to_loop->evaluate();
        }
    }

    void _impl_reset_internal() { to_loop->reset(); }

    std::string _impl_get_label() { return "Looper"; }

    std::string _impl_get_dot_graph_partial() {
        std::stringstream ss;

        ss << "subgraph cluster_" + std::to_string(get_uuid()) + " {\n";

        std::string loop_node = _impl_get_dot_graph_node_start();
        ss << loop_node + " [label=\"\", shape=point];\n";

        ss << to_loop->get_dot_graph_partial();

        ss << to_loop->get_dot_graph_node_end() << " -> " << loop_node
           << shambase::format(" [label=\"loop {} times\",weight=0];\n", n);
        ss << loop_node << " -> " << to_loop->get_dot_graph_node_start()
           << shambase::format(" [weight=1];\n", n);

        ss << shambase::format("label = \"{}\";\n", _impl_get_label());
        ss << "}\n";

        return ss.str();
    };

    inline virtual std::string _impl_get_dot_graph_node_start() {
        return shambase::format("loop_{}", to_loop->get_uuid());
    }
    inline virtual std::string _impl_get_dot_graph_node_end() {
        return to_loop->get_dot_graph_node_end();
    }

    std::string _impl_get_tex() {
        std::stringstream ss;
        ss << "Loop " << n << " times: {\n";
        ss << to_loop->get_tex_partial() << "\n";
        ss << "}\n";
        return ss.str();
    }
};

TestStart(Unittest, "tmp_graph_test", tmp_graph_test, 1) {

    int n                         = 3;
    float m_source                = 3;
    std::vector<float> h_source   = {1., 1. / 2., 1. / 4.};
    std::vector<float> rho_result = {};

    std::shared_ptr<FieldLoader> h_load = std::make_shared<FieldLoader>(h_source);
    std::shared_ptr<Field> h            = std::make_shared<Field>("h", "h");
    h_load->set_outputs(h);

    std::shared_ptr<FieldLoader> mass_load = std::make_shared<FieldLoader>(m_source, n);
    std::shared_ptr<Field> mass            = std::make_shared<Field>("m", "m");
    mass_load->set_outputs(mass);

    std::shared_ptr<FieldWriter> rho_write = std::make_shared<FieldWriter>(rho_result);
    std::shared_ptr<Field> rho             = std::make_shared<Field>("rho", "\\rho");
    rho_write->set_inputs(rho);

    std::shared_ptr<RhoOp> rho_op = std::make_shared<RhoOp>();
    rho_op->set_inputs(h, mass);
    rho_op->set_outputs(rho);

    std::shared_ptr<OperationSequence> load_seq
        = std::make_shared<OperationSequence>(OperationSequence{{h_load, mass_load}});

    std::shared_ptr<Looper> looper = std::make_shared<Looper>(rho_op, 2);

    std::shared_ptr<OperationSequence> seq
        = std::make_shared<OperationSequence>(OperationSequence{{load_seq, looper, rho_write}});

    seq->evaluate();
    std::cout << seq->get_dot_graph() << std::endl;
    std::cout << seq->get_tex() << std::endl;

    std::cout << "Rho result" << std::endl;
    std::cout << "size " << rho_result.size() << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << i << " " << rho_result[i] << std::endl;
    }
}

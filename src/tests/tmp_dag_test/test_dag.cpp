// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamdag/INode.hpp"
#include "shamdag/graph2dot.hpp"
#include "shamdag/graph2tex.hpp"
#include "shamdag/shamdag.hpp"
#include "shamtest/shamtest.hpp"
#include <unordered_map>
#include <iostream>
#include <memory>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <vector>

class Field : public IDataEdge {
    std::string name;
    std::string texsymbol;

    public:
    std::vector<float> field_data = {};
    Field(std::string name, std::string texsymbol)
        : name(name), texsymbol(texsymbol), field_data() {}

    inline std::string _impl_get_label() { return name; };
    inline std::string _impl_get_tex_symbol() { return texsymbol; };
};

class RhoOp : public INode {
    public:
    void _impl_evaluate_internal() {
        auto &h    = get_input<Field>(0);
        auto &mass = get_input<Field>(1);
        auto &rho  = get_output<Field>(0);

        rho.field_data.resize(h.field_data.size());
        for (int i = 0; i < h.field_data.size(); i++) {
            rho.field_data[i] = mass.field_data[i] / h.field_data[i];
        }
    }

    void _impl_reset_internal() { get_output<Field>(0).field_data = {}; }

    inline void set_inputs(std::shared_ptr<Field> h, std::shared_ptr<Field> mass) {
        __internal_set_inputs({h, mass});
    }
    inline void set_outputs(std::shared_ptr<Field> rho) { __internal_set_outputs({rho}); }

    inline std::string _impl_get_label() { return "Compute rho"; }
    inline std::string _impl_get_node_tex() {
        std::string h    = get_input<Field>(0).get_tex_symbol() + "_a";
        std::string mass = get_input<Field>(1).get_tex_symbol() + "_a";
        std::string rho  = get_output<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + rho + " = \\frac{" + mass + "}{" + h + "}\\]";
    }
};
class FieldLoader : public INode {
    public:
    std::vector<float> mirror;

    FieldLoader(std::vector<float> input) : mirror(input) {}
    FieldLoader(float val, int n) : mirror(n, val) {}

    void _impl_evaluate_internal() {
        auto &field      = get_output<Field>(0);
        field.field_data = mirror;
    }

    void _impl_reset_internal() {
        mirror.resize(0);
        auto &field      = get_output<Field>(0);
        field.field_data = {};
    }

    inline void set_inputs() { __internal_set_inputs({}); }
    inline void set_outputs(std::shared_ptr<Field> field) { __internal_set_outputs({field}); }

    inline std::string _impl_get_label() { return "Loader"; }
    inline std::string _impl_get_node_tex() {
        std::string field = get_output<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + field + " \\leftarrow field\\]";
    }
};

class FieldWriter : public INode {
    public:
    std::vector<float> &write_back;

    FieldWriter(std::vector<float> &write_back) : write_back(write_back) {}

    void _impl_evaluate_internal() {
        auto &field = get_input<Field>(0);
        write_back  = field.field_data;
    }

    void _impl_reset_internal() { write_back = {}; }

    inline void set_inputs(std::shared_ptr<Field> field) { __internal_set_inputs({field}); }
    inline void set_outputs() { __internal_set_outputs({}); }

    inline std::string _impl_get_label() { return "Writer"; }
    inline std::string _impl_get_node_tex() {
        std::string field = get_input<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + field + " \\rightarrow field\\]";
    }
};

TestStart(Unittest, "dag_stuff_testing", dag_stuff_testing, 1) {

    int n                         = 3;
    float m_source                = 3;
    std::vector<float> h_source   = {1., 1. / 2., 1. / 4.};
    std::vector<float> rho_result = {};

    std::shared_ptr<FieldLoader> h_load    = std::make_shared<FieldLoader>(h_source);
    std::shared_ptr<FieldLoader> mass_load = std::make_shared<FieldLoader>(m_source, n);
    std::shared_ptr<FieldWriter> rho_write = std::make_shared<FieldWriter>(rho_result);

    std::shared_ptr<Field> h    = std::make_shared<Field>("h", "h");
    std::shared_ptr<Field> mass = std::make_shared<Field>("m", "m");
    std::shared_ptr<Field> rho  = std::make_shared<Field>("rho", "\\rho");

    std::shared_ptr<RhoOp> rho_op = std::make_shared<RhoOp>();

    h_load->set_outputs(h);
    mass_load->set_outputs(mass);
    rho_op->set_inputs(h, mass);
    rho_op->set_outputs(rho);
    rho_write->set_inputs(rho);

    std::cout << get_node_graph_tex({rho_write}) << std::endl;
    std::cout << get_node_dot_graph({rho_write}) << std::endl;

    rho_write->evaluate();
    std::cout << "Rho result" << std::endl;
    std::cout << "size " << rho_result.size() << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << i << " " << rho_result[i] << std::endl;
    }
}

TestStart(Unittest, "dag_stuff_testing_reset", dag_stuff_testingreset, 1) {

    int n                         = 3;
    float m_source                = 3;
    std::vector<float> h_source   = {1., 1. / 2., 1. / 4.};
    std::vector<float> rho_result = {};
    std::vector<float> P_result   = {};
    std::vector<float> dt_result  = {};

    std::shared_ptr<FieldLoader> h_load    = std::make_shared<FieldLoader>(h_source);
    std::shared_ptr<FieldLoader> mass_load = std::make_shared<FieldLoader>(m_source, n);
    std::shared_ptr<FieldWriter> rho_write = std::make_shared<FieldWriter>(rho_result);

    std::shared_ptr<Field> h    = std::make_shared<Field>("h", "h");
    std::shared_ptr<Field> mass = std::make_shared<Field>("m", "m");
    std::shared_ptr<Field> rho  = std::make_shared<Field>("rho", "\\rho");

    std::shared_ptr<RhoOp> rho_op = std::make_shared<RhoOp>();

    h_load->set_outputs(h);
    mass_load->set_outputs(mass);
    rho_op->set_inputs(h, mass);
    rho_op->set_outputs(rho);
    rho_write->set_inputs(rho);

    std::cout << get_node_graph_tex({rho_write}) << std::endl;
    std::cout << get_node_dot_graph({rho_write}) << std::endl;

    auto print_node_states = [&]() {
        std::cout << "- h_load    eval=" << h_load->is_evaluated() << std::endl;
        std::cout << "- mass_load eval=" << mass_load->is_evaluated() << std::endl;
        std::cout << "- rho_op    eval=" << rho_op->is_evaluated() << std::endl;
        std::cout << "- rho_write eval=" << rho_write->is_evaluated() << std::endl;
    };

    print_node_states();
    rho_write->evaluate();
    std::cout << "Rho result" << std::endl;
    std::cout << "size " << rho_result.size() << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << i << " " << rho_result[i] << std::endl;
    }

    print_node_states();

    std::cout << " -- retry evaluate :" << std::endl;
    rho_write->evaluate();
    print_node_states();
    std::cout << " -- reset write :" << std::endl;
    rho_write->reset_up();
    print_node_states();
    std::cout << " -- retry evaluate :" << std::endl;
    rho_write->evaluate();
    print_node_states();
    std::cout << " -- reset only rhoop and up :" << std::endl;
    rho_op->reset_up();
    print_node_states();
    std::cout << " -- retry evaluate :" << std::endl;
    rho_write->evaluate();
    print_node_states();

    std::cout << " -- reset connected :" << std::endl;
    rho_op->reset_connected();
    print_node_states();
}

class Value : public IDataEdge {
    std::string name;
    std::string texsymbol;

    public:
    float value = {};
    Value(std::string name, std::string texsymbol) : name(name), texsymbol(texsymbol), value(0) {}

    inline std::string _impl_get_label() { return name; };
    inline std::string _impl_get_tex_symbol() { return texsymbol; };
};

class Setter : public INode {
    public:
    float &write_back;

    Setter(float &write_back) : write_back(write_back) {}

    void _impl_evaluate_internal() {
        auto &field = get_input<Value>(0);
        write_back  = field.value;
    }

    void _impl_reset_internal() { write_back = {}; }

    inline void set_inputs(std::shared_ptr<Value> value) { __internal_set_inputs({value}); }
    inline void set_outputs() { __internal_set_outputs({}); }

    inline std::string _impl_get_label() { return "Setter"; }
    inline std::string _impl_get_node_tex() {
        std::string field = get_input<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + field + " \\rightarrow field\\]";
    }
};

class Getter : public INode {
    public:
    float &input;

    Getter(float &input) : input(input) {}

    void _impl_evaluate_internal() {
        auto &field = get_output<Value>(0);
        field.value = input;
    }

    void _impl_reset_internal() {
        auto &field = get_output<Value>(0);
        field.value = {};
    }

    inline void set_inputs() { __internal_set_inputs({}); }
    inline void set_outputs(std::shared_ptr<Field> field) { __internal_set_outputs({field}); }

    inline std::string _impl_get_label() { return "Getter"; }
    inline std::string _impl_get_node_tex() {
        std::string field = get_output<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + field + " \\leftarrow field\\]";
    }
};

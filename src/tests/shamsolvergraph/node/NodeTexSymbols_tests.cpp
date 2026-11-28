// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamtest/shamtest.hpp"
#include <optional>
#include <string>

namespace {

    // ro and rw edges interleaved on purpose: the by-name accessors must not depend on the
    // declaration order matching the ro / rw slot order
#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::IDataEdge<u32>, in_a)                                              \
    X_RW(shamrock::solvergraph::IDataEdge<u32>, out_a)                                             \
    X_RO(shamrock::solvergraph::IDataEdge<u32>, in_b)                                              \
    X_RW(shamrock::solvergraph::IDataEdge<u32>, out_b)

    class TexSymbolsProbeNode : public shamrock::solvergraph::INode {
        public:
        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal() override {}
        std::string _impl_get_label() const override { return "TexSymbolsProbe"; }
        std::string _impl_get_tex() const override {
            std::string tex = "{out_a} = {in_a}, {out_b} = {in_b}, {not_an_edge}";
            replace_edges_tex_symbols(tex);
            return tex;
        }
    };

#undef NODE_EDGES

#define NODE_EDGES(X_RO, X_RW, X_RO_OPTIONAL, X_RW_OPTIONAL)                                       \
    X_RO_OPTIONAL(shamrock::solvergraph::IDataEdge<u32>, opt_in)                                   \
    X_RO(shamrock::solvergraph::IDataEdge<u32>, in)                                                \
    X_RW(shamrock::solvergraph::IDataEdge<u32>, out)

    class TexSymbolsOptionalProbeNode : public shamrock::solvergraph::INode {
        public:
        EXPAND_NODE_EDGES_OPTIONAL(NODE_EDGES)

        void _impl_evaluate_internal() override {}
        std::string _impl_get_label() const override { return "TexSymbolsOptionalProbe"; }
        std::string _impl_get_tex() const override {
            std::string tex = "{out} = {in} + {opt_in}";
            replace_edges_tex_symbols(tex);
            return tex;
        }
    };

#undef NODE_EDGES

} // namespace

NEW_TEST(Unittest, "shamsolvergraph/node/NodeTexSymbols", 1) {
    using namespace shamrock::solvergraph;

    {
        auto in_a  = IDataEdge<u32>::make_shared("in_a", "a_{in}");
        auto out_a = IDataEdge<u32>::make_shared("out_a", "a_{out}");
        auto in_b  = IDataEdge<u32>::make_shared("in_b", "b_{in}");
        auto out_b = IDataEdge<u32>::make_shared("out_b", "b_{out}");

        TexSymbolsProbeNode node;
        node.set_edges(in_a, out_a, in_b, out_b);

        auto symbols = node.get_edges_tex_symbols();
        REQUIRE_EQUAL(symbols.in_a, in_a->get_tex_symbol());
        REQUIRE_EQUAL(symbols.out_a, out_a->get_tex_symbol());
        REQUIRE_EQUAL(symbols.in_b, in_b->get_tex_symbol());
        REQUIRE_EQUAL(symbols.out_b, out_b->get_tex_symbol());

        std::string expected = sham::format(
            "{} = {}, {} = {}, {{not_an_edge}}",
            out_a->get_tex_symbol(),
            in_a->get_tex_symbol(),
            out_b->get_tex_symbol(),
            in_b->get_tex_symbol());
        REQUIRE_EQUAL(node.get_tex(), expected);
    }

    {
        auto in  = IDataEdge<u32>::make_shared("in", "x");
        auto out = IDataEdge<u32>::make_shared("out", "y");

        TexSymbolsOptionalProbeNode node;
        node.set_edges(std::nullopt, in, out);

        auto symbols = node.get_edges_tex_symbols();
        REQUIRE_EQUAL(symbols.in, in->get_tex_symbol());
        REQUIRE_EQUAL(symbols.out, out->get_tex_symbol());
        REQUIRE_EQUAL(symbols.opt_in, make_null_opt_edge()->get_tex_symbol());

        auto opt_in = IDataEdge<u32>::make_shared("opt_in", "z");
        node.set_edges(opt_in, in, out);

        std::string expected = sham::format(
            "{} = {} + {}", out->get_tex_symbol(), in->get_tex_symbol(), opt_in->get_tex_symbol());
        REQUIRE_EQUAL(node.get_tex(), expected);
    }
}

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamsolvergraph/LifetimeTracker.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/NodeSetEdge.hpp"
#include "shamsolvergraph/node/OperationSequence.hpp"
#include "shamtest/shamtest.hpp"
#include <nlohmann/json.hpp>
#include <typeinfo>

namespace {

    // Recorded events, populated by the hooks below. LifetimeTracker's hooks are raw
    // function pointers (not std::function), so they cannot capture `events` by reference -
    // it has to live at namespace scope.
    std::vector<nlohmann::json> events;

    void on_create_node(u64 uuid) {
        events.push_back({{"event", "create"}, {"type", "INode"}, {"uuid", uuid}});
    }
    void on_destroy_node(u64 uuid) {
        events.push_back({{"event", "destroy"}, {"type", "INode"}, {"uuid", uuid}});
    }
    void on_state_update_node(shamrock::solvergraph::INode &node) {
        // `typeid(node)` reports the dynamic type of `node`. If a state_update is ever fired
        // from a base class constructor (e.g. INode's own ctor) before the derived class's
        // constructor has run, this reports the base class, not the true derived type -- that
        // mismatch is exactly what the OperationSequence test below checks for.
        events.push_back(
            {{"event", "state_update"},
             {"type", "INode"},
             {"uuid", node.get_uuid()},
             {"dynamic_type", std::string(typeid(node).name())}});
    }
    void on_event_node(u64 uuid, std::string_view event) {
        events.push_back({{"event", event}, {"type", "INode"}, {"uuid", uuid}});
    }

    void on_create_edge(u64 uuid) {
        events.push_back({{"event", "create"}, {"type", "IEdge"}, {"uuid", uuid}});
    }
    void on_destroy_edge(u64 uuid) {
        events.push_back({{"event", "destroy"}, {"type", "IEdge"}, {"uuid", uuid}});
    }
    void on_state_update_edge(shamrock::solvergraph::IEdge &edge) {
        events.push_back(
            {{"event", "state_update"},
             {"type", "IEdge"},
             {"uuid", edge.get_uuid()},
             {"dynamic_type", std::string(typeid(edge).name())}});
    }

    // Installs the hooks above and clears `events` on construction, restores the hooks to
    // nullptr on destruction. This keeps a failed REQUIRE_EQUAL from leaking dangling static
    // hooks into whichever test runs next in the same binary.
    struct ScopedHooks {
        using INode = shamrock::solvergraph::INode;
        using IEdge = shamrock::solvergraph::IEdge;

        ScopedHooks() {
            events.clear();

            shamrock::solvergraph::LifetimeTracker<INode>::on_create       = &on_create_node;
            shamrock::solvergraph::LifetimeTracker<INode>::on_destroy      = &on_destroy_node;
            shamrock::solvergraph::LifetimeTracker<INode>::on_state_update = &on_state_update_node;
            shamrock::solvergraph::LifetimeTracker<INode>::on_event        = &on_event_node;

            shamrock::solvergraph::LifetimeTracker<IEdge>::on_create       = &on_create_edge;
            shamrock::solvergraph::LifetimeTracker<IEdge>::on_destroy      = &on_destroy_edge;
            shamrock::solvergraph::LifetimeTracker<IEdge>::on_state_update = &on_state_update_edge;
            // LifetimeTracker<IEdge>::on_event is intentionally left null: edges have no
            // "operation" concept, so it must never be invoked. If it ever is, this test
            // crashes on a null function-pointer call instead of silently passing.
        }

        ~ScopedHooks() {
            shamrock::solvergraph::LifetimeTracker<INode>::on_create       = nullptr;
            shamrock::solvergraph::LifetimeTracker<INode>::on_destroy      = nullptr;
            shamrock::solvergraph::LifetimeTracker<INode>::on_state_update = nullptr;
            shamrock::solvergraph::LifetimeTracker<INode>::on_event        = nullptr;

            shamrock::solvergraph::LifetimeTracker<IEdge>::on_create       = nullptr;
            shamrock::solvergraph::LifetimeTracker<IEdge>::on_destroy      = nullptr;
            shamrock::solvergraph::LifetimeTracker<IEdge>::on_state_update = nullptr;
        }
    };

    std::string dump(std::vector<nlohmann::json> &in) {
        std::string s = "";
        for (auto &j : in) {
            s += j.dump();
        }
        return s;
    }

} // namespace

NEW_TEST(Unittest, "shamsolvergraph/LifetimeTracker", 1) {
    using namespace shamrock::solvergraph;

    ScopedHooks hooks{};

    std::vector<nlohmann::json> expected;

    using NodeT = NodeSetEdge<IDataEdge<f64>>;

    // uuid captured after each object is created rather than hardcoded: WithUUID's counters
    // are global per-type and shared with every other test in the binary.
    u64 edge_uuid = 0;
    u64 node_uuid = 0;

    {
        // Step 1: creating an edge fires exactly one create event for that edge.
        auto edge = IDataEdge<f64>::make_shared("a", "a");
        edge_uuid = edge->get_uuid();
        expected.push_back({{"event", "create"}, {"type", "IEdge"}, {"uuid", edge_uuid}});
        REQUIRE_EQUAL(dump(events), dump(expected));

        // Step 2: creating a node fires exactly one create event for that node, immediately
        // followed by a state_update for itself (every node ctor ends with a self
        // state_update), before anything else happens to it.
        NodeT set_edge([](IDataEdge<f64> &e) {
            e.data = 1;
        });
        node_uuid = set_edge.get_uuid();
        expected.push_back({{"event", "create"}, {"type", "INode"}, {"uuid", node_uuid}});
        REQUIRE_EQUAL(dump(events), dump(expected));

        // Step 3: Binding the edges to the node fires at least one state_update.
        // Exactly how many fire, and for which objects, is an implementation
        // detail this test does not pin down.
        std::size_t before_bind = events.size();
        set_edge.set_edges(edge);
        // even if a state update was fired before we need a state update that register the new
        // edges so we check that at least one were added
        REQUIRE_EQUAL(events.size() > before_bind, true);
        // Check specifically for a state_update that is for `set_edge` itself: uuid alone would
        // already narrow it down here (only one node exists), but also checking dynamic_type
        // guards against a state_update fired too early (e.g. from a base class ctor), where
        // typeid() would report the wrong class -- see the OperationSequence test below.
        INode &node_ref               = set_edge;
        std::string node_dynamic_type = typeid(node_ref).name();
        bool has_state_update         = false;
        for (std::size_t i = before_bind; i < events.size(); i++) {
            has_state_update
                = has_state_update
                  || (events[i].at("event") == "state_update" && events[i].at("uuid") == node_uuid
                      && events[i].at("dynamic_type") == node_dynamic_type);
        }
        REQUIRE_EQUAL(has_state_update, true);

        // Resync `expected` with the actual events so the exact-match checks below still hold.
        expected.insert(expected.end(), events.begin() + std::ptrdiff_t(before_bind), events.end());
        REQUIRE_EQUAL(dump(events), dump(expected));

        // Step 4: moving the node into a shared_ptr transfers its identity: the destination
        // keeps the same uuid, no create event fires for it, and (checked in step 7 below) the
        // moved-from husk must not fire a destroy event when it goes out of scope.
        std::shared_ptr<NodeT> ptr = std::make_shared<NodeT>(std::move(set_edge));
        REQUIRE_EQUAL(ptr->get_uuid(), node_uuid);
        REQUIRE_EQUAL(dump(events), dump(expected));

        // Note here the definition of lifetime tracing is that the state must be up to date before
        // evaluation. Not sure how to enforce it though.

        // Step 5: evaluating the node brackets the operation, firing on_op with op_id 0 at
        // the start and op_id 1 at the end.
        ptr->evaluate();
        expected.push_back({{"event", "evaluate_begin"}, {"type", "INode"}, {"uuid", node_uuid}});
        expected.push_back({{"event", "evaluate_end"}, {"type", "INode"}, {"uuid", node_uuid}});
        REQUIRE_EQUAL(dump(events), dump(expected));
        REQUIRE_EQUAL(edge->data, 1.0);

        // Step 6: destroying the node fires exactly one destroy event for it. The edge is
        // still alive (held by `edge` below), so no edge destroy event fires here.
        ptr.reset();
        expected.push_back({{"event", "destroy"}, {"type", "INode"}, {"uuid", node_uuid}});
        REQUIRE_EQUAL(dump(events), dump(expected));

        // Step 7: destroying the last reference to the edge fires exactly one destroy event
        // for it.
        edge.reset();
        expected.push_back({{"event", "destroy"}, {"type", "IEdge"}, {"uuid", edge_uuid}});
        REQUIRE_EQUAL(dump(events), dump(expected));

        // `set_edge` (the moved-from husk from step 4) is destroyed here, at the end of this
        // block, before the final assertion below runs.
    }

    // Step 8: the moved-from husk's destruction above must not add a 10th event.
    REQUIRE_EQUAL(dump(events), dump(expected));
}

NEW_TEST(Unittest, "shamsolvergraph/LifetimeTracker_OperationSequence", 1) {
    using namespace shamrock::solvergraph;

    ScopedHooks hooks{};

    std::vector<nlohmann::json> expected;

    using NodeT = NodeSetEdge<IDataEdge<f64>>;

    // uuids captured after each object is created rather than hardcoded: WithUUID's counters
    // are global per-type and shared with every other test in the binary.
    u64 edge_uuid  = 0;
    u64 child_uuid = 0;
    u64 seq_uuid   = 0;

    // Step 1: creating the edge fires one create event for it.
    auto edge = IDataEdge<f64>::make_shared("a", "a");
    edge_uuid = edge->get_uuid();
    expected.push_back({{"event", "create"}, {"type", "IEdge"}, {"uuid", edge_uuid}});
    REQUIRE_EQUAL(dump(events), dump(expected));

    // Step 2: creating the child node fires one create event for it.
    auto child = std::make_shared<NodeT>([](IDataEdge<f64> &e) {
        e.data = 1;
    });
    child_uuid = child->get_uuid();
    expected.push_back({{"event", "create"}, {"type", "INode"}, {"uuid", child_uuid}});
    REQUIRE_EQUAL(dump(events), dump(expected));

    // Step 3: binding the edge to the child fires at least one state_update, same relaxed rule
    // as the plain-node test above: we do not pin down exactly how many fire.
    std::size_t before_bind = events.size();
    child->set_edges(edge);
    REQUIRE_EQUAL(events.size() > before_bind, true);
    bool child_state_updated = false;
    for (std::size_t i = before_bind; i < events.size(); i++) {
        child_state_updated = child_state_updated || (events[i].at("event") == "state_update");
    }
    REQUIRE_EQUAL(child_state_updated, true);
    // Resync `expected` with the actual events so the exact-match checks below still hold.
    expected.insert(expected.end(), events.begin() + std::ptrdiff_t(before_bind), events.end());
    REQUIRE_EQUAL(dump(events), dump(expected));

    // Step 4: wrapping the child into an OperationSequence fires one create event for the
    // sequence, immediately followed by a state_update for itself. A sequence owns no ro/rw
    // edges of its own (it only forwards evaluate() to its children), so it never goes through
    // __internal_set_ro_edges/__internal_set_rw_edges; instead, its constructor explicitly
    // fires this self state_update once its children are stored, via
    // INode::notify_self_state_update() -- the same mechanism the rule under test (step 5)
    // relies on.
    auto seq
        = std::make_shared<OperationSequence>("seq", std::vector<std::shared_ptr<INode>>{child});
    seq_uuid = seq->get_uuid();
    // Bound to a variable rather than written inline as typeid(*seq): typeid() only evaluates
    // its operand when it names a polymorphic glvalue, but *seq (a function call to
    // shared_ptr::operator*) trips a compiler warning about evaluating an expression with
    // side effects as a typeid operand.
    INode &seq_ref               = *seq;
    std::string seq_dynamic_type = typeid(seq_ref).name();
    expected.push_back({{"event", "create"}, {"type", "INode"}, {"uuid", seq_uuid}});
    expected.push_back(
        {{"event", "state_update"},
         {"type", "INode"},
         {"uuid", seq_uuid},
         {"dynamic_type", seq_dynamic_type}});
    REQUIRE_EQUAL(dump(events), dump(expected));

    // Step 5: the rule under test - a state_update must be recorded before evaluate() fires -
    // holds even for meta nodes like OperationSequence, which delegate evaluation entirely to
    // their children. Check for a state_update whose uuid is the sequence's own uuid AND whose
    // reported dynamic_type is genuinely OperationSequence, not just INode. This second check
    // matters because INode's own constructor runs before OperationSequence's: a naive fix that
    // fires a self state_update from inside INode's base ctor would still pass the uuid check,
    // but typeid() at that point reports the base class under construction (INode), not the
    // true derived type -- this catches that class of bug.
    bool seq_state_up_to_date_before_evaluate = false;
    for (auto &j : events) {
        seq_state_up_to_date_before_evaluate
            = seq_state_up_to_date_before_evaluate
              || (j.at("event") == "state_update" && j.at("uuid") == seq_uuid
                  && j.at("dynamic_type") == seq_dynamic_type);
    }
    REQUIRE_EQUAL(seq_state_up_to_date_before_evaluate, true);

    // Step 6: evaluating the sequence brackets both its own op events and the child's: the
    // sequence's evaluate_begin fires first, then the child's evaluate_begin/evaluate_end (as
    // the sequence's _impl_evaluate_internal() calls child->evaluate()), then the sequence's
    // evaluate_end.
    seq->evaluate();
    expected.push_back({{"event", "evaluate_begin"}, {"type", "INode"}, {"uuid", seq_uuid}});
    expected.push_back({{"event", "evaluate_begin"}, {"type", "INode"}, {"uuid", child_uuid}});
    expected.push_back({{"event", "evaluate_end"}, {"type", "INode"}, {"uuid", child_uuid}});
    expected.push_back({{"event", "evaluate_end"}, {"type", "INode"}, {"uuid", seq_uuid}});
    REQUIRE_EQUAL(dump(events), dump(expected));
    REQUIRE_EQUAL(edge->data, 1.0);
}

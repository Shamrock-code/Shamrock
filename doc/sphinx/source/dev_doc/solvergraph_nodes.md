# Solver graph nodes

A solver graph is a DAG of **nodes** connected by **edges**. Nodes read and write
edge data in `_impl_evaluate_internal()`. Edges carry typed values (fields,
scalars, indexes, ...). Most nodes can be found under the `shamrock::solvergraph` namespace.

- Nodes inherit `INode`.
- edges can be wired with `set_edges()`
- evaluation of the node from **edges** input and in-out is done using `evaluate()`.

## Node layout

```cpp
// 1. declare inputs (X_RO) and outputs (X_RW)
#define NODE_EDGES(X_RO, X_RW) \
    X_RO(shamrock::solvergraph::IDataEdge<T>, in) \
    X_RW(shamrock::solvergraph::IDataEdge<T>, out)

// 2. Create the class and inherit from INode
class MyNode : public shamrock::solvergraph::INode {
  public:
    // 3. generate set_edges() / get_edges()
    EXPAND_NODE_EDGES(NODE_EDGES)

    // 4. evaluation logic
    void _impl_evaluate_internal() override {
        auto edges = get_edges();
        // ... do stuff ...
    }

    // 5. labels for graph display and TeX output
    std::string _impl_get_label() const override { return "MyNode"; }
    std::string _impl_get_tex() const override { return ""; }
};
#undef NODE_EDGES
```

`EXPAND_NODE_EDGES` generates:

- `struct Edges` — `const T&` for read-only edges, `T&` for read-write edges
- `set_edges(shared_ptr...)` — wire before `evaluate()`
- `get_edges()` — typed access in `_impl_evaluate_internal()`

### Edge macros

| Macro | Role in `NODE_EDGES` | `set_edges` argument | `get_edges()` member |
|-------|----------------------|----------------------|----------------------|
| `X_RO` | required read-only input | `shared_ptr<T>` | `const T&` |
| `X_RW` | required read-write output | `shared_ptr<T>` | `T&` |

List inputs first, outputs last.

## Optional edges

An optional input may be absent at wiring time for nodes having multiple evaluation modes.
To use optional edges use `EXPAND_NODE_EDGES_OPTIONAL` when `NODE_EDGES`
contains any `X_*_OPTIONAL` macro.

The following changes for those :

```cpp
// Now we have to add X_RO_OPTIONAL, X_RW_OPTIONAL too
#define NODE_EDGES(X_RO, X_RW, X_RO_OPTIONAL, X_RW_OPTIONAL) \
    X_RO_OPTIONAL(shamrock::solvergraph::IDataEdge<u32>, opt_a) \
    X_RO_OPTIONAL(shamrock::solvergraph::IDataEdge<u32>, opt_b) \
    X_RW(shamrock::solvergraph::IDataEdge<u32>, out)

class OptionalEdgeProbeNode : public shamrock::solvergraph::INode {
  public:
    // Use EXPAND_NODE_EDGES_OPTIONAL instead of EXPAND_NODE_EDGES
    EXPAND_NODE_EDGES_OPTIONAL(NODE_EDGES)

    void _impl_evaluate_internal() override {
        auto edges = get_edges();
        // optional inputs: edges.opt_a.has_value(), edges.opt_b.has_value()
        // ...
    }

    std::string _impl_get_label() const override { return "OptionalEdgeProbe"; }
    std::string _impl_get_tex() const override { return ""; }
};
#undef NODE_EDGES
```

### Optional edge macros

| Macro | `set_edges` argument | `get_edges()` member |
|-------|----------------------|----------------------|
| `X_RO_OPTIONAL` | `optional<shared_ptr<T>>` or `nullopt` | `optional<reference_wrapper<const T>>` |
| `X_RW_OPTIONAL` | `optional<shared_ptr<T>>` or `nullopt` | `optional<reference_wrapper<T>>` |

### Wiring

Pass `std::nullopt` for absent optional inputs:

```cpp
node.set_edges(std::nullopt, std::nullopt, out);  // neither optional present
node.set_edges(a, std::nullopt, out);             // only opt_a
node.set_edges(a, b, out);                        // both present
node.evaluate();
```

Absent optionals are stored as
[`INullOptEdge`](../../../src/shamsolvergraph/include/shamsolvergraph/edge/INullOptEdge.hpp)
placeholders so slot indices stay stable. `get_ro_edge_optional` returns
`std::nullopt` for those slots.

## Composition nodes

`OperationSequence` and `OperationIf` wrap other `INode`s. Nested nodes are
passed to the constructor (graph structure). Data edges are still wired with
`set_edges`.

```cpp
auto then_node = std::make_shared<MyNode>(...); // optional
auto else_node = std::make_shared<MyNode>(...); // optional
auto cond = IDataEdge<bool>::make_shared("do_step", "do_step");

auto if_node = std::make_shared<OperationIf>("do step", then_node, else_node);
if_node->set_edges(cond);

cond->data = true;  // evaluate then_node (or skip if none)
if_node->evaluate();

cond->data = false; // evaluate else_node (or skip if none)
if_node->evaluate();
```

To gate several nodes, wrap them in an `OperationSequence` and pass that as
`then_node` / `else_node`. Else-only is `OperationIf("name", {}, else_node)`.

## Related files

- [`INode.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/node/INode.hpp)
- [`INullOptEdge.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/edge/INullOptEdge.hpp)
- [`OperationIf.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/node/OperationIf.hpp)
- [`OperationSequence.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/node/OperationSequence.hpp)
- [`OptionalEdges_tests.cpp`](../../../src/tests/shamsolvergraph/node/OptionalEdges_tests.cpp)
- [`OperationIf_tests.cpp`](../../../src/tests/shamsolvergraph/node/OperationIf_tests.cpp)

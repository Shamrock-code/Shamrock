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
- `get_edges_tex_symbols()` — returns `struct EdgesTexSymbols`, one `std::string`
  TeX symbol per edge, named after it
- `replace_edges_tex_symbols(tex)` — replaces every `{<edge name>}` placeholder in
  `tex` by the TeX symbol of that edge

### TeX output

In `_impl_get_tex()`, refer to edges by name rather than by slot index
(`get_ro_edge_base(i)` / `get_rw_edge_base(i)`): slot indices silently shift
when an edge is added, removed or reordered in `NODE_EDGES`, while names stay
in sync with it.

Name each edge placeholder after its edge in `NODE_EDGES`, and use
`replace_all` only for values that are not edges (node members, constants):

```cpp
std::string _impl_get_tex() const override {
    std::string tex = R"tex(
        {out}_i = f({in}_i), \quad N = {nvar}
    )tex";

    replace_edges_tex_symbols(tex); // {in}, {out} -> TeX symbols of those edges

    shambase::replace_all(tex, "{nvar}", std::to_string(nvar)); // node member

    return tex;
}
```

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

`OperationSequence`, `OperationIf`, `OperationFor`, `OperationForUntil` and
`OperationWhile` wrap other `INode`s. Nested nodes are passed to the
constructor (graph structure). Data edges are still wired with `set_edges`.

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

`OperationFor` evaluates a body node `n` times, `n` being read from an
`IDataEdge<u32>` wired with `set_edges`:

```cpp
auto body  = std::make_shared<MyNode>(...); // required, a null body throws
auto count = IDataEdge<u32>::make_shared("n_substeps", "m");

auto for_node = std::make_shared<OperationFor>("substeps", body);
for_node->set_edges(count);

count->data = 3; // evaluate body 3 times (0 evaluates nothing)
for_node->evaluate();
```

The count is read once, before the first iteration, so the body cannot change
the number of iterations. To repeat several nodes, wrap them in an
`OperationSequence` and pass that as the body.

`OperationForUntil` is a variant of `OperationFor` that also stops as soon as
its `IDataEdge<bool>` condition becomes false. It evaluates its (mandatory)
body node at most `max_iter` times, so it always ends even if the body never
sets the condition to false. It is equivalent to:

```cpp
u32 i = 0;
for (; i < max_iter && condition; i++) {
    body->evaluate(); // expected to update condition
}
iter_count = i; // if the optional iter_count edge is set
```

The condition is checked before each iteration, so the body may run zero
times. The body updates the condition, typically through its own read-write
edge pointing to the same `IDataEdge<bool>`. The optional `IDataEdge<u32>`
`iter_count` edge holds the current iteration index while the body runs, and
the number of evaluated iterations once the loop ends (the body must not write
it). Check the condition after the loop to know whether it stopped on the
condition or on `max_iter`.

```cpp
auto cond = IDataEdge<bool>::make_shared("not_converged", "not_converged");
auto iter_count = IDataEdge<u32>::make_shared("iter_count", "n_{iter}");
auto body = std::make_shared<MyIterationNode>(...); // updates cond->data

auto for_node = std::make_shared<OperationForUntil>("iterate", body, max_iter);
for_node->set_edges(cond, iter_count); // or (cond, std::nullopt)

cond->data = true;
for_node->evaluate();
// cond->data == true here means max_iter was reached before convergence
```

`OperationWhile` evaluates its (mandatory) body node as long as its
`IDataEdge<bool>` condition is true. The condition is checked before each
iteration, so the body may run zero times. The body must eventually set the
condition to false, typically through its own read-write edge pointing to the
same `IDataEdge<bool>`, otherwise the loop never ends.

```cpp
auto cond = IDataEdge<bool>::make_shared("not_converged", "not_converged");
auto body = std::make_shared<MyIterationNode>(...); // updates cond->data

auto while_node = std::make_shared<OperationWhile>("iterate", body);
while_node->set_edges(cond);

cond->data = true;
while_node->evaluate(); // evaluate body until cond->data becomes false
```

To loop over several nodes, wrap them in an `OperationSequence` and pass that
as the body.

## Related files

- [`INode.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/node/INode.hpp)
- [`INullOptEdge.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/edge/INullOptEdge.hpp)
- [`OperationIf.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/node/OperationIf.hpp)
- [`OperationFor.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/node/OperationFor.hpp)
- [`OperationFor.cpp`](../../../src/shamsolvergraph/src/OperationFor.cpp)
- [`OperationForUntil.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/node/OperationForUntil.hpp)
- [`OperationForUntil.cpp`](../../../src/shamsolvergraph/src/OperationForUntil.cpp)
- [`OperationSequence.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/node/OperationSequence.hpp)
- [`OperationWhile.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/node/OperationWhile.hpp)
- [`OperationWhile.cpp`](../../../src/shamsolvergraph/src/OperationWhile.cpp)
- [`OptionalEdges_tests.cpp`](../../../src/tests/shamsolvergraph/node/OptionalEdges_tests.cpp)
- [`OperationIf_tests.cpp`](../../../src/tests/shamsolvergraph/node/OperationIf_tests.cpp)
- [`OperationFor_tests.cpp`](../../../src/tests/shamsolvergraph/node/OperationFor_tests.cpp)
- [`OperationForUntil_tests.cpp`](../../../src/tests/shamsolvergraph/node/OperationForUntil_tests.cpp)
- [`OperationWhile_tests.cpp`](../../../src/tests/shamsolvergraph/node/OperationWhile_tests.cpp)

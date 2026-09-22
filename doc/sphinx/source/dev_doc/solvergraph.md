# Solver graph

In Shamrock, originally we were employing modules that were editing the content of global states (fields) that were stored in the scheduler or the solver storage. While this was very simple it has the annoying side effect that every module touches the global state and therefore may affect the behavior of other modules. They are not self contained! While this is manageable for a small code, it becomes very hard to track what is editing what and when in the code. This is the sole purpose of solvergraphs, to have an API to formulate operations in the code where every operation is self contained and can only, by design, read from its input and edit its outputs. On top of that the wiring of the graph is done at runtime which allows much more flexibility and can be explored for visualisation. Maybe this is sounding abstract right now ... So let's build it together!

First of the concept of having inputs and outputs is quite general so let's not reinvent the wheel and actually pull inspiration from known concepts. Let's start off by having a simple case:

::::{card}
A function $f$ take a floating point (f64) input $a$ and returns a floating point $b$.
Or shortly $f : a  \rightarrow b $
::::

Here we can already formalise a few things. We have $a$ and $b$ that are some tangible data they actually hold something, here a value. $f$ on the other hand is just a function or an application (or more generally a functor, yup the inspiration is from that part of math 😅) between an input and an output.

Formally in category theory people would draw a diagram like

```{graphviz}
digraph {
    rankdir=LR;
    a [shape=plaintext];
    b [shape=plaintext];
    a -> b [label="f"];
}
```

In our case we will write it like this

```{graphviz}
digraph {
    rankdir=LR;
    a [shape=plaintext];
    b [shape=plaintext];
    f [shape=box];
    a -> f [color=green];
    f -> b [color=red];
}
```

where green means input, red means output (if they are both in and out they fall in the output case).

Here if we think of it as a graph, $f$ is a node and $a$ and $b$ the edges. So if we express something in such a graph roughly speaking applications are **nodes** (or functors) and edges are some data, values, fields or whatever that is passed between applications (formally they are named category but we don't really care here). Essentially it can be anything that is tangible, we will just call them **edges** in solvergraphs since it is what lives between nodes.

Now just to improve readability we will put some circles around edge names

## A first node from a simple case

A good one to start is $c = a+b$. For this one we can write it like so:

```{graphviz}
digraph {
    rankdir=LR;
    a;
    b;
    c;
    sum [shape=box];
    a -> sum [color=green];
    b -> sum [color=green];
    sum -> c [color=red];
}
```

Since here it will be a singular value we can use `IDataEdge<f64>` which is a "Data Edge" of a singular "f64" (double precision floating point). So $f$ will have two `IDataEdge<f64>` as input and a single one as output.

When implementing it in c++ we use a macro to specify the inputs and outputs and here would do:

```c++
#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::IDataEdge<f64>, a)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<f64>, b)                                      \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IDataEdge<f64>, c)

```

Then we need our node to code the implementation

```c++
class Node_f : public shamrock::solvergraph::INode {

    public:
    EXPAND_NODE_EDGES(NODE_EDGES) // specify the inputs and outputs

    // What is done when the node evaluates
    inline void _impl_evaluate_internal() override {
         auto edges = get_edges(); // get actual access to the edges

         // do c = a+b, IDataEdge contain the data in a .data member value
         edges.c.data = edges.a.data + edges.b.data;
    }

    // This is the name that will be displayed on the graph when rendered
    inline std::string _impl_get_label() const override { return "Node_f"; }

    // This will print the TeX of the node, forget about it and ask an LLM it is faster :)
    std::string _impl_get_tex() const override {return "TODO"};
};
```

## Wiring node and edges together

Now to actually use it in practice do something like:

```c++
// make_shared(name, texsymbol): name is shown on the rendered solvergraph,
// texsymbol is the symbol used when the node prints its TeX formula
auto a = shamrock::solvergraph::IDataEdge<f64>::make_shared("a", "a");
auto b = shamrock::solvergraph::IDataEdge<f64>::make_shared("b", "b");
auto c = shamrock::solvergraph::IDataEdge<f64>::make_shared("c", "c");

a->data = 2.0;
b->data = 3.0;

auto node_f = std::make_shared<Node_f>();
node_f->set_edges(a, b, c); // wire a, b as inputs (read only) and c as output (read write)
node_f->evaluate();

// c->data now contains 5.0 (a->data + b->data)
```

While not yet available the goal is to be able to do the same from python such that user can poke around the solvergraph of the solver without needing to touch the internals of the code.

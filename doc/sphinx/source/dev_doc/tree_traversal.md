# Tree traversal

This page documents how the Shamrock tree (`shamtree`) is traversed on the device: the standard
stack-based traversal, and its "while-while" variant, which visits the same nodes in the same
order but processes the leaves after the threads of a warp have reconverged.

## The tree being traversed

`shamtree::CompressedLeafBVH` stores a Karras radix tree built on the reduced Morton codes of the
objects, plus an axis-aligned bounding box (AABB) per node. Node ids are laid out as follows:

- `0` is the root, and `[0, offset_leaf)` are the internal nodes.
- `[offset_leaf, offset_leaf + n_leaf)` are the leaves. If there is no internal node, the root is
  itself a leaf.
- `get_left_child(id)` and `get_right_child(id)` return the child ids, already offset when the
  child is a leaf, and `is_id_leaf(id)` is `id >= offset_leaf`.

The traversal is exposed at three levels, each one wrapping the previous one:

| Type (device accessor) | Header | Adds |
| --- | --- | --- |
| `KarrasTreeTraverserAccessed` | `shamtree/KarrasTreeTraverser.hpp` | the traversal loop on node ids |
| `CLBVHTraverserAccessed` | `shamtree/CLBVHObjectIterator.hpp` | the node AABB, passed to the node test |
| `CLBVHObjectIteratorAccessed` | `shamtree/CLBVHObjectIterator.hpp` | the loop over the objects of each hit leaf |

In a kernel, the object iterator is obtained with `tree.get_object_iterator()` and passed to
`sham::kernel_call` like a buffer:

```cpp
auto obj_it = tree.get_object_iterator();

sham::kernel_call(
    q, sham::MultiRef{rays_buf, obj_it, ...}, sham::MultiRef{out_buf}, nrays,
    [=](u32 gid, const Ray *rays, auto particle_looper, ..., T *out) {
        particle_looper.rtree_for(
            [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                return /* does the query touch this node? */;
            },
            [&](u32 id_b) {
                /* called for each object of each hit leaf */
            });
    });
```

## Standard traversal

`KarrasTreeTraverserAccessed::stack_based_traversal` is a depth-first traversal with an explicit
stack, run independently by each work-item (one query per work-item: a ray, a particle, a cell).
It takes three callbacks:

- `traverse_condition(node_id) -> bool`: the node test, called for every popped node, internal or
  leaf.
- `on_found_leaf(leaf_id)`: called for every leaf that passes the node test.
- `on_excluded_node(node_id)`: called for every node that fails it (unused by most callers, but
  needed e.g. for a multipole acceptance criterion).

```cpp
std::array<u32, tree_depth> id_stack; // per work-item, filled from the top
u32 stack_cursor       = tree_depth - 1;
id_stack[stack_cursor] = root_node;

while (stack_cursor < tree_depth) { // until the stack is empty
    u32 node = id_stack[stack_cursor];
    stack_cursor++;

    if (traverse_condition(node)) {
        if (is_id_leaf(node)) {
            on_found_leaf(node);
        } else {
            id_stack[--stack_cursor] = get_right_child(node);
            id_stack[--stack_cursor] = get_left_child(node);
        }
    } else {
        on_excluded_node(node);
    }
}
```

The right child is pushed first, so the left child is popped first: the hit leaves are visited
from left to right, i.e. in increasing Morton order. The stack size is `tree_depth`, which
`CLBVHTraverserAccessed` sets to `tree_depth_max` (the number of significant Morton bits plus one),
so a traversal can never overflow it.

The order in which a work-item calls its callbacks only depends on the tree and on the query, not
on the hardware or on how the work-items are scheduled. A kernel that accumulates into a per
work-item sum (as the column render does) therefore adds its terms in a fixed order, which keeps it
bitwise reproducible (see {doc}`determinism`).

A second overload takes a stack supplied by the caller instead of the internal `std::array`
(a functor `(u32 id) -> u32 &` and its capacity, at least the tree depth plus one), e.g. a slice of
a `sycl::local_accessor`. The SPH neighbour cache uses it to keep the stacks in work-group memory.
The loop is otherwise identical.

```{raw} html
:file: tree_traversal_column_flow.svg
```

The column render (`SPHColumnInteg`) as an example: one work-item per ray, the node test checks
whether the ray crosses the node AABB expanded by the largest smoothing length in the node, and
each hit leaf is scanned for the particles whose support the ray crosses. The highlighted boxes are
where the FP64 work is.

## Divergence of the standard traversal on GPUs

On a GPU the work-items run in warps (sub-groups of 32 on NVIDIA) that execute one instruction at
a time for all their lanes. In the single loop above, each loop iteration either tests a node or
processes a leaf, and different lanes of a warp reach their leaves at different iterations. When
some lanes process a leaf, the others wait, and the leaf processing (a loop over the objects of the
leaf, plus whatever the kernel does for each of them) is executed many times by the warp, with a
few active lanes each time.

```{raw} html
:file: tree_traversal_lane_usage.svg
```

Active lanes per warp instruction in the original column render (Nsight Compute source counters,
first of the four renders of `examples/sph/run_sph_rendering.py`, RTX 3070). The node test runs
with most lanes active; the leaf scan and the contribution, where most of the FP64 work is, run
with a handful.

In that kernel, at a given iteration only about 7 of the about 20 active lanes were at a leaf, and the kernel was bound by the FP64 pipe (about 86 % utilisation) with 4.8 of 32 lanes
active on average.

## While-while traversal

`stack_based_traversal_while_while` is the "while-while" form of the same traversal (Aila & Laine,
*Understanding the efficiency of ray traversal on GPUs*, HPG 2009). The internal nodes are
traversed in an inner loop that only exits once the lane has found its next hit leaf (or emptied
its stack). The leaf is processed after the inner loop, at which point the lanes of the warp have
reconverged, so the warp processes the leaves of all its lanes together.

```cpp
while (stack_cursor < tree_depth) {
    u32 found_leaf = _nindex;

    // inner loop: traverse until a hit leaf is found
    while (stack_cursor < tree_depth) {
        u32 node = id_stack[stack_cursor];
        stack_cursor++;

        if (traverse_condition(node)) {
            if (is_id_leaf(node)) {
                found_leaf = node;
                break;
            } else {
                id_stack[--stack_cursor] = get_right_child(node);
                id_stack[--stack_cursor] = get_left_child(node);
            }
        } else {
            on_excluded_node(node);
        }
    }

    // the lanes of the warp have reconverged here
    if (found_leaf != _nindex) {
        on_found_leaf(found_leaf);
    }
}
```

```{raw} html
:file: tree_traversal_while_while_lanes.svg
```

Lane × iteration for six lanes, with the same node tests and leaves per lane in both loops. In the
single loop the leaf work is scattered across iterations; in the while-while loop, lanes that
already have a leaf wait at the reconvergence point, and the warp then processes all the leaves at
once. The warp runs more iterations, but the extra ones are cheap node tests or waits, while the
expensive leaf scans go from 7 passes to 2.

### Why the results do not change

The pops, the pushes and the node tests are the same as in the standard traversal, and a leaf is
processed right after it is popped, before the next pop. Each work-item therefore calls
`traverse_condition`, `on_excluded_node` and `on_found_leaf` with the same arguments and in the
same order as with `stack_based_traversal`; only the moment at which a lane processes a leaf
relative to the other lanes of its warp changes. Switching a kernel from one to the other gives
bitwise identical results, whatever its callbacks do.

```{raw} html
:file: tree_traversal_visit_order.svg
```

### Usage

The while-while variant has the same signature as the standard one, with a `_while_while` suffix:

- `stack_based_traversal_while_while<tree_depth>` in `KarrasTreeTraverserAccessed`, with or
  without an explicit root node;
- `rtree_for_while_while` in `CLBVHTraverserAccessed`, which passes the node AABB to the node
  test;
- `rtree_for_while_while` in `CLBVHObjectIteratorAccessed`, which loops over the objects of each
  hit leaf.

There is no while-while overload taking a caller-supplied stack yet. The column render
(`SPHColumnInteg`) is currently the only user.

### When it helps

The while-while loop does not remove work; it regroups the leaf processing of the lanes of a warp,
at the cost of making lanes that found a leaf early wait for the others. It pays off when
processing a leaf is expensive compared to a node test, and the node test is cheap enough that
running it inside the divergent inner loop costs little.

Effect of switching from the standard to the while-while traversal, measured during a series of
optimisations of these kernels on a development branch (`examples/sph/run_sph_rendering.py` for the
renders). Times are Nsight Compute kernel durations with locked clocks on an RTX 3070, summed over
all the launches of the benchmark. For the column and azimuthal renders, the change was applied on
top of a bitwise identical symmetric `Y_3d` (different from the one on `main`, see
{doc}`sph_kernel_optimisations`); for the other kernels, on top of the original kernel. All outputs
were byte-identical before and after. The renders are `SPHColumnInteg`, `SPHAzymuthalInteg` and
`SPHInterpolation` (slices).

| Kernel | Benchmark | Before | After | Change |
| --- | --- | --- | --- | --- |
| Column render | 100k particles | 2.481 s | 1.833 s | −26.1 % |
| | 1M particles | 8.172 s | 5.155 s | −36.9 % |
| Azimuthal render | 100k particles | 18.716 s | 15.710 s | −16.1 % |
| | 1M particles | 51.314 s | 43.886 s | −14.5 % |
| Slice render | 100k particles | 633.0 ms | 633.4 ms | +0.1 % |
| | 1M particles | 1046 ms | 1041 ms | −0.4 % |
| SPH neighbour cache | HCP box, 1M particles | 1268 ms | 1268 ms | 0.0 % |
| Ramses block neighbour graph | box of 64³ blocks | 39.31 ms | 39.85 ms | +1.4 % |
| | AMR Sod, 48³ | 69.15 ms | 66.95 ms | −3.2 % |

- In the column and azimuthal renders, a hit leaf costs particle tests plus, for each accepted
  particle, an FP64 contribution (square root, kernel integral, divisions). In the slice render,
  the neighbour cache and the Ramses graph, it only costs a few distance or box tests.
- The column render gains the most, and gains more at 1M particles, where a ray meets about 2.3
  times more particles and the original kernel ran with fewer active lanes (3.2 instead of 4.8 out
  of 32).
- The azimuthal render gains less: its leaf processing also got more coherent, but its node test
  (ray against a ring, with two square roots) is about twice as expensive as the column one and
  now runs inside the divergent inner loop.
- Where a hit leaf only costs a few distance or box tests, the divergence removed was cheap and
  the change is neutral.

On the CPU (AdaptiveCpp OpenMP backend, 4 cores), the four column renders showed no measurable
change in wall-clock time.

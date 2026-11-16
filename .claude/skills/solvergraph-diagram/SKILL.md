---
name: solvergraph-diagram
description: >-
  Draw the changes a commit, commit range, branch, PR number or .diff/.patch
  file makes to a Shamrock solver graph (SPH, GSPH, Ramses, Zeus, ...) as a
  TikZ figure, delivered as .tex, compiled .pdf and a .txt copy of the source
  (for GitHub PR attachments). Use when the user asks for a diagram, figure,
  schematic or PDF of solver graph / node / edge / sequence changes.
---

# Solver graph change diagram

Output: `<short_topic>_solvergraph.{tex,pdf,txt}` in the scratchpad (or a
directory the user names; never committed). Start from `template.tex` next
to this file, a worked example: keep its preamble, styles, colours, layout
patterns and legend, replace the content. Its names, notes and
new/modified marks are specific to that change: never reuse one unchecked.

## 1. Get the change

- commit: `git show <sha>` (merge commit: `git diff <sha>^1 <sha>`);
- range/branch: `git diff $(git merge-base A B) B` (`A..B` can list
  commits merged in from main, and a fork's `origin/main` may be stale);
- patch file: read it;
- upstream PR `N`:
  `git fetch https://github.com/Shamrock-code/Shamrock +pull/N/head:pr-N main:refs/remotes/upstream/main`,
  then `git diff $(git merge-base upstream/main pr-N) pr-N`.

For a large diff, start with `git diff --stat` and keep the hunks in
`init_solver_graph`, the node classes they touch and the code deciding
what gets built (config, `evolve_once`). Read the full new versions of
those files, and the old ones where you claim something is unchanged or
removed.

## 2. Identify the affected part of the graph

First decide whether the graph itself changes (nodes, edges, order) or
only how/when it is built or run, and say so in the figure's first note.
The graph is built in `init_solver_graph()` (`src/shammodels/<model>/src/Solver.cpp`):

- nodes: `solver_graph.register_node("name", NodeType{ctor args})` then
  `.set_edges(...)`. The class's `NODE_EDGES(X_RO, X_RW[, X_RO_OPTIONAL, X_RW_OPTIONAL])`
  (`shamsolvergraph/node/INode.hpp`) gives read-only (`->`), read/write
  (`<->`) and optional edges; the `.set_edges` call may wire optional ones
  only under a condition. `NodeSetEdge`, `OperationIf`, ... set their edges
  by hand (`__internal_set_*_edges`): read the class. Edges created with
  `IDataEdge::make_shared(...)` and passed straight to `set_edges` are not
  registered: label them `(local edge)`.
- execution order: `OperationSequence(name, {nodes...})`. Runtime
  condition: `OperationIf(name, body)` on an `IDataEdge<bool>`. Build-time
  condition: C++ `if` around `push_back`, or `std::visit` over a config
  variant (draw the alternatives as small boxes in a frame labelled with
  the condition, under the step they build).
- solver_graph edges: `register_edge`, `FieldRefs`, `Field`, `Indexes`,
  `IDataEdge`, ... Some solvers (Ramses) keep most edges as
  `std::make_shared` members of `storage` instead of registering them:
  title that column "`storage` edges" and keep `(local edge)` for edges
  created inside one sequence (e.g. per-sequence `FieldRefs`). If nodes
  are not registered by name, label steps with the `OperationSequence`
  name or the node's `_impl_get_label()`.
- synchronized data edges: `sched.synchronized_data`, created by the
  `ensure_sync_edge` lambda at the top of `init_solver_graph` (serialized
  in dumps). Code elsewhere touching them (e.g. resizing in `Model.hpp`)
  is not a graph change: put it in a note under the sync frame.

Classify each step, node, edge and connection as **new**, **modified**
(same name, changed interface/ctor/behaviour/edges, or kind such as
Sequence -> If) or **removed**, and write the list down before drawing:

- an existing edge gaining a connection is not new: only the arrow is;
- an existing node newly pushed into a sequence is a new step in (a) but
  a plain node in (b);
- a node or edge whose build-time condition changes (built in more or
  fewer configs) is `modified`: give the old and new condition in a note;
- `(optional)` goes on the arrow of an optional connection; on the edge
  box only if the edge itself is registered conditionally.

Draw only what the change touches plus the unchanged neighbours needed to
read it (e.g. the node producing an edge a new node reads); big unchanged
nodes may be left out if the panel title says so.

## 3. Draw (copy the template's structure)

- **(a) execution order**: title names the sequence and what encloses it;
  `seq` boxes chained with `flow` arrows, at most ~8 per row, rows joined
  by the wrap-around arrow. An `OperationIf` body gets a `fit` frame around
  its steps (dashed green if new) and its condition edge as a `gedge`
  above the If step. Small notes (green `newc` / orange `modc`) say when new
  steps run and what modified ones now do; state a build-time flag once (e.g. in the
  frame label); mid-row, use one footnote under the sequence instead.
- **(b) data flow**: `gnode` compute nodes in the middle, in execution
  order; solver_graph edges on the left and `sedge` sync edges on the
  right, each column in a `fit` frame; edges produced by a drawn node sit
  between the columns. Connections: `ro`/`rw` unchanged, `ronew`/`rwnew`,
  `romodified`/`rwmodified` (e.g. read only became read/write),
  `roremoved`/`rwremoved`. Ctor/behaviour changes go as small orange
  lines inside the node; give shapes when useful
  (`$N\times n_{\rm dust}$`). Skip (b) if no connection changes. With no sync edge involved, drop the
  right column; it can hold edges created for the new nodes instead.
- **other panels** when the change lives elsewhere: `evolve_once` steps
  around `solver_sequence.evaluate()`, or the config choosing which nodes
  are built.
- **styles**: colour (and dashes) gives the diff state, line width the
  access. `new` green dashed, `modified` orange solid, `removed` red
  dashed with grey text, unchanged grey; read only thin (`semithick`),
  read/write thick (`line width=2.2pt`). Notes take the colour of what
  they describe; non-graph notes are grey. The legend lists exactly the
  styles used, including any you add (one per access for each state).
- **size**: past ~25 boxes, split into several pages (one `tikzpicture`
  per page of the `standalone` document) rather than shrinking text; each
  page has its own title and a legend of the styles used on that page.

Layout: level each input with the first node reading it; arrows may fan
out and cross arrows, never boxes. Pitfalls: escape `_` as `\_`; brace
`fit={...}` when it holds `$(...)$` coordinates; `gedge` labels wider than
30mm overflow their frame (move detail into notes); put notes inside the
`fit` of their frame; if (a) grows, wrap it in a `yshift` scope.

## 4. Build, look, fix

```bash
command -v pdflatex   # if missing (needs root, else ask the user):
# apt-get install -y --no-install-recommends texlive-latex-base texlive-pictures texlive-latex-extra
cd <outdir> && pdflatex -interaction=nonstopmode <topic>_solvergraph.tex
pdftoppm -png -r 120 <topic>_solvergraph.pdf <topic>_preview   # -> <topic>_preview-1.png
# zoom on small details with pdftoppm -png -r 200 -x X -y Y -W W -H H ...
```

Fix every LaTeX error (`grep -n '^!' <topic>_solvergraph.log`). Then
**Read the PNG** and fix overlaps, labels crossing arrows or boxes, boxes
outside frames, unreadable text or wrong new/modified marks; recompile and
look again until clean. Then `cp <topic>_solvergraph.tex <topic>_solvergraph.txt`,
delete `.aux`/`.log`, send the `.pdf` and `.txt` to the user (e.g.
SendUserFile when available) and give the absolute path of the `.tex`.
List briefly what is drawn as new / modified / removed so they can check.

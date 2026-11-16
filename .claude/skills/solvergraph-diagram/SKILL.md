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

Output: `<short_topic>_solvergraph.{tex,pdf,txt}` in the scratchpad (never
committed). Start from `template.tex` next to this file: a complete worked
example (sink accreted dust mass tracking, plus `if_has_sinks` turned from
an `OperationSequence` into an `OperationIf`). Keep its preamble, styles,
colours and legend; replace the body. The template's nodes, edges and notes
are an illustration only: never reuse a name or note without checking it in
the diff (similar features can be wired differently).

## 1. Get the change

- commit: `git show <sha>`; range/branch: `git diff $(git merge-base A B) B`
  (a fork's `origin/main` may be stale, so `A..B` can drag in unrelated
  history: diff from the merge-base with upstream main instead);
- patch file: read it;
- upstream PR `N`:
  `git fetch https://github.com/Shamrock-code/Shamrock pull/N/head:pr-N main:refs/remotes/upstream/main`,
  then `git diff $(git merge-base upstream/main pr-N) pr-N`.

Read the full new versions of the touched files, not only the hunks.

## 2. Identify the affected part of the graph

The graph is built in `init_solver_graph()` (`src/shammodels/<model>/src/Solver.cpp`):

- nodes: `solver_graph.register_node("name", NodeType{ctor args})` then
  `.set_edges(...)`. The node class's `NODE_EDGES(X_RO, X_RW[, X_RO_OPTIONAL, X_RW_OPTIONAL])`
  (see `shamsolvergraph/node/INode.hpp`) gives read-only (`->`) vs
  read/write (`<->`) and optional edges. Check the `.set_edges` call too:
  optional edges may only be wired under some condition. Some nodes
  (`NodeSetEdge`, `OperationIf`, ...) set their edges by hand
  (`__internal_set_*_edges`): read the class. Edges made locally
  (`IDataEdge::make_shared(...)`) and passed straight to `set_edges` are
  never registered: draw them as edges labelled `(local edge)`.
- execution order: `OperationSequence(name, {nodes...})`; conditional
  steps: `OperationIf(name, body)` gated on an `IDataEdge<bool>`, or C++
  `if` around `push_back` of a node (build-time condition: say which).
- solver_graph edges: `register_edge`, `FieldRefs`, `Field`, `Indexes`,
  `IDataEdge`, ... (left column).
- scheduler synchronized data edges: `sched.synchronized_data`, created by
  the `ensure_sync_edge` lambda at the top of `init_solver_graph` (right
  column; serialized in dumps). Code outside `init_solver_graph` touching
  them (e.g. resizing in `Model.hpp` on `add_sink` or dump load) is not a
  graph change: mention it in a note under the sync frame.

Classify each sequence step, node, edge and connection as **new**,
**modified** (same name, changed interface/ctor/behaviour/edges) or
**removed**, and write the list down before drawing. An existing edge
that gains a connection is not new: only the arrow is (`ronew`). An
existing node newly pushed into a sequence is a new step in (a) but a
plain node in (b). `(optional)` goes on the arrow for an
`EXPAND_NODE_EDGES_OPTIONAL` connection, on the box only if the edge
itself is registered conditionally. Draw only the
sequence(s) and nodes the change touches, plus the unchanged neighbours
needed to read them (e.g. the node producing an edge a new node reads).
Big unchanged nodes may be dropped if the panel title says so.

## 3. Draw (copy the template's structure)

- **(a) execution order**: title names the sequence and its enclosing
  `OperationIf`/sequence; `seq` boxes chained with `flow` arrows, at most
  ~8 per row, rows joined with the template's wrap-around arrow. Small
  `text=newc` notes say when new steps run and what modified steps now do;
  if new steps sit mid-row (notes would hit the wrap arrows), use one
  footnote under the sequence instead. A body sequence run by an
  `OperationIf` gets a frame around its steps (`fit`, dashed green if new),
  the `IDataEdge<bool>` condition as a `gedge` above the If step (condition
edges go in (a), not (b)). State a build-time flag and its meaning once
(e.g. in the frame label) rather than in every note. A change of
  kind (e.g. Sequence -> If) is `modified`, with a note saying before/after.
- **(b) data flow**: `gnode` compute nodes in a middle column in execution
  order; `gedge` inputs on the left grouped in a `fit` frame; `sedge`
  sync edges on the right in a `syncfill` frame. Edges produced by a node
  in the figure sit between the columns (template: `sink_accretion_table`).
  `ro`/`rw` for existing connections, `ronew`/`rwnew` for new ones,
  `roremoved`/`rwremoved` for removed ones. Put ctor/behaviour changes as
  small green lines inside the modified node. Mark optional edges
  `(optional)` and give shapes when useful (`$N\times n_{\rm dust}$`).
- styles: `new` (dashed green) for added things, `modified` (solid green)
  for existing things whose interface/behaviour changed, `removed` (dashed
  red, grey text) for removed ones. Drop legend entries the
  figure does not use; add `removed` ones only if something is removed.
- Place rows so each input sits level with the first node reading it;
  arrows from one edge to several nodes may fan out; arrows may cross
  arrows but not boxes. Escape `_` as `\_` everywhere. Brace `fit={...}`
  when it holds `$(...)$` coordinates with commas. `gedge` labels wider
  than 30mm stick out of their frame: keep them short, move detail into
  the node's notes. Include notes in `fit` frames. If (a) grows, shift it
  in a `yshift` scope rather than moving everything in (b).

## 4. Build, look, fix

```bash
command -v pdflatex || apt-get install -y --no-install-recommends \
  texlive-latex-base texlive-pictures texlive-latex-extra
cd <outdir> && pdflatex -interaction=nonstopmode <topic>_solvergraph.tex
pdftoppm -png -r 70 <topic>_solvergraph.pdf <topic>_preview
# 70 dpi can mislead: zoom on details with pdftoppm -png -r 160 -x X -y Y -W W -H H ...
```

`<outdir>` is the scratchpad unless the user names one; delete the
`.aux`/`.log`/preview PNGs once done.

Fix every LaTeX error (check the `.log`, `grep -n '^!'`). Then **Read the
PNG** and fix overlapping boxes, labels crossing arrows or boxes, boxes
sticking out of frames, unreadable text or the wrong new/modified
marking; recompile and look again until clean. Then
`cp <topic>_solvergraph.tex <topic>_solvergraph.txt`, send the `.pdf` and
`.txt` to the user (SendUserFile) and give the absolute path of the `.tex`.
Briefly list what is drawn as new / modified / removed so they can check.

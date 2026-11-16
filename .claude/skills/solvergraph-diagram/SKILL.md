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
example (sink accreted dust mass tracking). Keep its preamble, styles,
colours and legend; replace the body.

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
  optional edges may only be wired under some condition.
- execution order: `OperationSequence(name, {nodes...})`; conditional
  steps: `OperationIf(name, body)` gated on an `IDataEdge<bool>`, or C++
  `if` around `push_back` of a node (build-time condition: say which).
- solver_graph edges: `register_edge`, `FieldRefs`, `Field`, `Indexes`,
  `IDataEdge`, ... (left column).
- scheduler synchronized data edges: `sched.synchronized_data`, created by
  the `ensure_sync_edge` lambda at the top of `init_solver_graph` (right
  column; serialized in dumps).

Classify each sequence step, node, edge and connection as **new**,
**modified** (same name, changed interface/ctor/behaviour/edges) or
**removed**, and write the list down before drawing. Draw only the
sequence(s) and nodes the change touches, plus the unchanged neighbours
needed to read them (e.g. the node producing an edge a new node reads).

## 3. Draw (copy the template's structure)

- **(a) execution order**: title names the sequence and its enclosing
  `OperationIf`/sequence; `seq` boxes chained with `flow` arrows, at most
  ~8 per row, rows joined with the template's wrap-around arrow. Small
  `text=newc` notes say when new steps run and what modified steps now do.
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
  arrows from one edge to several nodes may fan out, but avoid arrows
  crossing boxes. Escape `_` as `\_` everywhere.

## 4. Build, look, fix

```bash
command -v pdflatex || apt-get install -y --no-install-recommends \
  texlive-latex-base texlive-pictures texlive-latex-extra
cd <scratchpad> && pdflatex -interaction=nonstopmode <topic>_solvergraph.tex
pdftoppm -png -r 70 <topic>_solvergraph.pdf <topic>_preview
```

Fix every LaTeX error (check the `.log`, `grep -n '^!'`). Then **Read the
PNG** and fix overlapping boxes, labels crossing arrows or boxes, boxes
sticking out of frames, unreadable text or the wrong new/modified
marking; recompile and look again until clean. Then
`cp <topic>_solvergraph.tex <topic>_solvergraph.txt`, send the `.pdf` and
`.txt` to the user (SendUserFile) and give the absolute path of the `.tex`.
Briefly list what is drawn as new / modified / removed so they can check.

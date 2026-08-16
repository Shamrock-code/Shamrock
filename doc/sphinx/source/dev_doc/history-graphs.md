# History graphs

CI metrics are stored on the
[`metrics-history`](https://github.com/Shamrock-code/Shamrock/tree/metrics-history)
branch. The charts below load those live series.

## Doxygen warnings

CI records the number of Doxygen warnings produced when the documentation is
built.

```{raw} html
<iframe
  src="../_static/doxygen_warnings.html"
  title="Doxygen warning count over time"
  width="100%"
  height="600"
  style="border: none;"
></iframe>
```

Raw JSON:
[doxygen_warnings.json](https://raw.githubusercontent.com/Shamrock-code/Shamrock/refs/heads/metrics-history/output/doxygen_warnings.json).

The org website (`https://shamrock-code.github.io/`) is same-origin with the
published Sphinx docs, so it can embed this chart with:

```html
<iframe
  src="https://shamrock-code.github.io/Shamrock/sphinx/_static/doxygen_warnings.html"
  title="Doxygen warning count over time"
  width="100%"
  height="600"
  style="border: none;"
></iframe>
```

## Build time

CI records ClangBuildAnalyzer compile times: the sum of frontend parsing and
backend codegen across translation units. That is cumulative compiler work, not
wall-clock time. The right axis shows how many translation units were profiled.
See [Profiling build performance / time](build-profiling.md).

```{raw} html
<iframe
  src="../_static/build_time_total.html"
  title="Build time and translation units over time"
  width="100%"
  height="640"
  style="border: none;"
></iframe>
```

Raw JSON:
[build_time_total.json](https://raw.githubusercontent.com/Shamrock-code/Shamrock/refs/heads/metrics-history/output/build_time_total.json).

The org website can embed this chart with:

```html
<iframe
  src="https://shamrock-code.github.io/Shamrock/sphinx/_static/build_time_total.html"
  title="Build time and translation units over time"
  width="100%"
  height="640"
  style="border: none;"
></iframe>
```

## Lines of code

CI counts lines in tracked source files, excluding git submodules. Exclusive
partitions (`code`, `examples`, `doc`) sum to `all`. Nested `shammodels/*`
counts are subsets of `code`, not extra buckets. File-type totals sum the
exclusive partitions only. Both y-axes use a log scale so small series stay
visible next to the totals.

```{raw} html
<iframe
  src="../_static/loc.html"
  title="Lines of code over time"
  width="100%"
  height="900"
  style="border: none;"
></iframe>
```

Raw JSON:
[loc.json](https://raw.githubusercontent.com/Shamrock-code/Shamrock/refs/heads/metrics-history/output/loc.json).

The org website can embed this chart with:

```html
<iframe
  src="https://shamrock-code.github.io/Shamrock/sphinx/_static/loc.html"
  title="Lines of code over time"
  width="100%"
  height="900"
  style="border: none;"
></iframe>
```

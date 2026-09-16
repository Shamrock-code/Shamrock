(function (root) {
  "use strict";

  const PLOT_EVENTS = [{ date: "2026-08-17T13:43:38Z", text: "force -j 4 in gh-runners" }];

  const THEME = {
    light: {
      bg: "#fff",
      fg: "#3c3836",
      muted: "#665c54",
      grid: "#d5c4a1",
      event: "#b57614",
      eventBg: "rgba(255, 255, 255, 0.9)",
      parsing: "#076678",
      codegen: "#af3a03",
      total: "#79740e",
      tus: "#8f3f71",
      palette: ["#076678", "#af3a03", "#79740e", "#8f3f71", "#9d0006", "#427b58", "#b57614", "#d65d0e"],
    },
    dark: {
      bg: "#282828",
      fg: "#ebdbb2",
      muted: "#a89984",
      grid: "#504945",
      event: "#fabd2f",
      eventBg: "rgba(40, 40, 40, 0.9)",
      parsing: "#83a598",
      codegen: "#fe8019",
      total: "#b8bb26",
      tus: "#d3869b",
      palette: ["#83a598", "#fe8019", "#b8bb26", "#d3869b", "#fb4934", "#8ec07c", "#fabd2f", "#d65d0e"],
    },
  };

  function themeColors(theme) {
    return THEME[theme === "dark" ? "dark" : "light"];
  }

  function metricsUrls(filename) {
    return [
      "https://raw.githubusercontent.com/Shamrock-code/Shamrock/refs/heads/metrics-history/output/" + filename,
      "https://cdn.jsdelivr.net/gh/Shamrock-code/Shamrock@metrics-history/output/" + filename,
    ];
  }

  function parentTheme() {
    try {
      if (window.parent && window.parent !== window) {
        const theme = window.parent.document.documentElement.getAttribute("data-theme");
        if (theme === "dark" || theme === "light") {
          return theme;
        }
      }
    } catch (err) {
      /* Cross-origin parent: fall back to prefers-color-scheme. */
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
  }

  function finiteNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function paddedRange(values, integer) {
    const nums = values.filter((value) => typeof value === "number" && Number.isFinite(value));
    if (nums.length === 0) {
      return undefined;
    }
    const min = Math.min(...nums);
    const max = Math.max(...nums);
    const span = max - min;
    const pad = span === 0 ? Math.max(Math.abs(min) * 0.05, integer ? 1 : 0.05) : span * 0.1;
    let lo = min - pad;
    let hi = max + pad;
    if (integer) {
      lo = Math.floor(lo);
      hi = Math.ceil(hi);
      if (lo === hi) {
        lo -= 1;
        hi += 1;
      }
    }
    if (min >= 0) {
      lo = Math.max(0, lo);
    }
    return [lo, hi];
  }

  function paddedLogRange(values) {
    const nums = values.filter(
      (value) => typeof value === "number" && Number.isFinite(value) && value > 0
    );
    if (nums.length === 0) {
      return undefined;
    }
    const logMin = Math.log10(Math.min(...nums));
    const logMax = Math.log10(Math.max(...nums));
    const span = logMax - logMin;
    const pad = span === 0 ? 0.1 : span * 0.1;
    return [logMin - pad, logMax + pad];
  }

  function axisStyle(theme) {
    const colors = themeColors(theme);
    return {
      gridcolor: colors.grid,
      zerolinecolor: colors.grid,
      linecolor: colors.muted,
      tickfont: { color: colors.muted },
    };
  }

  function eventDecorations(theme, xValues, events) {
    const colors = themeColors(theme);
    const color = colors.event;
    const bgcolor = colors.eventBg;
    const xs = Array.isArray(xValues) ? xValues : [];
    let xMin = null;
    let xMax = null;
    if (xs.length > 0) {
      xMin = Date.parse(xs[0]);
      xMax = Date.parse(xs[xs.length - 1]);
    }
    const shapes = [];
    const annotations = [];
    const markers = events || PLOT_EVENTS;
    for (let i = 0; i < markers.length; i++) {
      const event = markers[i];
      const eventMs = Date.parse(event.date);
      const nearEnd =
        xMin !== null && xMax !== null && xMax > xMin ? (eventMs - xMin) / (xMax - xMin) > 0.7 : true;
      shapes.push({
        type: "line",
        x0: event.date,
        x1: event.date,
        y0: 0,
        y1: 1,
        xref: "x",
        yref: "paper",
        line: { color: color, width: 1.5, dash: "dot" },
      });
      annotations.push({
        x: event.date,
        y: 0.98,
        xref: "x",
        yref: "paper",
        text: event.text,
        showarrow: false,
        xanchor: nearEnd ? "right" : "left",
        yanchor: "top",
        xshift: nearEnd ? -8 : 8,
        font: { color: color, size: 12 },
        bgcolor: bgcolor,
        bordercolor: color,
        borderwidth: 1,
        borderpad: 4,
      });
    }
    return { shapes: shapes, annotations: annotations };
  }

  function tracesXValues(traces) {
    if (traces && traces[0] && Array.isArray(traces[0].x)) {
      return traces[0].x;
    }
    return [];
  }

  function themedLayout(baseLayout, theme, extra) {
    const colors = themeColors(theme);
    const layout = Object.assign({}, baseLayout || {});
    const title = Object.assign({}, layout.title || {});
    title.font = Object.assign({ size: 18 }, title.font || {}, { color: colors.fg });
    layout.title = title;
    layout.paper_bgcolor = colors.bg;
    layout.plot_bgcolor = colors.bg;
    layout.font = Object.assign({}, layout.font || {}, { color: colors.fg });
    layout.xaxis = Object.assign({}, layout.xaxis || {}, axisStyle(theme));
    layout.yaxis = Object.assign({}, layout.yaxis || {}, axisStyle(theme));
    const traces = extra && extra.traces;
    if (traces) {
      const eventList = Array.isArray(extra.events) ? extra.events : undefined;
      const events = eventDecorations(theme, tracesXValues(traces), eventList);
      layout.shapes = events.shapes;
      layout.annotations = events.annotations;
    }
    return layout;
  }

  function plotlyTraces(payload) {
    if (!payload || !Array.isArray(payload.data)) {
      throw new Error("Unexpected dataset shape: missing Plotly data[]");
    }
    return payload.data.filter(function (trace) {
      return (
        trace &&
        Array.isArray(trace.x) &&
        Array.isArray(trace.y) &&
        trace.x.length === trace.y.length &&
        trace.x.length > 0
      );
    });
  }

  async function fetchDataset(urls) {
    const errors = [];
    for (let i = 0; i < urls.length; i++) {
      const url = urls[i];
      try {
        const response = await fetch(url, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(response.status + " " + response.statusText);
        }
        return await response.json();
      } catch (err) {
        errors.push(url + ": " + err.message);
      }
    }
    throw new Error(errors.join("; "));
  }

  function setStatus(message) {
    const status = document.getElementById("status");
    if (!status) {
      return;
    }
    if (message) {
      status.textContent = message;
      status.hidden = false;
    } else {
      status.hidden = true;
    }
  }

  function applyEventDecorations(plot, theme, events) {
    const xs = tracesXValues(plot.traces);
    const decorations = eventDecorations(theme, xs, events);
    plot.layout = Object.assign({}, plot.layout || {}, {
      shapes: decorations.shapes,
      annotations: decorations.annotations,
    });
    return plot;
  }

  function mount(options) {
    const chart = document.getElementById("chart");
    const loading = options.loading;
    const empty = options.empty;
    const error = options.error;
    const urls = options.urls;
    const makePlot = options.makePlot;
    const useEvents = options.events;

    if (loading) {
      setStatus(loading);
    }

    function plotFor(payload, theme) {
      const plot = makePlot(payload, theme) || {};
      plot.traces = plot.traces || [];
      plot.layout = plot.layout || {};
      if (useEvents) {
        applyEventDecorations(plot, theme);
      }
      return plot;
    }

    function bindTheme(restyle) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", restyle);
      try {
        if (window.parent && window.parent !== window) {
          const observer = new MutationObserver(restyle);
          observer.observe(window.parent.document.documentElement, {
            attributes: true,
            attributeFilter: ["data-theme"],
          });
        }
      } catch (err) {
        /* Ignore if the parent document is not readable. */
      }
    }

    async function main() {
      let theme = parentTheme();
      applyTheme(theme);

      let payload;
      try {
        payload = await fetchDataset(urls);
      } catch (err) {
        setStatus(error + "\n" + err.message);
        return;
      }

      const first = plotFor(payload, theme);
      if (!first.traces.length) {
        setStatus(empty);
        return;
      }

      await Plotly.newPlot(chart, first.traces, first.layout, {
        responsive: true,
        displaylogo: false,
      });
      setStatus("");

      bindTheme(function () {
        theme = parentTheme();
        applyTheme(theme);
        const next = plotFor(payload, theme);
        Plotly.react(chart, next.traces, next.layout);
      });
    }

    main();
  }

  root.MetricsChart = {
    PLOT_EVENTS: PLOT_EVENTS,
    metricsUrls: metricsUrls,
    themeColors: themeColors,
    parentTheme: parentTheme,
    applyTheme: applyTheme,
    finiteNumber: finiteNumber,
    paddedRange: paddedRange,
    paddedLogRange: paddedLogRange,
    axisStyle: axisStyle,
    eventDecorations: eventDecorations,
    themedLayout: themedLayout,
    fetchDataset: fetchDataset,
    plotlyTraces: plotlyTraces,
    mount: mount,
  };
})(window);

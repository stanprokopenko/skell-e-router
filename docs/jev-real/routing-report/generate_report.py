"""Build routing-report.html from the CSVs in data/.

Reads ONLY the CSVs (csv.DictReader). No jsonl, no summary json, no hardcoded
numbers. Run extract.py first.

Usage:  python generate_report.py
"""

import csv
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "routing-report.html")

ARMS = ["gemini_prod", "luna_low", "jev"]

# Categorical palette, fixed slot order, one slot per model, never reassigned.
# Slot 1 blue -> today's classifier, slot 2 aqua -> Luna, slot 3 Proko coral -> Jev.
# Validated with dataviz/scripts/validate_palette.js:
#   light "#2a78d6,#1baf7a,#fe5b50" --surface #ffffff --pairs all -> ALL CHECKS PASS
#   dark  "#3987e5,#199e70,#e5473c" --surface #242424 --pairs all -> ALL CHECKS PASS
SERIES_LIGHT = {"gemini_prod": "#2a78d6", "luna_low": "#1baf7a", "jev": "#fe5b50"}
SERIES_DARK = {"gemini_prod": "#3987e5", "luna_low": "#199e70", "jev": "#e5473c"}


def read(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def esc(text):
    return html.escape(str(text), quote=True)


def num(value):
    """CSV cell -> int or float."""
    f = float(value)
    return int(f) if f.is_integer() else f


def money(value):
    return "$%.3f" % float(value)


def pct(value):
    return "%.1f%%" % float(value)


def thousands(value):
    return "{:,}".format(int(float(value)))


# ---------------------------------------------------------------- load data
headline = read("headline.csv")
prod = read("prod_shaped.csv")
quart = read("jev_quartiles.csv")
cuts = [row["cut_pct"] for row in read("jev_quartile_cuts.csv")]
cats = read("category_accuracy.csv")
cat_group = read("category_grouping.csv")
policies = read("jev_policies.csv")
hybrid = read("hybrid.csv")
agreement = read("agreement.csv")
meta = read("meta.csv")[0]

H = {row["arm"]: row for row in headline}
P = {row["arm"]: row for row in prod}
LABEL = {row["arm"]: row["model_label"] for row in headline}

case_sets = [
    ("jev-wrong", "Every case Jev got wrong",
     "Jev's mistakes, least confident first. If Jev is going to hurt you, it is here.",
     read("cases_jev_wrong.csv")),
    ("jev-over-luna", "Jev right, Luna wrong",
     "Cases Jev got right that gpt-5.6-luna missed.",
     read("cases_jev_right_luna_wrong.csv")),
    ("jev-over-prod", "Jev right, today's classifier wrong",
     "Cases Jev got right that the model in production today missed.",
     read("cases_jev_right_gemini_wrong.csv")),
    ("all-wrong", "Cases all three got wrong",
     "The hard messages. These are where the labels themselves are worth a second look.",
     read("cases_all_wrong.csv")),
]


# ------------------------------------------------------------- html helpers
def table(headers, rows, classes="", aligns=None):
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)
    out = ['<div class="table-wrap"><table class="%s">' % classes, "<thead><tr>"]
    for head, align in zip(headers, aligns):
        out.append('<th class="a-%s">%s</th>' % (align, esc(head)))
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for cell, head, align in zip(row, headers, aligns):
            out.append('<td class="a-%s" data-label="%s">%s</td>' % (align, esc(head), cell))
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def swatch(arm):
    return '<span class="dot" data-arm="%s"></span>' % arm


def model_cell(arm):
    return '%s%s' % (swatch(arm), esc(LABEL[arm]))


def chart(chart_id, height, mobile_height=None):
    style = "height:%dpx" % height
    attr = ' data-mobile-height="%d"' % mobile_height if mobile_height else ""
    return ('<div class="chart-box" style="%s" data-height="%d"%s>'
            '<canvas id="%s"></canvas></div>' % (style, height, attr, chart_id))


def pill(kind, text):
    return '<span class="pill pill-%s">%s</span>' % (kind, esc(text))


def pred_pill(value):
    return pill("big" if value == "big" else "fast", value)


# ------------------------------------------------------------- chart config
chart_data = {
    "series": {"light": SERIES_LIGHT, "dark": SERIES_DARK},
    "cases": num(meta["cases"]),
    "labels": LABEL,
    "arms": ARMS,
    "counts": {
        "groups": ["Sent to the cheap model by mistake", "Sent to the expensive model by mistake"],
        "byArm": {a: [num(H[a]["missed_big"]), num(H[a]["false_big"])] for a in ARMS},
    },
    "prodShaped": {"byArm": {a: float(P[a]["accuracy_pct"]) for a in ARMS},
                   "correct": {a: num(P[a]["correct"]) for a in ARMS},
                   "n": num(P["jev"]["n"])},
    "latency": {"groups": ["Typical wait", "Slowest 1 in 20"],
                "byArm": {a: [float(H[a]["median_s"]), float(H[a]["p95_s"])] for a in ARMS}},
    "cost": {"byArm": {a: float(H[a]["cost_per_1000_usd"]) for a in ARMS}},
    "quartiles": {
        "labels": [r["quartile_label"] for r in quart],
        "accuracy": [float(r["accuracy_pct"]) for r in quart],
        "n": [num(r["n"]) for r in quart],
        "correct": [num(r["correct"]) for r in quart],
    },
    "categories": {
        "labels": [r["category"] for r in cats],
        "n": [num(r["n"]) for r in cats],
        "byArm": {a: [float(r["%s_accuracy_pct" % a]) for r in cats] for a in ARMS},
        "correctByArm": {a: [num(r["%s_correct" % a]) for r in cats] for a in ARMS},
    },
}

CSS = """
:root{
  color-scheme: light;
  --bg-app:#f7f7f7; --bg-panel:#ffffff; --bg-hover:rgba(18,18,18,0.06);
  --bg-inset:#f7f7f7;
  --accent:#fe5b50; --accent-hover:#e5473c; --accent-soft:#ffeceb;
  --text-primary:#242424; --text-muted:#757575; --text-bright:#ffffff;
  --border-subtle:rgba(18,18,18,0.10); --grid:rgba(18,18,18,0.10);
  --shadow-card:rgba(0,0,0,0.08) 0 8px 40px 0;
  --shadow-header:rgba(0,0,0,0.08) 0 0 16px 0;
  --s-gemini_prod:#2a78d6; --s-luna_low:#1baf7a; --s-jev:#fe5b50;
  --good:#3f7d20; --bad:#c23a30; --surface-gap:#ffffff;
}
[data-theme="dark"]{
  color-scheme: dark;
  --bg-app:#1c1c1c; --bg-panel:#242424; --bg-hover:rgba(247,247,247,0.06);
  --bg-inset:#1c1c1c;
  --accent:#e5473c; --accent-hover:#d2382e; --accent-soft:#3a2422;
  --text-primary:#e3e3e3; --text-muted:#adadad; --text-bright:#f7f7f7;
  --border-subtle:rgba(247,247,247,0.10); --grid:rgba(247,247,247,0.10);
  --shadow-card:rgba(0,0,0,0.16) 0 4px 16px 0;
  --shadow-header:rgba(0,0,0,0.08) 0 0 16px 0;
  --s-gemini_prod:#3987e5; --s-luna_low:#199e70; --s-jev:#e5473c;
  --good:#97c656; --bad:#ff8a80; --surface-gap:#242424;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;max-width:100%;overflow-x:hidden}
body{
  background:var(--bg-app); color:var(--text-primary);
  font-family:"Open Sans","Segoe UI",system-ui,-apple-system,sans-serif;
  font-size:16px; line-height:24px; -webkit-text-size-adjust:100%;
}
.wrap{max-width:1040px;margin:0 auto;padding:16px 16px 0}
h1,h2,h3{font-family:Rubik,"Open Sans","Segoe UI",system-ui,sans-serif;font-weight:500;margin:0}
h1{font-size:28px;line-height:34px;letter-spacing:-0.01em}
h2{font-size:20px;line-height:28px;margin-bottom:8px}
h3{font-size:15px;line-height:22px;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted);margin-bottom:8px}
p{margin:0 0 12px}
a{color:var(--accent);text-decoration:none}
.muted{color:var(--text-muted)}
.small{font-size:14px;line-height:20px}
.tiny{font-size:12px;line-height:16px}
.eyebrow{font-family:Rubik,sans-serif;font-size:12px;letter-spacing:0.12em;text-transform:uppercase;color:var(--text-muted)}

header.top{background:var(--bg-panel);box-shadow:var(--shadow-header);padding:20px 0 24px;margin-bottom:24px}
.theme-btn{
  border:1px solid var(--border-subtle);background:var(--bg-hover);color:var(--text-primary);
  font-family:Rubik,sans-serif;font-size:12px;font-weight:500;text-transform:uppercase;letter-spacing:0.06em;
  padding:8px 12px;border-radius:8px;cursor:pointer;transition:all .3s;white-space:nowrap;
}
.theme-btn:hover{background:var(--accent);color:var(--text-bright);border-color:var(--accent)}
.head-row{display:flex;gap:12px;align-items:flex-start;justify-content:space-between}

section{margin:0 0 32px}
.card{background:var(--bg-panel);border-radius:16px;box-shadow:var(--shadow-card);padding:20px}
.card + .card{margin-top:16px}

.hero-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
.hero-card{background:var(--bg-panel);border-radius:16px;box-shadow:var(--shadow-card);padding:20px;border-top:4px solid var(--bar)}
.hero-card .name{font-family:Rubik,sans-serif;font-size:16px;font-weight:500;display:flex;align-items:center;gap:8px}
.hero-card .sub{font-size:12px;line-height:16px;color:var(--text-muted);margin-bottom:12px;min-height:32px}
.hero-fig{font-size:44px;line-height:52px;font-weight:600;letter-spacing:-0.02em}
.hero-fig .of{font-size:16px;font-weight:400;color:var(--text-muted);letter-spacing:0}
.hero-label{font-size:13px;line-height:18px;color:var(--text-muted);margin-bottom:12px}
.stat-row{display:flex;justify-content:space-between;gap:8px;padding:7px 0;border-top:1px solid var(--border-subtle);font-size:14px}
.stat-row .v{font-weight:600;font-variant-numeric:tabular-nums}
.badge-best{display:inline-block;background:var(--accent-soft);color:var(--accent);font-size:11px;font-weight:600;
  text-transform:uppercase;letter-spacing:0.06em;padding:2px 8px;border-radius:9999px;margin-left:auto}

.dot{width:10px;height:10px;border-radius:9999px;display:inline-block;margin-right:8px;vertical-align:baseline;flex:none}
.dot[data-arm="gemini_prod"]{background:var(--s-gemini_prod)}
.dot[data-arm="luna_low"]{background:var(--s-luna_low)}
.dot[data-arm="jev"]{background:var(--s-jev)}

.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:0 0 12px;font-size:13px;color:var(--text-muted)}
.legend span.key{display:flex;align-items:center}
.chart-box{position:relative;width:100%;overflow:hidden;margin:4px 0 8px}
.chart-note{font-size:12px;line-height:16px;color:var(--text-muted);margin:0 0 12px}

.table-wrap{width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:9px 10px;border-bottom:1px solid var(--border-subtle);vertical-align:top}
th{font-family:Rubik,sans-serif;font-size:12px;font-weight:500;text-transform:uppercase;letter-spacing:0.04em;
   color:var(--text-muted);white-space:nowrap}
td{font-variant-numeric:tabular-nums}
td.a-left,th.a-left{text-align:left}
td.a-right,th.a-right{text-align:right}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--bg-hover)}
td .wrapme{font-variant-numeric:normal;display:block;max-width:100%;overflow-wrap:break-word}
td .wrapme.msg{overflow-wrap:anywhere}
table.cases th,table.cases td{text-align:left}
table.cases th:first-child{padding-left:10px}

.pill{display:inline-block;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;
  padding:2px 8px;border-radius:9999px;white-space:nowrap}
.pill-big{background:var(--accent-soft);color:var(--accent)}
.pill-fast{background:var(--bg-hover);color:var(--text-muted)}
.ok{color:var(--good);font-weight:600}
.no{color:var(--bad);font-weight:600}

details.eb{background:var(--bg-panel);border-radius:12px;box-shadow:var(--shadow-card);margin-bottom:12px;overflow:hidden}
details.eb > summary{
  list-style:none;cursor:pointer;padding:16px 20px;display:flex;gap:10px;align-items:center;justify-content:space-between;
  font-family:Rubik,sans-serif;font-size:15px;font-weight:500;
}
details.eb > summary::-webkit-details-marker{display:none}
details.eb > summary .count{font-size:12px;color:var(--text-muted);font-family:"Open Sans",sans-serif;font-weight:400}
details.eb > summary::after{content:"+";color:var(--accent);font-size:20px;line-height:1}
details.eb[open] > summary::after{content:"\\2212"}
details.eb .body{padding:0 20px 16px}
.show-all{margin-top:12px;border:1px solid var(--accent);background:transparent;color:var(--accent);
  font-family:Rubik,sans-serif;font-size:13px;font-weight:500;text-transform:uppercase;letter-spacing:0.05em;
  padding:8px 14px;border-radius:8px;cursor:pointer;transition:all .3s}
.show-all:hover{background:var(--accent);color:var(--text-bright)}
tr.extra{display:none}
.cases.shown tr.extra{display:table-row}

.callout{background:var(--bg-inset);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:12px 16px;margin:0 0 12px}
ul.clean{margin:0 0 12px;padding-left:20px}
ul.clean li{margin-bottom:6px}

footer{background:#242424;color:#ffffff;margin-top:40px;padding:32px 0}
footer .wrap{padding-bottom:32px}
footer h3{color:#ffffff}
footer code{background:rgba(255,255,255,0.08);padding:2px 6px;border-radius:4px;font-size:12px;overflow-wrap:anywhere}
footer p{color:#adadad;font-size:13px;line-height:20px}

@media (min-width:701px){
  table.compact td.a-right{white-space:nowrap}
  table.cases td[data-label="Message"]{width:40%}
  table.cases td[data-label="Type"],table.cases th:last-child{min-width:104px}
}
@media (max-width:700px){
  .wrap{padding:12px 12px 0}
  h1{font-size:24px;line-height:30px}
  h2{font-size:18px;line-height:24px}
  .card{padding:16px;border-radius:12px}
  .hero-card{padding:16px;border-radius:12px}
  .hero-fig{font-size:38px;line-height:44px}
  details.eb > summary{padding:14px 16px;font-size:14px}
  details.eb .body{padding:0 12px 14px}
  .table-wrap{overflow-x:visible}
  table.compact,table.compact tbody,table.compact tr,table.compact td{display:block;width:100%}
  table.compact thead{display:none}
  table.compact tr{border-bottom:1px solid var(--border-subtle);padding:10px 0}
  table.compact tr:last-child{border-bottom:none}
  table.compact td{border:none;padding:3px 0;display:flex;gap:12px;justify-content:space-between;align-items:baseline;text-align:right}
  table.compact td:first-child{display:block;text-align:left;font-weight:600;padding-bottom:6px}
  table.compact td:not(:first-child)::before{
    content:attr(data-label);color:var(--text-muted);font-size:12px;text-transform:uppercase;
    letter-spacing:0.04em;flex:none;text-align:left;font-weight:400}
  table.cases,table.cases tbody,table.cases tr,table.cases td{display:block;width:100%}
  table.cases thead{display:none}
  table.cases tr{border-bottom:1px solid var(--border-subtle);padding:12px 0}
  table.cases tr.extra{display:none}
  table.cases.shown tr.extra{display:block}
  table.cases tr:last-child{border-bottom:none}
  table.cases td{border:none;padding:2px 0;display:flex;gap:8px;justify-content:space-between;align-items:baseline}
  table.cases td[data-label="Message"]{display:block;padding-bottom:8px}
  table.cases td:not([data-label="Message"])::before{
    content:attr(data-label);color:var(--text-muted);font-size:12px;text-transform:uppercase;letter-spacing:0.04em;flex:none}
  table.compact th,table.compact td{padding:8px 6px;font-size:13px}
}
"""

JS = r"""
(function(){
  var D = window.__REPORT__;
  var charts = [];

  function cssVar(name){ return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function theme(){ return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light'; }
  function seriesColor(arm){ return D.series[theme()][arm]; }
  function textPrimary(){ return cssVar('--text-primary'); }
  function textMuted(){ return cssVar('--text-muted'); }
  function gridColor(){ return cssVar('--grid'); }
  function surface(){ return cssVar('--bg-panel'); }

  // Direct labels: value text at the cap of every bar, in text colors.
  // Formatters live outside chart options: Chart.js treats a function-valued
  // option as scriptable and would call it during option resolution.
  var FORMATTERS = {};
  var directLabels = {
    id: 'directLabels',
    afterDatasetsDraw: function(chart, args, opts){
      var fmt = FORMATTERS[chart.canvas.id];
      if(!fmt) return;
      var ctx = chart.ctx, horizontal = chart.options.indexAxis === 'y';
      ctx.save();
      ctx.font = '600 11px "Open Sans", "Segoe UI", system-ui, sans-serif';
      ctx.fillStyle = textPrimary();
      chart.data.datasets.forEach(function(ds, di){
        var meta = chart.getDatasetMeta(di);
        if(meta.hidden) return;
        meta.data.forEach(function(el, i){
          var text = fmt(ds.data[i], di, i);
          if(text === null || text === undefined || text === '') return;
          if(horizontal){
            ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
            var x = el.x + 6;
            if(x + ctx.measureText(text).width > chart.chartArea.right){
              ctx.textAlign = 'right'; x = el.x - 6; ctx.fillStyle = surface();
            } else { ctx.fillStyle = textPrimary(); }
            ctx.fillText(text, x, el.y);
          } else {
            ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
            ctx.fillStyle = textPrimary();
            ctx.fillText(text, el.x, el.y - 5);
          }
        });
      });
      ctx.restore();
    }
  };

  function baseOptions(extra){
    var o = {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 18, right: 8, bottom: 0, left: 0 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: theme() === 'dark' ? 'rgba(18,18,18,0.92)' : 'rgba(18,18,18,0.86)',
          titleColor: '#ffffff', bodyColor: '#ffffff', borderWidth: 0,
          cornerRadius: 6, padding: 10, displayColors: true, boxWidth: 8, boxHeight: 8, usePointStyle: true
        }
      },
      animation: { duration: 400 }
    };
    return Object.assign(o, extra || {});
  }

  function axes(opts){
    var o = opts || {};
    return {
      x: {
        stacked: false,
        grid: { display: o.vertical ? false : true, color: gridColor(), drawBorder: false, drawTicks: false, lineWidth: 1 },
        border: { display: false },
        ticks: { color: textMuted(), font: { size: 11 }, autoSkip: false,
                 maxRotation: 0, minRotation: 0,
                 callback: o.xCallback || (o.vertical ? function(v){ return wrapLabel(this.getLabelForValue(v)); } : undefined) },
        title: o.xTitle ? { display: true, text: o.xTitle, color: textMuted(), font: { size: 11 } } : undefined,
        beginAtZero: true,
        max: o.xMax,
        suggestedMax: o.xSuggestedMax
      },
      y: {
        grid: { display: o.vertical ? true : false, color: gridColor(), drawBorder: false, drawTicks: false, lineWidth: 1 },
        border: { display: false },
        ticks: { color: textMuted(), font: { size: 11 },
                 callback: o.yCallback || (o.vertical ? undefined : function(v){ return wrapLabel(this.getLabelForValue(v), 18); }) },
        title: o.yTitle ? { display: true, text: o.yTitle, color: textMuted(), font: { size: 11 } } : undefined,
        beginAtZero: true,
        max: o.yMax,
        suggestedMax: o.ySuggestedMax
      }
    };
  }

  function barStyle(color){
    return {
      backgroundColor: color, hoverBackgroundColor: color,
      borderRadius: 4, borderSkipped: 'start',
      maxBarThickness: 22, categoryPercentage: 0.7, barPercentage: 0.75
    };
  }

  function modelDatasets(values, opts){
    return D.arms.map(function(arm){
      return Object.assign({ label: D.labels[arm], data: values[arm], arm: arm }, barStyle(seriesColor(arm)), opts || {});
    });
  }

  function make(id, config, formatter){
    var el = document.getElementById(id);
    if(!el) return;
    FORMATTERS[id] = formatter || null;
    var c = new Chart(el.getContext('2d'), config);
    charts.push({ id: id, chart: c });
    return c;
  }

  function buildAll(){
    charts.forEach(function(c){ c.chart.destroy(); });
    charts = [];

    // 1. counts per model
    make('chartCounts', {
      type: 'bar',
      data: { labels: D.counts.groups, datasets: modelDatasets(D.counts.byArm) },
      options: baseOptions({
        scales: axes({ vertical: true, yTitle: 'Messages', ySuggestedMax: 60 }),
        plugins: Object.assign(baseOptions().plugins, {
          tooltip: Object.assign(baseOptions().plugins.tooltip, {
            callbacks: { label: function(c){ return c.dataset.label + ': ' + c.parsed.y + ' messages'; } }
          })
        })
      }),
      plugins: [directLabels]
    }, function(v){ return v; });

    // 2. production-shaped accuracy
    make('chartProd', {
      type: 'bar',
      data: {
        labels: D.arms.map(function(a){ return D.labels[a]; }),
        datasets: [Object.assign({
          label: 'Correct decisions', data: D.arms.map(function(a){ return D.prodShaped.byArm[a]; }),
          backgroundColor: D.arms.map(seriesColor), hoverBackgroundColor: D.arms.map(seriesColor)
        }, barStyle(null), { backgroundColor: D.arms.map(seriesColor) })]
      },
      options: baseOptions({
        scales: axes({ vertical: true, yTitle: 'Correct (%)', yMax: 100,
                       yCallback: function(v){ return v + '%'; } }),
        plugins: Object.assign(baseOptions().plugins, {
          tooltip: Object.assign(baseOptions().plugins.tooltip, {
            callbacks: { label: function(c){
              var arm = D.arms[c.dataIndex];
              return c.parsed.y.toFixed(1) + '% correct, ' + D.prodShaped.correct[arm] + ' of ' + D.prodShaped.n;
            } }
          })
        })
      }),
      plugins: [directLabels]
    }, function(v){ return v.toFixed(1) + '%'; });

    // 3. latency
    make('chartLatency', {
      type: 'bar',
      data: { labels: D.latency.groups, datasets: modelDatasets(D.latency.byArm) },
      options: baseOptions({
        scales: axes({ vertical: true, yTitle: 'Seconds', ySuggestedMax: 2,
                       yCallback: function(v){ return v + 's'; } }),
        plugins: Object.assign(baseOptions().plugins, {
          tooltip: Object.assign(baseOptions().plugins.tooltip, {
            callbacks: { label: function(c){ return c.dataset.label + ': ' + c.parsed.y.toFixed(2) + ' seconds'; } }
          })
        })
      }),
      plugins: [directLabels]
    }, function(v){ return v.toFixed(2) + 's'; });

    // 4. cost
    make('chartCost', {
      type: 'bar',
      data: {
        labels: D.arms.map(function(a){ return D.labels[a]; }),
        datasets: [Object.assign({ label: 'Cost per 1,000 calls',
          data: D.arms.map(function(a){ return D.cost.byArm[a]; }) },
          barStyle(null), { backgroundColor: D.arms.map(seriesColor), hoverBackgroundColor: D.arms.map(seriesColor) })]
      },
      options: baseOptions({
        scales: axes({ vertical: true, yTitle: 'US dollars per 1,000 calls',
                       yCallback: function(v){ return '$' + v.toFixed(2); } }),
        plugins: Object.assign(baseOptions().plugins, {
          tooltip: Object.assign(baseOptions().plugins.tooltip, {
            callbacks: { label: function(c){ return '$' + c.parsed.y.toFixed(3) + ' per 1,000 calls'; } }
          })
        })
      }),
      plugins: [directLabels]
    }, function(v){ return '$' + v.toFixed(3); });

    // 5. Jev accuracy by confidence quartile (single series -> Jev's own color)
    make('chartQuartile', {
      type: 'bar',
      data: {
        labels: D.quartiles.labels,
        datasets: [Object.assign({ label: 'Jev correct', data: D.quartiles.accuracy },
          barStyle(seriesColor('jev')))]
      },
      options: baseOptions({
        scales: axes({ vertical: true, yTitle: 'Correct (%)', yMax: 100,
                       yCallback: function(v){ return v + '%'; } }),
        plugins: Object.assign(baseOptions().plugins, {
          tooltip: Object.assign(baseOptions().plugins.tooltip, {
            callbacks: { label: function(c){
              return c.parsed.y.toFixed(1) + '% correct, ' + D.quartiles.correct[c.dataIndex] +
                     ' of ' + D.quartiles.n[c.dataIndex] + ' messages';
            } }
          })
        })
      }),
      plugins: [directLabels]
    }, function(v, di, i){ return v.toFixed(1) + '%'; });

    // 6. accuracy by message type (horizontal)
    make('chartCategory', {
      type: 'bar',
      data: {
        labels: D.categories.labels.map(function(l, i){ return l + ' (' + D.categories.n[i] + ')'; }),
        datasets: modelDatasets(D.categories.byArm, { maxBarThickness: 10 })
      },
      options: baseOptions({
        indexAxis: 'y',
        layout: { padding: { top: 4, right: 10, bottom: 0, left: 0 } },
        scales: axes({ vertical: false, xMax: 100, xTitle: 'Correct (%)',
                       xCallback: function(v){ return v + '%'; } }),
        plugins: Object.assign(baseOptions().plugins, {
          tooltip: Object.assign(baseOptions().plugins.tooltip, {
            callbacks: { label: function(c){
              var arm = c.dataset.arm;
              return c.dataset.label + ': ' + c.parsed.x.toFixed(1) + '% (' +
                     D.categories.correctByArm[arm][c.dataIndex] + ' of ' + D.categories.n[c.dataIndex] + ')';
            } }
          })
        })
      }),
      plugins: [directLabels]
    });
  }

  function wrapLabel(text, width){
    var max = width || (window.innerWidth <= 700 ? 14 : 20);
    text = String(text);
    if(text.length <= max) return text;
    var words = text.split(' '), lines = [], line = '';
    words.forEach(function(w){
      if((line + ' ' + w).trim().length > max){ if(line) lines.push(line); line = w; }
      else { line = (line + ' ' + w).trim(); }
    });
    if(line) lines.push(line);
    return lines;
  }

  function sizeCharts(){
    document.querySelectorAll('.chart-box').forEach(function(box){
      var h = window.innerWidth <= 700 && box.dataset.mobileHeight ? box.dataset.mobileHeight : box.dataset.height;
      box.style.height = h + 'px';
    });
  }

  // theme toggle
  var btn = document.getElementById('themeBtn');
  function setTheme(mode){
    document.documentElement.setAttribute('data-theme', mode);
    btn.textContent = mode === 'dark' ? 'Light mode' : 'Dark mode';
    try { localStorage.setItem('routingReportTheme', mode); } catch(e){}
    buildAll();
  }
  btn.addEventListener('click', function(){
    setTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  });

  // show-all buttons
  document.querySelectorAll('.show-all').forEach(function(b){
    b.addEventListener('click', function(){
      var tbl = document.getElementById(b.dataset.target);
      var shown = tbl.classList.toggle('shown');
      b.textContent = shown ? 'Show first 10 only' : b.dataset.label;
    });
  });

  var saved = null;
  try { saved = localStorage.getItem('routingReportTheme'); } catch(e){}
  document.documentElement.setAttribute('data-theme', saved === 'dark' ? 'dark' : 'light');
  btn.textContent = saved === 'dark' ? 'Light mode' : 'Dark mode';

  sizeCharts();
  buildAll();

  var t = null;
  window.addEventListener('resize', function(){
    clearTimeout(t);
    t = setTimeout(function(){ sizeCharts(); buildAll(); }, 200);
  });
})();
"""


# --------------------------------------------------------------- build body
parts = []
A = parts.append

best_correct = max(headline, key=lambda r: int(r["correct"]))["arm"]

# ---- header
A('<header class="top"><div class="wrap">')
A('<div class="head-row"><div>')
A('<div class="eyebrow">Benchmark, %s</div>' % esc(meta["run_date"]))
A("<h1>Chat routing: Jev vs Luna vs today's classifier</h1>")
A('</div><button class="theme-btn" id="themeBtn" type="button">Dark mode</button></div>')
A('<p class="small" style="margin-top:12px;max-width:70ch">'
  'The staff assistant in skell-e-web decides, for every message someone types, whether a cheap fast model can '
  'handle it or whether it needs the expensive one. Simple rules settle the obvious cases. '
  'An AI classifier decides the rest, and that classifier is what this test is about. '
  'We replayed %s real staff messages that a human had already labeled, and asked three models to make the call: '
  'the classifier running in production today, gpt-5.6-luna on the same prompt, and Jev, a model built only for '
  'picking between fixed choices. Every number on this page comes from that one run of %s calls.</p>'
  % (esc(meta["cases"]), esc(thousands(meta["total_calls"]))))
A('<p class="small muted" style="margin:0;max-width:70ch">Two kinds of mistake matter, and they cost different things. '
  'Sending a high-stakes message to the cheap model gives someone a weak answer. '
  'Sending a routine message to the expensive model just wastes money. Throughout this page, "fast" means the '
  'cheap model and "big" means the expensive one.</p>')
A("</div></header>")

A('<div class="wrap">')

# ---- 2. hero cards
A("<section>")
A("<h2>How each model did</h2>")
A('<p class="small muted">Each model decided all %s messages on its own, rules switched off.</p>'
  % esc(meta["cases"]))
A('<div class="hero-grid">')
for arm in ARMS:
    row = H[arm]
    A('<div class="hero-card" style="--bar:var(--s-%s)">' % arm)
    A('<div class="name">%s%s%s</div>' % (
        swatch(arm), esc(row["model_label"]),
        '<span class="badge-best">Most correct</span>' if arm == best_correct else ""))
    A('<div class="sub">%s</div>' % esc(row["description"]))
    A('<div class="hero-fig">%s<span class="of"> of %s</span></div>' % (esc(row["correct"]), esc(row["n"])))
    A('<div class="hero-label">messages routed correctly, %s</div>' % pct(row["accuracy_pct"]))
    A('<div class="stat-row"><span>Sent to the cheap model by mistake</span><span class="v">%s</span></div>'
      % esc(row["missed_big"]))
    A('<div class="stat-row"><span>Sent to the expensive model by mistake</span><span class="v">%s</span></div>'
      % esc(row["false_big"]))
    A('<div class="stat-row"><span>Typical wait</span><span class="v">%ss</span></div>'
      % esc("%.2f" % float(row["median_s"])))
    A('<div class="stat-row"><span>Cost per 1,000 calls</span><span class="v">%s</span></div>'
      % esc(money(row["cost_per_1000_usd"])))
    A("</div>")
A("</div></section>")

# ---- 3. counts chart
legend_html = '<div class="legend">%s</div>' % "".join(
    '<span class="key">%s%s</span>' % (swatch(a), esc(LABEL[a])) for a in ARMS)

A('<section><div class="card">')
A("<h2>The two kinds of mistake</h2>")
A('<p class="small">Every message has one right answer, and there are two ways to get it wrong. Jev sends %s '
  "high-stakes messages to the cheap model against Luna's %s, and %s routine messages to the expensive model "
  "against Luna's %s and today's %s. The correct totals sit in the cards above and in the table below.</p>"
  % (esc(H["jev"]["missed_big"]), esc(H["luna_low"]["missed_big"]), esc(H["jev"]["false_big"]),
     esc(H["luna_low"]["false_big"]), esc(H["gemini_prod"]["false_big"])))
A(legend_html)
A(chart("chartCounts", 320, 300))
A('<p class="chart-note">Mistake counts out of %s messages. Hover or tap a bar for the exact number.</p>'
  % esc(meta["cases"]))
A(table(
    ["Model", "Correct", "Correct %", "Cheap by mistake", "Expensive by mistake"],
    [[model_cell(a), esc(H[a]["correct"]), pct(H[a]["accuracy_pct"]),
      esc(H[a]["missed_big"]), esc(H[a]["false_big"])] for a in ARMS],
    classes="compact"))
A("</div></section>")

# ---- 4. production shaped
A('<section><div class="card">')
A("<h2>With the rules switched back on</h2>")
A('<p class="small">This is the score with the rules in front: the rules answer %s, the classifier answers the %s '
  'they leave open, and %s that name a model outright are dropped, so the total is %s.</p>'
  % (esc(meta["rules_settled"]), esc(meta["rules_unsettled"]),
     esc(meta["rules_explicit_model"]), esc(meta["prod_shaped_n"])))
A(legend_html)
A(chart("chartProd", 300, 280))
A('<p class="chart-note">Share of all %s scored messages decided correctly, rules included.</p>'
  % esc(meta["prod_shaped_n"]))
A(table(
    ["Model", "Correct", "Correct %", "Cheap by mistake", "Expensive by mistake"],
    [[model_cell(a), "%s of %s" % (esc(P[a]["correct"]), esc(P[a]["n"])), pct(P[a]["accuracy_pct"]),
      esc(P[a]["missed_big"]), esc(P[a]["false_big"])] for a in ARMS],
    classes="compact"))
A('<p class="chart-note" style="margin-top:8px">The gap narrows because the rules already catch a lot of what the '
  'models disagree about. Jev still leads, and it still makes the fewest money mistakes by a wide margin.</p>')
A("</div></section>")

# ---- 5. speed and cost
A("<section>")
A('<div class="card">')
A("<h2>How long each one takes</h2>")
A('<p class="small">This wait happens before the assistant starts answering, so the user feels all of it. '
  '"Slowest 1 in 20" is the bad case: 19 of 20 calls came back faster than this.</p>')
A(legend_html)
A(chart("chartLatency", 300, 280))
A(table(
    ["Model", "Typical wait", "Slowest 1 in 20", "Average"],
    [[model_cell(a), "%.2fs" % float(H[a]["median_s"]), "%.2fs" % float(H[a]["p95_s"]),
      "%.2fs" % float(H[a]["mean_s"])] for a in ARMS],
    classes="compact"))
A("</div>")
A('<div class="card">')
A("<h2>What each one costs</h2>")
A('<p class="small">The routing decision is a separate small call on top of the answer itself, so this is pure '
  'overhead. Jev is the cheapest of the three.</p>')
A(legend_html)
A(chart("chartCost", 300, 280))
A(table(
    ["Model", "Per 1,000 calls", "Cost of this whole run"],
    [[model_cell(a), money(H[a]["cost_per_1000_usd"]), "$%.3f" % float(H[a]["total_cost_usd"])] for a in ARMS],
    classes="compact"))
A('<p class="chart-note" style="margin-top:8px">Luna\'s price assumes no prompt caching, because each message '
  'carries different text and cache hits were near zero in this run.</p>')
A("</div></section>")

# ---- 6. confidence and categories
jev_errors = len(case_sets[0][3])
A('<section><div class="card">')
A("<h2>Jev tells you when it is unsure, and it is telling the truth</h2>")
A('<p class="small">Jev returns a confidence number with every answer. Split its %s decisions into four '
  'groups of roughly equal size by that number and the pattern is clean: %s of its %s mistakes sit in the two least confident groups, '
  'and on the quarter it is most sure about it did not miss a single one. That makes the confidence number usable '
  'as a safety valve rather than decoration.</p>'
  % (esc(meta["cases"]), esc(meta["jev_errors_bottom_two_quartiles"]), jev_errors))
A('<div class="legend"><span class="key">%sJev 1.13.0, split by how sure it was</span></div>' % swatch("jev"))
A(chart("chartQuartile", 300, 280))
A(table(
    ["How sure Jev was", "Messages", "Correct", "Correct %", "Cheap by mistake", "Expensive by mistake"],
    [[esc(r["quartile_label"]), esc(r["n"]), esc(r["correct"]), pct(r["accuracy_pct"]),
      esc(r["missed_big"]), esc(r["false_big"])] for r in quart],
    classes="compact"))
A('<p class="chart-note" style="margin-top:8px">Group boundaries sit at confidence %s.</p>'
  % esc(", ".join("%s%%" % c for c in cuts)))
A("</div>")

folded = [r["grouped_into_other"] for r in cat_group if r["grouped_into_other"] != "other"]
grouped_names = ", ".join(folded) + ", and the messages already tagged other" if folded else ""

A('<div class="card">')
A("<h2>Where each model fails</h2>")
A('<p class="small">The same messages, sorted by what kind of request they are. This is where you can see that the '
  'three models fail in different places. Jev leads on data pulls and lookups, the two biggest message types, '
  'and weakest on long writing, where it sends work to the cheap model that deserved the expensive one.</p>')
A(legend_html)
A(chart("chartCategory", 660, 800))
A('<p class="chart-note">Share of each message type routed correctly. The number in brackets is how many messages '
  'of that type there were. Types with fewer than %s messages are folded into "other": %s.</p>'
  % (esc(meta["category_min_n"]), esc(grouped_names)))
cat_headers = ["Message type", "Messages"] + [LABEL[a] for a in ARMS]
cat_rows = []
for r in cats:
    cells = [esc(r["category"]), esc(r["n"])]
    best = max(float(r["%s_accuracy_pct" % a]) for a in ARMS)
    for a in ARMS:
        value = float(r["%s_accuracy_pct" % a])
        mark = ' class="ok"' if value == best else ""
        cells.append('<span%s>%s</span>' % (mark, pct(value)))
    cat_rows.append(cells)
A(table(cat_headers, cat_rows, classes="compact"))
A("</div></section>")

# ---- 7. policies
A('<section><div class="card">')
A("<h2>Other ways to use Jev's answer</h2>")
A('<p class="small">Jev returns more than a choice. It also returns a probability and four yes/no judgements, so '
  'the decision can be assembled in several ways. These are those variants scored on the same %s messages.</p>'
  % esc(meta["cases"]))
pol_rows = []
for r in policies:
    pol_rows.append([
        "<strong>%s</strong><span class=\"wrapme small muted\">%s</span>" % (esc(r["policy_label"]), esc(r["explanation"])),
        "%s of %s" % (esc(r["correct"]), esc(r["n"])),
        pct(r["accuracy_pct"]), esc(r["missed_big"]), esc(r["false_big"]),
        money(H["jev"]["cost_per_1000_usd"]),
    ])
for r in hybrid:
    pol_rows.append([
        "<strong>%s</strong><span class=\"wrapme small muted\">%s</span>" % (esc(r["policy_label"]), esc(r["explanation"])),
        "%s of %s" % (esc(r["correct"]), esc(r["n"])),
        pct(r["accuracy_pct"]), esc(r["missed_big"]), esc(r["false_big"]),
        money(r["blended_cost_per_1000_usd"]),
    ])
A(table(["Way of deciding", "Correct", "Correct %", "Cheap by mistake", "Expensive by mistake", "Cost per 1,000"],
        pol_rows, classes="compact"))
A('<p class="chart-note" style="margin-top:8px">The cost column is worked out case by case: every message costs a '
  'Jev call, and the escalated ones cost a Luna call on top. routing-summary.json carries an older estimate for the '
  'three hybrid rows that averages the two prices instead, so it reads a little lower.</p>')
A('<p class="small" style="margin-top:12px">Two things stand out. Rebuilding the decision from the yes/no answers '
  "is much worse than just using Jev\'s own choice, so there is no point hand-tuning those. "
  'And sending the unsure cases to Luna does help a little, but only at the tightest cutoff. '
  'Escalating more makes it worse, because the messages Jev is unsure about are exactly the ones Luna '
  'over-sends to the expensive model.</p>')
A("</div></section>")

# ---- 8. agreement
A('<section><div class="card">')
A("<h2>How often the three agree with each other</h2>")
A('<p class="small">They agree with each other about as often as each one agrees with the human labels, which means '
  'they are not making the same mistakes. Jev is right where Luna is wrong %s times. Luna is right where Jev is '
  'wrong %s times.</p>' % (esc(meta["jev_right_luna_wrong"]), esc(meta["luna_right_jev_wrong"])))
A(table(["Pair", "Agree", "Agree %", "Disagree", "Both said expensive", "Both said cheap"],
        [[esc(r["pair"]), esc(r["agree"]), pct(r["agree_pct"]), esc(r["disagree"]),
          esc(r["both_big"]), esc(r["both_fast"])] for r in agreement],
        classes="compact"))
A("</div></section>")

# ---- 9. evidence
A("<section>")
A("<h2>Read the actual messages</h2>")
A('<p class="small muted">Each list is sorted with Jev\'s least confident decisions first, so the shakiest calls are '
  'at the top. Messages are the first 120 characters as stored in the results file.</p>')
case_headers = ["Message", "Human label", "Jev says", "Jev's confidence", "Luna says",
                "Today's model says", "Type"]
for slug, title, blurb, rows in case_sets:
    tid = "tbl-%s" % slug
    A("<details class=\"eb\"><summary><span>%s <span class=\"count\">%d cases</span></span></summary>"
      % (esc(title), len(rows)))
    A('<div class="body">')
    A('<p class="small muted">%s</p>' % esc(blurb))
    body = []
    for i, r in enumerate(rows):
        cells = [
            '<span class="wrapme msg">%s</span>' % esc(r["text_head"]),
            pred_pill(r["label"]),
            pred_pill(r["jev_prediction"]),
            esc("%.0f%%" % (float(r["jev_confidence"]) * 100)),
            pred_pill(r["luna_prediction"]),
            pred_pill(r["gemini_prediction"]),
            '<span class="wrapme small muted">%s</span>' % esc(r["category"]),
        ]
        body.append((i, cells))
    out = ['<div class="table-wrap"><table class="cases" id="%s">' % tid, "<thead><tr>"]
    for head in case_headers:
        out.append("<th>%s</th>" % esc(head))
    out.append("</tr></thead><tbody>")
    for i, cells in body:
        cls = ' class="extra"' if i >= 10 else ""
        out.append("<tr%s>" % cls)
        for cell, head in zip(cells, case_headers):
            out.append('<td data-label="%s">%s</td>' % (esc(head), cell))
        out.append("</tr>")
    out.append("</tbody></table></div>")
    A("".join(out))
    if len(rows) > 10:
        btn_label = "Show all %d" % len(rows)
        A('<button class="show-all" type="button" data-target="%s" data-label="%s">%s</button>'
          % (tid, esc(btn_label), esc(btn_label)))
    A("</div></details>")
A("</section>")

# ---- 10. what a swap looks like
A('<section><div class="card">')
A("<h2>What a production swap would look like</h2>")
A('<p>Jev takes the place of the classifier call inside the routing step of skell-e-web. Nothing else about the '
  'assistant changes. The rules stay in front of it and keep settling the obvious messages, so Jev only ever sees '
  'the ones the rules leave open. The state it reads is the same context the current prompt already builds: the '
  'setting, the last few turns, the new message and who sent it.</p>')
A('<p>If Jev errors or runs past the 2.5 second timeout, routing does what it does today and sends the message to '
  'the cheap model. There is no second classifier behind it now, and adding Jev does not create one. This run had '
  'no failures in %s calls.</p>' % esc(meta["cases"]))
A('<p>One optional addition is worth a look. Jev\'s confidence is honest enough that a tie-break rule, sending '
  'anything it is very unsure about to the expensive model unless the writer asked for something quick, cuts the '
  'quality mistakes to a tie with Luna for the lowest here, while keeping the money mistakes well below both text '
  'models. That row is in the table above as "Jev\'s answer, big when unsure", scored with the rules off across '
  'all %s messages.</p>' % esc(meta["cases"]))
A('<p>Before a full cutover, run Jev alongside the current classifier on live traffic for a week, log both '
  'decisions, and route on the old one. That gives a real disagreement list to read, on real traffic rather than a '
  'one-day sample, and it costs about %s per 1,000 messages to collect.</p>' % esc(money(H["jev"]["cost_per_1000_usd"])))
A("<h3>Caveats worth knowing before you decide</h3>")
A('<ul class="clean small">')
A("<li>The conversation context each model saw was rebuilt from the human labels, not from what that model itself "
  "decided earlier in the same conversation. Real production compounds its own mistakes. These numbers do not.</li>")
A("<li>Attachments were always empty, because the export does not carry them. Production sometimes has them.</li>")
A("<li>This is one day of exported chat history, %s messages. It is real traffic, but it is one sample.</li>"
  % esc(meta["cases"]))
A("<li>Luna's cost is effectively uncached pricing. Every message carries different text, so prompt caching almost "
  "never hit. A production deployment might do slightly better.</li>")
A("<li>The human labels are one person's judgement. The list of cases all three models got wrong is the place to "
  "check whether the label or the model was the problem.</li>")
A("</ul>")
A("</div></section>")

A("</div>")  # /wrap

# ---- 11. footer
A('<footer><div class="wrap">')
A("<h3>Where these numbers come from</h3>")
A("<p>Every number on this page comes from <code>docs/jev-real/routing-results.jsonl</code>, one row per message per "
  "model from the run on %s. Nothing is typed in by hand.</p>" % esc(meta["run_date"]))
A("<p>Raw results: <code>docs/jev-real/routing-results.jsonl</code><br>"
  "Aggregates: <code>docs/jev-real/routing-summary.json</code><br>"
  "Developer notes: <code>docs/jev-real/routing-notes.md</code><br>"
  "Extract step: <code>docs/jev-real/routing-report/extract.py</code> writes the CSVs in "
  "<code>docs/jev-real/routing-report/data/</code><br>"
  "This page: <code>docs/jev-real/routing-report/generate_report.py</code></p>")
A("<p>%s messages, %s calls, three models, run %s. Total spend on the run: $%.2f.</p>"
  % (esc(meta["cases"]), esc(thousands(meta["total_calls"])), esc(meta["run_date"]), float(meta["spend_usd"])))
A("</div></footer>")

body = "\n".join(parts)

doc = """<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chat routing: Jev vs Luna vs today's classifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&family=Rubik:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>__CSS__</style>
</head>
<body>
__BODY__
<script>window.__REPORT__ = __DATA__;</script>
<script>__JS__</script>
</body>
</html>
"""

doc = (doc.replace("__CSS__", CSS)
          .replace("__BODY__", body)
          .replace("__DATA__", json.dumps(chart_data))
          .replace("__JS__", JS))

with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(doc)

print("wrote %s (%.1f KB)" % (OUT, len(doc) / 1024.0))

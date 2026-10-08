/* Applicazioni Informatiche del Machine Learning — animation kit.
 *
 * One small library shared by every animations/lecture-NN.html. It carries no
 * lecture content, which is why it may be published before any lecture is due
 * (tools/publish_due.py lists animations/kit among the shared trees).
 *
 * The model of a widget is always the same, and the kit is built around it:
 *
 *     state  --step()-->  state  --render()-->  SVG
 *
 * The player calls step() to advance the algorithm by one unit (one gradient
 * step, one split, one k-means iteration) and render() to redraw the frame
 * from state. Nothing is tweened. "Step" therefore shows exactly one
 * iteration, "Reset" is exact, and a reader who prefers reduced motion sees
 * nothing move until they press a button.
 *
 * All data is synthetic, generated in the browser from a fixed seed, so every
 * reader sees the same picture and a reset reproduces it.
 *
 * Every function is on window.AIML. tools/check_animations.py drives each page
 * in Chrome through AIML._players and fails on any exception, so a widget that
 * builds a player must build it with AIML.player().
 */
(function () {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var MINUS = "−";

  // --- DOM -----------------------------------------------------------------

  function setAttrs(node, attrs) {
    if (!attrs) return node;
    Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k === "class") node.setAttribute("class", v);
      else node.setAttribute(k, v);
    });
    return node;
  }

  /** An SVG element. el("circle", {cx: 3, r: 2, class: "pt c0"}, parent) */
  function el(tag, attrs, parent) {
    var n = setAttrs(document.createElementNS(SVGNS, tag), attrs);
    if (parent) parent.appendChild(n);
    return n;
  }

  /** An HTML element. h("div", {class: "ctl"}, parent) */
  function h(tag, attrs, parent) {
    var n = setAttrs(document.createElement(tag), attrs);
    if (parent) parent.appendChild(n);
    return n;
  }

  function $(sel, root) {
    return typeof sel === "string" ? (root || document).querySelector(sel) : sel;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  // --- Numbers --------------------------------------------------------------

  /** A fixed-decimal number with a true minus sign, and no "-0.00". */
  function fmt(x, d) {
    if (d === undefined) d = 2;
    if (x === Infinity) return "∞";
    if (x === -Infinity) return MINUS + "∞";
    if (typeof x !== "number" || isNaN(x)) return "—";
    var s = x.toFixed(d);
    if (/^-0(\.0+)?$/.test(s)) s = s.slice(1);
    return s.replace("-", MINUS);
  }

  /** Three significant figures, switching to exponent form when tiny or huge. */
  function sig(x, n) {
    if (n === undefined) n = 3;
    if (!isFinite(x)) return fmt(x);
    if (x === 0) return "0";
    var a = Math.abs(x);
    if (a >= 1e5 || a < 1e-3) {
      var e = x.toExponential(n - 1).replace("e+", "e").replace("-", MINUS);
      return e.replace(/e(-?)/, function (_, m) { return "×10^" + (m ? MINUS : ""); })
              .replace(/\^(−?)(\d+)/, function (_, m, k) { return sup(m + k); });
    }
    return fmt(x, Math.max(0, n - 1 - Math.floor(Math.log10(a))));
  }
  function sup(s) {
    var map = { "0": "⁰", "1": "¹", "2": "²", "3": "³",
                "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷",
                "8": "⁸", "9": "⁹", "−": "⁻" };
    return s.split("").map(function (c) { return map[c] || c; }).join("");
  }

  function pct(x, d) { return fmt(100 * x, d === undefined ? 1 : d) + "%"; }

  function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function linspace(a, b, n) {
    var out = [];
    for (var i = 0; i < n; i++) out.push(n === 1 ? a : a + (b - a) * i / (n - 1));
    return out;
  }
  function range(n) { var o = []; for (var i = 0; i < n; i++) o.push(i); return o; }
  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }
  function mean(a) { return a.length ? sum(a) / a.length : NaN; }
  function variance(a) {
    var m = mean(a), s = 0;
    for (var i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
    return a.length ? s / a.length : NaN;
  }
  function std(a) { return Math.sqrt(variance(a)); }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function argmax(a) {
    var k = 0;
    for (var i = 1; i < a.length; i++) if (a[i] > a[k]) k = i;
    return k;
  }
  function sigmoid(z) {
    // Stable on both tails: exp of a large positive number never appears.
    if (z >= 0) { var e = Math.exp(-z); return 1 / (1 + e); }
    var f = Math.exp(z); return f / (1 + f);
  }
  /** Softmax with the max subtracted first -- the shift invariance of
   *  Lecture 17, which is what keeps exp() from overflowing. */
  function softmax(z, T) {
    T = T || 1;
    var m = Math.max.apply(null, z), e = z.map(function (v) { return Math.exp((v - m) / T); });
    var s = sum(e);
    return e.map(function (v) { return v / s; });
  }
  /** Matrix (array of rows) times vector. */
  function matvec(M, v) { return M.map(function (row) { return dot(row, v); }); }
  function matmul(A, B) {
    return A.map(function (row) {
      return B[0].map(function (_, j) {
        var s = 0;
        for (var k = 0; k < row.length; k++) s += row[k] * B[k][j];
        return s;
      });
    });
  }
  function transpose(A) { return A[0].map(function (_, j) { return A.map(function (r) { return r[j]; }); }); }

  /** Solve a small dense system Ax = b by Gaussian elimination with partial
   *  pivoting. Returns null when A is singular to working precision. */
  function solve(A, b) {
    var n = A.length, M = A.map(function (r, i) { return r.slice().concat([b[i]]); });
    for (var c = 0; c < n; c++) {
      var p = c;
      for (var r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-12) return null;
      var t = M[c]; M[c] = M[p]; M[p] = t;
      for (r = c + 1; r < n; r++) {
        var f = M[r][c] / M[c][c];
        for (var k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    var x = new Array(n);
    for (var i = n - 1; i >= 0; i--) {
      var s = M[i][n];
      for (var j = i + 1; j < n; j++) s -= M[i][j] * x[j];
      x[i] = s / M[i][i];
    }
    return x;
  }

  // --- Randomness -----------------------------------------------------------

  /** A seeded generator (mulberry32). Same seed, same data, every reader. */
  function rng(seed) {
    var a = (seed >>> 0) || 1;
    var spare = null;
    function u() {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    function n(mu, sd) {
      if (mu === undefined) mu = 0;
      if (sd === undefined) sd = 1;
      if (spare !== null) { var s = spare; spare = null; return mu + sd * s; }
      var x, y, r;
      do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0);
      var m = Math.sqrt(-2 * Math.log(r) / r);
      spare = y * m;
      return mu + sd * x * m;
    }
    return {
      u: u,
      n: n,
      uniform: function (lo, hi) { return lo + (hi - lo) * u(); },
      int: function (lo, hi) { return lo + Math.floor(u() * (hi - lo + 1)); },
      pick: function (arr) { return arr[Math.floor(u() * arr.length)]; },
      shuffle: function (arr) {
        for (var i = arr.length - 1; i > 0; i--) {
          var j = Math.floor(u() * (i + 1)), t = arr[i]; arr[i] = arr[j]; arr[j] = t;
        }
        return arr;
      }
    };
  }

  // --- Theme ----------------------------------------------------------------

  /** The current value of a colour token, e.g. color("c0") or color("accent"). */
  function color(name) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue("--" + name).trim();
  }
  /** [r, g, b] of a token, for raster drawing. */
  function rgb(name) {
    var c = color(name);
    if (c.indexOf("var(") === 0) c = color(c.slice(6, -1));
    var m = /^#([0-9a-f]{6})$/i.exec(c);
    if (!m) return [128, 128, 128];
    var v = parseInt(m[1], 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }
  var themeHandlers = [];
  var mq = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  if (mq) {
    var fire = function () { themeHandlers.forEach(function (f) { f(); }); };
    if (mq.addEventListener) mq.addEventListener("change", fire);
    else if (mq.addListener) mq.addListener(fire);
  }
  /** Call f whenever the reader's light/dark scheme changes. */
  function onTheme(f) { themeHandlers.push(f); }
  function isDark() { return !!(mq && mq.matches); }

  var reducedMotion = !!(window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  // --- Plot -----------------------------------------------------------------

  function niceTicks(lo, hi, n) {
    var span = hi - lo;
    if (!(span > 0)) return [lo];
    var step = Math.pow(10, Math.floor(Math.log10(span / n)));
    var err = (n * step) / span;
    if (err <= 0.15) step *= 10;
    else if (err <= 0.35) step *= 5;
    else if (err <= 0.75) step *= 2;
    var out = [], t = Math.ceil(lo / step - 1e-9) * step;
    for (; t <= hi + step * 1e-9; t += step) out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
    return out;
  }

  function tickFmt(ticks) {
    var step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
    var d = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step) - 1e-9));
    return function (v) { return fmt(v, d); };
  }

  /**
   * A 2-D plot in data coordinates.
   *
   *   var p = AIML.plot("#gd .viz", {x: [-3, 3], y: [0, 10], w: 560, h: 380,
   *                                  xlabel: "x", ylabel: "y"});
   *   p.dot(1, 2, {cls: "pt c0", shape: "circle"});
   *   p.path([[0, 0], [1, 1]], "line acc");
   *
   * Layers, bottom to top: back (raster fields), grid, axes, data, over, ui.
   * Everything a frame draws goes into data or over, and p.clear("data")
   * starts a frame. Axes and grid are drawn once.
   */
  function plot(target, o) {
    o = o || {};
    var host = $(target);
    var W = o.w || 560, H = o.h || 380;
    var m = Object.assign({ l: 52, r: 16, t: 14, b: 44 }, o.margin || {});
    if (o.noAxes) m = Object.assign({ l: 8, r: 8, t: 8, b: 8 }, o.margin || {});
    var xd = (o.x || [0, 1]).slice(), yd = (o.y || [0, 1]).slice();

    if (o.equal) {
      // One data unit is the same length on both axes: widen whichever
      // domain is short, about its centre.
      var pw = W - m.l - m.r, ph = H - m.t - m.b;
      var kx = (xd[1] - xd[0]) / pw, ky = (yd[1] - yd[0]) / ph;
      if (kx > ky) { var cy = (yd[0] + yd[1]) / 2, hy = kx * ph / 2; yd = [cy - hy, cy + hy]; }
      else { var cx = (xd[0] + xd[1]) / 2, hx = ky * pw / 2; xd = [cx - hx, cx + hx]; }
    }

    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
                          "aria-label": o.label || "", class: o.cls || null });
    if (o.title) el("title", { text: o.title }, svg);
    var where = o.into ? $(o.into) : host;
    if (o.caption) {
      var fig = h("figure", null, where);
      fig.appendChild(svg);
      h("figcaption", { html: o.caption }, fig);
    } else {
      where.appendChild(svg);
    }

    var x = function (v) { return m.l + (v - xd[0]) / (xd[1] - xd[0]) * (W - m.l - m.r); };
    var y = function (v) { return H - m.b - (v - yd[0]) / (yd[1] - yd[0]) * (H - m.t - m.b); };
    x.inv = function (px) { return xd[0] + (px - m.l) / (W - m.l - m.r) * (xd[1] - xd[0]); };
    y.inv = function (py) { return yd[0] + (H - m.b - py) / (H - m.t - m.b) * (yd[1] - yd[0]); };
    x.domain = xd; y.domain = yd;

    var id = "clip" + Math.random().toString(36).slice(2, 9);
    var defs = el("defs", null, svg);
    var cp = el("clipPath", { id: id }, defs);
    el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, cp);

    var layers = {};
    ["back", "grid", "axes", "data", "over", "ui"].forEach(function (k) {
      layers[k] = el("g", { class: "layer-" + k }, svg);
    });
    if (!o.noClip) {
      layers.back.setAttribute("clip-path", "url(#" + id + ")");
      layers.data.setAttribute("clip-path", "url(#" + id + ")");
    }

    var p = {
      svg: svg, W: W, H: H, m: m, x: x, y: y, layers: layers, defs: defs,
      clipId: id,
      /** The data rectangle in pixels. */
      box: { x0: m.l, y0: m.t, x1: W - m.r, y1: H - m.b },

      clear: function (name) {
        (name ? [name] : ["data", "over"]).forEach(function (k) { clear(layers[k]); });
        return p;
      },
      layer: function (name) { return layers[name || "data"]; },

      axes: function (ao) {
        ao = ao || {};
        clear(layers.axes); clear(layers.grid);
        if (o.noAxes) return p;
        var xt = Array.isArray(o.xticks) ? o.xticks : niceTicks(xd[0], xd[1], o.xticks || 6);
        var yt = Array.isArray(o.yticks) ? o.yticks : niceTicks(yd[0], yd[1], o.yticks || 5);
        var xf = o.xfmt || tickFmt(xt), yf = o.yfmt || tickFmt(yt);
        var g = el("g", { class: "grid" }, layers.grid);
        if (o.grid !== false) {
          xt.forEach(function (t) { el("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b }, g); });
          yt.forEach(function (t) { el("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }, g); });
        }
        var a = el("g", { class: "axis" }, layers.axes);
        el("rect", { class: "frame", x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, a);
        var tg = el("g", { class: "tick" }, a);
        xt.forEach(function (t) {
          el("line", { x1: x(t), x2: x(t), y1: H - m.b, y2: H - m.b + 4 }, tg);
          el("text", { x: x(t), y: H - m.b + 16, "text-anchor": "middle", text: xf(t) }, tg);
        });
        yt.forEach(function (t) {
          el("line", { x1: m.l - 4, x2: m.l, y1: y(t), y2: y(t) }, tg);
          el("text", { x: m.l - 7, y: y(t) + 3.5, "text-anchor": "end", text: yf(t) }, tg);
        });
        if (o.xlabel) el("text", { class: "axis-label", x: (m.l + W - m.r) / 2, y: H - 8,
                                   "text-anchor": "middle", text: o.xlabel }, a);
        if (o.ylabel) el("text", { class: "axis-label", x: 0, y: 0, "text-anchor": "middle",
                                   transform: "translate(13," + (m.t + H - m.b) / 2 + ") rotate(-90)",
                                   text: o.ylabel }, a);
        if (ao.zero || o.zero) {
          if (yd[0] < 0 && yd[1] > 0) el("line", { class: "zero", x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }, layers.grid);
          if (xd[0] < 0 && xd[1] > 0) el("line", { class: "zero", x1: x(0), x2: x(0), y1: m.t, y2: H - m.b }, layers.grid);
        }
        return p;
      },

      /** A polyline through data points [[x, y], ...]. */
      path: function (pts, cls, layer, extra) {
        var d = "", pen = false;
        for (var i = 0; i < pts.length; i++) {
          var px = pts[i];
          if (!px || !isFinite(px[0]) || !isFinite(px[1])) { pen = false; continue; }
          d += (pen ? "L" : "M") + x(px[0]).toFixed(2) + "," + y(px[1]).toFixed(2);
          pen = true;
        }
        return el("path", Object.assign({ d: d, class: cls || "line" }, extra || {}), layers[layer || "data"]);
      },
      /** A closed, filled polygon through data points. */
      poly: function (pts, cls, layer, extra) {
        var s = pts.map(function (q) { return x(q[0]).toFixed(2) + "," + y(q[1]).toFixed(2); }).join(" ");
        return el("polygon", Object.assign({ points: s, class: cls || "fill-soft" }, extra || {}), layers[layer || "data"]);
      },
      line: function (x1, y1, x2, y2, cls, layer, extra) {
        return el("line", Object.assign({ x1: x(x1), y1: y(y1), x2: x(x2), y2: y(y2), class: cls || "line thin" }, extra || {}),
                  layers[layer || "data"]);
      },
      /** A line across the whole plot, w0 + w1*x + w2*y = 0. */
      boundary: function (w0, w1, w2, cls, layer) {
        var pts = [];
        if (Math.abs(w2) > 1e-9) {
          pts = [[xd[0], -(w0 + w1 * xd[0]) / w2], [xd[1], -(w0 + w1 * xd[1]) / w2]];
        } else if (Math.abs(w1) > 1e-9) {
          pts = [[-w0 / w1, yd[0]], [-w0 / w1, yd[1]]];
        }
        return pts.length ? p.path(pts, cls || "line", layer) : null;
      },
      rect: function (x0, y0, x1, y1, cls, layer, extra) {
        var X0 = x(Math.min(x0, x1)), X1 = x(Math.max(x0, x1));
        var Y0 = y(Math.max(y0, y1)), Y1 = y(Math.min(y0, y1));
        return el("rect", Object.assign({ x: X0, y: Y0, width: Math.max(0, X1 - X0),
                                          height: Math.max(0, Y1 - Y0), class: cls || "fill-soft" }, extra || {}),
                  layers[layer || "data"]);
      },
      /** A marker. shape: circle (default), tri, sq, diamond, cross. */
      dot: function (px, py, d) {
        d = d || {};
        return marker(layers[d.layer || "data"], x(px), y(py), d.shape || "circle", d.r || 4.5, d.cls || "pt c0", d.extra);
      },
      text: function (px, py, s, d) {
        d = d || {};
        return el("text", Object.assign({ x: x(px) + (d.dx || 0), y: y(py) + (d.dy || 0),
                                          "text-anchor": d.anchor || "start", class: d.cls || "lbl", text: s },
                                        d.extra || {}),
                  layers[d.layer || "over"]);
      },
      /** An arrow from (x1, y1) to (x2, y2) in data coordinates. */
      arrow: function (x1, y1, x2, y2, d) {
        d = d || {};
        return arrowPx(layers[d.layer || "over"], x(x1), y(y1), x(x2), y(y2), d.cls || "arrow", d.head || 8);
      },

      /**
       * Shade the plot by a function of position. f(x, y) returns
       * [r, g, b, a] with a in 0..1. Drawn into a small canvas and scaled up,
       * so a 90-cell-wide field costs nothing and still looks smooth.
       */
      field: function (f, fo) {
        fo = fo || {};
        var nx = fo.res || 90, ny = Math.max(2, Math.round(nx * (H - m.t - m.b) / (W - m.l - m.r)));
        var cv = document.createElement("canvas");
        cv.width = nx; cv.height = ny;
        var ctx = cv.getContext("2d"), img = ctx.createImageData(nx, ny);
        for (var j = 0; j < ny; j++) {
          var vy = yd[1] - (j + 0.5) / ny * (yd[1] - yd[0]);
          for (var i = 0; i < nx; i++) {
            var vx = xd[0] + (i + 0.5) / nx * (xd[1] - xd[0]);
            var c = f(vx, vy) || [0, 0, 0, 0], k = 4 * (j * nx + i);
            img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2];
            img.data[k + 3] = Math.round(255 * (c[3] === undefined ? 1 : c[3]));
          }
        }
        ctx.putImageData(img, 0, 0);
        var L = layers[fo.layer || "back"];
        if (fo.replace !== false) clear(L);
        return el("image", { href: cv.toDataURL(), x: m.l, y: m.t, width: W - m.l - m.r,
                             height: H - m.t - m.b, preserveAspectRatio: "none",
                             style: fo.pixelated ? "image-rendering:pixelated" : null }, L);
      },

      /** The level set f(x, y) = level, by marching squares. */
      contour: function (f, level, cls, layer, res) {
        var nx = res || 70, ny = Math.round(nx * (H - m.t - m.b) / (W - m.l - m.r));
        var xs = linspace(xd[0], xd[1], nx + 1), ys = linspace(yd[0], yd[1], ny + 1);
        var v = ys.map(function (yy) { return xs.map(function (xx) { return f(xx, yy) - (level || 0); }); });
        var d = "";
        function cut(a, b, pa, pb) { var t = a / (a - b); return [pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])]; }
        for (var j = 0; j < ny; j++) for (var i = 0; i < nx; i++) {
          var c = [[xs[i], ys[j]], [xs[i + 1], ys[j]], [xs[i + 1], ys[j + 1]], [xs[i], ys[j + 1]]];
          var s = [v[j][i], v[j][i + 1], v[j + 1][i + 1], v[j + 1][i]];
          // A cell with a hole in it (f returned NaN to mask a region) is
          // skipped rather than read as negative, which would draw an edge
          // along the mask instead of along the level set.
          if (!(isFinite(s[0]) && isFinite(s[1]) && isFinite(s[2]) && isFinite(s[3]))) continue;
          var pts = [];
          for (var e = 0; e < 4; e++) {
            var a = s[e], b = s[(e + 1) % 4];
            if ((a > 0) !== (b > 0)) pts.push(cut(a, b, c[e], c[(e + 1) % 4]));
          }
          for (var q = 0; q + 1 < pts.length; q += 2)
            d += "M" + x(pts[q][0]).toFixed(1) + "," + y(pts[q][1]).toFixed(1) +
                 "L" + x(pts[q + 1][0]).toFixed(1) + "," + y(pts[q + 1][1]).toFixed(1);
        }
        return el("path", { d: d, class: cls || "line" }, layers[layer || "data"]);
      },

      /** Data coordinates of a pointer event. */
      at: function (ev) {
        var pt = svg.createSVGPoint();
        var src = ev.touches ? ev.touches[0] : ev;
        pt.x = src.clientX; pt.y = src.clientY;
        var q = pt.matrixTransform(svg.getScreenCTM().inverse());
        return [x.inv(q.x), y.inv(q.y)];
      },
      /** Is a data point inside the plotted domain? */
      inside: function (px, py) {
        return px >= xd[0] && px <= xd[1] && py >= yd[0] && py <= yd[1];
      },
      /** Clicks on the plot area, reported in data coordinates. */
      onClick: function (f) {
        svg.addEventListener("click", function (ev) {
          var q = p.at(ev);
          if (p.inside(q[0], q[1])) f(q[0], q[1], ev);
        });
        return p;
      },
      /**
       * Drag anything in the plot. hit(x, y) returns a handle (anything
       * truthy) or null; move(handle, x, y) is called as the pointer moves.
       */
      drag: function (hit, move, end) {
        var handle = null;
        svg.addEventListener("pointerdown", function (ev) {
          var q = p.at(ev);
          handle = hit(q[0], q[1]);
          if (handle) { svg.setPointerCapture(ev.pointerId); ev.preventDefault(); }
        });
        svg.addEventListener("pointermove", function (ev) {
          if (!handle) return;
          var q = p.at(ev);
          move(handle, clamp(q[0], xd[0], xd[1]), clamp(q[1], yd[0], yd[1]));
        });
        var stop = function () { if (handle && end) end(handle); handle = null; };
        svg.addEventListener("pointerup", stop);
        svg.addEventListener("pointercancel", stop);
        return p;
      }
    };
    if (o.axes !== false) p.axes();
    return p;
  }

  /** Tick label for a log10 axis: 2 -> "10²", -1 -> "10⁻¹", 0 -> "1". */
  function pow10(v) {
    if (v === 0) return "1";
    if (v === 1) return "10";
    return "10" + sup((v < 0 ? MINUS : "") + Math.abs(Math.round(v)));
  }

  /**
   * A quantity against iteration -- the loss curve every training widget
   * wants. The x axis grows in chunks of `n` as the run gets longer, so the
   * curve never runs off the right edge.
   *
   *   var c = AIML.curve("#gd-cost", {y: [-2, 3], log: true, ylabel: "J"});
   *   c.draw([{values: costs, cls: "line acc"}], [{y: Jmin, cls: "line ok dash thin", label: "minimum"}]);
   *
   * With log: true, y is given in decades (y: [-2, 3] is 0.01 to 1000) and
   * values are plotted as log10, clamped into the range.
   */
  function curve(target, o) {
    o = o || {};
    var host = $(target), p = null, xmax = 0, chunk = o.n || 50;
    function build(n) {
      clear(host);
      xmax = n;
      // Without an explicit width, the curve takes the width it is actually
      // shown at (within 360..860), so its labels stay at their true size
      // on a phone instead of being scaled down with an 860-wide viewBox.
      var w = o.w || clamp(Math.round(host.clientWidth || 860), 360, 860);
      p = plot(host, { x: [0, n], y: o.y || [0, 1], w: w, h: o.h || 150,
                       margin: o.margin || { l: 52, r: 16, t: 10, b: 36 },
                       xlabel: o.xlabel === undefined ? "iteration" : o.xlabel,
                       ylabel: o.ylabel || "", xticks: o.xticks,
                       yticks: o.yticks || (o.log ? range(o.y[1] - o.y[0] + 1).map(function (i) { return o.y[0] + i; }) : undefined),
                       yfmt: o.log ? pow10 : o.yfmt, caption: o.caption });
    }
    build(chunk);
    function Y(v) {
      if (!o.log) return v;
      if (!(v > 0)) return NaN;
      return clamp(Math.log10(v), o.y[0], o.y[1]);
    }
    return {
      get plot() { return p; },
      draw: function (series, refs) {
        var n = 0;
        series.forEach(function (s) { n = Math.max(n, s.values.length - 1); });
        if (n > xmax) build(Math.ceil(n / chunk) * chunk);
        p.clear();
        (refs || []).forEach(function (r) {
          var yy = Y(r.y);
          if (!isFinite(yy)) return;
          p.line(0, yy, xmax, yy, r.cls || "line ok dash thin");
          if (r.label) p.text(xmax, yy, r.label, { anchor: "end", dy: -5, cls: "lbl-sm" });
        });
        series.forEach(function (s) {
          p.path(s.values.map(function (v, k) { return [k, Y(v)]; }), s.cls || "line acc");
          if (s.label && s.values.length) {
            var last = s.values.length - 1;
            p.text(last, Y(s.values[last]), s.label, { dx: 6, dy: 4, cls: "lbl-sm" });
          }
        });
      }
    };
  }

  /** A blank SVG in pixel coordinates, for diagrams that are not plots. */
  function canvas(target, o) {
    o = o || {};
    var W = o.w || 560, H = o.h || 320;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": o.label || "" });
    var where = $(o.into || target);
    if (o.caption) {
      var fig = h("figure", null, where);
      fig.appendChild(svg);
      h("figcaption", { html: o.caption }, fig);
    } else where.appendChild(svg);
    var defs = el("defs", null, svg);
    var g = el("g", null, svg);
    return {
      svg: svg, W: W, H: H, g: g, defs: defs,
      clear: function () { clear(g); return this; },
      el: function (tag, attrs, parent) { return el(tag, attrs, parent || g); },
      arrow: function (x1, y1, x2, y2, cls, head, parent) {
        return arrowPx(parent || g, x1, y1, x2, y2, cls || "arrow", head || 8);
      },
      marker: function (px, py, shape, r, cls, parent) { return marker(parent || g, px, py, shape, r, cls); },
      text: function (px, py, s, cls, anchor, parent) {
        return el("text", { x: px, y: py, class: cls || "lbl", "text-anchor": anchor || "start", text: s }, parent || g);
      },
      at: function (ev) {
        var pt = svg.createSVGPoint();
        pt.x = ev.clientX; pt.y = ev.clientY;
        var q = pt.matrixTransform(svg.getScreenCTM().inverse());
        return [q.x, q.y];
      }
    };
  }

  function marker(parent, cx, cy, shape, r, cls, extra) {
    var n;
    if (shape === "tri") {
      var k = r * 1.25;
      n = el("polygon", { points: [cx, cy - k, cx + k * 0.95, cy + k * 0.7, cx - k * 0.95, cy + k * 0.7].join(" ") }, parent);
    } else if (shape === "sq") {
      n = el("rect", { x: cx - r * 0.9, y: cy - r * 0.9, width: r * 1.8, height: r * 1.8 }, parent);
    } else if (shape === "diamond") {
      var d = r * 1.25;
      n = el("polygon", { points: [cx, cy - d, cx + d, cy, cx, cy + d, cx - d, cy].join(" ") }, parent);
    } else if (shape === "cross") {
      n = el("path", { d: "M" + (cx - r) + "," + (cy - r) + "L" + (cx + r) + "," + (cy + r) +
                          "M" + (cx - r) + "," + (cy + r) + "L" + (cx + r) + "," + (cy - r),
                       style: "stroke-width:2;fill:none" }, parent);
    } else {
      n = el("circle", { cx: cx, cy: cy, r: r }, parent);
    }
    n.setAttribute("class", cls || "pt c0");
    if (extra) setAttrs(n, extra);
    return n;
  }

  /** The marker shape each class uses, everywhere: circle, triangle, square, diamond. */
  var SHAPES = ["circle", "tri", "sq", "diamond", "circle"];

  function arrowPx(parent, x1, y1, x2, y2, cls, head) {
    var g = el("g", null, parent);
    var dx = x2 - x1, dy = y2 - y1, L = Math.sqrt(dx * dx + dy * dy);
    if (L < 0.5) return g;
    var ux = dx / L, uy = dy / L, hh = Math.min(head, L * 0.6);
    el("line", { x1: x1, y1: y1, x2: x2 - ux * hh * 0.8, y2: y2 - uy * hh * 0.8, class: cls }, g);
    var bx = x2 - ux * hh, by = y2 - uy * hh, w = hh * 0.55;
    var headCls = (cls || "arrow").split(" ").indexOf("acc") >= 0 ? "fill-acc"
                : (cls || "").split(" ").indexOf("ok") >= 0 ? "fill-ok"
                : (cls || "").split(" ").indexOf("primary") >= 0 ? "fill-primary" : "arrow-head";
    el("polygon", { class: headCls, points: [x2, y2, bx - uy * w, by + ux * w, bx + uy * w, by - ux * w].join(" ") }, g);
    return g;
  }

  // --- Player ---------------------------------------------------------------

  var players = [];

  /**
   * Play / Step / Reset / speed, bound to a widget.
   *
   *   var pl = AIML.player("#gd .toolbar", {
   *     step:   function () { ...advance state...; return notDone; },
   *     reset:  function () { ...reinitialise state... },
   *     render: function () { ...redraw from state... },
   *     fps: 8
   *   });
   *
   * step() returning false means the algorithm has finished (converged, or
   * the tree is fully grown); the player stops and Play becomes Replay.
   * The player pauses itself when scrolled out of view.
   */
  function player(target, o) {
    var host = $(target);
    var fps = o.fps || 8, mult = 1, running = false, done = false, t = 0;
    var timer = null, last = 0, acc = 0, visible = true;

    var bPlay = h("button", { class: "b primary", type: "button", text: "Play" }, host);
    var bStep = h("button", { class: "b", type: "button", text: "Step", title: "Advance one " + (o.unit || "step") }, host);
    var bReset = h("button", { class: "b quiet", type: "button", text: "Reset" }, host);
    if (o.extraButtons) o.extraButtons(host);
    h("span", { class: "spacer" }, host);
    var sp = h("label", { class: "speed" }, host);
    h("span", { text: "Speed" }, sp);
    var sel = h("select", null, sp);
    [["0.25", "¼×"], ["0.5", "½×"], ["1", "1×"], ["2", "2×"], ["4", "4×"], ["10", "10×"]]
      .forEach(function (q) { h("option", { value: q[0], text: q[1], selected: q[0] === "1" ? "selected" : null }, sel); });
    sel.addEventListener("change", function () { mult = parseFloat(sel.value); });
    var counter = h("span", { class: "speed", "aria-live": "off" }, host);

    function label() {
      bPlay.textContent = running ? "Pause" : done ? "Replay" : (t ? "Resume" : "Play");
      bPlay.setAttribute("aria-pressed", running ? "true" : "false");
      counter.textContent = (o.unit || "step") + " " + t + (o.max ? " / " + o.max : "");
      bStep.disabled = done;
    }
    function advance() {
      if (done) return false;
      var more = o.step(t);
      t++;
      if (more === false || (o.max && t >= o.max)) { done = true; pause(); }
      return !done;
    }
    function frame(now) {
      if (!running) return;
      if (!last) last = now;
      acc += (now - last) / 1000 * fps * mult;
      last = now;
      var n = 0;
      while (acc >= 1 && running && n < 50) { acc -= 1; advance(); n++; }
      if (n) { o.render(); label(); }
      timer = requestAnimationFrame(frame);
    }
    function play() {
      if (done) reset();
      if (running) return;
      running = true; last = 0; acc = 1;
      label();
      timer = requestAnimationFrame(frame);
    }
    function pause() {
      running = false;
      if (timer) cancelAnimationFrame(timer);
      timer = null;
      label();
    }
    function stepOnce() { pause(); advance(); o.render(); label(); }
    function reset() {
      pause();
      t = 0; done = false;
      if (o.reset) o.reset();
      o.render();
      label();
    }
    bPlay.addEventListener("click", function () { running ? pause() : play(); });
    bStep.addEventListener("click", stepOnce);
    bReset.addEventListener("click", reset);

    if ("IntersectionObserver" in window) {
      var wasRunning = false;
      new IntersectionObserver(function (es) {
        es.forEach(function (e) {
          visible = e.isIntersecting;
          if (!visible && running) { wasRunning = true; pause(); }
          else if (visible && wasRunning) { wasRunning = false; play(); }
        });
      }).observe(host.closest("section") || host);
    }

    var api = {
      play: play, pause: pause, stepOnce: stepOnce, reset: reset,
      get running() { return running; }, get done() { return done; }, get t() { return t; },
      /** Re-run reset and render without touching the play state -- for a
       *  control that changes the problem (a new dataset, a new learning rate). */
      restart: function (keepPlaying) {
        var was = running;
        reset();
        if (keepPlaying && was) play();
      },
      label: label
    };
    players.push(api);
    reset();
    if (o.autoplay && !reducedMotion) play();
    return api;
  }

  // --- Controls -------------------------------------------------------------

  /**
   * A labelled range input with its value shown.
   *   slider(host, {label: "Learning rate η", min: 0.001, max: 1, value: 0.1,
   *                 log: true, fmt: v => v.toFixed(3), onInput: v => ...})
   * log: true maps the track logarithmically, which is what a learning rate
   * or a regularisation strength needs.
   */
  function slider(target, o) {
    var wrap = h("div", { class: "ctl" }, $(target));
    var lab = h("label", null, wrap);
    h("span", { html: o.label }, lab);
    var out = h("output", null, lab);
    var steps = o.log ? 1000 : null;
    var inp = h("input", { type: "range" }, wrap);
    if (o.log) {
      inp.min = 0; inp.max = steps; inp.step = 1;
    } else {
      inp.min = o.min; inp.max = o.max; inp.step = o.step || (o.max - o.min) / 100;
    }
    lab.setAttribute("for", inp.id = "s" + Math.random().toString(36).slice(2, 8));
    if (o.hint) h("span", { class: "hint", html: o.hint }, wrap);
    var f = o.fmt || function (v) { return sig(v); };
    function toVal(r) {
      return o.log ? Math.exp(Math.log(o.min) + (Math.log(o.max) - Math.log(o.min)) * r / steps) : parseFloat(r);
    }
    function toRaw(v) {
      return o.log ? Math.round(steps * (Math.log(v) - Math.log(o.min)) / (Math.log(o.max) - Math.log(o.min))) : v;
    }
    var api = {
      el: wrap, input: inp,
      get value() { return toVal(inp.value); },
      set value(v) { inp.value = toRaw(v); out.textContent = f(toVal(inp.value)); }
    };
    api.value = o.value;
    inp.addEventListener("input", function () {
      out.textContent = f(api.value);
      if (o.onInput) o.onInput(api.value);
    });
    if (o.onChange) inp.addEventListener("change", function () { o.onChange(api.value); });
    return api;
  }

  /** Segmented buttons: one of a few named options. options: [[value, label], ...] */
  function seg(target, o) {
    var wrap = h("div", { class: "ctl" }, $(target));
    if (o.label) h("span", { class: "lab", html: o.label }, wrap);
    var box = h("div", { class: "seg", role: "group" }, wrap);
    if (o.hint) h("span", { class: "hint", html: o.hint }, wrap);
    var value = o.value, buttons = [];
    o.options.forEach(function (q) {
      var b = h("button", { type: "button", html: q[1], "aria-pressed": q[0] === value ? "true" : "false" }, box);
      b.addEventListener("click", function () {
        value = q[0];
        buttons.forEach(function (bb) { bb.setAttribute("aria-pressed", bb === b ? "true" : "false"); });
        if (o.onChange) o.onChange(value);
      });
      buttons.push(b);
    });
    return { el: wrap, get value() { return value; } };
  }

  function select(target, o) {
    var wrap = h("div", { class: "ctl" }, $(target));
    var lab = h("label", null, wrap);
    h("span", { html: o.label }, lab);
    var s = h("select", null, wrap);
    o.options.forEach(function (q) {
      h("option", { value: q[0], text: q[1], selected: q[0] === o.value ? "selected" : null }, s);
    });
    if (o.hint) h("span", { class: "hint", html: o.hint }, wrap);
    s.addEventListener("change", function () { if (o.onChange) o.onChange(s.value); });
    return { el: wrap, get value() { return s.value; }, set value(v) { s.value = v; } };
  }

  function check(target, o) {
    var lab = h("label", { class: "check" }, $(target));
    var inp = h("input", { type: "checkbox" }, lab);
    inp.checked = !!o.value;
    h("span", { html: o.label }, lab);
    inp.addEventListener("change", function () { if (o.onChange) o.onChange(inp.checked); });
    return { el: lab, get value() { return inp.checked; }, set value(v) { inp.checked = !!v; } };
  }

  function button(target, text, onClick, cls) {
    var b = h("button", { class: "b " + (cls || "quiet"), type: "button", html: text }, $(target));
    b.addEventListener("click", onClick);
    return b;
  }

  /**
   * The live numbers. rows: [[key, label], ...]
   *   var r = readout(host, [["loss", "MSE"], ["w", "w"]]);
   *   r.set("loss", fmt(mse, 3));           r.set("loss", "diverged", "bad");
   */
  function readout(target, rows) {
    var dl = h("dl", { class: "readout" }, $(target));
    var cells = {};
    rows.forEach(function (r) {
      h("dt", { html: r[1] }, dl);
      cells[r[0]] = h("dd", { text: "—" }, dl);
    });
    return {
      el: dl,
      set: function (k, v, state) {
        var c = cells[k];
        if (!c) return;
        c.textContent = v;
        c.className = state || "";
      }
    };
  }

  /** The running commentary under the plot. say() takes HTML. */
  function narrate(target) {
    var box = $(target);
    if (!box.classList.contains("narrate")) box = h("div", { class: "narrate", "aria-live": "polite" }, box);
    return { el: box, say: function (html) { box.innerHTML = html; } };
  }

  /** A legend row. items: [{shape: "circle", cls: "pt c0", label: "class 0"},
   *                         {line: "line acc dash", label: "boundary"}] */
  function legend(target, items) {
    var box = h("div", { class: "legend" }, $(target));
    items.forEach(function (it) {
      var s = h("span", null, box);
      var sv = el("svg", { viewBox: "-7 -7 14 14" }, s);
      if (it.line) el("line", { x1: -6, x2: 6, y1: 0, y2: 0, class: it.line }, sv);
      else if (it.swatch) el("rect", { x: -6, y: -6, width: 12, height: 12, class: it.swatch }, sv);
      else marker(sv, 0, 0, it.shape || "circle", 4, it.cls || "pt c0");
      h("span", { html: it.label }, s);
    });
    return box;
  }

  // --- Mathematics in prose ----------------------------------------------------

  /** Render $...$ and $$...$$ under root with the vendored KaTeX. */
  function tex(root) {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(root || document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false
    });
  }
  /** One expression to an HTML string, for narration. */
  function k(s) {
    if (!window.katex) return s;
    try { return window.katex.renderToString(s, { throwOnError: false }); } catch (e) { return s; }
  }

  function ready(f) {
    if (document.readyState !== "loading") f();
    else document.addEventListener("DOMContentLoaded", f);
  }
  // Every page says it is work in progress, as the course site does. Written
  // here rather than into 24 pages so that removing it, when the material is
  // final, is one edit.
  function wip() {
    var main = document.querySelector("main");
    if (!main || document.querySelector(".wip")) return;
    var box = h("p", { class: "wip", role: "note",
      html: "<strong>Work in progress.</strong> This page is part of a course being written " +
            "while it is taught: it can change during the course, and very likely will." });
    main.insertBefore(box, main.firstChild);
  }
  ready(function () { wip(); tex(document.body); });

  window.AIML = {
    el: el, h: h, $: $, clear: clear,
    fmt: fmt, sig: sig, pct: pct, clamp: clamp, lerp: lerp, linspace: linspace, range: range,
    sum: sum, mean: mean, variance: variance, std: std, dot: dot, norm: norm, argmax: argmax,
    sigmoid: sigmoid, softmax: softmax, matvec: matvec, matmul: matmul, transpose: transpose, solve: solve,
    rng: rng, color: color, rgb: rgb, onTheme: onTheme, isDark: isDark, reducedMotion: reducedMotion,
    plot: plot, canvas: canvas, curve: curve, pow10: pow10, marker: marker, SHAPES: SHAPES, niceTicks: niceTicks,
    player: player, slider: slider, seg: seg, select: select, check: check, button: button,
    readout: readout, narrate: narrate, legend: legend, tex: tex, k: k, ready: ready,
    _players: players
  };
})();

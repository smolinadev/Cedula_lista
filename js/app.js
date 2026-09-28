
(() => {
"use strict";

/* ---------- Constantes ---------- */
const CARD_MM = { w: 85.6, h: 53.98 };
const OUT_W = 2022, OUT_H = 1276;          // 600 dpi a 100% → 300 dpi a 200%
const CORNER_MM = 3.18;
const MAX_SRC = 3200;
const PAPERS = { carta: { w: 215.9, h: 279.4 }, oficio: { w: 215.9, h: 330.2 } };
const TOP_MM = 15, GAP_MM = 10;

/* ---------- Estado ---------- */
const sides = { front: newSide(), back: newSide() };
function newSide() { return { src: null, quad: null, autoQuad: null, confident: false, edited: false, rot: 0, out: null, dataUrl: null, busy: false }; }
let cvReady = false;

const $ = (s, r = document) => r.querySelector(s);
const slotEl = side => $(`.slot[data-side="${side}"]`);

/* ---------- OpenCV (detección) ---------- */
function loadOpenCV() {
  if (window.cv) return waitCv();
  const s = document.createElement("script");
  s.src = "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js";
  s.async = true;
  s.onerror = () => setCvStatus("fail");
  s.onload = waitCv;
  document.head.appendChild(s);
}
function waitCv() {
  {
    let tries = 0;
    const settle = m => { if (m && m.Mat) { if (m !== window.cv) { try { delete m.then; } catch (e) {} window.cv = m; } onCvReady(); return true; } return false; };
    const c = window.cv;
    if (c && typeof c.then === "function" && !c.Mat) { try { c.then(m => settle(m)); } catch (e) {} }
    const t = setInterval(() => {
      if (cvReady) return clearInterval(t);
      if (settle(window.cv)) return clearInterval(t);
      if (++tries > 300) { clearInterval(t); setCvStatus("fail"); }
    }, 100);
  }
}
function onCvReady() {
  if (cvReady) return;
  cvReady = true;
  setCvStatus("ready");
  cvWaiters.splice(0).forEach(r => r());
  for (const side of ["front", "back"]) {
    const st = sides[side];
    if (st.src && !st.edited) { runDetection(side); renderSide(side); }
  }
}
function setCvStatus(s) {
  const dot = $("#cvDot"), txt = $("#cvText");
  dot.className = "dot" + (s === "ready" ? " ready" : s === "fail" ? " fail" : "");
  if (s === "fail") cvWaiters.splice(0).forEach(r => r());
  txt.textContent = s === "ready" ? "Detector listo · todo se procesa en este equipo"
    : s === "fail" ? "Sin detector automático (revisa internet). Puedes ajustar los bordes a mano."
    : "Cargando detector de bordes…";
}

/* Detector v2: combina contornos por color/brillo, líneas rectas (Hough) y GrabCut.
   Cada candidato se califica por: bordes reales a lo largo de sus 4 lados, proporción de cédula (1,586),
   ángulos cercanos a 90° y tamaño. Luego se afinan los bordes en alta resolución. */
function detectCards(srcCanvas) {
  if (!cvReady) return [];
  const cv = window.cv;
  const S = 720;
  const k = Math.min(1, S / Math.max(srcCanvas.width, srcCanvas.height));
  const sm = document.createElement("canvas");
  sm.width = Math.max(1, Math.round(srcCanvas.width * k)); sm.height = Math.max(1, Math.round(srcCanvas.height * k));
  sm.getContext("2d").drawImage(srcCanvas, 0, 0, sm.width, sm.height);
  const W = sm.width, H = sm.height, AREA = W * H, MIND = Math.min(W, H);
  const mats = [];
  const M = m => (mats.push(m), m);
  const cands = [];
  let sup = null, GX = null, GY = null;

  // Un punto apoya un lado solo si hay borde y el borde va en la misma dirección que el lado
  const sampleSup = (x, y, nx, ny) => {
    const xi = Math.round(x), yi = Math.round(y);
    if (xi < 1 || yi < 1 || xi >= W - 1 || yi >= H - 1) return (xi >= -2 && yi >= -2 && xi <= W + 1 && yi <= H + 1) ? 0.15 : 0;
    const i = yi * W + xi;
    if (!sup[i]) return 0;
    if (nx === undefined) return 1;
    const gx = GX[i], gy = GY[i], m = Math.hypot(gx, gy);
    if (m < 1e-3) return 0;
    return Math.abs(gx * nx + gy * ny) / m > 0.85 ? 1 : 0;
  };
  const scoreQuad = (raw, source) => {
    if (!raw || raw.length !== 4 || raw.some(p => !isFinite(p.x) || !isFinite(p.y))) return;
    const q = orderQuad(raw);
    const qa = polyArea(q), af = qa / AREA;
    if (af < 0.02 || af > 1.03) return;
    for (const p of q) if (p.x < -0.06 * W || p.y < -0.06 * H || p.x > 1.06 * W || p.y > 1.06 * H) return;
    const ratio = trueRatio(q, W, H);
    const rs = Math.exp(-Math.pow((ratio - 1.586) / 0.3, 2));
    if (rs < 0.05) return;
    let as = 1;
    for (let i = 0; i < 4; i++) {
      const a = q[(i + 3) % 4], b = q[i], c = q[(i + 1) % 4];
      const v1x = a.x - b.x, v1y = a.y - b.y, v2x = c.x - b.x, v2y = c.y - b.y;
      const cos = (v1x * v2x + v1y * v2y) / (Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y) + 1e-9);
      const dev = Math.abs(Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI - 90);
      as *= Math.max(0.001, 1 - Math.pow(dev / 40, 2));
    }
    as = Math.pow(as, 0.25);
    if (as < 0.2) return;
    const sides = [];
    for (let i = 0; i < 4; i++) {
      const a = q[i], b = q[(i + 1) % 4], len = dist(a, b);
      const n = Math.max(16, Math.round(len / 3));
      const snx = -(b.y - a.y) / (len || 1), sny = (b.x - a.x) / (len || 1);
      let hit = 0;
      for (let j = 0; j < n; j++) {
        const t = 0.06 + 0.88 * (j / (n - 1));
        hit += sampleSup(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, snx, sny);
      }
      sides.push(hit / n);
    }
    const sMean = sides.reduce((s, v) => s + v, 0) / 4, sMin = Math.min(...sides);
    const support = 0.55 * sMean + 0.45 * sMin;
    const size = Math.min(1, Math.sqrt(af / 0.12));
    // Textura interior: una cédula tiene texto y foto; un pedazo de mesa no
    let inner = 0, tot = 0;
    for (let gy = 1; gy < 12; gy++) for (let gx = 1; gx < 18; gx++) {
      const u = gx / 18, v = gy / 12;
      const x = (1 - v) * ((1 - u) * q[0].x + u * q[1].x) + v * ((1 - u) * q[3].x + u * q[2].x);
      const y = (1 - v) * ((1 - u) * q[0].y + u * q[1].y) + v * ((1 - u) * q[3].y + u * q[2].y);
      const xi = Math.round(x), yi = Math.round(y);
      if (xi >= 0 && yi >= 0 && xi < W && yi < H) { tot++; if (sup[yi * W + xi]) inner++; }
    }
    const tex = tot ? inner / tot : 0;
    const texF = 0.55 + 0.45 * Math.min(1, tex / 0.06);
    const shape = source.endsWith("-rect") ? 0.85 : 1;
    const score = support * support * rs * as * size * texF * shape;
    cands.push({ quad: q, score, support, sMin, rs, af, tex, source });
  };

  const hullQuads = (contour, source) => {
    const hull = M(new cv.Mat()); cv.convexHull(contour, hull, false, true);
    const peri = cv.arcLength(hull, true);
    for (const eps of [0.01, 0.02, 0.03, 0.05, 0.07]) {
      const ap = M(new cv.Mat()); cv.approxPolyDP(hull, ap, eps * peri, true);
      if (ap.rows === 4) {
        const q = []; for (let j = 0; j < 4; j++) q.push({ x: ap.data32S[j * 2], y: ap.data32S[j * 2 + 1] });
        scoreQuad(q, source); break;
      }
      if (ap.rows < 4) break;
    }
    scoreQuad(rectPoints(cv.minAreaRect(hull)), source + "-rect");
  };
  const addContours = (bin, mode, source, minFrac = 0.02, maxN = 10) => {
    const contours = M(new cv.MatVector()), hier = M(new cv.Mat());
    cv.findContours(bin, contours, hier, mode, cv.CHAIN_APPROX_SIMPLE);
    const list = [];
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i); mats.push(c);
      const a = cv.contourArea(c);
      if (a > AREA * minFrac) list.push({ c, a });
    }
    list.sort((x, y) => y.a - x.a);
    for (const { c } of list.slice(0, maxN)) hullQuads(c, source);
  };

  try {
    const rgba = M(cv.imread(sm));
    const rgb = M(new cv.Mat()); cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
    const lab = M(new cv.Mat()); cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
    const hsv = M(new cv.Mat()); cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
    const labCh = M(new cv.MatVector()); cv.split(lab, labCh);
    const hsvCh = M(new cv.MatVector()); cv.split(hsv, hsvCh);
    const chans = [labCh.get(0), labCh.get(1), labCh.get(2), hsvCh.get(1)];
    chans.forEach(c => mats.push(c));
    // a y b tienen poco rango: se amplifican alrededor de 128
    chans[1].convertTo(chans[1], -1, 2.5, -192);
    chans[2].convertTo(chans[2], -1, 2.5, -192);
    const blurred = chans.map(c => { const m = M(new cv.Mat()); cv.GaussianBlur(c, m, new cv.Size(5, 5), 0); return m; });

    // Umbrales de Canny según el ruido de la foto (una foto granulosa necesita umbrales más altos)
    const autoCanny = (c, lo0, hi0) => {
      const gx = M(new cv.Mat()), gy = M(new cv.Mat());
      cv.Sobel(c, gx, cv.CV_32F, 1, 0, 3); cv.Sobel(c, gy, cv.CV_32F, 0, 1, 3);
      const ax = gx.data32F, ay = gy.data32F, smp = [];
      for (let i = 0; i < ax.length; i += 7) smp.push(Math.abs(ax[i]) + Math.abs(ay[i]));
      smp.sort((x, y) => x - y);
      const med = smp[smp.length >> 1] || 0;
      const lo = Math.max(lo0, med * 3.2), hi = Math.max(hi0, lo * 2.6);
      const e = M(new cv.Mat()); cv.Canny(c, e, lo, hi);
      return e;
    };
    const edges = M(cv.Mat.zeros(H, W, cv.CV_8UC1));
    blurred.forEach((c, i) => {
      const e = i === 0 ? autoCanny(c, 22, 60) : autoCanny(c, 28, 70);
      cv.bitwise_or(edges, e, edges);
    });
    const strong = M(edges.clone());
    { // bordes tenues (cédula blanca sobre papel blanco)
      const soft = M(new cv.Mat()); cv.GaussianBlur(chans[0], soft, new cv.Size(9, 9), 0);
      cv.bitwise_or(edges, autoCanny(soft, 9, 26), edges);
      // A media resolución el ruido baja y aparecen bordes casi invisibles
      const half = M(new cv.Mat()); cv.resize(chans[0], half, new cv.Size(Math.round(W / 2), Math.round(H / 2)), 0, 0, cv.INTER_AREA);
      cv.GaussianBlur(half, half, new cv.Size(7, 7), 0);
      const eh = autoCanny(half, 6, 16);
      const up = M(new cv.Mat()); cv.resize(eh, up, new cv.Size(W, H), 0, 0, cv.INTER_NEAREST);
      cv.bitwise_or(edges, up, edges);
    }
    const k3 = M(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3)));
    const k5 = M(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5)));
    const k7 = M(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(7, 7)));
    const support = M(new cv.Mat()); cv.dilate(edges, support, k3);
    sup = support.data;
    // Dirección del gradiente: en cada píxel, el canal con el cambio más fuerte
    GX = new Float32Array(W * H); GY = new Float32Array(W * H);
    const MAG = new Float32Array(W * H);
    for (const c of blurred) {
      const gx = M(new cv.Mat()), gy = M(new cv.Mat());
      cv.Sobel(c, gx, cv.CV_32F, 1, 0, 3); cv.Sobel(c, gy, cv.CV_32F, 0, 1, 3);
      const ax = gx.data32F, ay = gy.data32F;
      for (let i = 0; i < ax.length; i++) {
        const m = ax[i] * ax[i] + ay[i] * ay[i];
        if (m > MAG[i]) { MAG[i] = m; GX[i] = ax[i]; GY[i] = ay[i]; }
      }
    }

    // A. Regiones por brillo, color amarillo (b) y saturación
    for (const idx of [0, 2, 3]) {
      const t = M(new cv.Mat()); cv.threshold(blurred[idx], t, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
      const ti = M(new cv.Mat()); cv.bitwise_not(t, ti);
      for (const m of [t, ti]) {
        cv.morphologyEx(m, m, cv.MORPH_OPEN, k5, new cv.Point(-1, -1), 2);
        cv.morphologyEx(m, m, cv.MORPH_CLOSE, k5, new cv.Point(-1, -1), 2);
        addContours(m, cv.RETR_EXTERNAL, "region" + idx, 0.02, 4);
      }
    }
    // B. Bordes cerrados
    const closed = M(new cv.Mat()); cv.morphologyEx(edges, closed, cv.MORPH_CLOSE, k7);
    addContours(closed, cv.RETR_LIST, "edges", 0.02, 12);

    // C. Líneas rectas: sirve cuando un dedo, un reflejo o una sombra cortan el borde
    const raw = [];
    for (const [map, thr, minL, gap] of [[strong, 30, 0.1, 20], [edges, 40, 0.12, 12]]) {
      const lines = M(new cv.Mat());
      cv.HoughLinesP(map, lines, 1, Math.PI / 180, thr, MIND * minL, gap);
      for (let i = 0; i < lines.rows; i++) raw.push(lines.data32S.slice(i * 4, i * 4 + 4));
    }
    let segs = [];
    for (const [x1, y1, x2, y2] of raw) {
      const len = Math.hypot(x2 - x1, y2 - y1); if (len < 1) continue;
      let nx = -(y2 - y1) / len, ny = (x2 - x1) / len;
      let th = Math.atan2(ny, nx);
      if (th < 0) { th += Math.PI; nx = -nx; ny = -ny; }
      if (th >= Math.PI) { th -= Math.PI; nx = -nx; ny = -ny; }
      segs.push({ nx, ny, th, rho: nx * x1 + ny * y1, len });
    }
    segs.sort((a, b) => b.len - a.len);
    const keep = [];
    const angDiff = (a, b) => { const d = Math.abs(a - b) % Math.PI; return Math.min(d, Math.PI - d); };
    const rhoOf = (s, ref) => (s.nx * ref.nx + s.ny * ref.ny) >= 0 ? s.rho : -s.rho;
    for (const s of segs) {
      if (keep.length >= 48) break;
      if (keep.some(q => angDiff(q.th, s.th) < 3 * Math.PI / 180 && Math.abs(rhoOf(s, q) - q.rho) < MIND * 0.012)) continue;
      // No dejar que las vetas de una mesa (muchas líneas paralelas) acaparen la lista
      if (keep.filter(q => angDiff(q.th, s.th) < 6 * Math.PI / 180).length >= 8) continue;
      keep.push(s);
    }
    const inter = (a, b) => {
      const det = a.nx * b.ny - a.ny * b.nx; if (Math.abs(det) < 1e-6) return null;
      return { x: (a.rho * b.ny - a.ny * b.rho) / det, y: (a.nx * b.rho - a.rho * b.nx) / det };
    };
    const pairs = [];
    for (let i = 0; i < keep.length; i++) for (let j = i + 1; j < keep.length; j++) {
      const a = keep[i], b = keep[j];
      if (angDiff(a.th, b.th) > 30 * Math.PI / 180) continue;
      if (Math.abs(rhoOf(b, a) - a.rho) < MIND * 0.08) continue;
      pairs.push([a, b]);
    }
    let tested = 0;
    for (let i = 0; i < pairs.length && tested < 60000; i++) for (let j = i + 1; j < pairs.length && tested < 60000; j++) {
      const [a, b] = pairs[i], [c, d] = pairs[j];
      const da = angDiff(a.th, c.th);
      if (da < 50 * Math.PI / 180) continue;
      const q = [inter(a, c), inter(a, d), inter(b, d), inter(b, c)];
      if (q.some(p => !p)) continue;
      tested++;
      scoreQuad(q, "lines");
    }

    // D. GrabCut si nada convence
    cands.sort((a, b) => b.score - a.score);
    if ((!cands.length || cands[0].score < 0.3) && typeof cv.grabCut === "function") {
      const gk = Math.min(1, 360 / Math.max(W, H));
      const g = M(new cv.Mat()); cv.resize(rgb, g, new cv.Size(Math.round(W * gk), Math.round(H * gk)), 0, 0, cv.INTER_AREA);
      const mask = M(new cv.Mat()), bgd = M(new cv.Mat()), fgd = M(new cv.Mat());
      const rect = new cv.Rect(Math.round(g.cols * 0.04), Math.round(g.rows * 0.04), Math.round(g.cols * 0.92), Math.round(g.rows * 0.92));
      cv.grabCut(g, mask, rect, bgd, fgd, 3, cv.GC_INIT_WITH_RECT);
      const fg = M(new cv.Mat(mask.rows, mask.cols, cv.CV_8UC1));
      for (let i = 0; i < mask.data.length; i++) fg.data[i] = (mask.data[i] === 1 || mask.data[i] === 3) ? 255 : 0;
      cv.morphologyEx(fg, fg, cv.MORPH_OPEN, k5, new cv.Point(-1, -1), 1);
      const contours = M(new cv.MatVector()), hier = M(new cv.Mat());
      cv.findContours(fg, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      let best = null, ba = 0;
      for (let i = 0; i < contours.size(); i++) { const c = contours.get(i); mats.push(c); const a = cv.contourArea(c); if (a > ba) { ba = a; best = c; } }
      if (best) {
        const scaled = M(new cv.Mat()); best.convertTo(scaled, cv.CV_32S, 1 / gk, 0);
        hullQuads(scaled, "grabcut");
        // GrabCut suele acertar aunque el borde sea tenue: se le da un piso
        for (const c of cands) if (c.source.startsWith("grabcut")) c.score = Math.max(c.score, 0.3 * c.rs);
      }
    }
  } catch (err) {
    console.warn("Detección falló", err);
  } finally {
    for (const m of mats) { try { m.delete(); } catch (e) {} }
  }
  if (!cands.length) return [];
  // Entre candidatos casi iguales, preferir el más grande (borde exterior de la cédula)
  cands.sort((a, b) => b.score - a.score);
  const top = cands[0];
  let best = top;
  // La foto o un recuadro interno de la cédula nunca debe ganarle al borde de la cédula que lo contiene
  for (const c of cands) {
    if (c.af <= best.af * 1.02) continue;
    if (!top.quad.every(p => pointInQuad(p, c.quad))) continue;
    const near = c.score >= top.score * 0.9 && c.af < top.af * 1.35;
    const solid = c.score >= top.score * 0.45 && c.sMin >= 0.45 && c.rs >= 0.6 && c.support >= 0.65;
    if (!(near || solid)) continue;
    // Si adentro hay otra cédula aparte de la primera, es un recuadro que abarca dos cédulas: no sirve
    const holdsAnother = cands.some(d => d !== top && d.score >= top.score * 0.5 && d.support >= 0.75 && d.rs >= 0.5 &&
      d.af / top.af > 0.5 && d.af / top.af < 2 && pointInQuad(centroid(d.quad), c.quad) &&
      !pointInQuad(centroid(d.quad), top.quad) && !d.quad.some(p => pointInQuad(p, top.quad)) &&
      dist(centroid(d.quad), centroid(top.quad)) > Math.max(dist(top.quad[0], top.quad[1]), dist(top.quad[1], top.quad[2])) * 0.8);
    if (!holdsAnother) best = c;
  }
  const result = [best];
  // ¿Hay una segunda cédula en la misma foto?
  const bc = centroid(best.quad);
  for (const c of cands) {
    if (c === best || c.score < Math.max(0.2, best.score * 0.5) || c.support < 0.75 || c.sMin < 0.55 || c.rs < 0.5 || c.tex < 0.05) continue;
    const cc = centroid(c.quad);
    if (pointInQuad(cc, best.quad) || pointInQuad(bc, c.quad)) continue;
    if (c.quad.some(p => pointInQuad(p, best.quad))) continue;
    const r = c.af / best.af; if (r < 0.5 || r > 2) continue;
    // Dos mitades de la misma cédula quedan pegadas; dos cédulas distintas están a una cédula de distancia
    const longSide = Math.max(dist(best.quad[0], best.quad[1]), dist(best.quad[1], best.quad[2]));
    if (dist(bc, cc) < longSide * 0.8) continue;
    result.push(c); break;
  }
  result.sort((a, b) => { const ca = centroid(a.quad), cb = centroid(b.quad); return Math.abs(ca.y - cb.y) > H * 0.15 ? ca.y - cb.y : ca.x - cb.x; });
  return result.map(c => {
    let quad = c.quad.map(p => ({ x: p.x / k, y: p.y / k }));
    quad = refineQuad(srcCanvas, quad);
    // Un pelo hacia adentro para que no quede ni una línea del fondo
    const cc = centroid(quad);
    quad = quad.map(p => ({ x: p.x + (cc.x - p.x) * 0.006, y: p.y + (cc.y - p.y) * 0.006 }));
    return { quad, confident: c.support > 0.7 && c.rs > 0.45 && c.score > 0.3 };
  });
}

// Proporción real del rectángulo corrigiendo la perspectiva (Zhang & He, "Whiteboard scanning")
function trueRatio(q, W, H) {
  const naive = quadRatio(q);
  try {
    const u0 = W / 2, v0 = H / 2;
    const P = p => [p.x, p.y, 1];
    const m1 = P(q[0]), m2 = P(q[1]), m3 = P(q[3]), m4 = P(q[2]);
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const k2 = dot(cross(m1, m4), m3) / dot(cross(m2, m4), m3);
    const k3 = dot(cross(m1, m4), m2) / dot(cross(m3, m4), m2);
    const n2 = m2.map((v, i) => k2 * v - m1[i]), n3 = m3.map((v, i) => k3 * v - m1[i]);
    if (Math.abs(n2[2]) < 1e-6 || Math.abs(n3[2]) < 1e-6) return naive; // sin perspectiva apreciable
    const f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0) +
                 (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)) / (n2[2] * n3[2]);
    const diag = Math.hypot(W, H);
    if (!(f2 > 0) || Math.sqrt(f2) < diag * 0.25 || Math.sqrt(f2) > diag * 6) return naive;
    const nrm = n => (Math.pow(n[0] - u0 * n[2], 2) + Math.pow(n[1] - v0 * n[2], 2)) / f2 + n[2] * n[2];
    const r = Math.sqrt(nrm(n2) / nrm(n3));
    if (!isFinite(r) || r <= 0) return naive;
    return Math.max(r, 1 / r);
  } catch (e) { return naive; }
}
function centroid(q) { return { x: q.reduce((s, p) => s + p.x, 0) / q.length, y: q.reduce((s, p) => s + p.y, 0) / q.length }; }
function pointInQuad(p, q) {
  let inside = false;
  for (let i = 0, j = q.length - 1; i < q.length; j = i++) {
    const a = q[i], b = q[j];
    if (((a.y > p.y) !== (b.y > p.y)) && (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)) inside = !inside;
  }
  return inside;
}

// Afina cada lado buscando el borde más fuerte a lo largo de la normal, en alta resolución
function refineQuad(srcCanvas, quad) {
  try {
    const hk = Math.min(1, 1600 / Math.max(srcCanvas.width, srcCanvas.height));
    const w = Math.round(srcCanvas.width * hk), h = Math.round(srcCanvas.height * hk);
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(srcCanvas, 0, 0, w, h);
    const d = ctx.getImageData(0, 0, w, h).data;
    const px = (x, y, ch) => {
      const xi = Math.max(0, Math.min(w - 1, Math.round(x))), yi = Math.max(0, Math.min(h - 1, Math.round(y)));
      return d[(yi * w + xi) * 4 + ch];
    };
    const q = quad.map(p => ({ x: p.x * hk, y: p.y * hk }));
    const diag = Math.hypot(w, h), R = Math.max(6, diag * 0.012);
    const lines = [];
    for (let i = 0; i < 4; i++) {
      const a = q[i], b = q[(i + 1) % 4], len = dist(a, b);
      const ux = (b.x - a.x) / len, uy = (b.y - a.y) / len;
      const nx = -uy, ny = ux;
      const pts = [];
      const n = Math.max(20, Math.round(len / 8));
      for (let j = 0; j < n; j++) {
        const t = 0.08 + 0.84 * j / (n - 1);
        const bx = a.x + (b.x - a.x) * t, by = a.y + (b.y - a.y) * t;
        let bestG = 0, bestO = 0;
        for (let o = -R; o <= R; o += 1) {
          const x = bx + nx * o, y = by + ny * o;
          let g = 0;
          for (let ch = 0; ch < 3; ch++) g += Math.abs(px(x + nx * 1.5, y + ny * 1.5, ch) - px(x - nx * 1.5, y - ny * 1.5, ch));
          if (g > bestG) { bestG = g; bestO = o; }
        }
        if (bestG > 30) pts.push({ x: bx + nx * bestO, y: by + ny * bestO });
      }
      if (pts.length < n * 0.35) { lines.push(null); continue; }
      let fit = fitLine(pts);
      for (let it = 0; it < 2; it++) {
        const res = pts.map(p => Math.abs((p.x - fit.x0) * fit.nx + (p.y - fit.y0) * fit.ny));
        const med = res.slice().sort((x, y) => x - y)[res.length >> 1];
        const kept = pts.filter((p, idx) => res[idx] <= Math.max(1.5, med * 2.5));
        if (kept.length < 6) break;
        fit = fitLine(kept);
      }
      lines.push(fit);
    }
    const out = q.map(p => ({ ...p }));
    for (let i = 0; i < 4; i++) {
      const l1 = lines[(i + 3) % 4], l2 = lines[i];
      if (!l1 || !l2) continue;
      const det = l1.nx * l2.ny - l1.ny * l2.nx; if (Math.abs(det) < 1e-6) continue;
      const r1 = l1.nx * l1.x0 + l1.ny * l1.y0, r2 = l2.nx * l2.x0 + l2.ny * l2.y0;
      const p = { x: (r1 * l2.ny - l1.ny * r2) / det, y: (l1.nx * r2 - r1 * l2.nx) / det };
      if (dist(p, q[i]) < diag * 0.02) out[i] = p;
    }
    return out.map(p => ({ x: p.x / hk, y: p.y / hk }));
  } catch (e) { return quad; }
}
function fitLine(pts) {
  const n = pts.length;
  const x0 = pts.reduce((s, p) => s + p.x, 0) / n, y0 = pts.reduce((s, p) => s + p.y, 0) / n;
  let sxx = 0, syy = 0, sxy = 0;
  for (const p of pts) { const dx = p.x - x0, dy = p.y - y0; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
  const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  return { x0, y0, nx: -Math.sin(ang), ny: Math.cos(ang) };
}
function detectQuad(srcCanvas) { const r = detectCards(srcCanvas); return r.length ? r[0] : null; }
function rectPoints(r) {
  const { center: c, size: s } = r;
  const a = r.angle * Math.PI / 180, cos = Math.cos(a), sin = Math.sin(a);
  const hw = s.width / 2, hh = s.height / 2;
  return [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => ({ x: c.x + x * cos - y * sin, y: c.y + x * sin + y * cos }));
}

/* ---------- Geometría ---------- */
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function polyArea(q) { let s = 0; for (let i = 0; i < q.length; i++) { const a = q[i], b = q[(i + 1) % q.length]; s += a.x * b.y - b.x * a.y; } return Math.abs(s) / 2; }
function quadRatio(q) { const w = (dist(q[0], q[1]) + dist(q[3], q[2])) / 2, h = (dist(q[0], q[3]) + dist(q[1], q[2])) / 2; return Math.max(w, h) / Math.max(1, Math.min(w, h)); }
// Ordena en sentido horario empezando arriba-izquierda y deja el lado largo arriba (cédula horizontal)
function orderQuad(pts) {
  const cx = pts.reduce((s, p) => s + p.x, 0) / 4, cy = pts.reduce((s, p) => s + p.y, 0) / 4;
  let q = pts.slice().sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));
  let start = 0, best = Infinity;
  q.forEach((p, i) => { if (p.x + p.y < best) { best = p.x + p.y; start = i; } });
  q = q.slice(start).concat(q.slice(0, start));
  const top = dist(q[0], q[1]) + dist(q[3], q[2]);
  const side = dist(q[1], q[2]) + dist(q[3], q[0]);
  if (side > top * 1.05) q = [q[3], q[0], q[1], q[2]];
  return q;
}
function defaultQuad(c) { const mx = c.width * 0.06, my = c.height * 0.06; return [{ x: mx, y: my }, { x: c.width - mx, y: my }, { x: c.width - mx, y: c.height - my }, { x: mx, y: c.height - my }]; }
function fullQuad(c) { return [{ x: 0, y: 0 }, { x: c.width, y: 0 }, { x: c.width, y: c.height }, { x: 0, y: c.height }]; }

// Homografía destino→origen
function homography(dst, src) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = dst[i], { x: u, y: v } = src[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  const n = 8;
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    for (let r = c + 1; r < n; r++) { const f = A[r][c] / A[c][c]; for (let k = c; k < n; k++) A[r][k] -= f * A[c][k]; b[r] -= f * b[c]; }
  }
  const h = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = b[r]; for (let k = r + 1; k < n; k++) s -= A[r][k] * h[k]; h[r] = s / A[r][r]; }
  return [...h, 1];
}
function warp(srcCanvas, quad) {
  const sw = srcCanvas.width, sh = srcCanvas.height;
  const sd = srcCanvas.getContext("2d").getImageData(0, 0, sw, sh).data;
  const out = document.createElement("canvas"); out.width = OUT_W; out.height = OUT_H;
  const octx = out.getContext("2d");
  const od = octx.createImageData(OUT_W, OUT_H), o = od.data;
  const H = homography([{ x: 0, y: 0 }, { x: OUT_W, y: 0 }, { x: OUT_W, y: OUT_H }, { x: 0, y: OUT_H }], quad);
  let i = 0;
  for (let y = 0; y < OUT_H; y++) {
    const yy = y + 0.5;
    for (let x = 0; x < OUT_W; x++, i += 4) {
      const xx = x + 0.5;
      const w = H[6] * xx + H[7] * yy + 1;
      let u = (H[0] * xx + H[1] * yy + H[2]) / w - 0.5;
      let v = (H[3] * xx + H[4] * yy + H[5]) / w - 0.5;
      if (u < 0) u = 0; else if (u > sw - 1.001) u = sw - 1.001;
      if (v < 0) v = 0; else if (v > sh - 1.001) v = sh - 1.001;
      const x0 = u | 0, y0 = v | 0, fx = u - x0, fy = v - y0;
      const p00 = (y0 * sw + x0) * 4, p10 = p00 + 4, p01 = p00 + sw * 4, p11 = p01 + 4;
      const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
      o[i] = sd[p00] * w00 + sd[p10] * w10 + sd[p01] * w01 + sd[p11] * w11;
      o[i + 1] = sd[p00 + 1] * w00 + sd[p10 + 1] * w10 + sd[p01 + 1] * w01 + sd[p11 + 1] * w11;
      o[i + 2] = sd[p00 + 2] * w00 + sd[p10 + 2] * w10 + sd[p01 + 2] * w01 + sd[p11 + 2] * w11;
      o[i + 3] = 255;
    }
  }
  octx.putImageData(od, 0, 0);
  return out;
}

/* ---------- Mejora de imagen ---------- */
function enhance(canvas) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, Hh = canvas.height;
  const img = ctx.getImageData(0, 0, W, Hh), d = img.data;
  // Niveles automáticos por percentiles (1% – 99%) sobre la luminancia
  const hist = new Uint32Array(256);
  for (let i = 0; i < d.length; i += 16) hist[(d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114) | 0]++;
  const total = d.length / 16;
  let lo = 0, hi = 255, acc = 0;
  for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc > total * 0.01) { lo = v; break; } }
  acc = 0;
  for (let v = 255; v >= 0; v--) { acc += hist[v]; if (acc > total * 0.01) { hi = v; break; } }
  if (hi - lo < 40) { lo = 0; hi = 255; }
  const lut = new Uint8ClampedArray(256);
  for (let v = 0; v < 256; v++) lut[v] = ((v - lo) * 255) / (hi - lo);
  for (let i = 0; i < d.length; i += 4) { d[i] = lut[d[i]]; d[i + 1] = lut[d[i + 1]]; d[i + 2] = lut[d[i + 2]]; }
  // Nitidez suave (unsharp 3×3)
  const src = new Uint8ClampedArray(d);
  const amt = 0.45, row = W * 4;
  for (let y = 1; y < Hh - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const p = y * row + x * 4;
      for (let c = 0; c < 3; c++) {
        const q = p + c;
        const blur = (src[q - row - 4] + src[q - row] * 2 + src[q - row + 4] + src[q - 4] * 2 + src[q] * 4 + src[q + 4] * 2 + src[q + row - 4] + src[q + row] * 2 + src[q + row + 4]) / 16;
        d[q] = src[q] + (src[q] - blur) * amt;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}
function roundCorners(canvas) {
  const r = OUT_W / CARD_MM.w * CORNER_MM;
  const tmp = document.createElement("canvas"); tmp.width = canvas.width; tmp.height = canvas.height;
  const t = tmp.getContext("2d");
  t.fillStyle = "#fff"; t.fillRect(0, 0, tmp.width, tmp.height);
  t.save(); t.beginPath();
  const w = tmp.width, h = tmp.height;
  t.moveTo(r, 0); t.lineTo(w - r, 0); t.arcTo(w, 0, w, r, r); t.lineTo(w, h - r); t.arcTo(w, h, w - r, h, r);
  t.lineTo(r, h); t.arcTo(0, h, 0, h - r, r); t.lineTo(0, r); t.arcTo(0, 0, r, 0, r); t.closePath();
  t.clip(); t.drawImage(canvas, 0, 0); t.restore();
  return tmp;
}
function rotate180(canvas) {
  const c = document.createElement("canvas"); c.width = canvas.width; c.height = canvas.height;
  const x = c.getContext("2d"); x.translate(c.width, c.height); x.rotate(Math.PI); x.drawImage(canvas, 0, 0);
  return c;
}

/* ---------- Carga ---------- */
async function fileToCanvas(file) {
  let bmp;
  try { bmp = await createImageBitmap(file, { imageOrientation: "from-image" }); }
  catch (e) {
    bmp = await new Promise((res, rej) => {
      const img = new Image(); const url = URL.createObjectURL(file);
      img.onload = () => { URL.revokeObjectURL(url); res(img); };
      img.onerror = () => { URL.revokeObjectURL(url); rej(new Error("formato")); };
      img.src = url;
    });
  }
  const w = bmp.width, h = bmp.height;
  const k = Math.min(1, MAX_SRC / Math.max(w, h));
  const c = document.createElement("canvas"); c.width = Math.round(w * k); c.height = Math.round(h * k);
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  return c;
}
let cvWaiters = [];
function waitForDetector(ms = 40000) {
  if (cvReady || $("#cvDot").classList.contains("fail")) return Promise.resolve();
  return new Promise(res => { cvWaiters.push(res); setTimeout(res, ms); });
}
async function processFiles(files, preferSide) {
  files = [...files].filter(f => !f.type || f.type.startsWith("image/"));
  if (!files.length) { toast("Eso no es una imagen. Usa fotos JPG o PNG."); return; }
  const canvases = [];
  for (const f of files.slice(0, 2)) {
    try { canvases.push(await fileToCanvas(f)); }
    catch (e) {
      const heic = /heic|heif/i.test(f.type || f.name || "");
      toast(heic ? "Este navegador no abre fotos HEIC de iPhone. Pide la foto por WhatsApp (llega en JPG) o conviértela." : "No se pudo abrir una de las fotos. Prueba con JPG o PNG.");
    }
  }
  if (canvases.length) await processCanvases(canvases, preferSide);
}
async function processCanvases(canvases, preferSide) {
  const targets = preferSide && canvases.length === 1 ? [preferSide] : ["front", "back"];
  targets.forEach(t => setBusy(t, true, cvReady ? "Enderezando…" : "Preparando detector…"));
  await waitForDetector();
  targets.forEach(t => setBusy(t, true, "Enderezando…"));
  await new Promise(r => setTimeout(r, 30));
  // Cada foto puede traer una o dos caras
  const found = [];
  for (const c of canvases) {
    const dets = detectCards(c);
    if (!dets.length) found.push({ src: c, quad: defaultQuad(c), auto: null, confident: false });
    else for (const d of (canvases.length === 1 ? dets : dets.slice(0, 1))) found.push({ src: c, quad: d.quad, auto: d.quad, confident: d.confident });
  }
  let order;
  if (found.length >= 2) order = ["front", "back"];
  else if (preferSide) order = [preferSide];
  else order = [!sides.front.src ? "front" : !sides.back.src ? "back" : "front"];
  ["front", "back"].forEach(t => { if (!order.includes(t)) setBusy(t, false); });
  order.forEach((side, i) => {
    const f = found[i]; if (!f) return;
    Object.assign(sides[side], newSide(), { src: f.src, quad: f.quad, autoQuad: f.auto && f.auto.map(p => ({ ...p })), confident: f.confident });
    renderSide(side);
  });
  if (found.length >= 2 && canvases.length === 1) toast("Encontré las dos caras en la misma foto");
}
function runDetection(side) {
  const st = sides[side];
  const det = detectQuad(st.src);
  if (det) { st.quad = det.quad; st.autoQuad = det.quad.map(p => ({ ...p })); st.confident = det.confident; }
  else { st.quad = defaultQuad(st.src); st.autoQuad = null; st.confident = false; }
}

/* ---------- Render ---------- */
function renderSide(side) {
  const st = sides[side];
  setBusy(side, true);
  requestAnimationFrame(() => setTimeout(() => {
    let out = warp(st.src, st.quad);
    if ($("#optEnhance").checked) enhance(out);
    if (st.rot) out = rotate180(out);
    if ($("#optRound").checked) out = roundCorners(out);
    st.out = out;
    st.dataUrl = out.toDataURL("image/jpeg", 0.93);
    setBusy(side, false);
    paintSlot(side);
    paintSheet();
  }, 10));
}
function rerenderAll() { for (const s of ["front", "back"]) if (sides[s].src) renderSide(s); }

function setBusy(side, on, label = "Procesando…") {
  const st = sides[side]; st.busy = on;
  const drop = $('[data-role="drop"]', slotEl(side));
  let b = $(".busy", drop);
  if (on && !b) { b = document.createElement("div"); b.className = "busy"; b.innerHTML = '<span class="spin" aria-hidden="true"></span><span class="busy-t"></span>'; drop.appendChild(b); }
  if (on) $(".busy-t", b).textContent = label;
  if (!on && b) b.remove();
}
function paintSlot(side) {
  const st = sides[side], el = slotEl(side);
  const drop = $('[data-role="drop"]', el), hint = $('[data-role="hint"]', el);
  const chip = $('[data-role="chip"]', el), actions = $('[data-role="actions"]', el);
  $("canvas.result", drop)?.remove();
  if (!st.out) {
    drop.classList.remove("has"); hint.hidden = false; actions.hidden = true;
    chip.className = "chip idle"; chip.textContent = "Sin foto";
    return;
  }
  drop.classList.add("has"); hint.hidden = true; actions.hidden = false;
  const c = document.createElement("canvas"); c.className = "result";
  c.width = 1011; c.height = 638; c.getContext("2d").drawImage(st.out, 0, 0, 1011, 638);
  drop.insertBefore(c, drop.firstChild);
  if (st.edited) { chip.className = "chip ok"; chip.textContent = "Ajustada a mano"; }
  else if (st.confident) { chip.className = "chip ok"; chip.textContent = "Detectada automáticamente"; }
  else { chip.className = "chip warn"; chip.textContent = "Revisa los bordes"; }
}

function layout() {
  const scale = parseFloat($('input[name="scale"]:checked').value);
  const paper = PAPERS[$('input[name="paper"]:checked').value];
  const w = CARD_MM.w * scale, h = CARD_MM.h * scale;
  const x = (paper.w - w) / 2;
  const items = [];
  const present = ["front", "back"].filter(s => sides[s].dataUrl);
  const order = present.length ? present : [];
  let y = TOP_MM;
  for (const s of ["front", "back"]) {
    items.push({ side: s, x, y, w, h, url: sides[s].dataUrl });
    y += h + GAP_MM;
  }
  return { paper, scale, items: items.filter(it => it.url || !order.length ? true : false), all: items };
}
function paintSheet() {
  const L = layout();
  const sheet = $("#sheet");
  sheet.style.aspectRatio = `${L.paper.w} / ${L.paper.h}`;
  sheet.innerHTML = "";
  const cut = $("#optCut").checked;
  for (const it of L.all) {
    const style = `left:${it.x / L.paper.w * 100}%;top:${it.y / L.paper.h * 100}%;width:${it.w / L.paper.w * 100}%;height:${it.h / L.paper.h * 100}%`;
    if (it.url) {
      const d = document.createElement("div"); d.className = "item" + (cut ? " cut" : ""); d.setAttribute("style", style);
      const img = document.createElement("img"); img.src = it.url; img.alt = it.side === "front" ? "Frente" : "Reverso";
      d.appendChild(img); sheet.appendChild(d);
    } else {
      const d = document.createElement("div"); d.className = "ph"; d.setAttribute("style", style);
      d.textContent = it.side === "front" ? "frente" : "reverso"; sheet.appendChild(d);
    }
  }
  const any = L.all.some(it => it.url);
  $("#printBtn").disabled = !any; $("#pdfBtn").disabled = !any;
  const paperName = $('input[name="paper"]:checked').value === "carta" ? "Carta" : "Oficio";
  $("#sheetMeta").textContent = `${paperName} · cédula a ${Math.round(L.scale * 100)}% = ${fmt(CARD_MM.w * L.scale)} × ${fmt(CARD_MM.h * L.scale)} mm`;
  $("#page-style").textContent = `@page { size: ${L.paper.w}mm ${L.paper.h}mm; margin: 0; }`;
}
const fmt = n => n.toFixed(1).replace(".", ",");

function buildPrintSheet() {
  const L = layout();
  const ps = $("#print-sheet");
  ps.style.width = L.paper.w + "mm"; ps.style.height = (L.paper.h - 0.5) + "mm";
  ps.innerHTML = "";
  const cut = $("#optCut").checked;
  for (const it of L.all) {
    if (!it.url) continue;
    const d = document.createElement("div"); d.className = "item" + (cut ? " cut" : "");
    d.style.left = it.x + "mm"; d.style.top = it.y + "mm"; d.style.width = it.w + "mm"; d.style.height = it.h + "mm";
    const img = document.createElement("img"); img.src = it.url; img.alt = "";
    d.appendChild(img); ps.appendChild(d);
  }
  return Promise.all([...ps.querySelectorAll("img")].map(img => img.decode ? img.decode().catch(() => {}) : null));
}

/* ---------- Acciones ---------- */
$("#printBtn").addEventListener("click", async () => {
  await buildPrintSheet();
  window.print();
});
$("#pdfBtn").addEventListener("click", () => {
  if (!window.jspdf) { toast("No cargó el generador de PDF. Revisa la conexión a internet y recarga la página."); return; }
  const L = layout();
  const doc = new window.jspdf.jsPDF({ unit: "mm", format: [L.paper.w, L.paper.h], orientation: "portrait" });
  const cut = $("#optCut").checked;
  for (const it of L.all) {
    if (!it.url) continue;
    doc.addImage(it.url, "JPEG", it.x, it.y, it.w, it.h, undefined, "FAST");
    if (cut) { doc.setLineDashPattern([1.5, 1], 0); doc.setLineWidth(0.2); doc.setDrawColor(120); doc.rect(it.x - 1, it.y - 1, it.w + 2, it.h + 2); }
  }
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
  doc.save(`cedula-${Math.round(L.scale * 100)}-${stamp}.pdf`);
  toast("PDF descargado");
});
$("#file-many").addEventListener("change", e => { const fs = [...e.target.files]; e.target.value = ""; processFiles(fs); });
{
  const bd = $("#bigdrop");
  bd.addEventListener("dragover", e => { e.preventDefault(); bd.classList.add("over"); });
  bd.addEventListener("dragleave", () => bd.classList.remove("over"));
  bd.addEventListener("drop", e => { e.preventDefault(); e.stopPropagation(); bd.classList.remove("over"); processFiles([...e.dataTransfer.files]); });
}
$("#swapBtn").addEventListener("click", () => {
  [sides.front, sides.back] = [sides.back, sides.front];
  paintSlot("front"); paintSlot("back"); paintSheet();
});
$("#resetBtn").addEventListener("click", () => {
  sides.front = newSide(); sides.back = newSide();
  for (const s of ["front", "back"]) { paintSlot(s); $(`#file-${s}`).value = ""; }
  paintSheet();
  toast("Listo para una nueva cédula");
});
for (const el of document.querySelectorAll('input[name="scale"],input[name="paper"],#optCut')) el.addEventListener("change", paintSheet);
for (const el of document.querySelectorAll("#optEnhance,#optRound")) el.addEventListener("change", rerenderAll);

for (const side of ["front", "back"]) {
  const el = slotEl(side);
  const drop = $('[data-role="drop"]', el);
  const input = $(`#file-${side}`);
  input.addEventListener("change", () => { const fs = [...input.files]; input.value = ""; processFiles(fs, fs.length === 1 ? side : undefined); });
  drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", e => {
    e.preventDefault(); e.stopPropagation(); drop.classList.remove("over");
    const files = [...e.dataTransfer.files];
    if (files.length) processFiles(files, files.length === 1 ? side : undefined);
  });
  el.addEventListener("click", e => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (!act) return;
    const st = sides[side];
    if (act === "edit") openEditor(side);
    if (act === "rotate") { st.rot = st.rot ? 0 : 180; renderSide(side); }
    if (act === "replace") input.click();
    if (act === "clear") { sides[side] = newSide(); paintSlot(side); paintSheet(); }
  });
}
// Soltar en cualquier parte de la página
document.addEventListener("dragover", e => e.preventDefault());
document.addEventListener("drop", e => {
  e.preventDefault();
  const files = [...(e.dataTransfer?.files || [])].filter(f => !f.type || f.type.startsWith("image/"));
  placeFiles(files);
});
function placeFiles(files) { if (files.length) processFiles(files); }
// Pegar (Ctrl+V desde WhatsApp Web, captura, etc.)
document.addEventListener("paste", e => {
  if (!$("#editor").hidden) return;
  const files = [...(e.clipboardData?.items || [])].filter(i => i.type.startsWith("image/")).map(i => i.getAsFile()).filter(Boolean);
  if (files.length) { e.preventDefault(); placeFiles(files); }
});

/* ---------- Editor de bordes ---------- */
const ed = { side: null, quad: null, drag: -1, scale: 1, hover: -1, pointer: null };
const edCanvas = $("#edCanvas"), edCtx = edCanvas.getContext("2d");
function openEditor(side) {
  const st = sides[side]; if (!st.src) return;
  ed.side = side; ed.quad = st.quad.map(p => ({ ...p })); ed.drag = -1; ed.pointer = null;
  $("#edTitle").textContent = side === "front" ? "Ajustar bordes · Frente" : "Ajustar bordes · Reverso";
  $("#edAuto").disabled = !cvReady;
  $("#editor").hidden = false;
  const maxW = Math.min(1600, st.src.width), k = maxW / st.src.width;
  edCanvas.width = Math.round(st.src.width * k); edCanvas.height = Math.round(st.src.height * k);
  ed.scale = k;
  drawEditor();
  $("#edApply").focus();
}
function closeEditor() { $("#editor").hidden = true; ed.side = null; }
function drawEditor() {
  const st = sides[ed.side]; if (!st) return;
  const k = ed.scale, c = edCtx, W = edCanvas.width, Hh = edCanvas.height;
  c.clearRect(0, 0, W, Hh);
  c.drawImage(st.src, 0, 0, W, Hh);
  const q = ed.quad.map(p => ({ x: p.x * k, y: p.y * k }));
  // Oscurecer fuera del cuadro
  c.save(); c.fillStyle = "rgba(0,0,0,.45)"; c.beginPath(); c.rect(0, 0, W, Hh);
  c.moveTo(q[0].x, q[0].y); for (let i = 3; i >= 0; i--) c.lineTo(q[i].x, q[i].y); c.closePath(); c.fill("evenodd"); c.restore();
  const lw = Math.max(2, W / 500);
  c.strokeStyle = "#4F8BFF"; c.lineWidth = lw; c.beginPath(); q.forEach((p, i) => i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)); c.closePath(); c.stroke();
  const r = Math.max(10, W / 90);
  q.forEach((p, i) => {
    c.beginPath(); c.arc(p.x, p.y, r, 0, Math.PI * 2);
    c.fillStyle = i === ed.drag || i === ed.hover ? "#FFFFFF" : "rgba(79,139,255,.9)"; c.fill();
    c.lineWidth = lw; c.strokeStyle = "#FFFFFF"; c.stroke();
  });
  // Lupa
  if (ed.drag >= 0) {
    const p = q[ed.drag], z = 3, R = Math.max(70, W / 12);
    const lx = p.x < W / 2 ? W - R - 16 : R + 16, ly = R + 16;
    c.save(); c.beginPath(); c.arc(lx, ly, R, 0, Math.PI * 2); c.clip();
    c.fillStyle = "#000"; c.fillRect(lx - R, ly - R, R * 2, R * 2);
    const sx = ed.quad[ed.drag].x, sy = ed.quad[ed.drag].y, span = (R / z) / k;
    c.drawImage(st.src, sx - span, sy - span, span * 2, span * 2, lx - R, ly - R, R * 2, R * 2);
    c.strokeStyle = "#FF4D4D"; c.lineWidth = 2;
    c.beginPath(); c.moveTo(lx - R, ly); c.lineTo(lx + R, ly); c.moveTo(lx, ly - R); c.lineTo(lx, ly + R); c.stroke();
    c.restore();
    c.beginPath(); c.arc(lx, ly, R, 0, Math.PI * 2); c.strokeStyle = "#fff"; c.lineWidth = 3; c.stroke();
  }
}
function edPos(e) {
  const r = edCanvas.getBoundingClientRect();
  return { x: (e.clientX - r.left) / r.width * edCanvas.width, y: (e.clientY - r.top) / r.height * edCanvas.height };
}
function nearest(p) {
  const k = ed.scale, W = edCanvas.width, thr = Math.max(40, W / 25);
  let best = -1, bd = Infinity;
  ed.quad.forEach((q, i) => { const d = Math.hypot(q.x * k - p.x, q.y * k - p.y); if (d < bd) { bd = d; best = i; } });
  return bd < thr ? best : -1;
}
edCanvas.addEventListener("pointerdown", e => {
  const p = edPos(e); ed.drag = nearest(p);
  if (ed.drag < 0) { // clic lejos: mueve la esquina más cercana
    let bd = Infinity; ed.quad.forEach((q, i) => { const d = Math.hypot(q.x * ed.scale - p.x, q.y * ed.scale - p.y); if (d < bd) { bd = d; ed.drag = i; } });
  }
  edCanvas.setPointerCapture(e.pointerId);
  moveCorner(p);
});
edCanvas.addEventListener("pointermove", e => {
  const p = edPos(e);
  if (ed.drag >= 0) moveCorner(p);
  else { const h = nearest(p); if (h !== ed.hover) { ed.hover = h; edCanvas.style.cursor = h >= 0 ? "grab" : "crosshair"; drawEditor(); } }
});
const endDrag = () => { if (ed.drag >= 0) { ed.drag = -1; drawEditor(); } };
edCanvas.addEventListener("pointerup", endDrag);
edCanvas.addEventListener("pointercancel", endDrag);
function moveCorner(p) {
  const st = sides[ed.side];
  ed.quad[ed.drag] = { x: Math.max(0, Math.min(st.src.width, p.x / ed.scale)), y: Math.max(0, Math.min(st.src.height, p.y / ed.scale)) };
  drawEditor();
}
$("#edCancel").addEventListener("click", closeEditor);
$("#edApply").addEventListener("click", () => {
  const st = sides[ed.side], side = ed.side;
  if (polyArea(ed.quad) < 1000) { toast("El recuadro quedó muy pequeño. Separa más las esquinas."); return; }
  st.quad = orderQuad(ed.quad); st.edited = true;
  closeEditor(); renderSide(side);
});
$("#edAuto").addEventListener("click", () => {
  const st = sides[ed.side];
  const det = detectQuad(st.src);
  if (det) { ed.quad = det.quad; drawEditor(); } else toast("No se encontró la cédula automáticamente. Ubica las esquinas a mano.");
});
$("#edFull").addEventListener("click", () => { ed.quad = fullQuad(sides[ed.side].src); drawEditor(); });
document.addEventListener("keydown", e => { if (e.key === "Escape" && !$("#editor").hidden) closeEditor(); });

/* ---------- Ejemplo ---------- */
$("#demoBtn").addEventListener("click", () => {
  processCanvases([makeDemoPhoto(false), makeDemoPhoto(true)]);
});
function makeDemoPhoto(back) {
  // Tarjeta genérica de ejemplo fotografiada torcida sobre una mesa
  const cw = 1011, ch = 638;
  const card = document.createElement("canvas"); card.width = cw; card.height = ch;
  const g = card.getContext("2d");
  const grad = g.createLinearGradient(0, 0, cw, ch);
  grad.addColorStop(0, back ? "#E9EEF2" : "#F3EFD9"); grad.addColorStop(1, back ? "#D4DDE5" : "#E4DDBA");
  g.fillStyle = grad; g.fillRect(0, 0, cw, ch);
  g.fillStyle = "#2B3A4A"; g.font = "600 44px sans-serif"; g.fillText("TARJETA DE EJEMPLO", 60, 90);
  g.font = "400 26px sans-serif"; g.fillStyle = "#4A5866"; g.fillText(back ? "Reverso · solo para probar" : "Frente · solo para probar", 60, 135);
  if (!back) {
    g.fillStyle = "#B8C2CC"; g.fillRect(60, 180, 240, 300);
    g.fillStyle = "#8C98A4"; g.beginPath(); g.arc(180, 290, 60, 0, Math.PI * 2); g.fill(); g.fillRect(100, 370, 160, 110);
    g.fillStyle = "#6C7885"; for (let i = 0; i < 5; i++) g.fillRect(340, 200 + i * 60, 420 - i * 40, 22);
  } else {
    g.fillStyle = "#6C7885"; for (let i = 0; i < 4; i++) g.fillRect(60, 190 + i * 55, 520 - i * 50, 20);
    for (let i = 0; i < 38; i++) g.fillRect(620 + i * 9, 190, (i % 3) + 2, 200);
    g.fillStyle = "#3D4A57"; g.font = "500 22px monospace";
    g.fillText("<<EJEMPLO<<<<<<<<<<<<<<<<<<<<<<<<", 60, 560);
  }
  g.strokeStyle = "rgba(0,0,0,.15)"; g.lineWidth = 4; g.strokeRect(2, 2, cw - 4, ch - 4);
  const W = 1600, H = 1200;
  const photo = document.createElement("canvas"); photo.width = W; photo.height = H;
  const p = photo.getContext("2d");
  const wood = p.createLinearGradient(0, 0, W, H);
  wood.addColorStop(0, "#6B4A30"); wood.addColorStop(1, "#3F2A1A");
  p.fillStyle = wood; p.fillRect(0, 0, W, H);
  for (let i = 0; i < 60; i++) { p.strokeStyle = `rgba(0,0,0,${0.05 + (i % 5) * 0.02})`; p.lineWidth = 2 + (i % 4); p.beginPath(); p.moveTo(0, i * 22); p.bezierCurveTo(W / 3, i * 22 + 30, (2 * W) / 3, i * 22 - 30, W, i * 22 + 10); p.stroke(); }
  const dst = back
    ? [{ x: 330, y: 300 }, { x: 1290, y: 210 }, { x: 1350, y: 870 }, { x: 290, y: 900 }]
    : [{ x: 260, y: 250 }, { x: 1250, y: 330 }, { x: 1180, y: 960 }, { x: 330, y: 880 }];
  // proyectar la tarjeta con la homografía inversa (origen = tarjeta)
  const Hm = homography(dst, [{ x: 0, y: 0 }, { x: cw, y: 0 }, { x: cw, y: ch }, { x: 0, y: ch }]);
  const cd = g.getImageData(0, 0, cw, ch).data;
  const pd = p.getImageData(0, 0, W, H);
  const o = pd.data;
  const minX = Math.min(...dst.map(d => d.x)), maxX = Math.max(...dst.map(d => d.x));
  const minY = Math.min(...dst.map(d => d.y)), maxY = Math.max(...dst.map(d => d.y));
  for (let y = minY; y < maxY; y++) for (let x = minX; x < maxX; x++) {
    const w = Hm[6] * x + Hm[7] * y + 1, u = (Hm[0] * x + Hm[1] * y + Hm[2]) / w, v = (Hm[3] * x + Hm[4] * y + Hm[5]) / w;
    if (u < 0 || v < 0 || u >= cw || v >= ch) continue;
    const si = ((v | 0) * cw + (u | 0)) * 4, di = (y * W + x) * 4;
    const shade = 0.88 + 0.12 * (x / W);
    o[di] = cd[si] * shade; o[di + 1] = cd[si + 1] * shade; o[di + 2] = cd[si + 2] * shade;
  }
  p.putImageData(pd, 0, 0);
  return photo;
}

/* ---------- Utilidades ---------- */
let toastT;
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => t.hidden = true, 3200); }

paintSheet();
loadOpenCV();

})();

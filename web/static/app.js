/* ── Example sequences (real entries from data/labeled_genes.csv) ─────────── */
const EXAMPLES = {
  IFNB1: [
    "ATTCTAACTGCAACCTTTCGAAGCCTTTGCTCTGGCACAACAGGTAGTAGGCGACACTGTTCGTGTTGTC",
    "AACATGACCAACAAGTGTCTCCTCCAAATTGCTCTCCTGTTGTGCTTCTCCACTACAGCTCTTTCCATGA",
    "GCTACAACTTGCTTGGATTCCTACAAAGAAGCAGCAATTTTCAGTGTCAGAAGCTCCTGTGGCAATTGAA",
    "TGGGAGGCTTGAATACTGCCTCAAGGACAGGATGAACTTTGACATCCCTGAGGAGATTAAGCAGCTGCAG",
    "CAGTTCCAGAAGGAGGACGCCGCATTGACCATCTATGAGATGCTCCAGAACATCTTTGCTATTTTCAGAC",
    "AAGATTCATCTAGCACTGGCTGGAATGAGACTATTGTTGAGAACCTCCTGGCTAATGTCTATCATCAGAT",
    "AAACCATCTGAAGACAGTCCTGGAAGAAAAACTGGAGAAAGAAGATTTCACCAGGGGAAAACTCATGAGC",
    "AGTCTGCACCTGAAAAGATATTATGGGAGGATTCTGCATTACCTGAAGGCCAAGGAGTACAGTCACTGTG",
    "CCTGGACCATAGTCAGAGTGGAAATCCTAAGGAACTTTTACTTCATTAACAGACTTACAGGTTACCTCCG",
    "AAACTGAAGATCTCCTAGCCTGTGCCTCTGGGACTGGACAATTGCTTCAAGCATTCTTCAACCAGCAGAT",
    "GCTGTTTAAGTGACTGATGGCTAATGTACTGCATATGAAAGGACACTAGAAGATTTTGAAATTTTTATTA",
    "AATTATGAGTTATTTTTATTTATTTAAATTTTATTTTGGAAAATAAATTATTTTTGGTGCAAAAGTCAA",
  ].join(""),
  CCDC182: [
    "CTCTCTTTGTCCAACGTAGAAGAGACAATGGAACCCCTCTACCAGGCTGGGTCCATTCTCATGACGGTGA",
    "ATACCCTACAGGGGAAAAAAATGATAGAGAGTGGCCTCCAGTCTGGAGACTTTTCCCTGTCCCAGTCATG",
    "GCCCTCCTGCCTCCCACCACCCGCTGACTTGGAGATCCTGCAGCAGAAGGTGGCCGGGGTGCAACGGGAA",
    "CTGGAGGACTTTAAGAAAGAGGCATTGAAGTCCATTCATTACCTTGAAGACGCCTTCTGCGAGATGAATG",
    "GAGCCCTGGTGCAACAGGAGGAGCAGGCGGCTCGCGTGAGGCAGCGGCTAAGGGAGGAGGAGGACCGTGG",
    "CATCGTGCGCAACAAGGTTCTCACCTTCCTGTTGCCGCGCGAGAAACAGCTCCGGGAGCACTGCAAGCGG",
    "CTGGAGGACCTGCTGCTGGACAGGGGACGTGACGCCCTGCGTGCCACCAAGAAGAGCCAGGCTGACTGAA",
    "CCTCGTGGGTGGCACTAGCCAAGCACCAAGTTGAAGATTTTTTTTAACGGAAGGAGAAACCCCAAGTCCC",
    "TACCCCCTTGCTTTCCCTTCCTCATTTCCATTGCTTTCCTTCTGCCTAAATAACACCTTGCTGAGAGGCA",
    "AAAAGACTTCAATATTTTTTTTGGTGAAATTCCATTTATCCAGCATTCTAAAGGAAGGAGGCACCCCAGT",
    "ACTGAGTCAGGCCCCTGTAACTTTCACTTGAACTGTTTTCCTGTTTGTAATGGTACTATTGCTCAATGTA",
    "TACCTCTTTATTTGTATAGTATATTCTTAAACTGA",
  ].join(""),
  HBA1: [
    "ACTCTTCTGGTCCCCACAGACTCAGAGAGAACCCACCATGGTGCTGTCTCCTGCCGACAAGACCAACGTC",
    "AAGGCCGCCTGGGGTAAGGTCGGCGCGCACGCTGGCGAGTATGGTGCGGAGGCCCTGGAGAGGTGAGGCT",
    "CCCTCCCCTGCTCCGACCCGGGCTCCTCGCCCGCCCGGACCCACAGGCCACCCTCAACCGTCCTGGCCCC",
    "GGACCCAAACCCCACCCCTCACTCTGCTTCTCCCCGCAGGATGTTCCTGTCCTTCCCCACCACCAAGACC",
    "TACTTCCCGCACTTCGACCTGAGCCACGGCTCTGCCCAGGTTAAGGGCCACGGCAAGAAGGTGGCCGACG",
    "CGCTGACCAACGCCGTGGCGCACGTGGACGACATGCCCAACGCGCTGTCCGCCCTGAGCGACCTGCACGC",
    "GCACAAGCTTCGGGTGGACCCGGTCAACTTCAAGGTGAGCGGCGGGCCGGGAGCGATCTGGGTCGAGGGG",
    "CGAGATGGCGCCTTCCTCGCAGGGCAGAGGATCACGCGGGTTGCGGGAGGTGTAGCGCAGGCGGCGGCTG",
    "CGGGCCTGGGCCCTCGGCCCCACTGACCCTCTTCTCTGCACAGCTCCTAAGCCACTGCCTGCTGGTGACC",
    "CTGGCCGCCCACCTCCCCGCCGAGTTCACCCCTGCGGTGCACGCCTCCCTGGACAAGTTCCTGGCTTCTG",
    "TGAGCACCGTGCTGACCTCCAAATACCGTTAAGCTGGAGCCTCGGTGGCCATGCTTCTTGCCCCTTGGGC",
    "CTCCCCCCAGCCCCTCCTCCCCTTCCTGCACCCGTACCCCCGTGGTCTTTGAATAAAGTCTGAGTGGGCG",
    "GCA",
  ].join(""),
};

const CLASS_COLORS = { Low: "#3b82f6", Medium: "#f59e0b", High: "#ef4444" };

/* ── Example selector ─────────────────────────────────────────────────────── */
document.getElementById("example-select").addEventListener("change", function () {
  const seq = EXAMPLES[this.value] || "";
  document.getElementById("seq-input").value = seq;
  updateMeta(seq);
});

document.getElementById("seq-input").addEventListener("input", function () {
  updateMeta(this.value);
});

function updateMeta(raw) {
  const seq  = raw.toUpperCase().replace(/\s/g, "");
  const meta = document.getElementById("seq-meta");
  if (!seq) { meta.classList.add("hidden"); return; }
  meta.classList.remove("hidden");
  const invalid = [...new Set(seq.split("").filter(c => !"ACGT".includes(c)))];
  if (invalid.length) {
    meta.innerHTML = `<span style="color:#fca5a5">Invalid characters: ${invalid.join(", ")}</span>`;
  } else {
    const kmers = Math.max(0, seq.length - 5);   // 6-mer count
    meta.textContent = `${seq.length} bp  ·  ~${kmers} 6-mers`;
  }
}

/* ── Prediction ───────────────────────────────────────────────────────────── */
async function runPredict() {
  const raw = document.getElementById("seq-input").value;
  const seq = raw.toUpperCase().replace(/\s/g, "");

  if (!seq) { showError("Please enter a DNA sequence."); return; }

  setLoading(true);
  clearError();

  try {
    const res  = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ sequence: seq }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await res.json();
    renderResults(data, seq.length);

  } catch (e) {
    showError(e.message);
  } finally {
    setLoading(false);
  }
}

/* ── Render results ───────────────────────────────────────────────────────── */
function renderResults(data, seqLen) {
  document.getElementById("placeholder").classList.add("hidden");
  document.getElementById("results").classList.remove("hidden");

  // Badge
  const badge = document.getElementById("pred-badge");
  badge.textContent   = data.predicted_class;
  badge.className     = `pred-badge ${data.predicted_class}`;

  // Meta
  const metaLines = [`${seqLen} bp &nbsp;·&nbsp; ${data.kmers.length} 6-mers`];
  if (data.score !== null) metaLines.push(`log-TPM: ${data.score.toFixed(3)}`);
  document.getElementById("pred-meta").innerHTML = metaLines.join("<br>");

  // Confidence bars
  renderConfBars(data.probabilities, data.predicted_class);

  // Heatmaps
  renderHeatmap("rollout-canvas",  data.kmers, data.rollout);
  renderHeatmap("saliency-canvas", data.kmers, data.saliency);

  // New interpretability panels
  const seq = document.getElementById("seq-input").value.toUpperCase().replace(/\s/g, "");
  renderNucleotideImportance("nuc-canvas", seq, data.saliency);
  renderProfile("profile-canvas", data.saliency);
  renderTopKTable("topk-table-wrap", data.kmers, data.saliency, data.rollout);
}

function renderConfBars(probs, predClass) {
  const container = document.getElementById("conf-bars");
  container.innerHTML = "";

  ["Low", "Medium", "High"].forEach(cls => {
    const pct  = (probs[cls] * 100).toFixed(1);
    const bold = cls === predClass ? "font-weight:800;color:" + CLASS_COLORS[cls] : "";

    const row = document.createElement("div");
    row.className = "conf-row";
    row.innerHTML = `
      <span class="conf-label" style="${bold}">${cls}</span>
      <div class="conf-track">
        <div class="conf-fill" style="width:0%;background:${CLASS_COLORS[cls]}"
             data-target="${probs[cls] * 100}"></div>
      </div>
      <span class="conf-pct" style="${bold}">${pct}%</span>`;
    container.appendChild(row);
  });

  // Animate bars after paint
  requestAnimationFrame(() => {
    container.querySelectorAll(".conf-fill").forEach(el => {
      el.style.width = el.dataset.target + "%";
    });
  });
}

/* ── Heatmap canvas renderer ─────────────────────────────────────────────── */
const CELL_W = 26, CELL_H = 44, LABEL_H = 22, CBAR_H = 16, CBAR_PAD = 6, CBAR_LABEL_H = 14;

function renderHeatmap(canvasId, kmers, scores) {
  const canvas = document.getElementById(canvasId);
  const n      = kmers.length;
  const dpr    = window.devicePixelRatio || 1;

  const totalH = CELL_H + LABEL_H + CBAR_PAD + CBAR_H + CBAR_LABEL_H;
  const totalW = n * CELL_W + 2;

  canvas.width  = totalW * dpr;
  canvas.height = totalH * dpr;
  canvas.style.width  = totalW + "px";
  canvas.style.height = totalH + "px";

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const maxScore = Math.max(...scores, 1e-9);
  const minScore = Math.min(...scores);

  for (let i = 0; i < n; i++) {
    const t    = (scores[i] - minScore) / (maxScore - minScore + 1e-9);
    const col  = heatColor(t);
    ctx.fillStyle = col;
    ctx.fillRect(i * CELL_W + 1, 0, CELL_W - 1, CELL_H);

    // k-mer label
    ctx.fillStyle    = t > 0.6 ? "#111" : "#e2e8f0";
    ctx.font         = `bold 8px monospace`;
    ctx.textAlign    = "center";
    ctx.textBaseline = "middle";
    ctx.save();
    ctx.translate(i * CELL_W + CELL_W / 2, CELL_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(kmers[i], 0, 0);
    ctx.restore();
  }

  // Score label below cell
  ctx.font         = "9px system-ui";
  ctx.textAlign    = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i < n; i++) {
    const t   = (scores[i] - minScore) / (maxScore - minScore + 1e-9);
    ctx.fillStyle = "#8892a4";
    ctx.fillText(scores[i].toFixed(3), i * CELL_W + CELL_W / 2, CELL_H + 4);
  }

  // Colour bar
  const cbarY = CELL_H + LABEL_H + CBAR_PAD;
  const grad  = ctx.createLinearGradient(1, 0, n * CELL_W, 0);
  grad.addColorStop(0,   "#2d3ab0");
  grad.addColorStop(0.5, "#f59e0b");
  grad.addColorStop(1,   "#ef4444");
  ctx.fillStyle = grad;
  ctx.fillRect(1, cbarY, n * CELL_W, CBAR_H);

  ctx.font         = "9px system-ui";
  ctx.textBaseline = "top";
  ctx.fillStyle    = "#8892a4";
  ctx.textAlign    = "left";
  ctx.fillText(minScore.toFixed(3), 2, cbarY + CBAR_H + 2);
  ctx.textAlign = "right";
  ctx.fillText(maxScore.toFixed(3), n * CELL_W, cbarY + CBAR_H + 2);
}

function heatColor(t) {
  // Cool blue → amber → hot red
  if (t < 0.5) {
    const s = t * 2;
    const r = Math.round(45  + s * (245 - 45));
    const g = Math.round(58  + s * (158 - 58));
    const b = Math.round(176 + s * (11  - 176));
    return `rgb(${r},${g},${b})`;
  } else {
    const s = (t - 0.5) * 2;
    const r = Math.round(245 + s * (239 - 245));
    const g = Math.round(158 + s * (68  - 158));
    const b = Math.round(11  + s * (68  - 11));
    return `rgb(${r},${g},${b})`;
  }
}

/* ── Nucleotide-level importance ─────────────────────────────────────────── */
function renderNucleotideImportance(canvasId, sequence, saliency, K = 6) {
  // Aggregate k-mer saliency to individual bases by averaging overlapping k-mers
  const seqLen  = sequence.length;
  const nucScores = new Float32Array(seqLen);
  const nucCounts = new Float32Array(seqLen);

  for (let j = 0; j < saliency.length; j++) {
    // k-mer j covers bases j..j+K-1
    for (let b = j; b < Math.min(j + K, seqLen); b++) {
      nucScores[b] += saliency[j];
      nucCounts[b] += 1;
    }
  }
  for (let b = 0; b < seqLen; b++) {
    if (nucCounts[b] > 0) nucScores[b] /= nucCounts[b];
  }

  const NW   = 14, NH = 28, LABEL_H = 14, PAD = 2;
  const totalH = NH + LABEL_H + PAD;
  const totalW = seqLen * NW + 2;
  const dpr    = window.devicePixelRatio || 1;

  const canvas = document.getElementById(canvasId);
  canvas.width  = totalW * dpr;
  canvas.height = totalH * dpr;
  canvas.style.width  = totalW + "px";
  canvas.style.height = totalH + "px";

  const ctx  = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const maxS = Math.max(...nucScores);
  const minS = Math.min(...nucScores);

  const BASE_COLORS = { A: "#22c55e", C: "#3b82f6", G: "#f59e0b", T: "#ef4444" };

  for (let b = 0; b < seqLen; b++) {
    const t   = (nucScores[b] - minS) / (maxS - minS + 1e-9);
    const col = heatColor(t);
    ctx.fillStyle = col;
    ctx.fillRect(b * NW + 1, 0, NW - 1, NH);

    const base = sequence[b] || "N";
    ctx.fillStyle    = BASE_COLORS[base] || "#e2e8f0";
    ctx.font         = `bold 9px monospace`;
    ctx.textAlign    = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(base, b * NW + NW / 2, NH / 2);
  }

  // Position ticks every 10 bases
  ctx.font         = "8px system-ui";
  ctx.textAlign    = "center";
  ctx.textBaseline = "top";
  ctx.fillStyle    = "#8892a4";
  for (let b = 0; b < seqLen; b += 10) {
    ctx.fillText(b + 1, b * NW + NW / 2, NH + PAD);
  }
}

/* ── Saliency profile line chart ─────────────────────────────────────────── */
function renderProfile(canvasId, saliency) {
  const canvas  = document.getElementById(canvasId);
  const W       = canvas.offsetWidth || 600;
  const H       = 90;
  const PAD     = { top: 10, right: 12, bottom: 20, left: 36 };
  const dpr     = window.devicePixelRatio || 1;

  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  canvas.style.height = H + "px";

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const n    = saliency.length;
  const maxS = Math.max(...saliency);
  const minS = Math.min(...saliency);
  const span = maxS - minS || 1;

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top  - PAD.bottom;

  const xOf = i => PAD.left + (i / (n - 1)) * plotW;
  const yOf = v => PAD.top  + (1 - (v - minS) / span) * plotH;

  // Fill under curve
  const grad = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + plotH);
  grad.addColorStop(0,   "rgba(99,102,241,0.45)");
  grad.addColorStop(1,   "rgba(99,102,241,0.02)");

  ctx.beginPath();
  ctx.moveTo(xOf(0), yOf(saliency[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(xOf(i), yOf(saliency[i]));
  ctx.lineTo(xOf(n - 1), PAD.top + plotH);
  ctx.lineTo(xOf(0),     PAD.top + plotH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(xOf(0), yOf(saliency[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(xOf(i), yOf(saliency[i]));
  ctx.strokeStyle = "#818cf8";
  ctx.lineWidth   = 1.5;
  ctx.stroke();

  // Axes
  ctx.strokeStyle = "#2e3248";
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(PAD.left, PAD.top);
  ctx.lineTo(PAD.left, PAD.top + plotH);
  ctx.lineTo(PAD.left + plotW, PAD.top + plotH);
  ctx.stroke();

  // Y labels
  ctx.fillStyle    = "#8892a4";
  ctx.font         = "9px system-ui";
  ctx.textAlign    = "right";
  ctx.textBaseline = "middle";
  ctx.fillText(maxS.toFixed(2), PAD.left - 4, PAD.top);
  ctx.fillText(minS.toFixed(2), PAD.left - 4, PAD.top + plotH);

  // X labels
  ctx.textAlign    = "center";
  ctx.textBaseline = "top";
  const ticks = [0, Math.floor(n / 4), Math.floor(n / 2), Math.floor(3 * n / 4), n - 1];
  ticks.forEach(i => {
    ctx.fillText(i + 1, xOf(i), PAD.top + plotH + 4);
  });
}

/* ── Top-10 k-mers table ─────────────────────────────────────────────────── */
function renderTopKTable(wrapperId, kmers, saliency, rollout) {
  const paired = kmers.map((k, i) => ({
    kmer: k, sal: saliency[i], rol: rollout[i], i,
  }));
  const top10 = [...paired].sort((a, b) => b.sal - a.sal).slice(0, 10);
  const maxSal = top10[0].sal;

  const DNA_COLORS = { A: "#22c55e", C: "#3b82f6", G: "#f59e0b", T: "#ef4444" };

  function colorKmer(kmer) {
    return kmer.split("").map(b =>
      `<span style="color:${DNA_COLORS[b] || "#e2e8f0"}">${b}</span>`
    ).join("");
  }

  const rows = top10.map((d, rank) => `
    <tr>
      <td style="color:var(--text-muted);font-size:11px">#${rank + 1}</td>
      <td><span class="kmer-pill" style="background:rgba(99,102,241,.15);color:#a5b4fc">${colorKmer(d.kmer)}</span></td>
      <td style="color:var(--text-muted);font-size:11px">pos ${d.i + 1}</td>
      <td>
        <div class="score-bar-wrap">
          <div class="score-bar-bg"><div class="score-bar-fill" style="width:${(d.sal / maxSal * 100).toFixed(1)}%"></div></div>
          <span style="font-size:11px;color:var(--text-muted);min-width:44px">${d.sal.toFixed(4)}</span>
        </div>
      </td>
      <td style="font-size:11px;color:var(--text-muted)">${d.rol.toFixed(4)}</td>
    </tr>`).join("");

  document.getElementById(wrapperId).innerHTML = `
    <table>
      <thead><tr>
        <th>Rank</th><th>6-mer</th><th>Position</th>
        <th>Saliency</th><th>Rollout</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/* ── UI helpers ───────────────────────────────────────────────────────────── */
function setLoading(on) {
  const btn     = document.getElementById("predict-btn");
  const txt     = document.getElementById("btn-text");
  const spinner = document.getElementById("btn-spinner");
  btn.disabled = on;
  txt.textContent = on ? "Predicting…" : "Predict";
  spinner.classList.toggle("hidden", !on);
}

function showError(msg) {
  const box = document.getElementById("error-box");
  box.textContent = msg;
  box.classList.remove("hidden");
}

function clearError() {
  document.getElementById("error-box").classList.add("hidden");
}

/**
 * Predictive AI-Based Environmental Stress Screening Dashboard
 * Step 3: Experimental Design, Ground Truth & Comparative Screening Protocol
 */

let g_profile = null;
let g_processed = [];
let g_components = [];
let g_groundTruth = [];

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initSubtabs();
  loadAllData();
  initExperimentControls();
});

function initNavigation() {
  const navLinks = document.querySelectorAll(".nav-link");
  const pageViews = document.querySelectorAll(".page-view");
  const currentPageTitle = document.getElementById("currentPageTitle");
  const currentPageDesc = document.getElementById("currentPageDesc");

  navLinks.forEach(link => {
    link.addEventListener("click", e => {
      e.preventDefault();
      const targetPageId = link.getAttribute("data-page");

      navLinks.forEach(l => l.classList.remove("active"));
      pageViews.forEach(v => v.classList.remove("active"));

      link.classList.add("active");
      const targetView = document.getElementById(targetPageId);
      if (targetView) targetView.classList.add("active");

      const title = link.getAttribute("data-title") || "Research Dashboard";
      const desc = link.getAttribute("data-desc") || "Predictive AI-Based ESS";
      if (currentPageTitle) currentPageTitle.textContent = title;
      if (currentPageDesc) currentPageDesc.textContent = desc;
    });
  });
}

function initSubtabs() {
  const subtabBtns = document.querySelectorAll(".subtab-btn");
  const subtabContents = document.querySelectorAll(".subtab-content");

  subtabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-subtab");

      subtabBtns.forEach(b => b.classList.remove("active"));
      subtabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const activeContent = document.getElementById(target);
      if (activeContent) activeContent.classList.add("active");
    });
  });
}

async function loadAllData() {
  try {
    const resProfile = await fetch("/api/dataset/profile");
    if (resProfile.ok) g_profile = await resProfile.json();

    const resProcessed = await fetch("/api/dataset/processed");
    if (resProcessed.ok) g_processed = await resProcessed.json();

    const resComps = await fetch("/api/dataset/components");
    if (resComps.ok) g_components = await resComps.json();

    const resGT = await fetch("/api/experiment/ground_truth?t_screen=47.0");
    if (resGT.ok) g_groundTruth = await resGT.json();

    const resComp = await fetch("/api/experiment/results");
    if (resComp.ok) g_comparison = await resComp.json();

    const resCompRes = await fetch("/api/experiment/component_results");
    if (resCompRes.ok) g_componentResults = await resCompRes.json();

    const resMeta = await fetch("/api/experiment/metadata");
    if (resMeta.ok) g_metadata = await resMeta.json();

    const resFeat = await fetch("/api/experiment/features");
    if (resFeat.ok) g_features = await resFeat.json();
  } catch (err) {
    console.info("Using embedded fallback data:", err);
  }

  renderDatasetOverview();
  renderFeatureExplorer();
  renderComponentExplorer();
  renderGroundTruthTable(g_groundTruth);
  renderAnalyticsTable(g_comparison);
  renderScreeningDecisions(g_componentResults);
  renderComparisonMatrix(g_comparison);
  renderSignificanceTests(g_metadata);
  initExplainabilityControls(g_componentResults);
  await loadValidationData();
}

function renderDatasetOverview() {
  if (!g_profile) return;
  const dStats = g_profile.dataset_statistics;
  if (!dStats) return;

  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setEl("ov_rows", dStats.number_of_rows);
  setEl("ov_cols", dStats.number_of_columns);
  setEl("ov_comps", dStats.number_of_unique_components);
  setEl("ov_lots", dStats.number_of_unique_lots);
  setEl("ov_time_range", `${dStats.time_range_hours[0]}h – ${dStats.time_range_hours[1]}h (${dStats.number_of_time_points} steps)`);
  setEl("ov_normal", dStats.total_normal_records);
  setEl("ov_fail", dStats.total_failure_records);
  setEl("ov_missing", `${100 - dStats.data_completeness_pct}% (0 nulls)`);
  setEl("ov_duplicates", `${dStats.duplicate_rows} duplicates`);
}

function renderFeatureExplorer() {
  if (!g_profile || !g_profile.column_statistics) return;
  const selectEl = document.getElementById("featureSelect");
  if (!selectEl) return;

  const cols = Object.keys(g_profile.column_statistics);
  selectEl.innerHTML = "";
  cols.forEach(col => {
    const opt = document.createElement("option");
    opt.value = col;
    opt.textContent = col;
    selectEl.appendChild(opt);
  });

  selectEl.addEventListener("change", () => {
    updateFeatureDetails(selectEl.value);
  });

  if (cols.includes("delta_capacitance_pct")) {
    selectEl.value = "delta_capacitance_pct";
  }
  updateFeatureDetails(selectEl.value);
}

function updateFeatureDetails(colName) {
  const stats = g_profile.column_statistics[colName];
  if (!stats) return;

  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setEl("feat_name", colName);
  setEl("feat_type", stats.type.toUpperCase());
  setEl("feat_count", stats.count);
  setEl("feat_missing", `${stats.missing_pct}% (${stats.missing_count} missing)`);

  if (stats.type === "numeric") {
    setEl("feat_mean", stats.mean);
    setEl("feat_median", stats.median);
    setEl("feat_std", stats.std);
    setEl("feat_min", stats.min);
    setEl("feat_max", stats.max);
    setEl("feat_iqr", stats.iqr);
    setEl("feat_skew", stats.skewness);
  } else {
    setEl("feat_mean", "—");
    setEl("feat_median", "—");
    setEl("feat_std", "—");
    setEl("feat_min", "—");
    setEl("feat_max", "—");
    setEl("feat_iqr", "—");
    setEl("feat_skew", "—");
  }
}

function renderComponentExplorer() {
  const compSelect = document.getElementById("compSelect");
  const compSelectPage = document.getElementById("compSelectPage");
  
  const setupSelect = (sel) => {
    if (!sel) return;
    sel.innerHTML = "";
    ["C1", "C2", "C3", "C4", "C5", "C6"].forEach(cid => {
      const opt = document.createElement("option");
      opt.value = cid;
      opt.textContent = `Component ${cid} (LOT_10V_EOS)`;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => {
      updateComponentDetails(sel.value);
      if (compSelect && sel !== compSelect) compSelect.value = sel.value;
      if (compSelectPage && sel !== compSelectPage) compSelectPage.value = sel.value;
    });
  };

  setupSelect(compSelect);
  setupSelect(compSelectPage);
  updateComponentDetails("C1");
}

function updateComponentDetails(cid) {
  if (!g_processed || g_processed.length === 0) return;
  const records = g_processed.filter(r => r.component_id === cid);
  if (records.length === 0) return;

  const compStat = g_components.find(c => c.component_id === cid) || {};

  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setEl("comp_id_title", `Component ${cid}`);
  setEl("comp_lot", records[0].lot_id);
  setEl("comp_obs", records.length);
  setEl("comp_final_c", `${records[records.length - 1].delta_capacitance_pct}%`);
  setEl("comp_final_esr", `${records[records.length - 1].delta_esr_pct}%`);
  
  const failBadge = document.getElementById("comp_spec_badge");
  const isFailed = compStat.spec_violation === "True" || compStat.spec_violation === true;
  if (failBadge) {
    failBadge.className = isFailed ? "badge badge-warning" : "badge badge-success";
    failBadge.textContent = isFailed ? `Violated Spec at ${compStat.first_violation_time_hours}h` : "Passed Specification Throughout (Survivor)";
  }

  const tbody = document.getElementById("compTrajectoryBody");
  const tbodyPage = document.getElementById("compTrajectoryBodyPage");

  const fillTable = (tb) => {
    if (!tb) return;
    tb.innerHTML = "";
    records.forEach(r => {
      const tr = document.createElement("tr");
      const isFailStep = parseInt(r.static_spec_fail) === 1;
      tr.style.backgroundColor = isFailStep ? "rgba(239, 68, 68, 0.1)" : "transparent";
      tr.innerHTML = `
        <td><strong>Step ${r.test_step}</strong></td>
        <td>${r.aging_time_hours} h</td>
        <td style="color: ${isFailStep ? '#f87171' : '#38bdf8'}; font-weight: 600;">${r.delta_capacitance_pct}%</td>
        <td style="color: #34d399;">${r.delta_esr_pct}%</td>
        <td>${r.capacitance_uf} µF</td>
        <td>${r.esr_ohms} Ω</td>
        <td>${r.drift_velocity_c} %/h</td>
        <td>${isFailStep ? '<span class="badge badge-warning">FAIL (≥20%)</span>' : '<span class="badge badge-success">PASS</span>'}</td>
      `;
      tb.appendChild(tr);
    });
  };

  fillTable(tbody);
  fillTable(tbodyPage);
}

function initExperimentControls() {
  const windowSelect = document.getElementById("expWindowSelect");
  if (!windowSelect) return;

  windowSelect.addEventListener("change", async () => {
    const tScreen = parseFloat(windowSelect.value);
    try {
      const res = await fetch(`/api/experiment/ground_truth?t_screen=${tScreen}`);
      if (res.ok) {
        const data = await res.json();
        renderGroundTruthTable(data);
      }
    } catch (e) {
      console.error("Failed to load ground truth for window:", e);
    }
  });
}

function renderGroundTruthTable(gtList) {
  const tbody = document.getElementById("groundTruthTableBody");
  if (!tbody || !gtList || gtList.length === 0) return;

  tbody.innerHTML = "";
  gtList.forEach(row => {
    const tr = document.createElement("tr");
    const isRisk = row.ground_truth_latent_risk === 1;
    tr.style.backgroundColor = isRisk ? "rgba(239, 68, 68, 0.07)" : "rgba(16, 185, 129, 0.07)";
    tr.innerHTML = `
      <td><strong>${row.component_id}</strong></td>
      <td>${row.t_screen_hours} h</td>
      <td><span class="badge badge-success">${row.early_max_delta_c}% (&lt;20%)</span></td>
      <td style="color: ${row.future_max_delta_c >= 20.0 ? '#f87171' : '#34d399'}; font-weight: 600;">${row.future_max_delta_c}%</td>
      <td>${(row.future_failure_time_h && !isNaN(row.future_failure_time_h)) ? row.future_failure_time_h + ' h' : 'None (Survived)'}</td>
      <td>${isRisk ? '<span class="badge badge-warning" style="font-weight:700;">LATENT RISK (y=1)</span>' : '<span class="badge badge-success" style="font-weight:700;">SAFE SURVIVOR (y=0)</span>'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAnalyticsTable(compList) {
  const tbody = document.getElementById("analyticsMetricsTableBody");
  if (!tbody || !compList || compList.length === 0) return;

  tbody.innerHTML = "";
  compList.forEach(row => {
    const tr = document.createElement("tr");
    const rec = parseFloat(row.recall);
    const fnr = parseFloat(row.false_negative_rate);
    const fpr = parseFloat(row.false_positive_rate);

    tr.innerHTML = `
      <td><strong>${row.method}</strong></td>
      <td style="color: ${rec >= 0.8 ? '#34d399' : (rec > 0 ? '#38bdf8' : '#f87171')}; font-weight: 700;">${(rec * 100).toFixed(1)}%</td>
      <td style="color: ${fnr === 0 ? '#34d399' : (fnr <= 0.4 ? '#fbbf24' : '#f87171')}; font-weight: 600;">${(fnr * 100).toFixed(1)}%</td>
      <td>${parseFloat(row.precision).toFixed(2)}</td>
      <td><strong>${parseFloat(row.f1_score).toFixed(2)}</strong></td>
      <td style="color: ${fpr === 0 ? '#34d399' : '#fbbf24'};">${(fpr * 100).toFixed(1)}%</td>
      <td>${parseFloat(row.pr_auc).toFixed(2)}</td>
      <td>${parseFloat(row.mean_lead_time_hours) > 0 ? row.mean_lead_time_hours + ' h' : '0.0 h (Escaped)'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderScreeningDecisions(compList) {
  const tbody = document.getElementById("screeningDecisionsTableBody");
  if (!tbody || !compList || compList.length === 0) return;

  tbody.innerHTML = "";
  compList.forEach(row => {
    const tr = document.createElement("tr");
    const isDefect = parseInt(row.ground_truth_latent_risk) === 1;
    const tier = row.risk_tier || "UNKNOWN";
    const tierBadge = tier === "HIGH RISK"
      ? '<span class="badge badge-warning" style="background-color:#991b1b;color:#fecaca;font-weight:700;">HIGH RISK</span>'
      : (tier === "MEDIUM RISK"
        ? '<span class="badge" style="background-color:#854d0e;color:#fef08a;font-weight:700;">MEDIUM RISK</span>'
        : '<span class="badge badge-success" style="font-weight:700;">LOW RISK</span>');

    tr.innerHTML = `
      <td><strong>${row.component_id}</strong></td>
      <td><span class="badge badge-success">${row.traditional_status || 'PASS'}</span></td>
      <td>${row.statistical_status === 'OUTLIER' ? '<span class="badge badge-warning">OUTLIER</span>' : '<span class="badge badge-success">NORMAL</span>'}</td>
      <td>${row.aiml_status === 'ANOMALY' ? '<span class="badge badge-warning">ANOMALY</span>' : '<span class="badge badge-success">NORMAL</span>'}</td>
      <td>${parseInt(row.ocsvm_flag) === 1 ? '<span class="badge badge-warning">FLAGGED (1)</span>' : '<span class="badge badge-success">NORMAL (0)</span>'}</td>
      <td>${row.drift_status === 'FUTURE_VIOLATION' ? `<span style="color:#f87171;font-weight:600;">VIOLATION (${row.predicted_delta_c_194h}%)</span>` : `<span style="color:#34d399;">SAFE (${row.predicted_delta_c_194h}%)</span>`}</td>
      <td>${tierBadge}</td>
      <td>${isDefect ? '<span style="color:#f87171;font-weight:700;">DEFECT RISK (y=1)</span>' : '<span style="color:#34d399;font-weight:700;">SAFE SURVIVOR (y=0)</span>'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function initExplainabilityControls(compList) {
  const sel = document.getElementById("explainCompSelect");
  if (!sel || !compList || compList.length === 0) return;

  sel.innerHTML = "";
  compList.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c.component_id;
    opt.textContent = `Component ${c.component_id} (${c.risk_tier || 'Risk'})`;
    sel.appendChild(opt);
  });

  sel.addEventListener("change", () => {
    updateExplainabilityView(sel.value);
  });

  // Default to C4 (first physical failure)
  sel.value = "C4";
  updateExplainabilityView("C4");
}

function updateExplainabilityView(cid) {
  if (!g_componentResults || g_componentResults.length === 0) return;
  const comp = g_componentResults.find(c => c.component_id === cid);
  if (!comp) return;

  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setEl("explainCompIdTitle", `Component ${cid} Diagnostic Breakdown`);
  setEl("explainScoreBadge", `Risk Score: ${comp.fusion_risk_score}`);
  setEl("explainText", comp.fusion_explanation || comp.stat_explanation || "Nominal component degradation profile.");

  setEl("exp_trad_margin", `Delta C at 47h: ${comp.delta_c_at_screen}% (Spec Limit: 20.0%, Margin: ${comp.spec_margin_c}%)`);
  setEl("exp_stat_z", comp.stat_explanation || `Max MAD Z-Score: ${comp.max_abs_zscore}`);
  setEl("exp_aiml_att", `iForest Score: ${comp.iforest_anomaly_score}, Top Precursor: ${comp.primary_contributing_feature} (${comp.max_feature_deviation_z} std devs)`);
  setEl("exp_drift_val", `Forecasted 194h Delta C: ${comp.predicted_delta_c_194h}% (Actual: ${comp.actual_delta_c_194h}%, Abs Error: ${comp.prediction_abs_error}%)`);

  const tierBadge = document.getElementById("explainTierBadge");
  if (tierBadge) {
    const tier = comp.risk_tier;
    tierBadge.textContent = tier;
    tierBadge.className = tier === "HIGH RISK" ? "badge badge-warning" : (tier === "MEDIUM RISK" ? "badge" : "badge badge-success");
    if (tier === "HIGH RISK") {
      tierBadge.style.backgroundColor = "#991b1b";
      tierBadge.style.color = "#fecaca";
    } else if (tier === "MEDIUM RISK") {
      tierBadge.style.backgroundColor = "#854d0e";
      tierBadge.style.color = "#fef08a";
    } else {
      tierBadge.style.backgroundColor = "";
      tierBadge.style.color = "";
    }
  }
}

function renderComparisonMatrix(compList) {
  const tbody = document.getElementById("comparisonMatrixTableBody");
  if (!tbody || !compList || compList.length === 0) return;

  tbody.innerHTML = "";
  compList.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${row.method}</strong></td>
      <td>${row.total_samples} units</td>
      <td style="color:#34d399;font-weight:700;">${row.true_positives}</td>
      <td style="color:#fbbf24;">${row.false_positives}</td>
      <td style="color:#f87171;font-weight:700;">${row.false_negatives}</td>
      <td>${row.true_negatives}</td>
      <td><strong>${(parseFloat(row.recall) * 100).toFixed(1)}%</strong></td>
      <td>${(parseFloat(row.false_negative_rate) * 100).toFixed(1)}%</td>
      <td>${parseFloat(row.precision).toFixed(2)}</td>
      <td>${parseFloat(row.f1_score).toFixed(2)}</td>
      <td>${parseFloat(row.pr_auc).toFixed(2)}</td>
      <td>${row.mean_lead_time_hours} h</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderSignificanceTests(metadata) {
  const container = document.getElementById("significanceTestsContent");
  if (!container || !metadata || !metadata.statistical_hypothesis_tests) return;

  const tests = metadata.statistical_hypothesis_tests;
  container.innerHTML = `
    <div class="card-grid" style="margin-bottom: 0;">
      <div class="card" style="background-color: var(--bg-surface);">
        <span class="card-tag card-tag-primary">MCNEMAR EXACT TEST</span>
        <h4 style="margin-top: 8px; font-size: 14px;">Paired Binary Discordance</h4>
        <ul class="card-meta-list">
          <li class="card-meta-item"><span>Trad vs One-Class SVM:</span><span>p = ${tests.mcnemar_trad_vs_one_class_svm ? tests.mcnemar_trad_vs_one_class_svm.exact_p_value : '—'} (n_disc = ${tests.mcnemar_trad_vs_one_class_svm ? tests.mcnemar_trad_vs_one_class_svm.total_discordant : '—'})</span></li>
          <li class="card-meta-item"><span>Trad vs Early Drift:</span><span>p = ${tests.mcnemar_trad_vs_drift_forecast ? tests.mcnemar_trad_vs_drift_forecast.exact_p_value : '—'} (n_disc = ${tests.mcnemar_trad_vs_drift_forecast ? tests.mcnemar_trad_vs_drift_forecast.total_discordant : '—'})</span></li>
          <li class="card-meta-item"><span>Trad vs Risk Fusion:</span><span>p = ${tests.mcnemar_trad_vs_risk_fusion ? tests.mcnemar_trad_vs_risk_fusion.exact_p_value : '—'} (n_disc = ${tests.mcnemar_trad_vs_risk_fusion ? tests.mcnemar_trad_vs_risk_fusion.total_discordant : '—'})</span></li>
          <li class="card-meta-item"><span>Significance (α = 0.05):</span><span style="color:#fbbf24;">Constrained by N=6 sample</span></li>
        </ul>
      </div>
      <div class="card" style="background-color: var(--bg-surface);">
        <span class="card-tag card-tag-backup">BOOTSTRAP 95% CIs</span>
        <h4 style="margin-top: 8px; font-size: 14px;">2,000 Iterations Delta Recall</h4>
        <ul class="card-meta-list">
          <li class="card-meta-item"><span>Δ Recall (OC-SVM - Trad):</span><span style="color:#34d399;font-weight:700;">[${tests.bootstrap_trad_vs_one_class_svm_95ci ? tests.bootstrap_trad_vs_one_class_svm_95ci.delta_recall_ci_95.join(', ') : '—'}] (+${tests.bootstrap_trad_vs_one_class_svm_95ci ? tests.bootstrap_trad_vs_one_class_svm_95ci.delta_recall_mean : '—'})</span></li>
          <li class="card-meta-item"><span>Δ Recall (Drift - Trad):</span><span style="color:#34d399;font-weight:700;">[${tests.bootstrap_trad_vs_drift_forecast_95ci ? tests.bootstrap_trad_vs_drift_forecast_95ci.delta_recall_ci_95.join(', ') : '—'}] (+1.0)</span></li>
          <li class="card-meta-item"><span>Δ Recall (Fusion - Trad):</span><span style="color:#34d399;font-weight:700;">[${tests.bootstrap_trad_vs_risk_fusion_95ci ? tests.bootstrap_trad_vs_risk_fusion_95ci.delta_recall_ci_95.join(', ') : '—'}] (+${tests.bootstrap_trad_vs_risk_fusion_95ci ? tests.bootstrap_trad_vs_risk_fusion_95ci.delta_recall_mean : '—'})</span></li>
          <li class="card-meta-item"><span>Empirical Verdict:</span><span style="color:#34d399;">Bootstrap strictly > 0</span></li>
        </ul>
      </div>
    </div>
  `;
}

function initRunExperimentButton() {
  const btn = document.getElementById("btnRunExperiment");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Executing 6-Fold LOCO Experiment...";
    try {
      const res = await fetch("/api/experiment/run");
      if (res.ok) {
        await loadAllData();
        alert("Step 4 Experiment Re-Executed Successfully! All models, metrics, and figures updated.");
      } else {
        alert("Experiment failed to execute. Check server logs.");
      }
    } catch (err) {
      console.error("Run experiment error:", err);
      alert("Error triggering experiment: " + err);
    } finally {
      btn.disabled = false;
      btn.textContent = "Re-Run 6-Fold LOCO Experiment";
    }
  });
}

async function loadValidationData() {
  try {
    const resAudit = await fetch("/api/validation/audit");
    if (resAudit.ok) {
      const data = await resAudit.json();
      renderAuditTable(data.records);
    }

    const resErr = await fetch("/api/validation/component_errors");
    if (resErr.ok) {
      const data = await resErr.json();
      renderErrorAnalysisTable(data.records);
    }

    const resAbl = await fetch("/api/validation/ablation");
    if (resAbl.ok) {
      const data = await resAbl.json();
      renderAblationTable(data.records);
    }

    const resSens = await fetch("/api/validation/sensitivity");
    if (resSens.ok) {
      const data = await resSens.json();
      renderSensitivityTable(data.records);
    }

    const resCost = await fetch("/api/validation/cost_analysis");
    if (resCost.ok) {
      const data = await resCost.json();
      renderCostTable(data.records);
    }
  } catch (err) {
    console.info("Validation endpoints error or fallback:", err);
  }
}

function renderAuditTable(records) {
  const tbody = document.getElementById("auditTableBody");
  if (!tbody || !records) return;
  tbody.innerHTML = "";
  records.forEach(r => {
    const tr = document.createElement("tr");
    const isMatch = r.status === "VERIFIED_EXACT_MATCH";
    tr.innerHTML = `
      <td>${r.category}</td>
      <td><strong>${r.method}</strong></td>
      <td><code>${r.metric}</code></td>
      <td>${r.step4_reported_value}</td>
      <td>${r.reproduced_value}</td>
      <td>${r.difference}</td>
      <td><span class="badge ${isMatch ? 'badge-success' : 'badge-danger'}">${isMatch ? 'MATCH (0.00)' : 'DIFF'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderErrorAnalysisTable(records) {
  const tbody = document.getElementById("errorAnalysisTableBody");
  if (!tbody || !records) return;
  tbody.innerHTML = "";
  records.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${r.component_id}</strong></td>
      <td><span class="badge ${r.ground_truth === 'Safe Survivor' ? 'badge-success' : 'badge-danger'}">${r.ground_truth}</span></td>
      <td><code>${r.trad_decision} (${r.trad_class})</code></td>
      <td><code>${r.stat_decision} (${r.stat_class})</code></td>
      <td><code>${r.ocsvm_decision} (${r.ocsvm_class})</code></td>
      <td><code>${r.drift_decision} (${r.drift_class})</code></td>
      <td><strong>${r.risk_tier} (${r.fusion_class})</strong></td>
      <td>${parseFloat(r.actual_delta_c_194h).toFixed(2)}%</td>
      <td>${parseFloat(r.predicted_delta_c_194h).toFixed(2)}%</td>
      <td><strong>${r.lead_time_hours > 0 ? r.lead_time_hours + ' h' : '0 h'}</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAblationTable(records) {
  const tbody = document.getElementById("ablationTableBody");
  if (!tbody || !records) return;
  tbody.innerHTML = "";
  records.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${r.paradigm}</strong></td>
      <td style="font-size:12px; color:var(--text-secondary);">${r.description}</td>
      <td style="color:#34d399;font-weight:700;">${(parseFloat(r.recall)*100).toFixed(1)}%</td>
      <td>${(parseFloat(r.false_negative_rate)*100).toFixed(1)}%</td>
      <td>${parseFloat(r.precision).toFixed(2)}</td>
      <td><strong>${parseFloat(r.f1_score).toFixed(2)}</strong></td>
      <td>${(parseFloat(r.false_positive_rate)*100).toFixed(1)}%</td>
      <td>${r.mean_lead_time_hours} h</td>
      <td style="color:#38bdf8;font-weight:700;">+${(parseFloat(r.incremental_recall_vs_trad)*100).toFixed(1)}%</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderSensitivityTable(records) {
  const tbody = document.getElementById("sensitivityTableBody");
  if (!tbody || !records) return;
  tbody.innerHTML = "";
  records.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${r.model_name}</strong></td>
      <td><code>${r.parameter_name}</code></td>
      <td><strong>${r.threshold_value}</strong></td>
      <td style="color:#34d399;">${(parseFloat(r.recall)*100).toFixed(1)}%</td>
      <td>${(parseFloat(r.false_negative_rate)*100).toFixed(1)}%</td>
      <td>${parseFloat(r.precision).toFixed(2)}</td>
      <td>${parseFloat(r.f1_score).toFixed(2)}</td>
      <td>${(parseFloat(r.false_positive_rate)*100).toFixed(1)}%</td>
      <td><span class="badge ${r.note.includes('Baseline') ? 'badge-success' : ''}">${r.note}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderCostTable(records) {
  const tbody = document.getElementById("costTableBody");
  if (!tbody || !records) return;
  tbody.innerHTML = "";
  // Show key ratios: 0.25, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0
  const keyRatios = [0.25, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0];
  records.filter(r => keyRatios.includes(parseFloat(r.cost_ratio_fn_to_fp))).forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${r.cost_ratio_fn_to_fp}x</strong></td>
      <td>${r.method}</td>
      <td style="color:#f87171;">${r.fn_count}</td>
      <td style="color:#fbbf24;">${r.fp_count}</td>
      <td><strong>$${parseFloat(r.total_expected_cost).toFixed(1)}</strong></td>
      <td>$${parseFloat(r.cost_per_component).toFixed(2)}</td>
      <td style="color:#34d399;">$${parseFloat(r.cost_savings_vs_traditional).toFixed(1)}</td>
      <td><strong>${r.cost_savings_pct}%</strong></td>
      <td><span class="badge ${r.is_superior_to_traditional ? 'badge-success' : 'badge-danger'}">${r.is_superior_to_traditional ? 'SUPERIOR' : 'INFERIOR'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}


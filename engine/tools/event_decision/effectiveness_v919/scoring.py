"""Masked features and small certified linear scorers; never consume review overlays."""
from __future__ import annotations

from collections import Counter
import numpy as np

from ..contracts import semantic_sha256
from ..models import sigmoid
from ..numerics import CertifiedRidgeLogisticScorer
from .metrics import choose_threshold, metrics, paired_bootstrap

BASES = ("independent_direct_nodes", "shared_unary_rowmax", "unary_ot", "conditional_rowmax", "conditional_ot_no_coherence", "conditional_ot_full")
LOCAL = ("direct_quality", "benign_fraction", "harmful_fraction", "unknown_fraction", "premise_unresolved_fraction")


def features(result):
    vals = {k: None for k in (*BASES, *LOCAL)}
    for k in BASES:
        v = result.get("baseline", {}).get("competitions", {}).get(k, {}).get("margin")
        if isinstance(v, (float, int)) and not isinstance(v, bool) and np.isfinite(v):
            vals[k] = float(v)
    c1, c2 = result.get("C1"), result.get("C2")
    if c1 is not None and c1["scan_complete"] and not c1["overflow"] and c1["observation_sufficient"]:
        vals["direct_quality"] = max((e["direct_mechanism_probability"] * e["direct_evidence_quality"] for e in c1["events"]), default=0.)
    if c2 is not None:
        events = c2["event_assessments"]
        if events:
            for v in ("benign", "harmful", "unknown"):
                vals[v + "_fraction"] = sum(e["discrimination"] == v for e in events) / len(events)
            vals["premise_unresolved_fraction"] = sum(e["premise_support"] != "supported" for e in events) / len(events)
        # No event is a distinct condition, not benign=1 or unknown=0.
    return {"window_uid": result["window_uid"], "values": vals,
            "local_valid": bool(vals["direct_quality"] is not None and (c2 is not None or result.get("C2_skipped") == "complete_empty_C1")),
            "source_result_sha256": semantic_sha256(result)}


def transform(rows, names, stats=None):
    x = np.array([[np.nan if r["values"][k] is None else r["values"][k] for k in names] for r in rows], dtype=float)
    observed = np.isfinite(x)
    if stats is None:
        mean = np.array([float(x[observed[:, i], i].mean()) if observed[:, i].any() else 0. for i in range(x.shape[1])])
        scale = np.array([float(x[observed[:, i], i].std()) if observed[:, i].any() else 1. for i in range(x.shape[1])])
        scale[scale < 1e-8] = 1.
        stats = {"mean": mean.tolist(), "scale": scale.tolist(), "fit_observed": observed.sum(axis=0).tolist()}
    standardized = (np.where(observed, x, np.array(stats["mean"])) - np.array(stats["mean"])) / np.array(stats["scale"])
    return np.c_[standardized, observed.astype(float)], stats


def fit_linear(rows, labels, names, groups, regularization, monotone=False):
    x, stats = transform(rows, names)
    counts = Counter(groups)
    weights = [1 / counts[g] for g in groups]
    model = CertifiedRidgeLogisticScorer(regularization=regularization, max_iter=30000, tolerance=1e-7,
                                      lower_bounds={0: 0.} if monotone else {})
    model.fit(x, labels, weights, fit_context={"uids": [r["window_uid"] for r in rows], "features": names})
    if not model.certificate_["certified"]:
        raise ValueError("Scorer not numerically certified")
    return {"names": names, "standardization": stats, "intercept": model.intercept_, "coef": model.coef_.tolist(),
            "numeric_certificate": model.certificate_, "lambda": regularization}


def linear_predict(rows, model):
    x, _ = transform(rows, model["names"], model["standardization"])
    return sigmoid(model["intercept"] + x @ np.array(model["coef"])).tolist()


def fit_models(inputs, labels, table, regularization=.1):
    lookup = {r["window_uid"]: r for r in table}
    lm = {r["window_uid"]: r for r in labels}
    roles = {role: [r for r in inputs if r["role"] == role and lm[r["window_uid"]]["loss_mask"]] for role in ("fit", "calibration", "validation")}
    for role, rs in roles.items():
        if {lm[r["window_uid"]]["target"] for r in rs} != {0, 1}:
            raise ValueError("Both supervised classes required in " + role)
    sources = {role: {r["source_group"] for r in rs} for role, rs in roles.items()}
    if sources["fit"] & (sources["calibration"] | sources["validation"]) or sources["calibration"] & sources["validation"]:
        raise ValueError("Source leakage between fit/calibration/validation")
    # Lambda is pre-registered, not tuned on calibration or validation outcomes.
    models = {"regularization": regularization, "fallback": "raw_same_base_margin_then_M0_else_missing",
              "feature_schema": "v919_six_values_plus_six_availability_masks", "models": {}, "thresholds": {}}
    for method, base, local in (("C_calibrated", "conditional_ot_full", False), ("D_local_OT", "conditional_ot_full", True), ("D_local_nodes", "independent_direct_nodes", True)):
        names = [base, *LOCAL] if local else [base]
        usable = [r for r in roles["fit"] if lookup[r["window_uid"]]["values"][base] is not None and (not local or lookup[r["window_uid"]]["local_valid"])]
        if {lm[r["window_uid"]]["target"] for r in usable} != {0, 1}:
            raise ValueError("Insufficient valid local features for " + method + "; do not masquerade a baseline as D")
        models["models"][method] = fit_linear([lookup[r["window_uid"]] for r in usable], [lm[r["window_uid"]]["target"] for r in usable], names,
                                              [r["source_group"] for r in usable], regularization, monotone=not local)
        models["models"][method]["fit_windows"] = len(usable)
    # Baselines mapped monotonically to [0,1]; D fallback uses calibrated C below.
    cal_rows = [lookup[r["window_uid"]] for r in roles["calibration"]]
    scores, fallback = predict(cal_rows, models)
    cy = [lm[r["window_uid"]]["target"] for r in roles["calibration"]]
    for method, s in scores.items():
        if any(v is None for v in s):
            raise ValueError("Calibration has unscorable windows: " + method)
        models["thresholds"][method] = choose_threshold(cy, s)
    models["fit_source_groups"] = sorted(sources["fit"])
    models["calibration_source_groups"] = sorted(sources["calibration"])
    models["label_scope"] = "weak_anchor_and_filename_A_supervision_not_frame_gold"
    return models


def predict(rows, models):
    scores, fallback = {}, {}
    for k in BASES:
        raw, fb = [], []
        for r in rows:
            v = r["values"][k]
            kind = "none"
            if v is None:
                v, kind = r["values"]["independent_direct_nodes"], "M0"
            raw.append(float(sigmoid(v)) if v is not None else None)
            fb.append(kind if v is not None else "unscorable")
        scores[k], fallback[k] = raw, fb
    for method, m in models["models"].items():
        raw = linear_predict(rows, m)
        base = m["names"][0]
        fb = []
        for i, r in enumerate(rows):
            usable = r["values"][base] is not None and (method == "C_calibrated" or r["local_valid"])
            if not usable:
                if method == "D_local_OT":
                    raw[i] = scores["C_calibrated"][i]
                    fb.append("C_calibrated_then_M0")
                else:
                    raw[i] = scores[base][i]
                    fb.append("raw_base_then_M0")
            else:
                fb.append("none")
        scores[method], fallback[method] = raw, fb
    return scores, fallback


def evaluate_pilot(inputs, labels, table, models, bootstrap=500):
    lm, fm = {r["window_uid"]: r for r in labels}, {r["window_uid"]: r for r in table}
    rs = [r for r in inputs if r["role"] == "validation" and lm[r["window_uid"]]["loss_mask"]]
    fs = [fm[r["window_uid"]] for r in rs]
    s, fb = predict(fs, models)
    y = [lm[r["window_uid"]]["target"] for r in rs]
    report = {"validation_windows": len(rs), "unlabelled_context_excluded_from_metrics_not_acquisition": sum(not lm[r["window_uid"]]["loss_mask"] for r in inputs),
              "scope": "source-disjoint development-exposed weak-window validation", "frame_AP": None,
              "methods": {}, "coverage": {}, "fallback": {}, "comparisons": {}}
    for k, values in s.items():
        report["coverage"][k] = sum(x is not None for x in values) / max(1, len(values))
        report["fallback"][k] = dict(Counter(fb[k]))
        report["methods"][k] = metrics(y, values, models["thresholds"][k]) if None not in values else None
    for a, b in (("conditional_ot_full", "D_local_OT"), ("C_calibrated", "D_local_OT"), ("D_local_nodes", "D_local_OT"), ("shared_unary_rowmax", "unary_ot")):
        if None not in s[a] and None not in s[b]:
            report["comparisons"][b + "_minus_" + a] = paired_bootstrap(y, s[a], s[b], [r["source_group"] for r in rs], repeats=bootstrap)
    report["per_class"] = {}
    for code in ("A", "B1", "B2", "B4", "B5", "B6", "G"):
        ix = [i for i, r in enumerate(rs) if (code in lm[r["window_uid"]]["codes"] or code == "A" and lm[r["window_uid"]]["pure_normal"])]
        report["per_class"][code] = {k: metrics([y[i] for i in ix], [v[i] for i in ix], models["thresholds"][k]) for k, v in s.items() if ix and all(v[i] is not None for i in ix)}
    c, d = report["methods"].get("C_calibrated"), report["methods"].get("D_local_OT")
    ci = report["comparisons"].get("D_local_OT_minus_C_calibrated", {}).get("AP_delta_B_minus_A_CI95")
    normal = report["per_class"]["A"]
    report["ready_for_dense_method_test"] = bool(c and d and ci and ci[0] > 0 and d["balanced_accuracy"] >= c["balanced_accuracy"] and
        normal and normal["D_local_OT"]["FP"] <= normal["C_calibrated"]["FP"] and sum(f["local_valid"] for f in fs) / len(fs) >= .95)
    report["gate_rule"] = "paired_AP_CI_lower>0_vs_C_calibrated; BA_not_lower; A_FP_not_higher; local_valid>=.95; six_class_scope checked separately"
    return report

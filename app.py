"""Streamlit app: live monitor, results and how-it-works in one web page.

    streamlit run app.py

Deploy for free on Streamlit Community Cloud: push this folder (including models/ and outputs/) to GitHub.
"""
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import config
import fuzzy
from build_dashboard import build_html

st.set_page_config(page_title="Predictive Maintenance Monitor", layout="wide")


def load_results(domain):
    p = config.MODELS_DIR / domain / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


st.title("Predictive Maintenance Monitor")
st.caption("Hybrid soft computing: Genetic Algorithm + Neural Network + Fuzzy Logic")

tab_live, tab_results, tab_how = st.tabs(["Live monitor", "Results", "How it works"])

with tab_live:
    html, found = build_html()
    if not found:
        st.info("No trained models found yet. The monitor below shows a simulated preview. Run train.py to use real results.")
    if hasattr(st, "iframe"):
        st.iframe(html, height=1900)
    else:
        components.html(html, height=1900, scrolling=True)

with tab_results:
    domains = [d for d in config.DOMAINS if load_results(d)]
    if not domains:
        st.warning("No results yet. Run train.py for a machine type first.")
    for d in domains:
        r = load_results(d)
        st.header(r["label"])
        if r["quick_run"] or "synthetic" in r["note"].lower():
            st.warning("These numbers come from a quick or test run. Re-run train.py without --quick on the real data before reporting them.")
        st.caption(r["note"])
        c1, c2, c3 = st.columns(3)
        best = r["models"][-1]
        c1.metric("Test RMSE (proposed model)", f"{best['rmse']:.2f}", f"± {best['rmse_std']:.2f}", delta_color="off")
        c2.metric("Sensors kept by GA", f"{r['selected_count']} of {r['features_total']}")
        c3.metric("Critical alert lead time", f"{r['alerts']['fuzzy']['mean_critical_lead']:.0f} {r['unit']}")
        st.caption(r["test_metric_note"])

        st.subheader("Model comparison (ablation)")
        st.dataframe(pd.DataFrame([{"Model": m["name"], "RMSE": round(m["rmse"], 2), "± std": round(m["rmse_std"], 2),
                                    "RMSE vs capped life": round(m["rmse_capped"], 2), "NASA score": round(m["nasa"], 0)}
                                   for m in r["models"]]), hide_index=True, width="stretch")

        st.subheader("Alert quality: fuzzy logic vs fixed thresholds")
        rows = []
        for name, a in (("Fuzzy (proposed)", r["alerts"]["fuzzy"]), ("Fixed thresholds", r["alerts"]["threshold_baseline"])):
            rows.append({"Method": name, f"Mean warning lead ({r['unit']})": round(a["mean_warning_lead"], 1),
                         f"Mean critical lead ({r['unit']})": round(a["mean_critical_lead"], 1),
                         "Critical issued in time (%)": round(a["critical_in_time_pct"], 0),
                         "False warnings in healthy zone (%)": round(a["false_warning_rate_pct"], 1)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.subheader("Selected inputs and tuned network")
        st.write(f"GA kept: {', '.join(r['selected'])}")
        hp = r["hyperparameters"]
        st.write(f"Tuned ANN: hidden layers {hp['hidden']}, L2 regularisation {hp['alpha']}, learning rate {hp['lr']}")

        out = config.OUTPUT_DIR / d
        figs = [("ga_vs_pso.png", "GA vs PSO convergence"), ("model_comparison.png", "Model comparison"),
                ("pred_vs_true_test.png", "Predicted vs true life left (test)"), ("alert_timeline_example.png", "Alert timeline of one held-out unit"),
                ("fuzzy_membership.png", "Fuzzy membership functions"), ("fuzzy_surface.png", "Fuzzy rule surface")]
        cols = st.columns(2)
        for i, (f, cap) in enumerate(figs):
            if (out / f).exists():
                cols[i % 2].image(str(out / f), caption=cap)

with tab_how:
    st.markdown("""
**Pipeline**

1. **Genetic Algorithm** picks the sensors that help predict remaining life and drops the rest.
2. **Neural network (MLP)** reads those sensors and predicts the life left. A second GA run tunes its layers, regularisation and learning rate.
3. **Fuzzy logic** combines predicted life left with the wear rate (how fast the sensors are drifting) and returns a risk score, shown as Normal, Warning or Critical.

**Fuzzy rules**
""")
    for text, _ in fuzzy.RULES:
        st.markdown(f"- {text}")
    st.markdown("""
**Reading the monitor**

- Each cycle is one flight (aircraft) or one 10-minute vibration reading (factory machine).
- The coloured traces are the three most informative sensors the GA kept.
- The dashed line is the true life left, the cyan line is the network's prediction.
- Pick a different engine or run from the drop-down, switch machine type at the top, and turn sound on to hear the unit wear out.
""")

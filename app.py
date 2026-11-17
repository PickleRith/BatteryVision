"""BatteryVision Streamlit app.

    streamlit run app.py

Every view separates PUBLISHED ANNOTATIONS (ground truth from the CoatingVision
authors) from MODEL PREDICTIONS (produced locally by models trained in this repo).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src/ importable without `pip install -e .` (e.g. on Streamlit Community Cloud).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from batteryvision import analytics, config
from batteryvision.dataset import CLASS_LABELS, CLASSES, load_image, load_manifest, load_mask
from batteryvision.inference import load_model, predict
from batteryvision.visualization import color_hex, error_overlay, mask_to_rgb, overlay_masks

st.set_page_config(page_title="BatteryVision", layout="wide")

COLOR_MAP = {CLASS_LABELS[c]: color_hex(c) for c in CLASSES}
SAMPLES_DIR = config.resolve("data/samples")
METRICS_DIR = config.artifacts_dir() / "metrics"
PLOT_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13))


# ---------------------------------------------------------------- data access
@st.cache_data
def get_manifest() -> pd.DataFrame | None:
    p = config.manifest_path()
    return load_manifest(p) if p.exists() else None


def image_root() -> tuple[Path | None, str]:
    """Full dataset if downloaded, else the small bundled sample set."""
    root = config.data_root()
    if (root / "segmentation").exists():
        return root, "full"
    if (SAMPLES_DIR / "samples.csv").exists():
        return SAMPLES_DIR, "samples"
    return None, "none"


@st.cache_data
def browsable_samples() -> pd.DataFrame | None:
    root, mode = image_root()
    if mode == "full":
        return get_manifest()
    if mode == "samples":
        return load_manifest(SAMPLES_DIR / "samples.csv")
    return None


@st.cache_resource
def get_model(path: str):
    return load_model(path)


def available_models() -> dict[str, Path]:
    return {p.stem: p for p in sorted(config.models_dir().glob("*.pt")) if not p.stem.startswith("smoke")}


def read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def legend_html(classes=CLASSES) -> str:
    chips = " ".join(
        f"<span style='display:inline-flex;align-items:center;margin-right:14px'>"
        f"<span style='width:12px;height:12px;border-radius:3px;background:{color_hex(c)};"
        f"display:inline-block;margin-right:6px'></span>{CLASS_LABELS[c]}</span>" for c in classes)
    return f"<div style='font-size:0.9rem'>{chips}</div>"


def sidebar_model() -> str | None:
    models = available_models()
    st.sidebar.markdown("### Model")
    if not models:
        st.sidebar.info("No trained model in `models/`. Inspection shows published annotations only.")
        return None
    names = list(models)
    default = names.index("unet_v2") if "unet_v2" in names else 0
    name = st.sidebar.selectbox("Prediction model", names, index=default, key="model_name")
    m = get_model(str(models[name]))
    st.sidebar.caption(f"**{m.kind}** · {m.meta['architecture']}")
    return name


# ---------------------------------------------------------------- pages
def page_inspect():
    st.title("Inspect an image")
    st.caption("Model-predicted visible defects on electrode-coating imagery. "
               "This is a research prototype, not a pass/fail or safety decision.")
    model_name = st.session_state.get("model_name")
    models = available_models()
    model = get_model(str(models[model_name])) if model_name in models else None
    root, mode = image_root()
    samples = browsable_samples()

    src_opts = ["Dataset example", "Upload an image"] if samples is not None else ["Upload an image"]
    src = st.radio("Image source", src_opts, horizontal=True)
    image, gt, sid, row = None, None, "upload", None

    if src == "Dataset example":
        c1, c2, c3 = st.columns([1, 1, 2])
        splits = [s for s in ("test", "val", "train") if s in set(samples["split"])]
        split = c1.selectbox("Split", splits, help="Test images were never used for training or threshold selection.")
        cls = c2.selectbox("Annotated defect", ["any"] + CLASSES,
                           format_func=lambda c: "Any" if c == "any" else CLASS_LABELS[c])
        d = samples[samples.split == split]
        if cls != "any":
            d = d[d[cls] == 1]
        if d.empty:
            st.warning("No images match this filter.")
            return
        sid = c3.selectbox("Sample", d["sample_id"].tolist())
        row = d[d.sample_id == sid].iloc[0]
        image = load_image(root / row.image_path)
        gt = load_mask(root / row.mask_path)
        if mode == "samples":
            st.caption("Showing the bundled sample set. Download the full dataset to browse all 2,227 images.")
    else:
        up = st.file_uploader("PNG or JPEG electrode-coating image", type=["png", "jpg", "jpeg"])
        if up is None:
            st.info("Upload an image to run the model. Images are processed locally and never sent to a third party.")
            return
        try:
            image = np.array(Image.open(up).convert("RGB"))
        except Exception as e:  # noqa: BLE001 - surface any decode error to the user
            st.error(f"Could not read this file as an image ({e}).")
            return
        if image.shape[:2] != (480, 640):
            st.warning(f"This image is {image.shape[1]}x{image.shape[0]}; the model was trained on 640x480 "
                       "CoatingVision patches. Results on other imagery are unvalidated.")
        sid = Path(up.name).stem

    res = predict(model, image, sid, gt) if model else None

    st.markdown(legend_html(), unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].markdown("**Original image**")
    cols[0].image(image, width="stretch")

    cols[1].markdown(f"**Model prediction** · `{model_name}`" if model else "**Model prediction**")
    if res is None:
        cols[1].info("No model loaded.")
    elif res["predicted_masks"] is not None:
        cols[1].image(overlay_masks(image, res["predicted_masks"]), width="stretch")
    else:
        cols[1].info("Localization unavailable; classification only.")

    cols[2].markdown("**Published ground-truth annotation**")
    if gt is not None:
        cols[2].image(overlay_masks(image, gt), width="stretch")
        cols[2].caption("From the CoatingVision authors. Not a model output.")
    else:
        cols[2].info("No annotation for uploaded images.")

    if res is None:
        return

    if res["review_required"]:
        st.warning("**Human review suggested** (demo heuristic: a class score is within "
                   f"±{model.review_band:.3f} of its decision threshold; band set on validation).\n\n- "
                   + "\n- ".join(res["review_reasons"]))
    else:
        st.info("Not flagged for review by the demo heuristic (no class score near its threshold). "
                "This is not a pass/fail decision.")

    thr = model.image_thresholds
    table = []
    for c in CLASSES:
        r = {
            "Defect": CLASS_LABELS[c],
            "Model probability": round(res["class_probabilities"][c], 3),
            "Threshold (from val)": thr[c],
            "Model predicts": "yes" if res["predicted_labels"][c] else "no",
        }
        if res["predicted_coverage_pct"] is not None:
            r["Predicted coverage (% of pixels)"] = round(res["predicted_coverage_pct"][c], 3)
        if row is not None:
            r["Annotated (ground truth)"] = "yes" if row[c] else "no"
            r["Annotated coverage (% of pixels)"] = round(float(row[f"{c}_coverage_pct"]), 3)
        table.append(r)
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    if model.kind == "segmentation":
        st.caption("Probability = highest pixel probability for that class channel. Coverage is a share of image "
                   "pixels, not a physical area (no calibrated scale is available).")
    else:
        st.caption("Image-level classifier: it reports presence only and cannot measure defect area or location.")

    if res["predicted_masks"] is not None and gt is not None:
        with st.expander("Per-class comparison: prediction vs annotation"):
            st.caption("Green = predicted and annotated · Red = annotated but missed · Blue = predicted, not annotated")
            cc = st.columns(4)
            for k, c in enumerate(CLASSES):
                cc[k].markdown(f"**{CLASS_LABELS[c]}**")
                cc[k].image(error_overlay(image, res["predicted_masks"][..., k], gt[..., k]), width="stretch")


def page_analytics():
    st.title("Dataset analytics")
    df = get_manifest()
    if df is None:
        st.error("Manifest not found. Run `python scripts/prepare_data.py` first.")
        return
    st.caption("Published CoatingVision annotations (all counts computed from the manifest). "
               "Labels are multi-label, so class counts overlap.")
    f1, f2 = st.columns(2)
    splits = f1.multiselect("Split", ["train", "val", "test"], default=["train", "val", "test"])
    cls_filter = f2.multiselect("Must include defect", CLASSES, format_func=CLASS_LABELS.get)
    d = df[df.split.isin(splits)]
    for c in cls_filter:
        d = d[d[c] == 1]
    if d.empty:
        st.warning("No images match these filters.")
        return

    k = st.columns(4)
    k[0].metric("Images", f"{len(d):,}")
    k[1].metric("Acquisition runs", d["run_id"].nunique())
    k[2].metric("Images with >1 defect type", f"{(d[CLASSES].sum(axis=1) > 1).sum():,}")
    k[3].metric("No annotated defect", f"{(d[CLASSES].sum(axis=1) == 0).sum():,}")

    a, b = st.columns(2)
    prev = analytics.prevalence(d)
    prev["Defect"] = prev["class"].map(CLASS_LABELS)
    fig = px.bar(prev, x="images", y="Defect", orientation="h", color="Defect", color_discrete_map=COLOR_MAP,
                 text=prev.apply(lambda r: f"{r.images:,} ({r.pct_of_images:.1f}%)", axis=1),
                 title="Images per defect type")
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0,
                      hovertemplate="%{y}: %{x:,} images<extra></extra>")
    fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="images", bargap=0.35,
                      yaxis=dict(categoryorder="array", categoryarray=[CLASS_LABELS[c] for c in CLASSES][::-1]),
                      **PLOT_LAYOUT)
    a.plotly_chart(fig, width="stretch")

    co = analytics.cooccurrence(d)
    labels = [CLASS_LABELS[c] for c in CLASSES]
    fig = go.Figure(go.Heatmap(z=co.values, x=labels, y=labels, colorscale=[[0, "#cde2fb"], [1, "#104281"]],
                               text=co.values, texttemplate="%{text}", xgap=2, ygap=2,
                               hovertemplate="%{y} & %{x}: %{z} images<extra></extra>", showscale=False))
    fig.update_layout(title="Co-occurrence (images with both labels)", yaxis=dict(autorange="reversed"), **PLOT_LAYOUT)
    b.plotly_chart(fig, width="stretch")

    cov = analytics.coverage_long(d)
    cov["Defect"] = cov["class"].map(CLASS_LABELS)
    cov = cov[cov.coverage_pct > 0]
    fig = px.box(cov, x="coverage_pct", y="Defect", color="Defect", color_discrete_map=COLOR_MAP, log_x=True,
                 points="outliers", title="Annotated mask coverage per image (log scale)",
                 category_orders={"Defect": labels})
    fig.update_layout(showlegend=False, xaxis_title="% of image pixels", yaxis_title=None, **PLOT_LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption("Coverage is the percentage of image pixels annotated with that class, computed only over images where "
               "the class is present. It is **not** a physical defect area: no calibrated scale is available.")

    c, e = st.columns(2)
    combos = analytics.label_combinations(d).head(10)
    combos["labels"] = combos["labels"].str.replace("_", " ")
    c.markdown("**Most common label combinations**")
    c.dataframe(combos, hide_index=True, width="stretch")
    runs = analytics.run_summary(d).rename(columns={**CLASS_LABELS, "run_id": "Run (from file name)"})
    e.markdown("**Per acquisition run** (as named in the source file names)")
    e.dataframe(runs, hide_index=True, width="stretch")

    with st.expander("Split summary and sample table"):
        st.dataframe(df.groupby("split")[CLASSES].sum().reindex(["train", "val", "test"]).assign(
            images=df.groupby("split").size()).rename(columns=CLASS_LABELS), width="stretch")
        show = ["sample_id", "split", "run_id", "frame", *CLASSES, *[f"{c}_coverage_pct" for c in CLASSES]]
        st.dataframe(d[show], hide_index=True, width="stretch", height=320)


def _img_level_table(m: dict) -> pd.DataFrame:
    rows = []
    for c in CLASSES:
        x = m["image_level"][c]
        rows.append({"Defect": CLASS_LABELS[c], "Support (test images)": x["support"], "Precision": fmt(x["precision"]),
                     "Recall": fmt(x["recall"]), "F1": fmt(x["f1"]), "Average precision": fmt(x["average_precision"]),
                     "FN (missed)": x["fn"], "FP (false alarms)": x["fp"]})
    return pd.DataFrame(rows)


def page_evaluation():
    st.title("Model evaluation")
    models = available_models()
    name = st.session_state.get("model_name")
    m = read_json(METRICS_DIR / f"{name}_test_metrics.json") if name else None
    log = read_json(METRICS_DIR / f"{name}_train_log.json") if name else None
    if m is None:
        st.info("No held-out results for the selected model. Run `python scripts/evaluate.py` after training.")
    else:
        meta = get_model(str(models[name])).meta if name in models else (log["meta"] if log else {})
        st.caption(f"Held-out **test split**: {m['n_images']} images from frame blocks never seen in training or "
                   "threshold selection. Thresholds were frozen on the validation split.")
        with st.expander("Model card / training settings", expanded=False):
            st.json({k: meta.get(k) for k in ("architecture", "input_size_wh", "loss", "optimizer", "epochs",
                                              "batch_size", "augmentation", "seed", "device", "selected_epoch",
                                              "selection_metric", "threshold_selection", "pixel_thresholds", "image_thresholds", "thresholds",
                                              "review_band", "manifest_sha256_16") if k in meta})

        st.subheader("Image-level detection (per class)")
        st.caption(f"Score used: {m['image_level_score']}. Macro F1 over classes with support: "
                   f"{fmt(m['image_level_macro_f1'])}.")
        st.dataframe(_img_level_table(m), hide_index=True, width="stretch")

        pr = read_json(METRICS_DIR / f"{name}_test_pr.json")
        if pr:
            fig = go.Figure()
            for c in CLASSES:
                if pr.get(c):
                    fig.add_trace(go.Scatter(x=pr[c]["recall"], y=pr[c]["precision"], mode="lines",
                                             name=f"{CLASS_LABELS[c]} (n={m['image_level'][c]['support']})",
                                             line=dict(color=color_hex(c), width=2),
                                             hovertemplate="recall %{x:.2f} · precision %{y:.2f}<extra></extra>"))
            fig.update_layout(title="Precision-recall curves, test split", xaxis_title="Recall",
                              yaxis_title="Precision", xaxis_range=[0, 1.02], yaxis_range=[0, 1.02],
                              legend=dict(orientation="h", y=-0.2), height=420, **PLOT_LAYOUT)
            st.plotly_chart(fig, width="stretch")

        if "pixel_level" in m:
            st.subheader("Pixel-level segmentation (640x480, against published masks)")
            rows = []
            for c in CLASSES:
                x = m["pixel_level"][c]
                rows.append({"Defect": CLASS_LABELS[c], "Images with class": x["images_gt_present"],
                             "Dice (all pixels)": fmt(x["dice"]), "IoU": fmt(x["iou"]),
                             "Pixel precision": fmt(x["pixel_precision"]), "Pixel recall": fmt(x["pixel_recall"]),
                             "Mean per-image Dice (class present)": fmt(x["mean_image_dice_gt_present"]),
                             "Predicted on images without class": fmt(x["empty_fp_rate"], 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(m["pixel_level_notes"])
            if "pinhole_objects" in m:
                po = m["pinhole_objects"]
                st.caption(f"Pinhole objects: {po['gt_detected']}/{po['gt_objects']} annotated pinholes detected "
                           f"(recall {fmt(po['object_recall'])}); {po['pred_true']}/{po['pred_objects']} predicted "
                           f"blobs matched (precision {fmt(po['object_precision'])}); 3 px tolerance.")

    comparison_section()
    if m is not None:
        failure_gallery(name, models)
    lim = config.resolve("artifacts/limitations.md")
    if lim.exists():
        st.subheader("What the model misses, and why it is not production-certified")
        st.markdown(lim.read_text(encoding="utf-8"))


def comparison_section():
    rows = []
    base = read_json(METRICS_DIR / "baseline_pinhole.json")
    if base:
        t = base["test"]
        rows.append({"Method": "OpenCV top-hat baseline (pinhole only)", "Pinhole image precision": fmt(t["image_level"]["precision"]),
                     "Pinhole image recall": fmt(t["image_level"]["recall"]), "Pinhole image F1": fmt(t["image_level"]["f1"]),
                     "Pinhole object recall": fmt(t["object_recall"]), "Pinhole object precision": fmt(t["object_precision"]),
                     "Localizes?": "yes (pinholes)"})
    for name in available_models():
        m = read_json(METRICS_DIR / f"{name}_test_metrics.json")
        if not m:
            continue
        p = m["image_level"]["pinhole"]
        po = m.get("pinhole_objects", {})
        rows.append({"Method": name, "Pinhole image precision": fmt(p["precision"]), "Pinhole image recall": fmt(p["recall"]),
                     "Pinhole image F1": fmt(p["f1"]), "Pinhole object recall": fmt(po.get("object_recall")),
                     "Pinhole object precision": fmt(po.get("object_precision")),
                     "Localizes?": "yes (4 classes)" if m["model_kind"] == "segmentation" else "no"})
    if rows:
        st.subheader("Baseline vs learned models: pinholes, test split")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Pinholes are the one class all three methods address. The classical baseline's parameters were "
                   "tuned on train+val; all methods are scored on the same test images.")


def failure_gallery(name: str, models: dict):
    per = METRICS_DIR / f"{name}_test_per_image.csv"
    root, mode = image_root()
    if not per.exists() or mode != "full":
        if mode != "full":
            st.info("Failure gallery needs the full dataset in `data/raw/CoatingVision`.")
        return
    st.subheader("Failure cases")
    p = pd.read_csv(per)
    c1, c2 = st.columns(2)
    cls = c1.selectbox("Defect", CLASSES, format_func=CLASS_LABELS.get, index=2, key="fail_cls")
    kind = c2.radio("Error type", ["False negatives (missed)", "False positives (false alarm)"], horizontal=True)
    if kind.startswith("False negatives"):
        sel = p[(p[f"{cls}_true"] == 1) & (p[f"{cls}_pred"] == 0)].sort_values(f"{cls}_prob")
    else:
        sel = p[(p[f"{cls}_true"] == 0) & (p[f"{cls}_pred"] == 1)].sort_values(f"{cls}_prob", ascending=False)
    st.caption(f"{len(sel)} test images. Showing up to 6, most confident mistakes first.")
    if sel.empty:
        return
    model = get_model(str(models[name]))
    man = get_manifest().set_index("sample_id")
    cols = st.columns(3)
    for i, r in enumerate(sel.head(6).itertuples(index=False)):
        mr = man.loc[r.sample_id]
        img = load_image(root / mr.image_path)
        gt = load_mask(root / mr.mask_path)
        with cols[i % 3]:
            st.markdown(f"`{r.sample_id}` · p({CLASS_LABELS[cls].lower()}) = {getattr(r, f'{cls}_prob'):.2f}")
            if model.kind == "segmentation":
                res = predict(model, img, r.sample_id)
                st.image(error_overlay(img, res["predicted_masks"][..., CLASSES.index(cls)], gt[..., CLASSES.index(cls)]),
                         width="stretch")
                st.caption("Green = hit · Red = missed annotation · Blue = false alarm")
            else:
                st.image(overlay_masks(img, gt), width="stretch")
                st.caption("Overlay shows the published annotation (classifier has no localization).")


def page_methodology():
    st.title("Methodology")
    md = config.resolve("artifacts/methodology.md")
    if md.exists():
        st.markdown(md.read_text(encoding="utf-8"))
    else:
        st.info("artifacts/methodology.md not found.")


# ---------------------------------------------------------------- navigation
st.sidebar.markdown("## BatteryVision")
st.sidebar.caption("Electrode-coating defect inspection on the public CoatingVision dataset.")
sidebar_model()
nav = st.navigation([
    st.Page(page_inspect, title="Inspect image", default=True),
    st.Page(page_analytics, title="Dataset analytics"),
    st.Page(page_evaluation, title="Model evaluation"),
    st.Page(page_methodology, title="Methodology"),
])
st.sidebar.markdown("---")
st.sidebar.caption("Data: Sampath et al., *CoatingVision*, Scientific Data (2026), CC BY 4.0. "
                   "Independent PyTorch implementation; not the authors' code.")
nav.run()

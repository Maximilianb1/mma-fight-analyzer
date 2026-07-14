"""Streamlit demo for the frozen MMA fight-analysis pipeline.

Run from the repository root:
    streamlit run tools/demo_app.py
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mma import identity as ident  # noqa: E402
from mma.pipeline import FightAnalyzer  # noqa: E402


GATE_CKPT = REPO / "outputs/gate/gate.pt"
PHASE_CKPT = REPO / "outputs/phase/deployment_phase_final.pt"
PRESSURE_CKPT = REPO / "outputs/phase/deployment_pressure_final.pt"
DEMO_DIR = REPO / "outputs/demo"
UPLOAD_DIR = DEMO_DIR / "uploads"
HELDOUT_NAME = "Paddy Pimblett vs Michael Chandler"
IDENTITY_PIPELINE_VERSION = "comparative_color_temporal_v2"
HELDOUT_CLIP_DIRS = [
    REPO.parent / "Creating the data set/mma_labeler/Done" / HELDOUT_NAME,
    REPO / "data/raw" / HELDOUT_NAME,
]


def safe_stem(name):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._")
    return stem or "fight"


def find_holdout_clips():
    for folder in HELDOUT_CLIP_DIRS:
        if folder.is_dir():
            clips = sorted(folder.glob("clip_*.mp4"))
            if clips:
                return clips
    return []


def save_upload(upload):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(f"{upload.name}:{upload.size}".encode()).hexdigest()[:10]
    path = (
        UPLOAD_DIR
        / f"{token}_{safe_stem(upload.name)}{Path(upload.name).suffix.lower()}"
    )
    if not path.exists() or path.stat().st_size != upload.size:
        with path.open("wb") as handle:
            handle.write(upload.getbuffer())
    return path


def source_controls():
    heldout = find_holdout_clips()
    source_kind = st.radio(
        "Video source",
        ["Held-out fight", "Upload video", "Local video path"],
        horizontal=True,
    )
    if source_kind == "Held-out fight":
        if not heldout:
            st.error(
                "The held-out clip folder was not found. Select Upload video or Local video path."
            )
            return None
        total_gb = sum(p.stat().st_size for p in heldout) / 1024**3
        st.caption(
            f"{len(heldout)} ordered 5-second clips ({total_gb:.2f} GB). "
            "They are processed continuously so fighter tracking persists across clips; "
            "their audio tracks are concatenated into the final annotated video."
        )
        return {
            "kind": "clips",
            "paths": heldout,
            "name": HELDOUT_NAME,
            "f1_name": "Michael Chandler",
            "f2_name": "Paddy Pimblett",
            "f1_color": "black",
            "f2_color": "red",
            "key": f"heldout:{len(heldout)}:{heldout[-1].stat().st_mtime_ns}",
        }
    if source_kind == "Upload video":
        upload = st.file_uploader(
            "Fight video", type=["mp4", "mov", "mkv", "avi", "webm"]
        )
        if upload is None:
            return None
        path = save_upload(upload)
        return {
            "kind": "video",
            "path": path,
            "name": Path(upload.name).stem,
            "key": f"upload:{path.name}:{path.stat().st_size}",
        }
    local_value = st.text_input("Absolute or repository-relative video path")
    if not local_value:
        return None
    path = Path(local_value).expanduser()
    if not path.is_absolute():
        path = (REPO / path).resolve()
    if not path.is_file():
        st.error(f"Video not found: {path}")
        return None
    return {
        "kind": "video",
        "path": path,
        "name": path.stem,
        "key": f"path:{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}",
    }


def new_analyzer(f1_name, f2_name, identity_choice=None, f1_color=None, f2_color=None):
    return FightAnalyzer(
        GATE_CKPT,
        PHASE_CKPT,
        pressure_ckpt=PRESSURE_CKPT,
        interactive=False,
        f1_color=f1_color,
        f2_color=f2_color,
        f1_name=f1_name,
        f2_name=f2_name,
        out_dir=DEMO_DIR,
        identity_choice=identity_choice,
        web_compatible=True,
    )


def prepare_prompt(analyzer, source):
    if source["kind"] == "clips":
        return analyzer.find_identity_prompt_in_clips(source["paths"])
    return analyzer.find_identity_prompt(source["path"])


def run_pipeline(analyzer, source, output_path, callback, finalize_callback):
    if source["kind"] == "clips":
        return analyzer.process_clips(
            source["paths"], output_path, callback, finalize_callback
        )
    return analyzer.process(source["path"], output_path, callback, finalize_callback)


def result_table(results, f1_name, f2_name):
    rows = []
    for item in results:
        pressure = item.get("pressure")
        if not item.get("pressure_reliable", True):
            pressure = "Uncertain — identity unavailable"
        else:
            pressure = {"Fighter 1": f1_name, "Fighter 2": f2_name}.get(
                pressure, pressure
            )
        rows.append(
            {
                "Time": f"{item['start_s'] // 60:02d}:{item['start_s'] % 60:02d}",
                "Segment": "Non-fight" if item["excluded"] else "Fight",
                "Gate confidence": (
                    item["gate_prob_excluded"]
                    if item["excluded"]
                    else 1 - item["gate_prob_excluded"]
                ),
                "Phase": item.get("phase") or "—",
                "Phase confidence": item.get("phase_conf"),
                "Pressure": pressure or "—",
                "Pressure confidence": item.get("pressure_conf"),
                "Identity method": item.get("identity_method", "—"),
                "Identity coverage": item.get("identity_both_coverage"),
            }
        )
    return pd.DataFrame(rows)


def show_results(run, f1_name, f2_name):
    output_path = Path(run["output_path"])
    results = run["results"]
    fight_results = [r for r in results if not r["excluded"]]
    phases = Counter(r["phase"] for r in fight_results if r.get("phase"))
    pressures = Counter(
        r["pressure"]
        for r in fight_results
        if r.get("pressure") and r.get("pressure_reliable", True)
    )
    pressure_name = {
        "Fighter 1": f1_name,
        "Fighter 2": f2_name,
        "Mutual": "Mutual",
    }

    st.divider()
    st.header("Annotated result")
    if output_path.is_file():
        st.video(str(output_path), format="video/mp4")
        size_mb = output_path.stat().st_size / 1024**2
        st.caption(f"Saved to `{output_path}` ({size_mb:.1f} MB)")
        prepare_download = st.checkbox(
            f"Prepare full annotated video download ({size_mb:.1f} MB)",
            help="Large Streamlit downloads are loaded into server memory only when enabled.",
            key=f"prepare_download_{output_path.name}",
        )
        if prepare_download:
            with output_path.open("rb") as video_file:
                st.download_button(
                    "Download full annotated video",
                    data=video_file,
                    file_name=output_path.name,
                    mime="video/mp4",
                    type="primary",
                )
    else:
        st.error(f"Output video is missing: {output_path}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("5-second windows", len(results))
    c2.metric("Fight", len(fight_results))
    c3.metric("Non-fight", len(results) - len(fight_results))
    c4.metric("Most common phase", phases.most_common(1)[0][0] if phases else "—")
    if pressures:
        label, count = pressures.most_common(1)[0]
        st.caption(
            f"Most frequent pressure prediction: **{pressure_name.get(label, label)}** "
            f"({count}/{len(fight_results)} fight windows)."
        )

    table = result_table(results, f1_name, f2_name)
    st.subheader("Window-by-window timeline")
    st.dataframe(
        table.style.format(
            {
                "Gate confidence": "{:.1%}",
                "Phase confidence": lambda value: (
                    "—" if pd.isna(value) else f"{value:.1%}"
                ),
                "Pressure confidence": lambda value: (
                    "—" if pd.isna(value) else f"{value:.1%}"
                ),
                "Identity coverage": lambda value: (
                    "—" if pd.isna(value) else f"{value:.0%}"
                ),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    json_path = Path(run["json_path"])
    st.download_button(
        "Download prediction timeline (JSON)",
        data=json_path.read_bytes(),
        file_name=json_path.name,
        mime="application/json",
    )


st.set_page_config(page_title="MMA Fight Analyzer", page_icon="🥊", layout="wide")
st.title("🥊 MMA Fight Phase & Pressure Analyzer")
st.write(
    "Run the frozen final pipeline on the untouched hold-out fight or another video. "
    "Every 5-second window is classified as fight/non-fight; fight windows also receive "
    "phase and pressure predictions."
)

missing = [p for p in (GATE_CKPT, PHASE_CKPT, PRESSURE_CKPT) if not p.is_file()]
if missing:
    st.error(
        "Required final checkpoints are missing:\n\n"
        + "\n".join(f"- `{p}`" for p in missing)
    )
    st.stop()

device_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
st.caption(
    f"Runtime: **{device_label}** · Gate: frozen OOF threshold · "
    "Phase: multi-task R(2+1)D · Pressure: pressure-only R(2+1)D · No smoothing"
)
if not torch.cuda.is_available():
    st.warning("No CUDA GPU detected. A full fight will run slowly on CPU.")

source = source_controls()
if source is None:
    st.stop()

if st.session_state.get("demo_source_key") != source["key"]:
    for key in ("demo_analyzer", "demo_prompt", "demo_run"):
        st.session_state.pop(key, None)
    st.session_state.demo_source_key = source["key"]

default_f1 = source.get("f1_name", "Fighter 1")
default_f2 = source.get("f2_name", "Fighter 2")
name_col1, name_col2 = st.columns(2)
f1_name = name_col1.text_input(
    "Fighter 1 — name left of the broadcast timer", default_f1
)
f2_name = name_col2.text_input(
    "Fighter 2 — name right of the broadcast timer", default_f2
)

st.subheader("One-time fighter identity")
identity_mode = st.radio(
    "Identity method",
    ["Visual A/B prompt", "Known shorts colors"],
    horizontal=True,
    help="The visual prompt is recommended for a new video. Shorts colors are useful for unattended runs.",
)

identity_choice = None
f1_color = f2_color = None
ready = False
if identity_mode == "Visual A/B prompt":
    if st.button("Find a frame with both fighters", type="primary"):
        with st.spinner("Loading the frozen models and finding two fighters..."):
            try:
                analyzer = new_analyzer(
                    f1_name,
                    f2_name,
                    f1_color=source.get("f1_color"),
                    f2_color=source.get("f2_color"),
                )
                prompt = prepare_prompt(analyzer, source)
                if prompt is None:
                    st.error(
                        "No suitable two-fighter frame was found. Try Known shorts colors or another video."
                    )
                else:
                    st.session_state.demo_analyzer = analyzer
                    st.session_state.demo_prompt = prompt
            except Exception as exc:
                st.exception(exc)
    prompt = st.session_state.get("demo_prompt")
    if prompt is not None:
        rgb = cv2.cvtColor(prompt["image_bgr"], cv2.COLOR_BGR2RGB)
        st.image(
            rgb,
            caption=f"First usable live-fight window, starting at {prompt['start_s']} seconds",
            use_container_width=True,
        )
        answer = st.radio(
            f"Which box is {f1_name} (Fighter 1, left of the timer)?",
            ["A", "B"],
            horizontal=True,
        )
        identity_choice = answer
        ready = True
else:
    colors = list(ident.COLOR_RANGES)
    default_color_1 = colors.index("black") if source["name"] == HELDOUT_NAME else 0
    default_color_2 = colors.index("red") if source["name"] == HELDOUT_NAME else 1
    color_col1, color_col2 = st.columns(2)
    f1_color = color_col1.selectbox(f"{f1_name} shorts", colors, index=default_color_1)
    f2_color = color_col2.selectbox(f"{f2_name} shorts", colors, index=default_color_2)
    ready = f1_color != f2_color
    if not ready:
        st.error("Choose different shorts colors for the two fighters.")

if st.button("Run full inference", type="primary", disabled=not ready):
    try:
        if identity_mode == "Visual A/B prompt":
            analyzer = st.session_state.demo_analyzer
            analyzer.f1_name, analyzer.f2_name = f1_name, f2_name
            # The held-out fight has trusted metadata; arbitrary videos learn
            # their two colors from the user-confirmed prompt when possible.
            analyzer.f1_color = source.get("f1_color")
            analyzer.f2_color = source.get("f2_color")
            analyzer.reset_identity(identity_choice)
        else:
            analyzer = new_analyzer(
                f1_name, f2_name, f1_color=f1_color, f2_color=f2_color
            )

        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        stem = safe_stem(source["name"])
        output_path = DEMO_DIR / f"{stem}_labeled.mp4"
        json_path = DEMO_DIR / f"{stem}_predictions.json"
        progress = st.progress(0.0, text="Starting inference...")

        def update_progress(done, total, info):
            # Reserve the last 5% for the browser-video encoding pass.
            fraction = min(0.95 * done / total, 0.95) if total else 0.0
            label = (
                "NON-FIGHT"
                if info["excluded"]
                else f"{info['phase']} · {info['pressure']}"
            )
            progress.progress(fraction, text=f"Window {done}/{total or '?'}: {label}")

        def show_finalizing():
            progress.progress(
                0.97,
                text="Finalizing browser-ready H.264 video — this can take a few minutes...",
            )

        results = run_pipeline(
            analyzer, source, output_path, update_progress, show_finalizing
        )
        payload = {
            "source": source["name"],
            "identity": {"fighter_1": f1_name, "fighter_2": f2_name},
            "settings": {
                "window_seconds": 5,
                "gate_checkpoint": str(GATE_CKPT),
                "phase_checkpoint": str(PHASE_CKPT),
                "pressure_checkpoint": str(PRESSURE_CKPT),
                "temporal_smoothing": "none",
                "identity_pipeline": IDENTITY_PIPELINE_VERSION,
            },
            "windows": results,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        st.session_state.demo_run = {
            "output_path": str(output_path),
            "json_path": str(json_path),
            "results": results,
            "source_key": source["key"],
            "identity_pipeline": IDENTITY_PIPELINE_VERSION,
        }
        progress.progress(1.0, text="Inference complete")
        st.success("The annotated video and prediction timeline are ready.")
    except Exception as exc:
        st.exception(exc)

run = st.session_state.get("demo_run")
if run is not None and run.get("identity_pipeline") != IDENTITY_PIPELINE_VERSION:
    st.session_state.pop("demo_run", None)
    run = None
if run is None:
    stem = safe_stem(source["name"])
    existing_video = DEMO_DIR / f"{stem}_labeled.mp4"
    existing_json = DEMO_DIR / f"{stem}_predictions.json"
    if existing_video.is_file() and existing_json.is_file():
        payload = json.loads(existing_json.read_text(encoding="utf-8"))
        if (
            payload.get("settings", {}).get("identity_pipeline")
            == IDENTITY_PIPELINE_VERSION
        ):
            st.success(
                f"A completed result is already saved ({existing_video.stat().st_size / 1024**2:.1f} MB)."
            )
        else:
            st.warning(
                "The saved result used the old appearance-only tracker. Run inference again "
                "to generate a result with comparative shorts-color identity correction."
            )
            payload = None
        if payload is not None and st.button("Open last completed result"):
            st.session_state.demo_run = {
                "output_path": str(existing_video),
                "json_path": str(existing_json),
                "results": payload["windows"],
                "source_key": source["key"],
                "identity_pipeline": IDENTITY_PIPELINE_VERSION,
            }
            st.rerun()

run = st.session_state.get("demo_run")
if run is not None and run.get("source_key") == source["key"]:
    show_results(run, f1_name, f2_name)

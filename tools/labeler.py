import os
import subprocess
import shutil
import pandas as pd
import streamlit as st
import cv2

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
CLIP_DURATION = 5  # seconds
VIDEO_DIR = "fights"
OUTPUT_CLIPS_DIR = "labeled_clips"  # <-- New parent directory for permanent slices

PHASE_LABELS = [
    "Striking",
    "Grappling/Ground Work",
    "Clinch",
    "Transition/Takedown",
    "Neutral/Measuring Distance",
]

st.set_page_config(layout="wide", page_title="MMA Phase Annotator")

# Inject Custom CSS to blow up the font sizes for scannability
st.markdown(
    """
    <style>
    div[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] p {
        font-size: 22px !important;
        font-weight: 500 !important;
        padding-bottom: 5px;
    }
    .main h3 {
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    div[data-testid="stForm"] button {
        padding: 10px 20px !important;
        font-size: 18px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_video_duration(path):
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0:
        return 0
    return frame_count / fps


def extract_clip(input_video, start_time, duration, output_path):
    if os.path.exists(output_path):
        return

    command = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-i",
        input_video,
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-y",
        output_path,
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ==========================================
# APPLICATION LOGIC
# ==========================================
st.title("🥊 MMA Fight Phase & Pressure Annotator")

if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)
if not os.path.exists(OUTPUT_CLIPS_DIR):
    os.makedirs(OUTPUT_CLIPS_DIR)

available_fights = [
    f for f in os.listdir(VIDEO_DIR) if f.endswith((".mp4", ".mkv", ".avi"))
]

if not available_fights:
    st.info(
        f"Please drop your full fight MP4 videos inside the `{VIDEO_DIR}/` directory to begin."
    )
    st.stop()

selected_fight = st.sidebar.selectbox("Select Fight Video", available_fights)
video_path = os.path.join(VIDEO_DIR, selected_fight)
fight_id = os.path.splitext(selected_fight)[0]
csv_path = os.path.join(VIDEO_DIR, f"{fight_id}_labels.csv")

# Create a unique folder for this fight's clips
fight_clips_folder = os.path.join(OUTPUT_CLIPS_DIR, fight_id)
if not os.path.exists(fight_clips_folder):
    os.makedirs(fight_clips_folder)

# Fighter Identity Inputs in Sidebar
st.sidebar.write("---")
st.sidebar.subheader("Fighter Profiles")
f1_identity = st.sidebar.text_input("Fighter 1 (Left on screen)", "Fighter 1")
f2_identity = st.sidebar.text_input("Fighter 2 (Right on screen)", "Fighter 2")

PRESSURE_OPTIONS_DISPLAY = [
    f"🔴 {f1_identity} Applying Pressure",
    f"🔵 {f2_identity} Applying Pressure",
    "⚪ Mutual / Equal Pressure",
]

total_duration = get_video_duration(video_path)
max_clips = int(total_duration // CLIP_DURATION)

if os.path.exists(csv_path):
    df_labels = pd.read_csv(csv_path)
else:
    df_labels = pd.DataFrame(
        columns=[
            "clip_index",
            "start_time",
            "end_time",
            "phase_label",
            "pressure_label",
            "excluded",
            "saved_filename",
        ]
    )

if (
    "clip_idx" not in st.session_state
    or st.session_state.get("current_fight") != selected_fight
):
    st.session_state.clip_idx = len(df_labels)
    st.session_state.current_fight = selected_fight

if st.session_state.clip_idx >= max_clips:
    st.success(
        f"🎉 Fully labeled! All {max_clips} clips processed for {selected_fight}."
    )
    st.stop()

current_start = st.session_state.clip_idx * CLIP_DURATION
current_end = current_start + CLIP_DURATION

# Temporary path for player preview
temp_clip_dir = "temp_clips"
if not os.path.exists(temp_clip_dir):
    os.makedirs(temp_clip_dir)
temp_clip_path = os.path.join(temp_clip_dir, f"{fight_id}_temp_curr.mp4")

with st.spinner("Cutting clip..."):
    extract_clip(video_path, current_start, CLIP_DURATION, temp_clip_path)

# ==========================================
# UI LAYOUT
# ==========================================
col_video, col_controls = st.columns([1.6, 1])

with col_video:
    st.subheader(
        f"Clip {st.session_state.clip_idx + 1} / {max_clips} ({current_start}s - {current_end}s)"
    )
    if os.path.exists(temp_clip_path):
        with open(temp_clip_path, "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes, loop=True, autoplay=True)
    else:
        st.error("Failed to generate video preview clip.")

with col_controls:
    st.subheader("Annotation Matrix")

    with st.form("annotation_form", clear_on_submit=False):
        phase_choice = st.radio("1. Action Phase Class", PHASE_LABELS)
        st.write("---")
        pressure_choice = st.radio("2. Pressure Dynamics", PRESSURE_OPTIONS_DISPLAY)
        st.write("---")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit_btn = st.form_submit_button("💾 Save & Next", type="primary")
        with col_btn2:
            skip_btn = st.form_submit_button("⏭️ Skip / Exclude")

# ==========================================
# DATA SAVING & FILE COPYING
# ==========================================
if submit_btn or skip_btn:
    if skip_btn:
        final_pressure = "None"
        filename_suffix = "_excluded.mp4"
    else:
        filename_suffix = ".mp4"
        if pressure_choice == PRESSURE_OPTIONS_DISPLAY[0]:
            final_pressure = "Fighter 1"
        elif pressure_choice == PRESSURE_OPTIONS_DISPLAY[1]:
            final_pressure = "Fighter 2"
        else:
            final_pressure = "Mutual"

    # Format file names as clip_0000.mp4, clip_0001.mp4 to preserve perfect order
    clip_filename = f"clip_{st.session_state.clip_idx:04d}{filename_suffix}"
    permanent_clip_path = os.path.join(fight_clips_folder, clip_filename)

    # Save video cut directly out of the temp folder
    if os.path.exists(temp_clip_path):
        shutil.move(temp_clip_path, permanent_clip_path)

    new_row = {
        "clip_index": st.session_state.clip_idx,
        "start_time": current_start,
        "end_time": current_end,
        "phase_label": "None" if skip_btn else phase_choice,
        "pressure_label": final_pressure,
        "excluded": True if skip_btn else False,
        "saved_filename": clip_filename,  # Tracking the exact file link
    }

    df_labels = pd.concat([df_labels, pd.DataFrame([new_row])], ignore_index=True)
    df_labels.to_csv(csv_path, index=False)

    st.session_state.clip_idx += 1
    st.rerun()

# Sidebar Metadata & Smart Undo
st.sidebar.markdown("---")
st.sidebar.subheader("Progress Metrics")
st.sidebar.text(f"Total Clips: {max_clips}")
st.sidebar.text(f"Labeled/Reviewed: {len(df_labels)}")

if len(df_labels) > 0 and st.sidebar.button("↩️ Undo Last Label"):
    # Identify the last saved clip path before deleting the row
    last_row = df_labels.iloc[-1]
    last_filename = last_row["saved_filename"]
    last_clip_filepath = os.path.join(fight_clips_folder, last_filename)

    # Remove the physical video file to clean the folder layout
    if os.path.exists(last_clip_filepath):
        os.remove(last_clip_filepath)

    # Drop from dataframe and save
    df_labels = df_labels.iloc[:-1]
    df_labels.to_csv(csv_path, index=False)

    # Roll back counter state
    st.session_state.clip_idx = max(0, st.session_state.clip_idx - 1)
    if os.path.exists(temp_clip_path):
        os.remove(temp_clip_path)
    st.rerun()

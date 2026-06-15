import streamlit as st

from swing_detect import analyze_file

st.title("Swing Detector")

st.text(
    """
    Select one or more audio files to be analyzed. You can optionally restrict the analysis to only look at a certain section of the audio files.
    """
)

offset: float | None = None
duration: float | None = None

with st.expander("Audio Analysis Options", expanded=True):
    analyze_whole_file = st.toggle("Analyze whole audio file", value=True)

    offset = st.number_input(
        "Start at offset (in seconds)", 0,
        help="Start analysis at this time within the audio file (in seconds).",
        disabled=analyze_whole_file)

    duration = st.number_input(
        "Duration (in seconds)", 0, value=30,
        help="Only analyze up to this much audio (in seconds).",
        disabled=analyze_whole_file)

if analyze_whole_file:
    offset = None
    duration = None

audio_files = st.file_uploader(
    label="Audio File",
    help="Select one or more audio files to be analyzed.",
    type="audio",
    accept_multiple_files=True,
)

for audio_file in audio_files:
    results = analyze_file(audio_file, offset=offset, duration=duration)
    st.text(results.to_json(indent=2))

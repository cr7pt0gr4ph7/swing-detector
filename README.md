# Swing Detector

Automatically detect how much a rhythm in a given audio file is [swung](https://en.wikipedia.org/wiki/Swing_time) by determining how the offbeats are positioned relative to the surrounding main beats.

This project uses the [librosa](http://librosa.org/) library for estimating the positions of the main beats as well as for detecting the intervening note onset events.

## CLI Usage

```bash
python swing_detect.py file1.mp3 file2.mp3 ...
```

## Local Streamlit Usage

```bash
streamlit run swing_detect_streamlit.py
```

# Swing Detector

Automatically detect how much a rhythm in a given audio file is [swung](https://en.wikipedia.org/wiki/Swing_time) by determining how the offbeats are positioned relative to the surrounding main beats.

This project uses the [librosa](http://librosa.org/) library for estimating the positions of the main beats as well as for detecting the intervening note onset events.
[mutagen](https://github.com/quodlibet/mutagen) is used for writing the metadata back to the audio files if requested by the user.

## CLI Usage

```bash
python swing_detect.py file1.mp3 file2.mp3 file3.flac ...
```

Use `--help` to get an overview of the available CLI options.

```bash
python swing_detect.py --help
```

## Local Streamlit Usage

```bash
streamlit run swing_detect_streamlit.py
```

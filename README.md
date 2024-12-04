# Audio Transcription with Faster Whisper

This project provides a script to record, process, and transcribe audio using the Faster Whisper model. The workflow includes options to record from your microphone, download audio from YouTube, or load an existing audio file. It then splits the audio into chunks and transcribes them using a Faster Whisper model, finally saving the transcription result in a text file.

## Features

- **Audio Recording**: Records audio using `sounddevice` library with an option to use headphones (e.g., WH-CH720N).
- **Audio Download**: Download audio from a YouTube URL and convert it to WAV format.
- **Audio Chunking**: Split long audio files into smaller chunks for efficient transcription.
- **Transcription**: Use the Faster Whisper model to transcribe the audio chunks.
- **Output**: Save transcriptions into text files, handling multiple transcription runs.

## Requirements

- Python 3.10
- Install the required Python packages:

```pip install numpy sounddevice soundfile faster-whisper yt-dlp```

Faster Whisper Model should be installed for transcription.

## Usage

You can run the script with various arguments depending on your needs.
Command Line Arguments

    -m, --model: Specify the model size for transcription. Choices include:
        t: Tiny
        s: Small
        b: Base
        m: Medium
        l: Large-v3 (default: m)
    -c, --chunk_duration: Set the duration (in seconds) for each audio chunk. Default is 30 seconds.
    -d, --headphones: Use headphones as the audio input device (e.g., WH-CH720N).
    -l, --load: Load an existing audio file instead of recording. You can provide a URL to download audio or specify a path to a local file.

## Example Commands

    Record and transcribe audio with default settings:

```python transcribe.py -m m -c 30```

Use headphones to record audio and transcribe:

```python transcribe.py -m m -c 30 -d```

Download audio from YouTube and transcribe:

```python transcribe.py -m m -c 30 -l "https://www.youtube.com/watch?v=VIDEO_ID"``` 

Load an existing audio file for transcription:

    ```python transcribe.py -m m -c 30 -l "path_to_audio.wav"```

## Workflow Overview

    Audio Recording:
        The script uses sounddevice to record audio from the default input device or a specified headphone device.
        The audio is saved as a .wav file in the audio/ directory.

    Audio Download:
        If a YouTube URL is provided, yt-dlp is used to download the best available audio and convert it to .wav format.

    Audio Chunking:
        The audio file is split into chunks of a specified duration (default is 30 seconds) to optimize the transcription process.

    Transcription:
        The Faster Whisper model is used to transcribe each chunk of audio.
        Transcription includes timestamps and the corresponding transcribed text for each chunk.

    Result Storage:
        The transcriptions are saved to text files in the output/ directory.

## Example Output

Transcriptions will be saved in files named output_0.txt, output_1.txt, etc. An example transcription might look like:

[0.00s -> 30.00s] This is an example transcription of the first chunk.
[30.00s -> 60.00s] The second chunk begins here and continues.

## Model Options

The faster-whisper model supports various model sizes. You can choose the model based on your hardware capabilities:

    t: Tiny
    s: Small
    b: Base
    m: Medium
    l: Large-v3 (default)

# 🎙️ Audio Transcription Suite with Faster Whisper

--- NOTE: This proyect should be dockerized to avoid dependencies related to OS, but is not implemented yet ---

## 🚀 Quick Start
```bash
# List available recording devices
python src/main.py --list-devices

# Record using device 2 with large model
python src/main.py -m l -d 2

# Transcribe YouTube video with summarization
python src/main.py -l "https://youtu.be/EXAMPLE" -m l -s
```

## 🔧 Full Command Reference
| Flag | Parameter       | Description                          | Default |
|------|-----------------|--------------------------------------|---------|
| `-m` | `--model`       | Whisper model size                   | m       |
| `-d` | `--device`      | Input device ID                      | Auto    |
| `-c` | `--chunk-dur`   | Chunk duration (seconds)             | 30      |
| `-l` | `--load`        | File path/YouTube URL                | None    |
| `-s` | `--summarize`   | Enable GPT-4 analysis                | Off     |

## 📂 File Structure
```
project-root/
├── audio/               # Raw recordings/chunks
├── output/              # Transcripts & summaries
├── prompts/             # Custom AI templates
└── models/              # Whisper model cache
```

## 🧠 AI Template Example
Create `prompts/medical.txt`:
```txt
title=Medical Analysis
Analyze patient discussion:
1. List symptoms
2. Identify medications
3. Flag concerns

[transcription here]
```

## 💻 Real-World Usage
### 1. Conference Recording
```bash
python src/main.py -d 4 -c 120 -s
```
Outputs: Timestamped transcript + bullet-point summary

### 2. Podcast Episode Analysis
```bash
python src/main.py -l "https://youtu.be/PODCAST_ID" -m l
```
Process: YouTube → WAV → 30s chunks → Full transcript

### 3. Historical Archive Processing
```bash
python src/main.py -l archive_tape.wav -m m -c 45
```
Features: Medium model, 45s chunks for context

## ⚠️ Requirements
1. `.env` file:
```env
OPENAI_API_KEY=your_key_here
```

2. Hardware:
- 4GB RAM minimum
- GPU recommended for large models

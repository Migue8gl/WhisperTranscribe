# 🎤 Audio Transcriber Pro: Whisper-Powered Transcription

## 🚀 Quick Start

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **List Audio Devices**:
   ```bash
   python src/main.py --list-devices
   ```

3. **Basic Recording**:
   ```bash
   python src/main.py -m l -d 2
   ```

## ✨ Key Features
- 🎧 Device Selection
- ⏱ 30-120s Chunking
- 🌐 YouTube Integration
- 📝 AI Summarization

## 🔧 Full Usage
```bash
python src/main.py [OPTIONS]
```

### 🔍 Options Table
| Short | Long          | Description        |
|-------|---------------|--------------------|
| -m    | --model       | Model size (t/s/b/m/l/d) |
| -d    | --device      | Input device ID    |

## 💻 Examples
### 🎙 Record Audio
```bash
python src/main.py -m l -d 2 -c 60
```

### 🌐 YouTube Download
```bash
python src/main.py -l "https://youtu.be/EXAMPLE"
```

## 📂 Output Files
```
output/
├── transcript_0.txt
└── summary.md
```

## 🌟 Pro Tips
1. For sample rate errors:
   ```bash
   python src/main.py -d 2 -c 60
   ```
   
2. Add OpenAI key:
   ```env
   OPENAI_API_KEY=your_key_here
   ```

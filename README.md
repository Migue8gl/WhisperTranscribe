# 🎤 Audio Transcriber Pro: Whisper-Powered Transcription

## 🚀 Ultimate Command Cheat Sheet

### 🔥 Top Summarization Examples

**1. Academic Lecture Analysis**
```bash
python src/main.py -l "lecture.mp3" -m l -s --prompt ./prompts/academic_template.txt
```
*Outputs:* Detailed chapter summaries + key concepts glossary

**2. Podcast Highlights**
```bash
python src/main.py -l "https://youtu.be/podcast123" -m l -s --temperature 0.7
```
*Features:* Guest quotes + episode highlights + discussion topics

**3. Meeting Minutes Generation**
```bash
python src/main.py -d 3 -c 120 -s --prompt ./prompts/business_meeting.txt
```
*Creates:* Action items + Decisions made + Next steps

**4. Interview Analysis**
```bash
python src/main.py -l interview.wav -s --max_tokens 2000
```
*Produces:* Key insights + Quotes + Sentiment analysis

### 🎛️ Advanced Template Control

**Custom Prompt Template** (`./prompts/custom_template.txt`):
```txt
title=My Custom Analysis
Analyze this conversation:
1. Identify main topics
2. Extract 5 key points
3. Create timeline of events
4. Highlight controversial statements

[transcription here]
```

**Usage**:
```bash
python src/main.py -l audio.mp3 -s --prompt ./prompts/custom_template.txt
```

### 📜 Template Gallery

| Template File              | Use Case                          | Output Features                     |
|----------------------------|-----------------------------------|--------------------------------------|
| `legal_discussion.txt`      | Court recordings                  | Timeline, Evidence list             |
| `medical_consult.txt`       | Doctor-patient talks              | Symptoms list, Treatment plan       |
| `tech_interview.txt`        | Coding interviews                 | Code challenges, Solution analysis  |
| `creative_writing.txt`      | Story recordings                  | Character map, Plot structure       |

Create new templates by copying `prompt_schema_md.txt` and modifying the instructions!

### 🌐 Multi-Format Example

**YouTube Tech Review → Markdown Report**
```bash
python src/main.py -l "https://youtu.be/tech_review" -m l -s --prompt ./prompts/tech_analysis.txt
```

*Sample Output* (`output/tech-review-2023.md`):
```markdown
## GPU Comparison Analysis - 2023

### Key Specifications
| Model       | VRAM | Clock Speed | Price  |
|-------------|------|-------------|--------|
| RTX 4090    | 24GB | 2.52GHz     | $1599  |

### Performance Highlights
- 4K gaming: 128 avg FPS in Cyberpunk 2077
- Thermal max: 72°C under load
- Value rating: 8.5/10

### Conclusion
"Best for 4K enthusiasts despite premium pricing"
```

## 💡 Pro Template Tips

1. **Control Output Format**:
   ```txt
   title=Meeting Minutes
   Format in bullet points with emojis:
   - 📅 Date: [auto-insert-date]
   - 🎯 Objectives: [transcription here]
   ```

2. **Multi-Stage Analysis**:
   ```txt
   First analyze sentiment, then extract facts:
   
   Sentiment Score: [1-10]
   Key Facts:
   1. 
   2. 
   ```

3. **Language Control**:
   ```txt
   Write in Spanish using markdown:
   ## Resumen Ejecutivo
   - Puntos clave
   - Recomendaciones
   ```

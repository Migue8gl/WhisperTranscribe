import os
import whisper

try:
    if not os.path.isdir("audio"):
        os.mkdir("audio")
    print('Listening... Press Ctrl+C to stop')
    os.system('arecord -vv --format=cd -t wav audio/audio.wav > /dev/null 2>&1')
except KeyboardInterrupt:
    pass

model = whisper.load_model("large", download_root="models")
print("\nTranscribing...")
result = model.transcribe("audio/audio.wav", fp16=False, verbose=False)

if not os.path.isdir("output"):
        os.mkdir("output")
with open("output/output.txt", "w") as f:
    f.write(result["text"])

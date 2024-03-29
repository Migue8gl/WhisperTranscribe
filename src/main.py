import os
import sys

from faster_whisper import WhisperModel


def run_faster_whisper(model_name):
    model = WhisperModel(model_name, download_root="models")
    print("\nTranscribing...")
    result = ''
    segments, _ = model.transcribe("audio/audio.wav")
    for segment in segments:
        result += "%s" % (segment.text)
    return result


def transcribe_audio(model_name):
    result = ''
    try:
        if not os.path.isdir("audio"):
            os.mkdir("audio")
        print('Listening... Press Ctrl+C to stop')
        os.system(
            'arecord -vv --format=cd -t wav audio/audio.wav > /dev/null 2>&1')
    except KeyboardInterrupt:
        pass

    result = run_faster_whisper(model_name)

    if not os.path.isdir("output"):
        os.mkdir("output")
    with open("output/output.txt", "w") as f:
        f.write(result)

    print("\n----- RESULT SAVED IN OUTPUT DIR -----\n")
    print(result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        model_name = 'large-v3'  # Default model name
    else:
        model_name = sys.argv[1]
    transcribe_audio(model_name)

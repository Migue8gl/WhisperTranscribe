import time

import whisper
from faster_whisper import WhisperModel


def run_faster_whisper(model_name):
    model = WhisperModel(model_name, download_root="models")
    print("\nTranscribing...")
    result = ''
    segments, _ = model.transcribe("audio/audio.wav")
    for segment in segments:
        result += "%s" % (segment.text)
    return result


def run_whisper(model_name):
    model = whisper.load_model(model_name, download_root="models")
    result = model.transcribe("audio/audio.wav", verbose=False, fp16=False)
    return result['text']


def measure_execution_time(name, func, model_name='large-v3', samples=1):
    start = time.time()
    for _ in range(samples):
        _ = func(model_name)
    end = time.time()

    print("%s: %f seconds" % (name, (end - start) / samples))


def main():
    measure_execution_time("faster_whisper", run_faster_whisper)
    measure_execution_time("whisper", run_whisper)


if __name__ == "__main__":
    main()

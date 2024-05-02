import os
import sys

from faster_whisper import WhisperModel

models = {
    't': 'tiny',
    's': 'small',
    'b': 'base',
    'm': 'medium',
    'l': 'large-v3',
}


def run_faster_whisper(model_name):
    model = WhisperModel(model_name, download_root="models")
    print("\nTranscribing...")
    result = ''
    segments, _ = model.transcribe("audio/audio.wav")
    for segment in segments:
        result += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end,
                                             segment.text)
    return result


def transcribe_audio(model_name, headphones=False):
    result = ''
    try:
        if not os.path.isdir("audio"):
            os.mkdir("audio")
        print('Listening... Press Ctrl+C to stop')
        if headphones:
            os.system(
                'parec --device=alsa_output.usb-Device_xxx-xxx.analog-stereo.monitor --format=s16le | ffmpeg -y -f s16le -ar 44100 -ac 2 -i - -acodec pcm_s16le audio/audio.wav > /dev/null 2>&1'
            )
        else:
            os.system(
                'parec --device=alsa_input.pci-0000_00_1f.3.analog-stereo --format=s16le | ffmpeg -y -f s16le -ar 44100 -ac 2 -i - -acodec pcm_s16le audio/audio.wav > /dev/null 2>&1'
            )
    except KeyboardInterrupt:
        pass

    result = run_faster_whisper(model_name)

    output_dir = "output"
    output_file = os.path.join(output_dir, "output_0.txt")

    if not os.path.isdir(output_dir):
    	os.mkdir(output_dir)

    # Find the next available output file
    i = 1
    while os.path.exists(output_file):
    	i += 1
    	output_file = os.path.join(output_dir, f"output_{i}.txt")

    with open(output_file, "w") as f:
    	f.write(result)

    print("\n----- RESULT SAVED IN OUTPUT DIR -----\n")
    print(result)


if __name__ == "__main__":
    args = sys.argv[1:]  # Skip the script name
    kwargs = {}
    for i in range(len(args)):
        if args[i].startswith('-'):
            if i + 1 < len(args) and not args[i + 1].startswith('-'):
                kwargs[args[i]] = args[i + 1]
            else:
                kwargs[args[i]] = None

    model_name = kwargs.get('-m',
                            'm')  # Get model name from command line arguments
    headphones = bool(kwargs.get(
        '-h', False))  # Get headphones value from command line arguments

    transcribe_audio(models[model_name], headphones)

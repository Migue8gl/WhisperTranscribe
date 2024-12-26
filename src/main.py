import argparse
import os
import re
import threading
import wave
from queue import Queue
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import yt_dlp
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI

if os.path.exists(".env"):
    load_dotenv(".env")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# Mapping for model size abbreviations
models = {
    "t": "tiny",
    "s": "small",
    "b": "base",
    "m": "medium",
    "l": "large-v3",
    "d": "distil-large-v2",
}

# For yt URL detection
youtube_regex = r"^https?:\/\/(?:www\.)?(youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)[\w\-]{11}(?:[&?][\w=]*)*$"


def record_audio(
    filename: str,
    headphones: bool = False,
    duration: Optional[int] = None,
    samplerate: int = 44100,
):
    """
    Record audio using sounddevice's InputStream for more efficient continuous recording.

    Parameters:
    - filename (str): Path to save the recorded audio file.
    - headphones (bool): If True, tries to use the WH-CH720N device for recording.
    - duration (int): Recording duration in seconds; if None, recording will continue until stopped.
    - samplerate (int): Sample rate for recording; if incompatible, falls back to device default.
    """
    try:
        # List available devices
        devices = sd.query_devices()
        device_id = None

        # Select device based on headphones parameter
        if headphones:
            # Loop through devices to find 'WH-CH720N'
            for i, device in enumerate(devices):
                if (
                    "wh-ch720n" in device["name"].lower()
                    and device["max_output_channels"] >= 2
                ):
                    device_id = i
                    break
            if device_id is None:
                raise ValueError("WH-CH720N headphones not found for recording")
        else:
            try:
                device_id = sd.default.device[0]
            except AttributeError:
                raise ValueError("No default input device set in sounddevice")

        # Fetch device info for channel and sample rate verification
        device_info = sd.query_devices(device_id, "input")
        default_samplerate = int(device_info["default_samplerate"])

        if samplerate != default_samplerate:
            print(
                f"Specified sample rate {samplerate} Hz not supported. Using default: {default_samplerate} Hz",
            )
            samplerate = default_samplerate

        channels = min(2, device_info["max_input_channels"])
        print(f"Recording on device: {devices[device_id]['name']} at {samplerate} Hz")

        # Create a queue to store the recorded data
        q: Queue = Queue()
        stop_event = threading.Event()
        recorded_data = []

        # Callback function to handle incoming audio data
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            q.put(indata.copy())

        # Create an input stream
        with sd.InputStream(
            device=device_id,
            channels=channels,
            samplerate=samplerate,
            dtype=np.int16,
            callback=audio_callback,
        ):
            print("Recording... Press Ctrl+C to stop.")

            try:
                # Handle fixed duration recording
                if duration:
                    total_frames = int(duration * samplerate)
                    collected_frames = 0

                    while collected_frames < total_frames:
                        data = q.get()
                        recorded_data.append(data)
                        collected_frames += len(data)

                    # Trim excess data
                    final_data = np.concatenate(recorded_data)
                    final_data = final_data[:total_frames]

                # Handle continuous recording
                else:
                    while not stop_event.is_set():
                        data = q.get()
                        recorded_data.append(data)

            except KeyboardInterrupt:
                print("\nRecording stopped by user.")
                stop_event.set()

            finally:
                # Combine all recorded data
                final_data = np.concatenate(recorded_data)

                # Save the recording
                sf.write(filename, final_data, samplerate)
                print(f"Recording saved to {filename}")

    except Exception as e:
        print(f"Error recording audio: {str(e)}")
        raise


def split_audio(input_file: str, chunk_duration: int = 30) -> List[Dict[str, Any]]:
    """
    Split an audio file into chunks of specified duration.

    Args:
        input_file (str): Path to the input WAV file.
        chunk_duration (int): Duration of each chunk in seconds.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing file paths and start times of each chunk.
    """
    with wave.open(input_file, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()

        frames_per_chunk = int(chunk_duration * framerate)
        chunks = []
        chunk_id = 0

        os.makedirs("audio/chunks", exist_ok=True)

        while True:
            frames = wf.readframes(frames_per_chunk)
            if not frames:
                break

            chunk_file = f"audio/chunks/chunk_{chunk_id}.wav"
            with wave.open(chunk_file, "wb") as chunk_wf:
                chunk_wf.setnchannels(n_channels)
                chunk_wf.setsampwidth(sampwidth)
                chunk_wf.setframerate(framerate)
                chunk_wf.writeframes(frames)

            chunks.append({"file": chunk_file, "start_time": chunk_id * chunk_duration})
            chunk_id += 1

    return chunks


def run_faster_whisper(model_name: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Transcribe audio chunks using the Faster Whisper model.

    Args:
        model_name (str): Model name or path for Faster Whisper model.
        chunks (List[Dict[str, Any]]): List of audio chunks with file paths and start times.

    Returns:
        str: The complete transcription result.
    """
    model = WhisperModel(model_name, download_root="models", compute_type="float16")
    print("\nTranscribing in batches...")

    full_result = ""
    for chunk in chunks:
        print(f"Processing chunk starting at {chunk['start_time']}s...")
        segments, _ = model.transcribe(chunk["file"])

        for segment in segments:
            adjusted_start = segment.start + chunk["start_time"]
            adjusted_end = segment.end + chunk["start_time"]
            full_result += (
                f"[{adjusted_start:.2f}s -> {adjusted_end:.2f}s] {segment.text}\n"
            )

        os.remove(chunk["file"])

    return full_result


def download_audio(audio_url: str):
    ydl_opts = {
        "format": "bestaudio/best",  # Download the best audio available
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",  # Use ffmpeg to extract audio
                "preferredcodec": "wav",  # Output audio codec (WAV format)
            },
        ],
        "outtmpl": "./audio/audio.%(ext)s",  # Store audio in the './audio' directory
        "cookiesfrombrowser": ("firefox",),
    }
    output_file = "./audio/audio.wav"
    if os.path.exists(output_file):
        os.remove(output_file)
    print(f"Existing file {output_file} has been deleted.")

    try:
        # Initialize yt-dlp and download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.download([audio_url])

        # Check the result of the download
        if result == 0:
            print("Download and conversion completed successfully!")
        else:
            print(f"Download failed with error code {result}")
    except Exception as e:
        print(f"An error occurred: {e}")


def transcribe_audio(
    model_name: str,
    headphones: bool = False,
    chunk_duration: int = 30,
    summarize: bool = False,
    load_audio: Optional[str] = None,
) -> None:
    """
    Record, split, and transcribe audio, saving the result to a text file.

    Args:
        model_name (str): The model size identifier (e.g., 'tiny', 'small').
        headphones (bool): Whether to use headphones as the audio input.
        chunk_duration (int): Duration in seconds for each audio chunk.
        load_audio (Optional[str]): Audio file to transcribe. If not provided, recording mode is set.
    """
    os.makedirs("audio", exist_ok=True)
    audio_file = "audio/audio.wav"

    if load_audio is None:
        # Record audio using sounddevice
        record_audio(audio_file, headphones)
    else:
        # Load audio file instead of recording
        if not isinstance(load_audio, str):
            raise Exception(
                "Audio provided is neither a path to an audio file or a link to download it",
            )
        if bool(re.match(youtube_regex, load_audio)):
            download_audio(load_audio)
            audio_file = audio_file
        else:
            audio_file = load_audio

    # Split audio and transcribe chunks
    chunks = split_audio(audio_file, chunk_duration)
    result = run_faster_whisper(model_name, chunks)

    # Save result to file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "output_0.txt")
    i = 1
    while os.path.exists(output_file):
        output_file = os.path.join(output_dir, f"output_{i}.txt")
        i += 1

    with open(output_file, "w") as f:
        f.write(result)

    print("\n----- RESULT SAVED IN OUTPUT DIR -----\n")
    print(result)

    if summarize:
        prompt_file = "./prompts/prompt_schema_md.txt"
        output_resume = generate_resume(prompt_file, result)

        match = re.search(r"title=(.*)\n", output_resume)
        if match:
            title = match.group(1).strip().lower() + ".md"
            output_resume = re.sub(r"title=.*\n", "", output_resume)
        else:
            title = f"resume_output_{i}.md"

        output_resume_dir = os.path.join(output_dir, title)

        print("\n----- RESUME SAVED IN OUTPUT DIR -----\n")
        with open(output_resume_dir, "w") as f:
            f.write(output_resume)
            

def generate_resume(prompt_file: str, transcription: str):
    """
    Loads a prompt template from a file, combines it with a transcription,
    and generates a summary using OpenAI's API.

    Args:
        prompt_file (str): Path to the input prompt template file.
        transcription (str): Transcription text to analyze.

    Returns:
        Optional[str]: The generated summary or None if an error occurs.
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Read the prompt template from the file
        with open(prompt_file, encoding="utf-8") as file:
            prompt_template = file.read()

        # Replace the placeholder with the transcription
        updated_prompt = prompt_template.replace("[transcription here]", transcription)


        # Create the messages array for the API call
        messages = [
            {
                "role": "system",
                "content": "Eres un asistente especializado en analisis y sintesis de conversaciones. Debes responder de forma esructurada y directamente como te ordende el usuario en su esquema",
            },
            {"role": "user", "content": updated_prompt},
        ]

        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.15,
            max_tokens=4000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        # Extract and return the response text
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return None

    except FileNotFoundError:
        print(f"Error: Prompt file '{prompt_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None


def main() -> None:
    """
    Main function to parse arguments and execute transcription.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Faster Whisper.",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=models.keys(),
        default="m",
        help="Specify the model size: 't' for tiny, 's' for small, 'b' for base, 'm' for medium, 'l' for large-v3",
    )
    parser.add_argument(
        "-c",
        "--chunk_duration",
        type=int,
        default=30,
        help="Duration of each audio chunk in seconds (default: 30)",
    )
    parser.add_argument(
        "-d",
        "--headphones",
        action="store_true",
        help="Use headphones as audio input device",
    )
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        help="Load audio file instead of recording",
    )
    parser.add_argument(
        "-s",
        "--summarize",
        action="store_true",
        help="Use LLM summarization functionality",
    )
    args = parser.parse_args()

    transcribe_audio(
        models[args.model],
        args.headphones,
        args.chunk_duration,
        args.summarize,
        args.load,
    )


if __name__ == "__main__":
    main()

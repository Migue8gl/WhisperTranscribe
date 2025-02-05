import argparse
import os
import re
import threading
import wave
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

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

models = {
    "t": "tiny",
    "s": "small",
    "b": "base",
    "m": "medium",
    "l": "large-v3",
    "d": "distil-large-v2",
}

youtube_regex = r"^https?:\/\/(?:www\.)?(youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)[\w\-]{11}(?:[&?][\w=]*)*$"

def list_input_devices():
    """List all available input devices with their IDs"""
    devices = sd.query_devices()
    print("\nAvailable input devices:")
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            print(f"ID {i}: {device['name']} (Input channels: {device['max_input_channels']})")

def get_supported_sample_rates(device_id: int):
    """Get supported sample rates for a device"""
    common_rates = [48000, 44100, 32000, 24000, 22050, 16000, 11025, 8000]
    supported_rates = []
    
    for rate in common_rates:
        try:
            sd.check_input_settings(device=device_id, samplerate=rate)
            supported_rates.append(rate)
        except sd.PortAudioError:
            continue
    return supported_rates

def record_audio(
    filename: str,
    device_id: Optional[int] = None,
    duration: Optional[int] = None,
    samplerate: int = 44100,
):
    try:
        devices = sd.query_devices()
        
        # Device validation
        if device_id is not None:
            if device_id >= len(devices):
                raise ValueError(f"Device ID {device_id} does not exist (max ID: {len(devices)-1})")
            device_info = devices[device_id]
        else:
            device_id = sd.default.device[0]
            device_info = sd.query_devices(device_id)

        # Input capability check
        if device_info["max_input_channels"] <= 0:
            raise ValueError(f"Device {device_id} ({device_info['name']}) is not an input device")

        # Sample rate validation
        supported_rates = get_supported_sample_rates(device_id)
        if not supported_rates:
            raise ValueError(f"No supported sample rates found for device {device_id}")

        # Use device's default rate if available, otherwise closest supported rate
        device_default_rate = int(device_info["default_samplerate"])
        if samplerate not in supported_rates:
            closest_rate = min(supported_rates, key=lambda x: abs(x - samplerate))
            print(f"Warning: {samplerate}Hz not supported. Using closest supported rate: {closest_rate}Hz")
            samplerate = closest_rate

        # Final device configuration
        channels = device_info["max_input_channels"]
        print(f"\nSelected device: {device_info['name']}")
        print(f"Channels: {channels}, Sample rate: {samplerate}Hz")

        q = Queue()
        stop_event = threading.Event()
        recorded_data = []

        def audio_callback(indata, frames, time, status):
            q.put(indata.copy())

        with sd.InputStream(
            device=device_id,
            channels=channels,
            samplerate=samplerate,
            dtype=np.int16,
            callback=audio_callback,
        ):
            print("Recording... Press Ctrl+C to stop")
            try:
                if duration:
                    total_frames_needed = duration * samplerate
                    current_frames = 0
                    while current_frames < total_frames_needed:
                        chunk = q.get()
                        chunk_frames = chunk.shape[0]
                        if current_frames + chunk_frames > total_frames_needed:
                            needed = total_frames_needed - current_frames
                            recorded_data.append(chunk[:needed])
                            current_frames += needed
                        else:
                            recorded_data.append(chunk)
                            current_frames += chunk_frames
                else:
                    while not stop_event.is_set():
                        try:
                            chunk = q.get(timeout=1)
                            recorded_data.append(chunk)
                        except Empty:
                            continue

            except KeyboardInterrupt:
                stop_event.set()
                print("\nRecording stopped by user")

            finally:
                while True:
                    try:
                        recorded_data.append(q.get_nowait())
                    except Empty:
                        break

                if recorded_data:
                    audio_data = np.concatenate(recorded_data)
                    if duration:
                        max_frames = duration * samplerate
                        audio_data = audio_data[:max_frames]
                    sf.write(filename, audio_data, samplerate)
                    print(f"Audio saved to {filename}")
                else:
                    print("No audio data recorded.")

    except Exception as e:
        print(f"Recording error: {str(e)}")
        raise
        
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

# Rest of the functions remain the same as previous versions
def split_audio(input_file: str, chunk_duration: int = 30) -> List[Dict[str, Any]]:
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
    model = WhisperModel(model_name, download_root="models")
    print("\nTranscribing in batches...")
    full_result = ""
    for chunk in chunks:
        print(f"Processing chunk starting at {chunk['start_time']}s...")
        segments, _ = model.transcribe(chunk["file"])
        for segment in segments:
            adjusted_start = segment.start + chunk["start_time"]
            adjusted_end = segment.end + chunk["start_time"]
            full_result += f"[{adjusted_start:.2f}s -> {adjusted_end:.2f}s] {segment.text}\n"
        os.remove(chunk["file"])
    return full_result

def transcribe_audio(
    model_name: str,
    device_id: Optional[int] = None,
    chunk_duration: int = 30,
    summarize: bool = False,
    load_audio: Optional[str] = None,
    output_name: Optional[str] = None
) -> None:
    os.makedirs("audio", exist_ok=True)
    audio_file = "audio/audio.wav"

    if load_audio is None:
        record_audio(audio_file, device_id=device_id)
    else:
        if not isinstance(load_audio, str):
            raise Exception("Invalid audio source provided")
        if bool(re.match(youtube_regex, load_audio)):
            download_audio(load_audio)
            audio_file = audio_file
        else:
            audio_file = load_audio

    chunks = split_audio(audio_file, chunk_duration)
    result = run_faster_whisper(model_name, chunks)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "output_0.txt")
    i = 1
    if output_name:
        output_file = output_name
    else:
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
        title = match.group(1).strip().lower() + ".md" if match else f"resume_output_{i}.md"
        output_resume = re.sub(r"title=.*\n", "", output_resume) if match else output_resume

        output_resume_dir = os.path.join(output_dir, title)
        with open(output_resume_dir, "w") as f:
            f.write(output_resume)
        print("\n----- RESUME SAVED IN OUTPUT DIR -----\n")

def generate_resume(prompt_file: str, transcription: str):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        with open(prompt_file, encoding="utf-8") as file:
            prompt_template = file.read()
        updated_prompt = prompt_template.replace("[transcription here]", transcription)
        messages = [
            {"role": "system", "content": "You are a specialized conversation analysis assistant."},
            {"role": "user", "content": updated_prompt},
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=4000,
        )
        return response.choices[0].message.content if response.choices else None
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None

def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio using Faster Whisper.")
    parser.add_argument("-m", "--model", choices=models.keys(), default="m",
                      help="Model size: t(tiny), s(small), b(base), m(medium), l(large-v3), d(distil-large-v2)")
    parser.add_argument("-c", "--chunk_duration", type=int, default=30,
                      help="Duration of audio chunks in seconds (default: 30)")
    parser.add_argument("-d", "--device", type=int,
                      help="Specify input device ID (use --list-devices to see available IDs)")
    parser.add_argument("--list-devices", action="store_true",
                      help="List available input devices and exit")
    parser.add_argument("-s", "--summarize", action="store_true",
                      help="Generate AI summary of transcription")
    parser.add_argument("-l", "--load", type=str,
                      help="Load audio file/YouTube URL instead of recording")
    parser.add_argument("-n", "--name", type=str,
                      help="Output name for transcription file")

    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    transcribe_audio(
        models[args.model],
        device_id=args.device,
        chunk_duration=args.chunk_duration,
        summarize=args.summarize,
        load_audio=args.load,
        output_name=args.name
    )

if __name__ == "__main__":
    main()

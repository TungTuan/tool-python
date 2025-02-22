
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import FileResponse, JSONResponse
import os
import torch
from TTS.api import TTS
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import timedelta
import pydub
import numpy as np
import wave
# from utils import have_pyrubberband
# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class VitsConfig:
    def __init__(self, sample_rate=22050, noise_scale=0.667, length_scale=1.0):
        """
        VITS configuration for speech generation.
        Parameters:
            - sample_rate: The audio sample rate.
            - speaker_id: ID for the speaker in multi-speaker models.
            - noise_scale: Controls voice expressiveness.
            - length_scale: Adjusts speech speed.
        """
        self.sample_rate = sample_rate
        self.noise_scale = noise_scale
        self.length_scale = length_scale

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/generate/")
def generate_text(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=50)
    return {"response": tokenizer.decode(output[0], skip_special_tokens=True)}

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
whisper_model = whisper.load_model("turbo")

# Directory for storing output files
OUTPUT_DIR = "output_files"
SRT_DIR = "SrtFiles"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SRT_DIR, exist_ok=True)

def get_wave_header(frame_input=b"", channels=1, sample_width=2, sample_rate=24000):
    """
    Generates the wave header and appends the frame input for streaming WAV file.
    """
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()

def prepare_speech(sr=24000):
    """
    Prepares speech with the specified sample rate.
    """
    return get_wave_header(sample_rate=sr)

def get_no_audio(return_as_byte=True, return_nonbyte_as_file=False, sr=None):
    """
    Generates a non-audio response.
    """
    if return_as_byte:
        return b""
    else:
        if return_nonbyte_as_file:
            return None
        else:
            assert sr is not None
            return sr, np.array([]).astype(np.int16)

def combine_audios(audios, audio=None, channels=1, sample_width=2, sr=24000, expect_bytes=True, verbose=False):
    """
    Combines multiple audio files or streams into one.
    """
    no_audio = get_no_audio(sr=sr)
    have_audio = any(x not in [no_audio, None, ''] for x in audios) or audio not in [no_audio, None, '']
    if not have_audio:
        return no_audio

    if audio or audios:
        is_bytes = expect_bytes
        if audios:
            is_bytes |= isinstance(audios[0], (bytes, bytearray))
        if audio:
            is_bytes |= isinstance(audio, (bytes, bytearray))
        assert audio is None or isinstance(audio, (bytes, bytearray))
        combined_wav = pydub.AudioSegment.empty()
        for x in audios:
            if x is not None:
                s = io.BytesIO(x) if is_bytes else x
                combined_wav += pydub.AudioSegment.from_raw(s, sample_width=sample_width, frame_rate=sr, channels=channels)
        if audio is not None:
            s = io.BytesIO(audio) if is_bytes else audio
            combined_wav += pydub.AudioSegment.from_raw(s, sample_width=sample_width, frame_rate=sr, channels=channels)
        if is_bytes:
            combined_wav = combined_wav.export(format='raw').read()
        return combined_wav
    return audio

def chunk_speed_change(chunk, sr, tts_speed=1.0):
    """
    Adjusts the speed of the audio.
    - For slowing down, `tts_speed` < 1.0.
    - For speeding up, `tts_speed` > 1.0.
    """
    if tts_speed == 1.0:
        return chunk

    # if have_pyrubberband:
    #     import pyrubberband as pyrb
    #     chunk = pyrb.time_stretch(chunk, sr, tts_speed)
    #     chunk = (chunk * 32767).astype(np.int16)
    #     return chunk

    if tts_speed < 1.0:
        return chunk

    # Speed-up using pydub
    s = io.BytesIO(chunk)
    channels = 1
    sample_width = 2
    audio = pydub.AudioSegment.from_raw(s, sample_width=sample_width, frame_rate=sr, channels=channels)
    chunk = pydub_to_np(speedup(audio, tts_speed, 150))
    return chunk

def pydub_to_np(audio: pydub.AudioSegment) -> (np.ndarray, int):
    """
    Converts a pydub audio segment to numpy array with int16 data type.
    """
    return np.array(audio.get_array_of_samples(), dtype=np.int16).reshape((-1, audio.channels))

@app.post("/generate-audio-news/")
async def generate_audio(
    text: str = Form(...),
    speaker_wav: UploadFile = None,
    language: str = Form("en"),
    sample_rate: int = Form(22050),
    noise_scale: float = Form(0.667),
    length_scale: float = Form(1.0),
):
    """
    Generate audio file from text.
    Parameters:
        - text: The input text to convert to speech.
        - speaker_wav: A speaker wav file for voice cloning (optional).
        - language: Language of the text (default is English).
        - sample_rate: Audio sample rate (default: 22050).
        - speaker_id: Speaker ID for multi-speaker models (default: 0).
        - noise_scale: Voice expressiveness (default: 0.667).
        - length_scale: Speech speed (default: 1.0).
    Returns:
        - A downloadable audio file.
    """
    try:
        # Save the uploaded speaker wav file (if provided)
        speaker_wav_path = None
        if speaker_wav:
            speaker_wav_path = os.path.join(OUTPUT_DIR, speaker_wav.filename)
            with open(speaker_wav_path, "wb") as f:
                f.write(await speaker_wav.read())

        # Generate output audio file path
        output_file_path = os.path.join(OUTPUT_DIR, "output.wav")

        # Generate speech
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav_path,
            language=language,
            file_path=output_file_path,
            # Pass parameters only if required by the TTS method
            sample_rate=sample_rate if "sample_rate" in tts.tts_to_file.__code__.co_varnames else None,
            noise_scale=noise_scale if "noise_scale" in tts.tts_to_file.__code__.co_varnames else None,
            length_scale=length_scale if "length_scale" in tts.tts_to_file.__code__.co_varnames else None,
        )

        # Return the generated audio file
        return FileResponse(output_file_path, media_type="audio/wav", filename="output.wav")

    except Exception as e:
        return {"error": str(e)}

@app.post("/generate-audio/slow/")
async def generate_audio(
    text: str = Form(...),
    speaker_wav: UploadFile = None,
    language: str = Form("en"),
    speed: float = Form(0.3)  # 0.3 means 70% slower (0.3 is 30% of original speed)
):
    """
    Generate an audio file from text with the option to slow down the voice.
    Parameters:
        - text: The input text to convert to speech.
        - speaker_wav: A speaker wav file for voice cloning (optional).
        - language: Language of the text (default is English).
        - speed: Speed of the speech (default is 0.3 for 70% slower).
    Returns:
        - A downloadable audio file.
    """
    try:
        # Save the uploaded speaker wav file (if provided)
        speaker_wav_path = None
        if speaker_wav:
            speaker_wav_path = os.path.join(OUTPUT_DIR, speaker_wav.filename)
            with open(speaker_wav_path, "wb") as f:
                f.write(await speaker_wav.read())

        # Generate output audio file path
        output_file_path = os.path.join(OUTPUT_DIR, "output.wav")

        # Generate speech (you might need to modify tts.tts_to_file if needed)
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav_path,
            language=language,
            file_path=output_file_path,
        )

        # Read the generated audio file into a byte array
        with open(output_file_path, "rb") as f:
            audio_bytes = f.read()

        # Slow down the audio (70% slower means 0.3 speed)
        audio_chunk = chunk_speed_change(audio_bytes, sr=24000, tts_speed=speed)

        # Save the modified audio to a new file
        modified_output_path = os.path.join(OUTPUT_DIR, "output_slowed.wav")
        with open(modified_output_path, "wb") as f:
            f.write(audio_chunk)

        # Return the slowed-down audio file
        return FileResponse(modified_output_path, media_type="audio/wav", filename="output_slowed.wav")

    except Exception as e:
        return {"error": str(e)}


@app.post("/generate-audio/")
async def generate_audio(
    text: str = Form(...),
    speaker_wav: UploadFile = None,
    language: str = Form("en")
):
    """
    Generate audio file from text.
    Parameters:
        - text: The input text to convert to speech.
        - speaker_wav: A speaker wav file for voice cloning (optional).
        - language: Language of the text (default is English).
    Returns:
        - A downloadable audio file.
    """
    try:
        # Save the uploaded speaker wav file (if provided)
        speaker_wav_path = None
        if speaker_wav:
            speaker_wav_path = os.path.join(OUTPUT_DIR, speaker_wav.filename)
            with open(speaker_wav_path, "wb") as f:
                f.write(await speaker_wav.read())

        # Generate output audio file path
        output_file_path = os.path.join(OUTPUT_DIR, "output.wav")

        # Generate speech
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav_path,
            language=language,
            file_path=output_file_path,
        )

        # Return the generated audio file
        return FileResponse(output_file_path, media_type="audio/wav", filename="output.wav")

    except Exception as e:
        return {"error": str(e)}

@app.post("/transcribe-audio/")
async def transcribe_audio(file: UploadFile):
    """
    Transcribe audio file to text.
    Parameters:
        - file: The audio file to be transcribed.
    Returns:
        - Detected language and transcribed text.
    """
    try:
        # Save the uploaded audio file
        audio_file_path = os.path.join(OUTPUT_DIR, file.filename)
        with open(audio_file_path, "wb") as f:
            f.write(await file.read())

        # Load audio and preprocess
        audio = whisper.load_audio(audio_file_path)
        audio = whisper.pad_or_trim(audio)

        # Generate log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels).to(model.device)

        # Detect the language
        _, probs = whisper_model.detect_language(mel)
        detected_language = max(probs, key=probs.get)

        # Decode the audio
        options = whisper.DecodingOptions()

        result = whisper.decode(whisper_model, mel, options)

        srt_content = []
        for i, segment in enumerate(result["segments"], start=1):
            start_time = format_time_customize(segment["start"])
            end_time = format_time_customize(segment["end"])
            text = segment["text"].strip()
            srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")

        # Save SRT file
        srt_file_path = os.path.join(OUTPUT_DIR, f"{file.filename.rsplit('.', 1)[0]}.srt")
        with open(srt_file_path, "w", encoding="utf-8") as srt_file:
            srt_file.writelines(srt_content)

        # Return detected language and transcription
        return {
            "detected_language": detected_language,
            "transcription": result.text,
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def format_time_ass(seconds: float) -> str:
    """
    Format time in seconds to ASS timestamp format (h:mm:ss.cs).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    centiseconds = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02}:{int(seconds):02}.{centiseconds:02}"


@app.post("/transcribe-audio-ass/")
async def transcribe_audio(file: UploadFile):
    """
    Transcribe audio file to text and generate ASS subtitles.
    Parameters:
        - file: The audio file to be transcribed.
    Returns:
        - Detected language and transcribed text.
    """
    try:
        # Save the uploaded audio file
        audio_file_path = os.path.join(OUTPUT_DIR, file.filename)
        with open(audio_file_path, "wb") as f:
            f.write(await file.read())

        # Transcribe audio and get detailed segments
        result = whisper_model.transcribe(audio_file_path, task="transcribe", language=None)  # Add options as needed
        detected_language = result["language"]  # Detected language

        # Generate ASS header and styles
        ass_header = """
        [Script Info]
        Title: Transcription
        ScriptType: v4.00+
        Collisions: Normal
        PlayDepth: 0
        Timer: 100.0000

        [V4+ Styles]
        Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
        Style: Default,Arial,36,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,1,1,0,2,10,10,10,1
        """

        # Generate dialogue section
        ass_dialogues = ["[Events]\n", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"]
        for segment in result["segments"]:
            start_time = format_time_ass(segment["start"])
            end_time = format_time_ass(segment["end"])
            text = segment["text"].strip()
            ass_dialogues.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")

        # Save ASS file
        ass_file_path = os.path.join(OUTPUT_DIR, f"{file.filename.rsplit('.', 1)[0]}.ass")
        with open(ass_file_path, "w", encoding="utf-8") as ass_file:
            ass_file.write(ass_header)
            ass_file.writelines(ass_dialogues)

        # Return detected language and transcription
        return FileResponse(ass_file_path, media_type="text/plain", filename=os.path.basename(ass_file_path))

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)




def format_time_customize(seconds):
    """
    Format time in seconds to HH:MM:SS,SSS format for SRT.
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def transcribe_audio(path, filename):
    """
    Transcribe audio and return the SRT file path.
    Parameters:
        - path: The audio file path to transcribe.
        - filename: The filename to use for the output SRT file.
    Returns:
        - The path to the generated SRT file.
    """
    model = whisper.load_model("base")  # Load the Whisper model
    print("Whisper model loaded.")

    # Transcribe audio
    transcribe = model.transcribe(audio=path)
    segments = transcribe['segments']

    srt_filename = os.path.join(SRT_DIR, f"{filename.rsplit('.', 1)[0]}.srt")
    
    # Create the SRT file
    with open(srt_filename, 'w', encoding='utf-8') as srtFile:
        for segment in segments:
            # start_time = str(timedelta(seconds=int(segment['start'])))
            # end_time = str(timedelta(seconds=int(segment['end'])))
            start_time = format_time_customize(segment["start"])
            end_time = format_time_customize(segment["end"])
            text = segment['text']
            
            # Clean up text if it starts with a space
            text = text[1:] if text[0] == ' ' else text
            
            # Format segment as SRT
            segment_id = segment['id'] + 1
            segment_str = f"{segment_id}\n{start_time.replace('days', '').strip()} --> {end_time.replace('days', '').strip()}\n{text}\n\n"
            srtFile.write(segment_str)

    return srt_filename

@app.post("/transcribe-audio-srt/")
async def transcribe_audio_api(file: UploadFile = File(...)):
    """
    API endpoint to transcribe an uploaded audio file to an SRT file.
    Parameters:
        - file: The audio file to transcribe.
    Returns:
        - The generated SRT file.
    """
    try:
        # Save the uploaded audio file
        audio_file_path = os.path.join(OUTPUT_DIR, file.filename)
        with open(audio_file_path, "wb") as f:
            f.write(await file.read())

        # Transcribe audio and generate SRT file
        srt_filename = transcribe_audio(audio_file_path, file.filename)

        # Return the SRT file as a response
        return FileResponse(srt_filename, media_type="text/plain", filename=srt_filename)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
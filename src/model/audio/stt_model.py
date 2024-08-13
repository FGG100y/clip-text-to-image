import io
import platform
import numpy as np
import soundfile as sf

import speech_recognition as sr
from faster_whisper import WhisperModel

# Get rid of ALSA lib error messages in Linux ================================
if platform.system() == "Linux":
    from ctypes import CFUNCTYPE, c_char_p, c_int, cdll

    # Define error handler
    error_handler = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

    # Don't do anything if there is an error message
    def py_error_handler(filename, line, function, err, fmt):
        pass

    # Pass to C
    c_error_handler = error_handler(py_error_handler)
    asound = cdll.LoadLibrary("libasound.so")
    asound.snd_lib_error_set_handler(c_error_handler)
# Get rid of ALSA lib error messages in Linux END ============================


def load_faster_whisper():
    modeldir = "models/hfLLMs/faster-whisper-large-v3"
    model = WhisperModel(model_size_or_path=modeldir)  # local_files_only=True
    print("Load whisper model from:", modeldir)
    return model


def transcribe_fast(
    model, language="zh", duration=50, adjust_duration=5, verbose=False
):
    """Using faster-whisper"""

    # obtain audio from the microphone
    rec = sr.Recognizer()
    with sr.Microphone() as source:
        rec.adjust_for_ambient_noise(source, duration=adjust_duration)
        print(">>> It's faster-whisper listening, say something:")
        audio = rec.listen(source, phrase_time_limit=duration)

    # from speech_recognition().recognize_whisper: (well done)
    wav_bytes = audio.get_wav_data(convert_rate=16000)  # 16k for whisper
    wav_stream = io.BytesIO(wav_bytes)
    audio_array, sampling_rate = sf.read(wav_stream)
    audio_array = audio_array.astype(np.float32)

    text = ""
    segments, info = model.transcribe(audio_array, language=language)
    for segment in segments:
        text += segment.text
        if verbose:
            print(
                "[%.2fs -> %.2fs] %s"
                % (segment.start, segment.end, segment.text)
            )
        if verbose == 2:
            print("model transciption info:", info)
    return text

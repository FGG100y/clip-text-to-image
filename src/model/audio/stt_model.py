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
    """Using fast-whisper"""

    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=adjust_duration)
        print(">>> It's faster-whisper listening, say something:")
        audio = r.listen(source, phrase_time_limit=duration)

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


#  # TODO save wav of user for coqui-xtts cloning?
#  def transcribe(language="chinese", duration=5):
#      """fixed interval stt"""
#      # obtain audio from the microphone
#      r = sr.Recognizer()
#      with sr.Microphone() as source:
#          r.adjust_for_ambient_noise(source)
#          print("It's whisper listening, say something:")
#          audio = r.listen(source, phrase_time_limit=duration)
#
#      # speech_recognition using `whisper` (which load .cache/whisper/model-X.pt)
#      try:
#          text = r.recognize_whisper(audio, model="large", language=language)
#          print("Whisper thinks you said:", text)
#      except sr.UnknownValueError:
#          print("Whisper could not understand audio")
#      except sr.RequestError as e:  # noqa
#          print("Could not request results from Whisper")
#
#      return text
#
#
#  def listen_and_respond(prompt, language="zh-cn"):
#      """Listen for user input, transcribe it, and respond accordingly."""
#      r = sr.Recognizer()
#      # Continuously listen for user input until they confirm or cancel
#      with sr.Microphone() as source:
#          print("Listening...")
#          r.adjust_for_ambient_noise(source, duration=0.5)
#          audio = None
#
#          while True:
#              try:
#                  # Listen for user input and transcribe it
#                  audio = r.listen(source)
#                  text = r.recognize_whisper(
#                      audio, model="large", language=language
#                  )
#                  break
#              except sr.WaitTimeoutError:
#                  print("Timed out waiting for user input.")
#              except sr.UnknownValueError:
#                  print("Unable to recognize speech.")
#              except sr.RequestError as e:
#                  print(f"Speech recognition service error: {e}")
#
#      # Check if the user confirmed or canceled the interaction
#      if text.lower() in {"yes", "yeah", "sure", "ok", "okay"}:
#          print("User confirmed.")
#          return True
#      elif text.lower() in {"no", "nope", "nevermind", "cancel", "quit"}:
#          print("User canceled.")
#          return False
#      else:
#          print(f"Unrecognized response: {text}")
#          return
#
#
#  if __name__ == "__main__":
#      # Example usage
#      listen_and_respond("Are you ready to begin? Say yes or no.")
#
#      #  # create a speech recognition object
#      #  recognizer = sr.Recognizer()
#      #
#      #  with sr.Microphone() as source:
#      #      print("Say something!")
#      #      # read the audio data from the default microphone
#      #      audio_data = recognizer.record(source, duration=5)

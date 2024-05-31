import speech_recognition as sr
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

class ASR:
  def __init__(self):
    # self.recognizer = sr.Recognizer()
    self.client = OpenAI()

  def speech_to_text(self, audio_file):
    audio = open(audio_file, "rb")
    transcription = self.client.audio.transcriptions.create(
      model="whisper-1", 
      file=audio
    )
    return transcription.text
  
  # def speech_to_text(self, audio_file: str, language: str = "ko-KR") -> str:
  #   """
  #   Recognize the audio file.
  #   Args:
  #     audio_file (str): The audio file to recognize.
  #     language (str): The language of the audio file. e.g. "ko-KR", "en-US", "ja-JP"
  #   """
  #   with sr.AudioFile(audio_file) as source:
  #     audio = self.recognizer.record(source)
  #     text = None
  #     try:
  #       text = self.recognizer.recognize_google(audio, language=language)
  #     except sr.UnknownValueError:
  #       text = ""
  #     except sr.RequestError as e:
  #       text = ""
  #     return text
  
if __name__ == "__main__":
  asr = ASR()
  # Test speech to text
  print(asr.speech_to_text("./contents/voice.wav", language="ko-KR"))
from openai import OpenAI
import dotenv

dotenv.load_dotenv()
client = OpenAI()

audio_file= open("contents/user_voice.wav", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)
print(transcription.text)
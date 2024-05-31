from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import ffmpeg #ffmpeg-python
import dotenv
dotenv.load_dotenv()

from agent import Agent

app = FastAPI()

# CORS
origins = [
  "*"
]
app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.mount("/contents", StaticFiles(directory="contents"), name="contents")

# Agent Definition
agent = Agent(ai_name="김민지")


@app.get("/")
async def main():
  return {"message": "Hello World"}

@app.post("/query")
async def query(voice: UploadFile = File()):
  # 1. 유저 목소리 저장
  file_path = f"./contents/user_voice.webm"
  with open(file_path, "wb") as f:
    f.write(await voice.read())
  f.close()
  
  # x. 포맷 변환 (webm -> wav)
  (
    ffmpeg
    .input(file_path)
    .output("./contents/user_voice.wav", loglevel="quiet")
    .run(overwrite_output=True)
    
  )
  file_path = "./contents/user_voice.wav"
  
  # x. 목소리를 텍스트로 변환
  input_text = agent.speech_to_text(file_path)
  
  # x. 인식 실패 시 응답안함.
  if(input_text.strip() == ""):
    return {"status": False, "input_text": "", "output_text": "", "output_voice": ""}

  # x. 적절한 응답 추론
  output_text = agent.query(user_name="나영채", user_input=input_text)
  # output_text = "테스트 출력입니다."
  print(output_text)
  
  # x. 텍스트를 목소리로 변환 및 저장
  agent.text_to_speech(output_text, output_file="contents/chloe_voice.wav")

  # x. 결과 반환
  return {"status": True, "input_text": input_text, "output_text": output_text, "output_voice": "chloe_voice.wav"}

if __name__ == "__main__":
  print(r'''
 ____       __          ___                                    __            __            _     
/\  _`\    /\ \        /\_ \                                 /'__`\        /'__`\        /' \    
\ \ \/\_\  \ \ \___    \//\ \      ___      __              /\ \/\ \      /\ \/\ \      /\_, \   
 \ \ \/_/_  \ \  _ `\    \ \ \    / __`\  /'__`\            \ \ \ \ \     \ \ \ \ \     \/_/\ \  
  \ \ \L\ \  \ \ \ \ \    \_\ \_ /\ \L\ \/\  __/             \ \ \_\ \ __  \ \ \_\ \ __    \ \ \ 
   \ \____/   \ \_\ \_\   /\____\\ \____/\ \____\             \ \____//\_\  \ \____//\_\    \ \_\
    \/___/     \/_/\/_/   \/____/ \/___/  \/____/              \/___/ \/_/   \/___/ \/_/     \/_/                                                                                                                                
''') # https://snskeyboard.com/asciitext/
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=9145)
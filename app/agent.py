import time
from asr import ASR
from llm import LLM
from tts import TTS
from memory import Memoery


class Agent:
  def __init__(self, ai_name: str):
    self.ai_name = ai_name
    self.asr = ASR()
    self.llm_4 = LLM(model="gpt-4")
    self.llm_3 = LLM(model="gpt-3.5-turbo")
    self.tts = TTS() # model='mio/tokiwa_midori'
    self.mem = Memoery()
  
  def speech_to_text(self, input_file: str) -> str:
    """
    음성 인식 및 텍스트 변환
    """
    return self.asr.speech_to_text(input_file)
  
  def query(self, user_name:str, user_input: str):
    """
    응답생성
    """
    # x. 유저의 입력 저장
    self.mem.addChat(user_name, self.ai_name, user_input)
    
    # x. 최근 기억 로드
    recent_chat_history = self.mem.getRecentChat()
    formatted_recent_chat_history = [
      f"[{chat.timestemp}] {chat.subject}: {chat.content}" for chat in recent_chat_history
    ]
    recent_chat_history = "\n".join(formatted_recent_chat_history)
    print("최근==========\n", recent_chat_history)

    # x. 관련 기억 로드
    relevant_chat_memory = self.mem.getReleventMemory(user_input)
    formatted_relevant_chat_memory = [
      f"[{mem.timestemp}] {mem.subject}: {mem.content}" for mem in relevant_chat_memory
    ]
    relevant_chat_memory = "\n".join(formatted_relevant_chat_memory)
    print("관련==========\n", relevant_chat_memory)
    
    # x. AI의 응답 출력
    ai_output = self.llm_4.response(
      user_input=user_input,
      ai_name=self.ai_name,
      user_name=user_name,
      relevant_memory=relevant_chat_memory,
      current_time=f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
      chat_history=recent_chat_history
    )
    
    # x. AI의 응답 저장
    self.mem.addChat(self.ai_name, user_name, ai_output)
    
    print()
    
    return ai_output
  
  def text_to_speech(self, text: str, output_file: str) -> None:
    """
    음성 생성 및 저장
    """
    self.tts.text_to_speech(text, output_file)
  
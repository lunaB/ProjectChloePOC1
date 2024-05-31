import os
import dotenv
dotenv.load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

class LLM:
  def __init__(self, model="gpt-4"):
    """
    Args:
      llm (str): The language model to use. e.g. "gpt4", "llama3"
    """
    self.model = None
    if model == "gpt-4":
      from langchain_openai import ChatOpenAI
      self.model = ChatOpenAI(model="gpt-4")
    elif model == "gpt-3.5-turbo":
      from langchain_openai import ChatOpenAI
      self.model = ChatOpenAI(model="gpt-3.5-turbo")
    elif model == "test":
      from transformers import AutoTokenizer
      from transformers import AutoModelForCausalLM
      from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
      from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

      model = AutoModelForCausalLM.from_pretrained("yanolja/EEVE-Korean-Instruct-2.8B-v1.0")
      tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-2.8B-v1.0")
      pipe = pipeline("text-generation", model=model,
                      tokenizer=tokenizer, max_new_tokens=512)
      self.model = HuggingFacePipeline(pipeline=pipe)

    else:
      pass

  def load_prompt(self, prompt_file:str) -> PromptTemplate:
    """
    프롬프트 로딩
    Args:
      prompt_file (str): 프롬프트 파일명
    """
    with open(f"./templates/{prompt_file}", "r") as file:
      template = file.read()
    return PromptTemplate.from_template(template)

  def translate(self, text:str, src:str, dest:str) -> str:
    """
    번역 수행
    Args:
      text (str): 번역할 텍스트
      src (str): 원본 언어. e.g. "English"
      dest (str): 번역 언어. e.g. "Japanese"
    """
    prompt_template = self.load_prompt("translate.txt")
    output_parser = StrOutputParser()
    chain = (prompt_template | self.model | output_parser).with_config({"run_name": "Translate"})
    return chain.invoke({"text": text, "src": src, "dest": dest})
  
  def emotion(self, user_input:str, script:str) -> str:
    """
    감정 분석
    Args:
      user_input (str): 
      script (str): 
    """
    prompt_template = self.load_prompt("emotion.txt")
    output_parser = StrOutputParser()
    chain = (prompt_template | self.model | output_parser).with_config({"run_name": "Emotion"})
    return chain.invoke({"user_input": user_input, "script": script})
  
  def response(self, user_input:str, ai_name:str, user_name:str, relevant_memory:str, current_time:str, chat_history:str) -> str:
    """
    응답 생성
    Args:
      user_input (str): 
      ai_name (str): 
      user_name (str): 
      relevant_memory (str): 
      current_time (str): 
      chat_history (str): 
    """
    prompt_template = self.load_prompt("response.txt")
    output_parser = StrOutputParser()
    chain = (prompt_template | self.model | output_parser).with_config({"run_name": "Response"})
    return chain.invoke({
      "ai_name": ai_name,
      "user_name": user_name,
      "user_input": user_input,
      "relevant_memory": relevant_memory,
      "current_time": current_time,
      "chat_history": chat_history
    })
    
    
  

if __name__ == "__main__":
  # llm = LLM(model="gpt-3.5-turbo")
  # Test Translation
  # print(llm.translate("Hello my name is jin.", "English", "Japanese"))
  # Test Response
  # print(llm.response("Hello, world!"))
  
  
  from transformers import AutoTokenizer
  from transformers import AutoModelForCausalLM

  model = AutoModelForCausalLM.from_pretrained("yanolja/EEVE-Korean-Instruct-2.8B-v1.0")
  tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-2.8B-v1.0")

  print("load ok")

  prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {prompt}\nAssistant:\n"
  text = '안녕하세요.'
  model_inputs = tokenizer(prompt_template.format(prompt=text), return_tensors='pt')
  print(tokenizer)
  outputs = model.generate(**model_inputs, max_new_tokens=256)
  output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
  print(output_text)

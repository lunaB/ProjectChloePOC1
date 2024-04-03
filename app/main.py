print(r'''
 ____       __          ___                                    __            __            _     
/\  _`\    /\ \        /\_ \                                 /'__`\        /'__`\        /' \    
\ \ \/\_\  \ \ \___    \//\ \      ___      __              /\ \/\ \      /\ \/\ \      /\_, \   
 \ \ \/_/_  \ \  _ `\    \ \ \    / __`\  /'__`\            \ \ \ \ \     \ \ \ \ \     \/_/\ \  
  \ \ \L\ \  \ \ \ \ \    \_\ \_ /\ \L\ \/\  __/             \ \ \_\ \ __  \ \ \_\ \ __    \ \ \ 
   \ \____/   \ \_\ \_\   /\____\\ \____/\ \____\             \ \____//\_\  \ \____//\_\    \ \_\
    \/___/     \/_/\/_/   \/____/ \/___/  \/____/              \/___/ \/_/   \/___/ \/_/     \/_/                                                                                                                                
''') # https://snskeyboard.com/asciitext/

import speech_recognition as sr
import soundfile
import playsound
from espnet2.bin.tts_inference import Text2Speech

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
import uuid
import time

import warnings
warnings.filterwarnings("ignore")

import logging
# Logger Debug
log_format = '(%(filename)s : %(lineno)d) >> %(levelname)s - %(message)s'
logging.basicConfig(format=log_format, level=logging.WARNING)
# disable logging
logging.disable(logging.WARNING)

import chromadb
from chromadb.utils import embedding_functions

from dotenv import load_dotenv

# load dotenv
load_dotenv()
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
print("[Info] Load enviroment")

# embedding model
ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=HUGGING_FACE_API_KEY,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("[Info] Embedding model loaded")

# database load
db = chromadb.PersistentClient(path="chroma")
collection_history = db.get_or_create_collection(name="history", embedding_function=ef) # all history (chat, etc)
collection_inference = db.get_or_create_collection(name="inference", embedding_function=ef) # inference about user
print("[Info] Database loaded")

# tts = Text2Speech.from_pretrained("mio/tokiwa_midori")
tts = Text2Speech.from_pretrained("imdanboy/kss_jets")

print('[Info] TTS model loaded')

# veriable setting
user_name = "나영채" # "ナ ヨンチェ"
ai_name = "김민지" # みどり

# response chain
response_template = '''
Information about "{ai_name}":
She is Korean.
Bright personality, first-year student at Sejong University, loves sleeping, and has a lot of curiosity.
Close friends with {user_name}.

{user_name}'s information:
{inferenced_user_info}

Relevant history:
{relevant_history}

Context:
The following is a conversation between "{ai_name}" and "{user_name}"
Relevant history refers to records that individuals can consult for their conversations. It's okay not to refer to the records depending on the situation. 
The current time is ${current_time}.

Conversation:
{chat_history}
{ai_name} :
'''.strip()
response_prompt_template = PromptTemplate.from_template(response_template)
llm = ChatOpenAI(model="gpt-4")
output_parser = StrOutputParser()
response_chain = (response_prompt_template | llm | output_parser).with_config({"run_name": "Generate Response"})

# 실행동안의 대화 히스토리를 담을 배열
history = [] # {timestemp: "", name: "", text: ""}
def add_history(timestemp, name, text):
    history.append({"timestemp": timestemp, "name": name, "text": text})

def get_hisory(max_n=50):
    return "\n".join([f"{d['name']} : {d['text']}" for d in history[-max_n:]])

# 로그 생성
log_file_name = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
def add_log(text):
    with open(f'data/{log_file_name}.txt', 'a+') as file:
        file.write(text+"\n")

# start listening
print("[Info] Start listening")
start_timestemp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
currunt_turn_start_chat_id = str(uuid.uuid4())
collection_history.add(
    embeddings=ef("Start of conversation"), 
    documents=["Start of conversation"],
    metadatas=[{"id": currunt_turn_start_chat_id, "type": "start", "timestemp": start_timestemp}],
    ids=[currunt_turn_start_chat_id]
)
add_log(f"[{start_timestemp}] Start of conversation ==========")
try:
    r = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            # Auto Speech Recognition
            audio = r.listen(source)
            user_text = None
            try:
                user_text = r.recognize_google(audio, language="ko-KR")
            except sr.UnknownValueError:
                # 오디오에서 발화를 인식하지 못함.
                print("[Run] Unrecognized utterance....")
                continue
            except sr.RequestError as e:
                print("[Error] Could not request results from Google Speech Recognition service; {0}".format(e))
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
                
            # timestemp
            user_timestemp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

            # Add log
            log_form_text = f"[{user_timestemp}] {user_name} : {user_text}"
            add_log(log_form_text)
            add_history(user_timestemp, user_name, user_text)
            print(log_form_text)

            # Add user chat history
            currunt_turn_user_chat_id = str(uuid.uuid4())
            collection_history.add(
                embeddings=ef(user_text), 
                documents=[user_text],
                metadatas=[{"id": currunt_turn_user_chat_id, "type": "chat", "role": "user", "timestemp": user_timestemp}],
                ids=[currunt_turn_user_chat_id],
            )

            # Select inference history
            inferenced_user_info = ""

            # Select relevant history
            # res = collection_history.query(
            #     query_embeddings=ef("こんにちは"),
            #     n_results=5,
            #     where={
            #         "type": {
            #             "$in": ['chat_user', 'chat_ai']
            #         }
            #     }
            # )

            # Select relevant history
            query_result = collection_history.query(
                query_embeddings=ef(user_text),
                n_results=5,
                where={
                    "$and": [
                        {
                            "type": {
                                "$in": ['chat']
                            }
                        },
                        {
                            "role": {
                                "$in": ['user', 'ai']
                            }
                        },
                        {
                            "id": {
                                "$ne": currunt_turn_user_chat_id
                            }
                        }
                    ]
                }
            )
            relevant_history_arr = []
            for text, meta in zip(query_result["documents"][0], query_result["metadatas"][0]):
                if meta['role'] == 'user':
                    relevant_history_arr.append(f"[{meta['timestemp']}] {user_name} : {text}")
                elif meta['role'] == 'ai':
                    relevant_history_arr.append(f"[{meta['timestemp']}] {ai_name} : {text}")
            relevant_history = "\n".join(relevant_history_arr)
            
            # Select chat history
            chat_history = get_hisory(30)

            # Generate Response
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            response_text = response_chain.invoke({
                "user_name": user_name, 
                "ai_name": ai_name,
                "inferenced_user_info": inferenced_user_info, 
                "relevant_history": relevant_history, 
                "chat_history": chat_history,
                "current_time": current_time
            })
            
            # timestemp
            response_timestemp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            
            # Add Log
            log_form_text = f"[{response_timestemp}] {ai_name} : {response_text}"
            add_log(log_form_text)
            add_history(response_timestemp, ai_name, response_text)
            print(log_form_text)

            # Add AI chat history
            db_form_text = f"{ai_name} : {response_text}"
            currunt_turn_ai_chat_id = str(uuid.uuid4())
            collection_history.add(
                embeddings=ef(db_form_text), 
                documents=[db_form_text],
                metadatas=[{"id": currunt_turn_ai_chat_id, "type": "chat", "role": "ai", "timestemp": response_timestemp}],
                ids=[currunt_turn_ai_chat_id],
            )

            # Text to speech
            tts_output = tts(response_text)
            soundfile.write("contents/tts.wav", tts_output['wav'].numpy(), tts.fs, 'PCM_32')
            playsound.playsound("contents/tts.wav")

except KeyboardInterrupt:
    print("[Info] Stop listening")
except Exception as e:
    print("[Error] ", e)
finally:
    end_timestemp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    currunt_turn_end_chat_id = str(uuid.uuid4())
    collection_history.add(
        embeddings=ef("End of conversation"), 
        documents=["End of conversation"],
        metadatas=[{"id": currunt_turn_end_chat_id, "type": "end", "timestemp": end_timestemp}],
        ids=[currunt_turn_end_chat_id]
    )
    add_log(f"[{end_timestemp}] End of conversation ==========\n\n")

    # Debug (reset database)
    # print("[Debug] Reset database")
    # db.delete_collection("history")
    # db.delete_collection("inference")

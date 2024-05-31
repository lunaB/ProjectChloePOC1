import os
import time
from typing import List
import uuid
import sqlite3
import chromadb
from pydantic import BaseModel


from chromadb.utils import embedding_functions
from dotenv import load_dotenv
load_dotenv()

HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

class MemoryFragment(BaseModel):
  id: int
  timestemp: str
  mem_type: str
  subject: str
  object: str
  content: str
  parent_id: int
  level: int
  

class Memoery:
  def __init__(self):
    self.init()
    
  def init(self):
    # 0. rdb, vector db 초기화
    self.db = sqlite3.connect("./storage/sqlite3/history.db")
    self.vector_db = chromadb.PersistentClient("./storage/chroma")
    
    # 1. rdb cursor 초기화
    self.cur = self.db.cursor()
    self.cur.execute('''
      CREATE TABLE IF NOT EXISTS memory
        (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          timestemp DATETIME DEFAULT (DATETIME('now', 'localtime')),
          mem_type TEXT,
          subject TEXT,
          object TEXT,
          content TEXT,
          parent_id INTEGER,
          level INTEGER
        )
    ''')
    
    # 2. vector embedding 초기화
    self.ef = embedding_functions.HuggingFaceEmbeddingFunction(
      api_key=HUGGING_FACE_API_KEY,
      model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 3. vector db collection 초기화
    self.collection = self.vector_db.get_or_create_collection(
      "memory", embedding_function=self.ef)
    
  def reset(self):
    # 1. rdb 제거
    self.cur.execute('''
      DROP TABLE IF EXISTS memory
    ''')
    
    # 2. vector db 제거
    self.vector_db.delete_collection("memory")
    
  ############################################# 
  # Util
  #############################################
  def memoryListToDict(self, memory_list: List[MemoryFragment]):
    return [mem.model_dump() for mem in memory_list]
    
    
  ############################################# 
  # Generator
  #############################################
  def getRecentChat(self, n=25):
    """
      최근 대화
      Args:
        n (int): 
      Return:
        List[MemoryFragment]: 
    """
    return list(reversed(self.selectOrderByTimestampOnlyChat(n)))
  
  def getReleventMemory(self, content: str, n=10):
    """
      관련 기록
      Args:
        content (str): 
        n (int): 
      Return:
        List[MemoryFragment]: 
    """
    return self.vectorSearch(content)
  
  def addChat(self, subject:str, object:str, content:str):
    """
      채팅 추가
    """
    self.insert(
      timestemp=self.getTimestamp(),
      mem_type="chat",
      subject=subject,
      object=object,
      content=content,
      parent_id=0,
      level=0
    )
    
    
  ############################################# 
  # Data
  #############################################
  
  def getTimestamp(self):
    """
      Datetime 형식 현재시간
    """
    return f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    
  def insert(self, timestemp:str, mem_type: str, subject: str, object: str, content: str, parent_id: int = 0, level: int = 0):
    """
      메모리에 기억 추가
      Args:
        timestemp (str): 시간 (YYYY-MM-DD HH:MM:SS)
        mem_type (str): 메모리 타입
        subject (str): 생성 주체
        object (str): 대상
        content (str): 내용
        parent_id (int): 부모 id
        level (int): 추론 레벨
    """
    # 0. timestemp 추가
    if timestemp == None:
      timestemp = self.getTimestamp()
    
    # 1. rdb에 추가
    self.cur.execute('''
      INSERT INTO memory (timestemp, mem_type, subject, object, content, parent_id, level)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestemp, mem_type, subject, object, content, parent_id, level))
    insert_id = self.cur.lastrowid # 마지막으로 추가된 row의 id
    
    # 2. vector db에 추가
    self.collection.add(
      embeddings=self.ef(content),
      documents=[content],
      metadatas=[{
        "id": insert_id, 
        "timestemp": timestemp, 
        "mem_type": mem_type, 
        "subject": subject, 
        "object": object, 
        "parent_id": parent_id, 
        "level": level
      }],
      ids=[str(insert_id)]
    )
    
  def keywordSearch(self, keyword: str, n=5):
    """
      키워드를 이용한 메모리 검색
      Args:
        keyword (str): 검색할 키워드
        n (int): 반환할 결과 수
    """
    # 1. SQL LIKE 쿼리를 이용한 키워드 검색
    self.cur.execute('''
      SELECT * FROM memory
      WHERE content LIKE ?
      ORDER BY timestemp DESC
      LIMIT ?
    ''', ('%' + keyword + '%', n))
    
    # 2. 결과 반환
    return self.cur.fetchall()
    
  def vectorSearch(self, content: str, n=5):
    """
      # semantic search
      Embedding을 이용한 메모리 검색
      Args:
        content (str): 검색할 내용
        n (int): 반환할 결과 수
      Return:
        List[MemoryFragment]: MemoryFragment 리스트
    """
    
    # 1. 컨텐츠 임베딩
    content_embedding = self.ef(content)
    
    # 2. 벡터 유사도 검색
    rows = self.collection.query(
      query_embeddings=content_embedding,
      n_results=n
    )
    
    # 벡터 검색 결과를 MemoryFragment 객체로 변환
    memory_fragments = []
    for result in rows['metadatas']:
      for metadata in result:
        memory_fragments.append(MemoryFragment(
          id=metadata['id'],
          timestemp=metadata['timestemp'],
          mem_type=metadata['mem_type'],
          subject=metadata['subject'],
          object=metadata['object'],
          content=rows['documents'][0][result.index(metadata)],
          parent_id=metadata['parent_id'],
          level=metadata['level']
        ))
    return memory_fragments
    
  def selectOrderByTimestamp(self, n=5):
    """
      메모리 전체 조회
      Args:
        n (int): 반환할 결과 수
    """
    
    self.cur.execute('''
      SELECT * 
      FROM memory 
      ORDER BY timestemp DESC LIMIT ?
    ''', (n,))
    return self.cur.fetchall()
  
  def selectOrderByTimestampOnlyChat(self, n=5):
    """
      최근 대화 정보 
      Args:
        n (int): 반환할 결과 수
    """
    self.cur.execute('''
      SELECT * 
      FROM memory 
      WHERE mem_type = "chat"
      ORDER BY timestemp DESC 
      LIMIT ?
    ''', (n,))
    
    return [MemoryFragment(
      id=row[0], 
      timestemp=row[1], 
      mem_type=row[2], 
      subject=row[3],
      object=row[4],
      content=row[5],
      parent_id=row[6],
      level=row[7]
    ) for row in self.cur.fetchall()]
  
  def getLastId(self):
    """
      마지막 id 
    """
    self.cur.execute('''
        SELECT seq 
        FROM sqlite_sequence 
        WHERE name = 'memory'
      ''')
    return self.cur.fetchone()
    
if __name__ == "__main__":
  mem = Memoery()
  # 현재시간 검색
  mem.reset()
  mem.init()
  # # 생성
  # mem.insert(
  #   timestemp=mem.getTimestamp(), 
  #   mem_type="chat",
  #   subject="나영채",
  #   object="김민지",
  #   content="안녕 요즘 재미있는일 있니?",
  #   parent_id=0,
  #   level=0
  # )
  # mem.insert(
  #   timestemp=mem.getTimestamp(), 
  #   mem_type="chat",
  #   subject="김민지",
  #   object="나영채",
  #   content="재미있는일 많지~",
  #   parent_id=0,
  #   level=0
  # )
  # OK
  # print(mem.getRecentChat())
  # OK
  # print(mem.vectorSearch("안녕하세요"))
  

# format /kaggle/input/2024-1-nlp-2/Korean_movie_reviews_2016.txt
data = """
부산 행 때문 너무 기대하고 봤   0
한국 좀비 영화 어색하지 않게 만들어졌 놀랍   1
조금 전 보고 왔 지루하다 언제 끝나 이 생각 드   0
평 밥 끼 먹자 돈 니 내고 미친 놈 정신사 좀 알 싶어 그래 밥 먹다 먹던 숟가락 대가리 존나 때릴 지도 모르 일단 면상 확인하고 싶다 전화번호 남겨 놔 진짜   1
점수 대가 과 이 엑소 팬 어중간 점수 줄리 없겠 클레멘타인 이후 최고 평점 조작 영화 아닐 생각 드네   0
평점 너무 높 공감 이해 안되는 스토리   0
경 각심 일 깨워 준 영화 였   1
"""

import pandas as pd
import numpy as np

with open('/kaggle/input/2024-1-nlp-2/Korean_movie_reviews_2016.txt', encoding='utf-8') as f:
    docs = [doc.strip().split('\t') for doc in f]
    docs = [(doc[0], int(doc[1])) for doc in docs if len(doc) == 2]
    texts, labels = zip(*docs)
words_list = [doc.strip().split() for doc in texts]
print(words_list[:2])
# total text tokens
total_words = []
for words in words_list:
    total_words.extend(words)
    
from collections import Counter
c = Counter(total_words)

# 빈도를 기준으로 상위 10000개의 단어들만 선택
max_features = 10000
common_words = [word for word, count in c.most_common(max_features)]
# 각 단어에 대해서 index 생성하기
words_dic = {}
for i, word in enumerate(common_words):
  words_dic[word] = i

# 각 index에 대해서 단어 기억하기
# 각 문서를 상위 10000개 단어들에 대해서 index 번호로 표현하기
# [['부산','행'],[]] .. 예시로 [[0,1,2,3,4,5], [...],...]
indexed_words = [[words_dic[word] for word in words if word in words_dic] for words in words_list]
filtered_indexed_words = [words for words in indexed_words if words]


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical

# 2.1 X - input data 처리 (text tokens id_index to padded X)
max_len = 100  # maximum sequence length
X = sequence.pad_sequences(indexed_words, maxlen=max_len)


# 2.2 y - label data 처리 (one_hot_encoded y)
from sklearn.model_selection import train_test_split
y = to_categorical(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2.3 Train / Test Split

from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()

#Write your code - model build
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(64, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer=RMSprop(lr=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop # you can add more optimizers

#Write your code - model setting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])

def predict_sentiment(text):
  # Tokenize the input text
  words = text.strip().split()
  # Convert words to indices using the words_dic dictionary
  indexed_words = [words_dic[word] for word in words if word in words_dic]
  # Pad the indexed_words sequence
  padded_sequence = sequence.pad_sequences([indexed_words], maxlen=max_len)
  # Make predictions using the trained model
  predictions = model.predict(padded_sequence)
  # Convert predictions to sentiment labels
  sentiment_labels = ['Negative', 'Positive']
  sentiment = sentiment_labels[np.argmax(predictions)]
  return sentiment
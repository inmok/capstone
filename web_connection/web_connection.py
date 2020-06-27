#!/usr/bin/env python

import pandas as pd
from konlpy.tag import Komoran
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#LSTM
from tensorflow.keras.models import load_model
import pymysql

import cgitb
import cgi
#php에서 python 에러코드 확인해주는 코드
cgitb.enable()
#임계값에 따라 리턴값 분류하는 함수
def jg(result):
    if (result > 0.5):
        i = 1
    else:
        i = 0
    return i

kor=Komoran(userdic='C:/Users/INMO/PycharmProjects/capstone_pjt/user_dic.txt')

stopwords = ['의', '가', '이', '은', '들', '는', '과', '도', '를', '으로', '이', 'ㅋ',
             '자', '에', '와', '한', '하다', '을', '다', '에서', '하고', 'ㄴ', 'ㄹ',
             '아', '하', '있', '았', '것', '나', '라', '고', '지', '게', '어', '되', '보',
             '면', '거', '네', 'ㅁ', '었', '아서', '겠', '로', '만', 'ㅂ시다', 'ㄴ가',
             '는데', 'ㄴ다', '왜', '어서', '어요', 'ㅂ니다', '으면', '라고', 'ㄴ데', '요',
             '그렇', '부터', 'ㄴ다고', '처럼', '라는', '는지', '습니다', '이다', '죠', '네요',
             'ㅡ', '으니', 'ㄴ다는', 'ㄹ까', 'ㄴ지', '구나', '그리고', 'ㄴ다는데']

conn = pymysql.connect(host='localhost', user='root', password='inmo123', db='capstone')
curs = conn.cursor()

sql = "SELECT contents FROM comments ORDER BY id DESC limit 1"
curs.execute(sql)

rows = curs.fetchall()

rows = " ".join(str(x) for x in rows)


df = pd.DataFrame({'document': [rows]})


df['document'] = df['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

train_data = pd.read_csv('C:/Users/INMO/PycharmProjects/capstone_pjt/data_set_fin.csv')

#한글만 남기고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

X=train_data['document']
y=train_data['label']

train_data_document = []
for sentence in train_data['document']:
    temp_X = []
    temp_X = kor.morphs(sentence) # 토큰화
    #stopword에 등록한 조사 제거
    temp_X = [word for word in temp_X if not word in stopwords]
    train_data_document.append(temp_X)

X_input = []
for sentence in df['document']:
    temp_X = []
    temp_X = kor.morphs(sentence)  # 토큰화
    # stopword에 등록한 조사 제거
    temp_X = [word for word in temp_X if not word in stopwords]
    X_input.append(temp_X)

#정수 인코딩과정 텍스트가 1234 같은 정수로 바뀜
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data_document)

# threshold 의 정수값보다 낮게 반복되는 단어를 제거
threshold = 2
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

# 단어의 등장 빈도수가 threshold보다 작으면
    if (value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

    # 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거. 0번 패딩 토큰을 고려하여 +1
vocab_size = total_cnt - rare_cnt + 1

tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(train_data_document)
X_input = tokenizer.texts_to_sequences(X_input)

# input 빈샘플제거
drop_test = [index for index, sentence in enumerate(X_input) if len(sentence) < 1]
X_input = np.delete(X_input, drop_test, axis=0)

#X_input = tokenizer.texts_to_sequences(X_input)
#패딩
max_len = 64
X_input = pad_sequences(X_input, maxlen=max_len)

#모델 가져오기
model = load_model('C:/Users/INMO/PycharmProjects/capstone_pjt/model_best.h5')
#input 값을 모델에 넣어 결과 확인
result = model.predict(X_input)
#결과값이 0.5보다 크면 1, 작으면 0으로 분류
result_fin = jg(result)
#결과출력
print(result_fin)

conn.close()
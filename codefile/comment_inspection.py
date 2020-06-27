import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import re
from konlpy.tag import Komoran
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#LSTM
from tensorflow.keras.models import load_model

kor=Komoran(userdic='./user_dic.txt')

stopwords = ['의', '가', '이', '은', '들', '는', '과', '도', '를', '으로', '이', 'ㅋ',
             '자', '에', '와', '한', '하다', '을', '다', '에서', '하고', 'ㄴ', 'ㄹ',
             '아', '하', '있', '았', '것', '나', '라', '고', '지', '게', '어', '되', '보',
             '면', '거', '네', 'ㅁ', '었', '아서', '겠', '로', '만', 'ㅂ시다', 'ㄴ가',
             '는데', 'ㄴ다', '왜', '어서', '어요', 'ㅂ니다', '으면', '라고', 'ㄴ데', '요',
             '그렇', '부터', 'ㄴ다고', '처럼', '라는', '는지', '습니다', '이다', '죠', '네요',
             'ㅡ', '으니', 'ㄴ다는', 'ㄹ까', 'ㄴ지', '구나', '그리고', 'ㄴ다는데']

train_data = pd.read_csv('data_set_fin.csv')
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

train_data_document = []
for sentence in train_data['document']:
    temp_X = []
    temp_X = kor.morphs(sentence) # 토큰화
    #stopword에 등록한 조사 제거
    temp_X = [word for word in temp_X if not word in stopwords]
    train_data_document.append(temp_X)



# 악플이라면 악플메세지를 출력하고 아니면 입력한 댓글데이터를 출력하는 함수
def cybershield(comment, result):
    if (result == 0):
        return print("악플입니다. 등록하실 수 없는 댓글입니다")
    else:
        return print(comment)


while True:
    # 댓글 직접 입력
    comment_input = input('댓글 입력 : ')
    #print(type(input))
    if comment_input == 1:
        print('프로그램 종료')
        break
    df = pd.DataFrame({'document': [comment_input]})
    #print(type(df['document']))
    # 입력받은 댓글 한글만 남기고 제거
    df['document'] = df['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

    X_input = []
    for sentence in df['document']:
        temp_X = []
        temp_X = kor.morphs(sentence)  # 토큰화
        # stopword에 등록한 조사 제거
        temp_X = [word for word in temp_X if not word in stopwords]
        X_input.append(temp_X)

    #print('토큰화 완료된 input data')
    #print(X_input)

    # 형태소 정수인코딩
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data_document)
    #print(len(tokenizer.word_index))

    # 단어 빈도수가 너무 낮으면 학습에 영향을 줄 수 있으므로 제거해준다
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
    #print(vocab_size)

    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(train_data_document)
    X_input = tokenizer.texts_to_sequences(X_input)


    # input 빈샘플제거
    drop_test = [index for index, sentence in enumerate(X_input) if len(sentence) < 1]
    X_input = np.delete(X_input, drop_test, axis=0)

    #print('정수화 완료된 input data')
    #print(X_input)
    #print(np.shape(X_input))

    # 패딩
    max_len = 64
    X_input = pad_sequences(X_input, maxlen=max_len)
    #print('패딩 결과')
    #print(X_input)

    # 베스트 모델을 모델로 사용
    model = load_model('model_fin_th0.h5')

    result = model.predict(X_input)
    #print("댓글데이터에 대한 결과값")
    #print(result)

    # result값이 0.5보다 크면 1로 작으면 0(악플)로 판단
    if (result > 0.5):
        result=1
    else:
        result=0

    cybershield(comment_input, result)

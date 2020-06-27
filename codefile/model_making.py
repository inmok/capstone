import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import re
from konlpy.tag import Komoran
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#LSTM
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


train_data = pd.read_csv('data_set_fin.csv')
print(type(train_data['document']))
print(train_data[:5])

#한글만 남기고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")


X=train_data['document']
y=train_data['label']

#전체 train중 20%를 test data로 미리 빼둠
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1594, stratify=y)


kor=Komoran(userdic='./user_dic.txt')

#제거 할 형태소
stopwords = ['의', '가', '이', '은', '들', '는', '과', '도', '를', '으로', '이', 'ㅋ',
             '자', '에', '와', '한', '하다', '을', '다', '에서', '하고', 'ㄴ', 'ㄹ',
             '아', '하', '있', '았', '것', '나', '라', '고', '지', '게', '어', '되', '보',
             '면', '거', '네', 'ㅁ', '었', '아서', '겠', '로', '만', 'ㅂ시다', 'ㄴ가',
             '는데', 'ㄴ다', '왜', '어서', '어요', 'ㅂ니다', '으면', '라고', 'ㄴ데', '요',
             '그렇', '부터', 'ㄴ다고', '처럼', '라는', '는지', '습니다', '이다', '죠', '네요',
             'ㅡ', '으니', 'ㄴ다는', 'ㄹ까', 'ㄴ지', '구나', '그리고', 'ㄴ다는데']

train_data_document = []
for sentence in X_train:
    temp_X = []
    temp_X = kor.morphs(sentence) # 토큰화
    #stopword에 등록한 조사 제거
    temp_X = [word for word in temp_X if not word in stopwords]
    train_data_document.append(temp_X)

test_data_document = []
for sentence in X_test:
    temp_X = []
    temp_X = kor.morphs(sentence) # 토큰화
    #stopword에 등록한 조사 제거
    temp_X = [word for word in temp_X if not word in stopwords]
    test_data_document.append(temp_X)

#형태소 정수인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data_document)

#빈도수가 높은 단어부터 출력
#print(tokenizer.word_index)

#단어 빈도수가 너무 낮으면 학습에 영향을 줄 수 있으므로 제거해준다
#threshold값 미만으로 반복되는 단어를 제거
threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# 전체 단어 개수 중 빈도수 3이하인 단어 개수는 제거. 0번 패딩 토큰을 고려하여 +1
vocab_size = total_cnt - rare_cnt + 1
'''
print('전체 단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)


토크나이저의 인자로 단어집합의 크기인 vocab_size를 넘겨주어
토크나이저는 텍스트 시퀀스를 숫자 시퀀스로 변환하는 정수 인코딩과정에서
이보다 큰 숫자가 부여된 단어를 뺀다
'''

#형태소 정수인코딩(vocab_size 설정 후)
#새로 설정된 vocab_size로 다시 토큰화
tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(train_data_document)
train_data_document = tokenizer.texts_to_sequences(train_data_document)
test_data_document = tokenizer.texts_to_sequences(test_data_document)

train_data_label = np.array(y_train)
test_data_label = np.array(y_test)

#빈 샘플 제거
drop_train = [index for index, sentence in enumerate(train_data_document) if len(sentence) < 1]
drop_test = [index for index, sentence in enumerate(test_data_document) if len(sentence) < 1]
#train 빈샘플제거
train_data_document = np.delete(train_data_document, drop_train, axis=0)
train_data_label = np.delete(train_data_label, drop_train, axis=0)

#print(len(train_data_document))
#print(len(train_data_label))

#test 빈샘플제거
test_data_document = np.delete(test_data_document, drop_test, axis=0)
test_data_label = np.delete(test_data_label, drop_test, axis=0)

#print(len(test_data_document))
#print(len(test_data_label))
'''
print('댓글의 최대 길이 :',max(len(l) for l in X_train))
print('댓글의 평균 길이 :',sum(map(len, X_train))/len(X_train))
#댓글 길이 히스토그램
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
'''

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

#패딩
max_len = 64  #패딩 최대길이를 설정
below_threshold_len(max_len, train_data_document)

train_data_document = pad_sequences(train_data_document, maxlen = max_len)
test_data_document = pad_sequences(test_data_document, maxlen = max_len)


#LSTM 모델링
model = Sequential()
model.add(Embedding(vocab_size, 256, input_length = 64))  #임베딩 벡터의 차원을 256으로 설정
model.add(LSTM(128, return_sequences = True))
model.add(LSTM(128, return_sequences = False))
model.add(Dense(1, activation='sigmoid')) #활성함수 sigmoid 사용

#과적합이 되었을 때 epoch를 멈추는 EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
#ModelCheckpoint를 사용하여 검증데이터의 정확도가 이전보다 좋아질 경우에만 모델을 저장
mc = ModelCheckpoint('model_case_N.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#훈련데이터의 20%를 검증데이터로 사용하면서 정확도를 확인한다
history = model.fit(train_data_document, train_data_label, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.25)

loaded_model = load_model('model_case_N.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(test_data_document, test_data_label)[1]))




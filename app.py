import streamlit as st
import tweepy
from tensorflow import keras
from gensim import models
import tensorflow as tf
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import plotly.express as px
import gzip
emotion_to_n={'슬픔':0, '중립':1, '걱정':2, '놀람':3, '기쁨/행복/사랑':4, '분노/증오':5}
ab_dic={'lol': 'laughing', 'rofl': 'laughing', 'brb': 'be right back', 'ily': 'i love you', 'ty': 'thank you', 'imy': 'i miss you', 'yolo': 'you only live once', 'fomo': 'fear of missing out', 'idk': 'i do not know', 'idc': 'i do not care', 'ffs': 'for freaks sake', 'smh': 'shake my head', 'ngl': 'not going to lie', 'w': 'with', 'abt': 'about', 'u': 'you only live once', 'r': 'are', 'gtg': 'going to go', 'nvm': 'never mind', 'bcoz': 'because', 'coz': 'because', 'bcos': 'because', 'cld': 'could', 'ez': 'easy', 'fbm': 'fine by me', 'ftw': 'for the win', 'fyi': 'for your information', 'ik': 'i know', 'wfh': 'work from home', 'lmfao': 'laughing my freaking ass off', 'lmk': 'let me know', 'af': 'as freak', 'aight': 'alright', 'awol': 'away without leaving', 'irl ': 'in real life', 'bt': 'bad trip', 'bb': 'baby', 'btw': 'by the way ', 'cu': 'see you', 'idgaf': "i don't give a freak", 'dgaf': "don't give a freak", 'df': 'the freak ', 'dis': 'this', 'dm': 'direct message', 'dnt': "don't ", 'dw': "don't worry", 'enf': 'enough', 'eta': 'estimated time of arrival', 'fu': 'freak you', 'fwm': 'fine with me', 'gg': 'good game', 'gn': 'good night', 'gm': 'good morning', 'gr8': 'great', 'grl': 'girl', 'grw': 'get ready with me', 'h8': 'hate', 'hbd': 'happy birthday', 'hbu': 'how about you', 'hru': 'how are you', 'hw': 'homework', 'idts': "i don't think so", 'ig': 'instagram', 'ilysm': 'i love you so much', 'imo': 'in my opinion', 'jk': 'just kidding', 'k': 'okay', 'ldr': 'long distance relationship', 'l2g': 'like to go', 'ly': 'love you', 'mfw': 'my face when', 'm8': 'mate', 'nbd': 'no big deal', 'nsfw': 'not safe for work', 'nm': 'nothing much', 'np': 'no problem', 'nw': 'no way', 'og': 'original gangster', 'ofc': 'ofcourse', 'omg': 'oh my god', 'omfg': 'oh my freaking god', 'ootd': 'outfit of the day', 'otb': 'off to bed', 'otw': 'off to work', 'pm': 'private message', 'ppl': 'people', 'prob': 'probably', 'qt': 'cutie', 'rly': 'really', 'sh': 'same here', 'sis': 'sister', 'bro': 'brother', 'sry': 'sorry', 'sup': "what's up", 'tbh': 'to be honest', 'thnk': 'thank you', 'thx': 'thanks', 'ttly': 'totally', 'ttyl': 'talk to you later', 'ur': 'you are', 'wb': 'welcome back', 'whatevs': 'whatever', 'wyd': 'what are you doing', 'wdyk': 'what do you know', 'wru': 'where are you', 'wtf': 'what the freak', 'wtg': 'way to go', 'wywh': 'wish you were here', 'xd': 'laugh', 'xoxo': 'hugs and kisses', 'xo': 'hugs and kisses', 'y': 'why', 'tryna': 'trying to be '}
text_or_graph=st.sidebar.selectbox('결과 보는 방법:', ('그래프', '트윗 내용'))
@st.cache(allow_output_mutation=True)
def load_wordmodel(model):
    
    wordmodel=models.KeyedVectors.load_word2vec_format(model, binary=True)
    return wordmodel
wordmodel=load_wordmodel('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz')
def word_to_num(w):
 
  p=word_tokenize(w)
  words=[]
  for j in p:
      lower_j=j.lower()
      if lower_j in ab_dic.keys():
        for l in ab_dic[lower_j].split(' '):
          words.append(l)
      else:
        words.append(lower_j)
  res=[]
  vocab=list(wordmodel.index_to_key)
  for k in words:
    if k in vocab:
       res.append(wordmodel[k])
  if len(res)<=50 and len(res)>0:
    left=50-len(res)
    unsplit=tf.zeros(300*left)
    split=tf.split(unsplit, left)
    return tf.concat([split, res], axis=0)

model=keras.models.load_model('tweet_emotion_classify.h5')
nltk.download('wordnet')
nltk.download('punkt')
if 'newdata' not in st.session_state:
  st.session_state.newdata={}
elif len(st.session_state.newdata)>=12:
  newdata=st.session_state.newdata
  X_train=[]
  Y_train=[]
  for i in range(len(list(newdata.keys()))):
    X_train.append(word_to_num(list(newdata.keys())[i]))
    Y_train.append(tf.keras.utils.to_categorical(list(newdata.values())[i], num_classes=6))
  X_train=tf.convert_to_tensor(X_train)
  Y_train=tf.convert_to_tensor(Y_train)

  model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(0.000003), metrics='accuracy')
  model.fit(X_train, Y_train, epochs=3)
  keras.models.save_model(model, 'tweet_emotion_classify.h5', overwrite=True, include_optimizer=True)
  st.session_state.newdata={}
keyword=st.text_input("검색할 키워드")
search_again=st.button('새로/다시 검색하기')
st.session_state.search_again=False

if search_again:
  st.session_state.search_again=True
@st.cache(allow_output_mutation=True)
def connect_api():

    consumer_key='wMjTpovuMgBQJcs6hKhGmo7mO'
    consumer_secret='w279RgGLl2994NImyduKqlE8EiXAvhGxrXlPdoYmfsAxit8oDj'
    access_token='1297870587067379712-p002dQkdG6KUNykX6cssGMrrGe6Jyw'
    access_token_secret='BBLjBHHkLRXkY9cLWw7hx1ogAuXTWAcJOMGCOcMiKaqNA'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    return api

api=connect_api()

emotion_type=['슬픔', '중립', '걱정', '놀람', '기쁨/행복/사랑', '분노/증오']

if search_again:
  tweets={}
  emotion_dic={0:'슬픔', 1:'중립', 2:'걱정', 3:'놀람', 4:'기쁨/행복/사랑', 5:'분노/증오'}
  if keyword != '':
      search=api.search_tweets(q=keyword, lang='en', count=100)
      
      for tweet in search:
          num_tweet=word_to_num(tweet.text)
          if num_tweet==None:
            continue
          n=tf.math.argmax(model.predict(tf.convert_to_tensor([num_tweet]))[0]).numpy()
          tweets[tweet.text]=emotion_dic[n]
  emotion=tweets.values()

  st.session_state.tweets=tweets
  st.session_state.emotion=emotion
elif 'tweets' in st.session_state:
  tweets=st.session_state.tweets
  emotion=st.session_state.emotion
from collections import Counter
if text_or_graph=='그래프':
  if 'emotion' in globals():
    if emotion!=[]:

      emotion_freq=dict(Counter(emotion))
      keys=emotion_freq.keys()
      for k in emotion_type:
        if k not in keys:
          emotion_freq[k]=0

      df=pd.DataFrame(dict(number_of_tweets=emotion_freq.values(), emotion=emotion_freq.keys()))
      fig=px.scatter_polar(df, r='number_of_tweets', theta='emotion')
      fig.update_traces(fill='toself')
      fig.update_layout(polar=dict(
          radialaxis=dict(
            visible=False
          ),
        ),showlegend=False)
      st.write(fig)
      piefig=px.pie(df, values='number_of_tweets', names='emotion')
      st.write(piefig)
else:
  st.session_state.fix=False
  text_df=pd.DataFrame((tweets.keys(), tweets.values()))
  text_df=text_df.transpose()
  pd.set_option('display.max_rows', None)
  st.write(text_df)
  st.write('분류 오류 신고하기: 분류에 잘못된 것이 있다면 몇번째 데이터인지와 알맞은 정답을 선택해주세요.')
  fix_n_list=list(range(len(tweets.keys())))
  fix_n_list.insert(0, '선택하세요')
  fix_n=st.selectbox(label='',options=fix_n_list)
  fix_emotion=st.selectbox(label='',options=['선택하세요','슬픔', '중립', '걱정', '놀람', '기쁨/행복/사랑', '분노/증오'])
  fix_button=st.button('신고')
  if fix_button:
    if fix_n!='선택하세요' and fix_emotion!='선택하세요':
      newdata=st.session_state.newdata
      newdata[list(tweets.keys())[int(fix_n)]]=emotion_to_n[fix_emotion]
      st.session_state.newdata=newdata
      print(st.session_state.newdata)
      

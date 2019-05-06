from flask import Flask,render_template,url_for,request

import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf
import pickle,re,string
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences

import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./Model_Saved/model1"))
from loadmodel1 import * 
sys.path.append(os.path.abspath("./Model_Saved/model2"))
from loadmodel2 import * 
sys.path.append(os.path.abspath("./Model_Saved/model3"))
from loadmodel3 import * 


# Function to load document into the notebook
def load_document(fileName):
    file=open(fileName,'r')
    text_data=file.read()
    file.close()
    return text_data

def clean_document(document,m_type="mlp"):
    document=document.lower()
    #split the review into tokens by white space
    tokens=document.split()
    # regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # removetokens which are not alphabetis
    if m_type=="mlp":
        tokens = [word for word in tokens if word.isalpha()]
        # remove stop words
        ##  A stop word is a commonly used word (such as “the”, “a”, “an”, “in”)
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # remove out short tokens
        tokens = [word for word in tokens if len(word) > 1]
    
    return tokens



def predict_sentiment(review, vocab, tokenizer, model):
    tokens = clean_document(review)
    tokens = [w for w in tokens if w in vocab]
    line = ' '.join(tokens)
    print(line)
    encoded = tokenizer.texts_to_matrix([line], mode='freq')
    yhat = model.predict(encoded, verbose=0)
    percent_pos = yhat[0,0]
    if percent_pos < 0.49:
        return (1-percent_pos), 'NEGATIVE'
    else:
        return percent_pos, 'POSITIVE'


def predict_cnn_sentiment(review, vocab, tokenizer,max_length, model):
    tokens = clean_document(review)
    tokens = [w for w in tokens if w in vocab]
    line = ' '.join(tokens)
    print(line)
    padded=encode_documents(tokenizer,max_length,[line])
    yhat = model.predict(padded, verbose=0)
    percent_pos = yhat[0,0]
    print(percent_pos)
    if percent_pos < 0.49:
        return (1-percent_pos), 'NEGATIVE'
    else:
        return percent_pos, 'POSITIVE'

def predict_ncnn_sentiment(review, vocab, tokenizer,max_length, model):
    tokens = clean_document(review)
    tokens = [w for w in tokens if w in vocab]
    line = ' '.join(tokens)
    print(line)
    padded=encode_documents(tokenizer,max_length,[line])
    # predict sentiment
    yhat = model.predict([padded,padded,padded], verbose=0)
    percent_pos = yhat[0,0]
    print(percent_pos)
    if percent_pos < 0.49:
        return (1-percent_pos), 'NEGATIVE'
    else:
#         if percent_pos >0.5 and percent_pos<0.5045:
#             return percent_pos, 'NEUTRAL'
#         else:
        return percent_pos, 'POSITIVE'
#     if round(percent_pos)==0:
#         return (1-percent_pos), 'NEGATIVE'
#     return percent_pos, 'POSITIVE'

def encode_documents(tokenizer ,max_length,docs):
    encoded=tokenizer.texts_to_sequences(docs)
    padded=pad_sequences(encoded,maxlen=max_length,padding='post')
    return padded


app = Flask(__name__)

global  model1,graph1,model2,graph2,model3,graph3
model1,graph1= init_model_1()
model2,graph2= init_model_2()
model3,graph3= init_model_3()

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    #Load Vocabulary
    vocab_data=load_document("vocab.txt")
    vocab_data=vocab_data.split()
    vocab=set(vocab_data)
    # # Loading Model1
    # model1_json_file = open('model1mlp_json.json','r')
    # loaded_model1_json = json_file.read()
    # json_file.close()
    # loaded_model1 = model_from_json(loaded_model1_json)
	# #load woeights into new model
    # loaded_model1.load_weights("model1mlp.h5")
    # print("Loaded Model from disk")
    # loaded_model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    with open('tokenizermodel1.pickle', 'rb') as handle:
        tokenizer_model_1 = pickle.load(handle)
    with open('tokenizer_model_2.pickle', 'rb') as handle:
        tokenizer_model_2 = pickle.load(handle)
    with open('tokenizer_model_3.pickle', 'rb') as handle:
        tokenizer_model_3 = pickle.load(handle)
    if request.method == 'POST':
        message = request.form['message']
        data = message
        with graph1.as_default():
            my_prediction1 = predict_sentiment(data,vocab,tokenizer_model_1,model1)
        with graph2.as_default():
            my_prediction2 = predict_cnn_sentiment(data,vocab,tokenizer_model_2,1244,model2)
        with graph3.as_default():
            my_prediction3 = predict_ncnn_sentiment(data,vocab,tokenizer_model_3,1244,model3)
    return render_template('result.html',prediction1 = my_prediction1,prediction2 = my_prediction2,prediction3 = my_prediction3,mess=data)

	#compile and evaluate loaded model
	# loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


if __name__ == '__main__':
	app.run(debug=True)
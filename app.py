from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

global data
data=pd.read_csv('train_data.csv',encoding = "ISO-8859-1")

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    
    x=data['message']
    y=data['is_vulgar']

# Create a pipeline
    model_0 = Pipeline([
        ("tf-idf", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])

# Fit the pipeline to the training data
    model_0.fit(X=x, 
            y=y);
    global req
    req=request.form.get('a')
    global real
    real=model_0.predict([req])
# Second model
    y=data['is_vulgar1']
    bert_preprocess = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')
    
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)
# Neural network layers
    l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)
# Use inputs and outputs to construct a final model
    model = tf.keras.Model(inputs=[text_input], outputs = [l])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=30, batch_size = 32)

    y_predicted = model.predict([req])
    data1=np.round(y_predicted)
    return render_template('after.html', data=real,data1=data1)

@app.route('/finpg', methods=['POST'])
def finpg():
    global data
    option=request.form.get('opt')
    check=request.form.get('check')
    print(option)
    print(check)
    if(np.int(option)==3 and np.int(check)==1):
        if(np.int(real[0])==0):
            new_row = {'message':req, 'is_vulgar':0,'is_vulgar1':0}
            data=data.append(new_row, ignore_index=True)
        elif(np.int(real[0])==1):
            new_row = {'message':req, 'is_vulgar':1,'is_vulgar1':1}
            data=data.append(new_row, ignore_index=True)
    elif(np.int(option)==3 and np.int(check)==0):
        if(np.int(real[0])==0):
            new_row = {'message':req, 'is_vulgar':1,'is_vulgar1':1}
            data=data.append(new_row, ignore_index=True)
        elif(np.int(real[0])==1):
            new_row = {'message':req, 'is_vulgar':0,'is_vulgar1':0}
            data=data.append(new_row, ignore_index=True)
    elif(np.int(option)==0 and np.int(check)==1):
        if(np.int(real[0])==0):
            new_row = {'message':req, 'is_vulgar':0,'is_vulgar1':0}
            data=data.append(new_row, ignore_index=True)
        elif(np.int(real[0])==1):
            new_row = {'message':req, 'is_vulgar':1,'is_vulgar1':1}
            data=data.append(new_row, ignore_index=True)
    elif(np.int(option)==0 and np.int(check)==0):
        if(np.int(real[0])==0):
            new_row = {'message':req, 'is_vulgar':1,'is_vulgar1':1}
            data=data.append(new_row, ignore_index=True)
        elif(np.int(real[0])==1):
            new_row = {'message':req, 'is_vulgar':0,'is_vulgar1':0}
            data=data.append(new_row, ignore_index=True)
    elif(np.int(option)==1 and np.int(check)==1):
        if(np.int(real[0])==0):
            new_row = {'message':req, 'is_vulgar':0,'is_vulgar1':0}
            data=data.append(new_row, ignore_index=True)
        elif(np.int(real[0])==1):
            new_row = {'message':req, 'is_vulgar':1,'is_vulgar1':1}
            data=data.append(new_row, ignore_index=True)
    elif(np.int(option)==1 and np.int(check)==0):
        if(np.int(real[0])==0):
            new_row = {'message':req, 'is_vulgar':1,'is_vulgar1':1}
            data=data.append(new_row, ignore_index=True)
        elif(np.int(real[0])==1):
            new_row = {'message':req, 'is_vulgar':0,'is_vulgar1':0}
            data=data.append(new_row, ignore_index=True)
    return render_template('finpg.html',  tables=[data.to_html(classes='data')], titles=data.columns.values)

if __name__ == "__main__":
    import random, threading, webbrowser

    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)

    threading.Timer(1.25, lambda: webbrowser.open(url) ).start()

    app.run(port=port, debug=False)
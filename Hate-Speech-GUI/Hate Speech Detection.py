import pandas as pd
import preprocessor as p
import string
import emoji
import nltk
from nltk import TweetTokenizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, pipelines
import transformers
#import torch
from googletrans import Translator
from flask import Flask, app, request, render_template

#defining stopwords
eng = list(pd.read_csv('Hate-Speech-GUI/english_stopwords.txt', sep = '/n', header= None, engine= 'python')[0])
mal = list(pd.read_csv('Hate-Speech-GUI/malay_stopwords.txt', sep = '/n', header= None, engine= 'python')[0])

def text_preprocessing(text, language):
    #remove user mentions, URLs and reserved words
    p.set_options(p.OPT.MENTION, p.OPT.URL, p.OPT.RESERVED)
    text = p.clean(text)

    #remove smart quotes
    text = text.replace('“', '')
    text = text.replace('”', '')

    #lowercasing text
    text = text.lower()

    #demojize text
    text = emoji.demojize(text, delimiters=(" ", " "))

    #Defining stop_words
    if language == 'en':
        stop_words = eng
    elif language == 'ms':
        stop_words = mal
    else:
        stop_words = []
    
    #Defining tokenizer 
    tokenizer = TweetTokenizer()

    #Removing stop words
    text = [word for word in tokenizer.tokenize(text) if not word in stop_words]

    #Joining words back into sentence string
    text = ' '.join(text)

    return(text)

#Defining tokenizer
pretrained_model = 'bert-base-multilingual-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model)

#Defining fine-tuned model
model_path = "Hate-Speech-GUI/Multilingual Model"
model = BertForSequenceClassification.from_pretrained(model_path)

#Defining classifier
classify_hate_speech = transformers.pipeline('text-classification', tokenizer= bert_tokenizer, model= model)

#Defining web app using Flask
app = Flask("__main__", static_url_path='')
message = ""
@app.route("/")
def main():

    return render_template("hate_speech_gui.html")

@app.route("/", methods=['POST'])
def hate_speech_detection():
    text = request.form['input_sentence']

    #detect language
    t = Translator()
    detected = t.detect(text)
    lang = detected.lang
    
    #remove punctuations
    nopunc_text = [char for char in text if char not in string.punctuation]
    nopunc_text = ''.join(nopunc_text)
    
    #text preprocessing
    preprocessed_text = text_preprocessing(nopunc_text, lang)

    #classify text 
    result = classify_hate_speech(preprocessed_text)

    #return result
    msg = ""
    if result[0]['label'] == 'LABEL_0':
        msg = "Good news! No hate speech is detected."
        colour1 = 'green'
    
    else:
        msg = "Hate speech is detected in the input. Report to relevant authorities!"
        colour1 = 'red'

    return render_template("hate_speech_gui.html", input=text, output=msg, colour=colour1)
    
    
if __name__ == "__main__":
    app.run(debug=True)

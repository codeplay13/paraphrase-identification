import re
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

## constant variables
bow_threshold = 0.5  # threshold value for cosine_similarity for BOW. If similarity > threshold -> 1, else 0
minilm_threshold = 0.75 # threshold value for cosine_similarity for minilm.

## load model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


## function to preprocess/clean text
def clean_text(text):
    """
    The function to clean or preprocess the input text by applying following operations:
        1. Remove special characters, punctuations and digits
        2. Convert to lowercase
        3. Lemmatization
    Note: Return value is string and does NOT contain list of strings.

    Parameters:
        text (str): Input string to be processed

    Returns:
        Preprocessed or cleaned text.
    """
    # remove special characters, punctuations, digits
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # convert to lowercase
    text = text.lower()
    
    # Convert to list from string
    text = text.split()
    
    # lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text] 
    text = " ".join(text)
    return text


## function to build corpus from the dataset
def build_corpus_from_df(df_list, col_list):
    """
    The function to build corpus from dataframes.

    Parameters:
        df_list (list): List of Dataframe(s) which contains the text columns to build the corpus
        col_list (list): List of text column(s) from which the corpus is to be built

    Returns:
        corpus (list): List of all the sentences from input Dataframes and Columns
    """
    corpus = list()
    for df in df_list:
        for col in col_list:
            corpus += df[col].values.astype(str).tolist()
    return corpus


## function to vectorize the column texts and get calculate cosine_similarities
def transform_get_score(vectorizer, sent1, sent2):
    """
    The function to vectorize two input sentences and return the cosine similarity score.

    Parameters:
        vectorizer (): Model to transform the text. For eg. CountVectorizer
        sent1 (str): Input sentence 1
        sent2 (str): Input sentence 2

    Returns:
        similarity score between the two input sentences.
    """
    return cosine_similarity(vectorizer.transform([str(sent1)]), vectorizer.transform([str(sent2)]))[0][0]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    try:
        sent1 = request.json['sent_1']
        sent2 = request.json['sent_2']
        model_name = request.json['model']
        if model_name == "bow":
            similarity_score = transform_get_score(vectorizer, sent1, sent2)
            prediction = 1 if similarity_score > bow_threshold else 0
        elif model_name == "minilm":
            sent1_embedding = model.encode(sent1, convert_to_tensor=True)
            sent2_embedding = model.encode(sent2, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(sent1_embedding, sent2_embedding)
            prediction = 1 if similarity_score > minilm_threshold else 0
        response = {
            'output': prediction
        }
        return jsonify(response)
    except Exception as ex:
        res = dict({'message': str(ex)})
        return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')


## read all csv files
train_df = pd.read_csv("data/train.tsv", sep="\t", error_bad_lines=False, warn_bad_lines=False)
test_df = pd.read_csv("data/test.tsv", sep="\t", error_bad_lines=False, warn_bad_lines=False)
dev_df = pd.read_csv("data/dev.tsv", sep="\t", error_bad_lines=False, warn_bad_lines=False)

corpus = build_corpus_from_df([train_df, test_df, dev_df], ["#1 String", "#2 String"])  # build corpus
vectorizer = CountVectorizer(stop_words='english').fit(corpus)  # fit CountVectorizer on the corpus


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=True)
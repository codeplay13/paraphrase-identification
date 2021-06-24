import re
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

## constant variables
threshold = 0.5  # threshold value for cosine_similarity. If similarity > threshold -> 1, else 0

## read all csv files
print("Loading data..")
train_df = pd.read_csv("data/train.tsv", sep="\t", error_bad_lines=False, warn_bad_lines=False)
test_df = pd.read_csv("data/test.tsv", sep="\t", error_bad_lines=False, warn_bad_lines=False)
dev_df = pd.read_csv("data/dev.tsv", sep="\t", error_bad_lines=False, warn_bad_lines=False)


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


print("Building Corpus..")
corpus = build_corpus_from_df([train_df, test_df, dev_df], ["#1 String", "#2 String"])  # build corpus
print("Extracting featues..")
vectorizer = CountVectorizer(stop_words='english').fit(corpus)  # fit CountVectorizer on the corpus

## performance evaulation on dev.tsv
print("Predicting..")
similarity_score = dev_df.apply(lambda x: transform_get_score(vectorizer, x["#1 String"], x["#2 String"]), axis=1).values # get similarity scores
predicted_values = np.where(similarity_score>threshold, 1, 0) # apply threshold to convert values into {0, 1}
actual_values = dev_df['Quality'].values # get actual values/labels
f1_score_value = f1_score(actual_values, predicted_values) # calculate F1-Score

print(f"\nF1-Score on test data is {round(f1_score_value, 3)}, for threshold {threshold}") # print F1-Score with threshold value
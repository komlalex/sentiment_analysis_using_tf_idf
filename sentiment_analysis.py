import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer 

raw_df = pd.read_csv("./data/train.tsv", sep="\t") 
test_df = pd.read_csv("./data/test.tsv", sep="\t")
sub_df = pd.read_csv("./data/sampleSubmission.csv") 

raw_df.Sentiment.value_counts(normalize=True).sort_index().plot(kind="bar")
#plt.show() 

""" Implement TF-IDF Technique """
#print(word_tokenize("This is the (real) deal"))
stemmer = SnowballStemmer(language="english") 
english_stopwords = stopwords.words("english")
selected_stopwords = english_stopwords[:115]

def tokenize(text):
    return [stemmer.stem(token) for token in word_tokenize(text) if token.isalpha()] 
 

vectorizer = TfidfVectorizer(
    lowercase=True, 
    tokenizer=tokenize, 
    #stop_words= selected_stopwords, 
    ngram_range=(1, 2), 
    max_features=2000
) 

vectorizer.fit(raw_df.Phrase)  

#print(vectorizer.get_feature_names_out()[:400]) 

"""
Transform Training and Test Data 
- Transform phrases from training set 
- Transform phrases from test set 
- Look at some examples 
"""

inputs = vectorizer.transform(raw_df.Phrase)
test_inputs = vectorizer.transform(test_df.Phrase.apply(lambda x: np.str_(x)))

"""
Split Training and Validation Sets
Tip: Don't use a random sample for validation set (why?)
"""
TRAIN_SIZE = 110_000 
train_inputs = inputs[:TRAIN_SIZE]
train_targets = raw_df.Sentiment[:TRAIN_SIZE] 

val_inputs = inputs[TRAIN_SIZE:]
val_targets = raw_df.Sentiment[TRAIN_SIZE:] 

"""
Train Logistic Regression Model
"""
MAX_ITER = 1000
model = LogisticRegression(max_iter=MAX_ITER, solver="sag") 
model.fit(train_inputs, train_targets) 

train_preds = model.predict(train_inputs)
acc_score = accuracy_score(train_targets, train_preds)
print(f"Accuracy on Train dataset: {acc_score}") 

val_preds = model.predict(val_inputs) 
acc_score = accuracy_score(val_targets, val_preds) 
print(f"Accuracy on Val dataset: {acc_score}") 

"""
Study Predictions on Sample Inputs
""" 
small_df = raw_df.sample(20) 

small_inputs = vectorizer.transform(small_df.Phrase.apply(lambda x: np.str_(x)))

small_preds = model.predict(small_inputs)

#print(small_df)
#print(small_preds) 

"""
Make Predictions on test dataset 
- Make predictions on Test Dataset
- Generate and add predictions to CSV file 
"""
test_preds = model.predict(test_inputs) 

sub_df.Sentiment = test_preds 
sub_df.to_csv("submission.csv")
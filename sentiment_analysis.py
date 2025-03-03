import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
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
    stop_words= selected_stopwords, 
    ngram_range=(1, 2), 
    max_features=2500
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

print(test_inputs.toarray()[0][:100])




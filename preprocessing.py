# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import nltk
from collections import Counter
from nltk import tokenize
import re
import pickle

from mongodbConnector import MongodbConnector
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from nltk.stem.wordnet import WordNetLemmatizer

from collections import Counter
from sklearn.model_selection import train_test_split

from scipy import sparse


#nltk.download('wordnet')

def expand_contractions(sentence, contraction_mapping): 
     
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),  
                                      flags=re.IGNORECASE|re.DOTALL) 
    def expand_match(contraction): 
        match = contraction.group(0) 
        first_char = match[0] 
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())                        
        expanded_contraction = first_char+expanded_contraction[1:] 
        return expanded_contraction 
         
    expanded_sentence = contractions_pattern.sub(expand_match, sentence) 
    return expanded_sentence 

def buildingPickle():
    nltk.download('wordnet')

    lmtzr = WordNetLemmatizer()
    with open('helper/WordNetLemmatizer.pickle', 'wb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        pickle.dump(lmtzr, f, pickle.HIGHEST_PROTOCOL)
    
    """
    Expanding contraction 
    """
    CONTRACTION_MAP = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                       "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                       "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                       "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                       "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                       "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                       "he'll've": "he he will have", "he's": "he is", "how'd": "how did", 
                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                       "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                       "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                       "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                       "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                       "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                       "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                       "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                       "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                       "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                       "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                       "this's": "this is",
                       "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                       "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                       "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                       "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                       "we're": "we are", "we've": "we have", "weren't": "were not", 
                       "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                       "what's": "what is", "what've": "what have", "when's": "when is", 
                       "when've": "when have", "where'd": "where did", "where's": "where is", 
                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                       "who's": "who is", "who've": "who have", "why's": "why is", 
                       "why've": "why have", "will've": "will have", "won't": "will not", 
                       "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                       "you'll've": "you will have", "you're": "you are", "you've": "you have" } 
    
    with open('helper/contraction_map.pickle', 'wb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        pickle.dump(CONTRACTION_MAP, f, pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    
    print("Building Pickle")
    buildingPickle()    
    
    print("Get data from MongoDB")
    # Get data from MongoDB
    DBmanager = MongodbConnector()
    DBmanager.setCollection("storypoint")
    data = DBmanager.getData()
    
    # Put it into Pandas
    pd_data = pd.DataFrame.from_records(data,columns=['project','title','user_story','point'])
    
    # list of project
    project_list = list(pd_data.project.unique())
    
    
    # Contraction Map
    with open('helper/contraction_map.pickle', 'rb') as f:
        CONTRACTION_MAP = pickle.load(f)
        
    # Lematization model
    with open('helper/WordNetLemmatizer.pickle', 'rb') as f:
        lmtzr = pickle.load(f)
    
    # Concatinate title and user_story
    pd_data['concat'] = pd_data['title'] + ". " + pd_data['user_story']
    
    corpus = pd_data['concat']
    # Save CSV file for later use
    pd_data.to_csv("data_csv/data",index=False)
    
#    vectorizer = CountVectorizer(stop_words='english',strip_accents="ascii")
    vectorizer = TfidfVectorizer(stop_words='english',strip_accents="ascii",lowercase=False)
    vectorizer.fit(corpus)
    
    # TF_IDF features
    with open('features/tf_idf_vectorizer.pickle', 'wb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        pickle.dump(vectorizer, f, pickle.HIGHEST_PROTOCOL)    
        
    tf_idf_features = vectorizer.transform(corpus)
    
    # TF_IDF features
    sparse.save_npz("features/tf_idf_matrix.npz", tf_idf_features)

    
#    with open('features/tf_idf.pickle', 'wb') as f:
#        # The protocol version used is detected automatically, so we do not
#        # have to specify it.
#        pickle.dump(tf_idf_features, f, pickle.HIGHEST_PROTOCOL)    
        
#    tokenizer = vectorizer.build_tokenizer()
#    preprocessing = vectorizer.build_preprocessor()
    analyzer = vectorizer.build_analyzer()    
    dictionary = vectorizer.vocabulary_
    
    length = len(dictionary)
    reverse_dictionary = {k+3:v for v,k in dictionary.items()}
    reverse_dictionary[0] = '<PAD>'
    reverse_dictionary[1] = '<START>'
    reverse_dictionary[2] = '<OOV>'
    dictionary = {k:v for v,k in reverse_dictionary.items()}
    
    # Save dictionary
    with open('helper/dictionary.pickle', 'wb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
    
    # Tokenizer
#    token = corpus.apply(lambda x: tokenizer(x))
#    length_token = sum(token.apply(lambda x: len(x)))
    
    corpus = corpus.apply(lambda x: analyzer(x))
    
#    corpus = corpus.apply(lambda x: tokenizer(x))
    
    # Lemmatization -> Not using because IOS and IO are totally different
#    corpus = corpus.apply(lambda x: [lmtzr.lemmatize(i) for i in x])
    
    # Expand Contraction
    corpus = corpus.apply(lambda x: [expand_contractions(i,CONTRACTION_MAP) for i in x])
     
    corpus = corpus.to_frame()
    corpus['project'] = pd_data['project']
    
    # Convert to     
    corpus.to_pickle('helper/corpus.pickle')
    
    
    
    

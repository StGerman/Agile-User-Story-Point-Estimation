#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:20:21 2018

@author: bking
"""

from gensim.test.utils import common_texts,get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
import pandas as pd
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--proc', default='8', type=str, help='Number of proccessor') 
args = parser.parse_args()
proc = int(args.proc)

print("================== Word2Vec Feature Extraction ===========================")
print(parser.print_help())
print("==========================================================================")

#path = get_tmpfile("word2vec.model")
def Word2VecFeature(embedding_size = 200):


    corpus = pd.read_pickle('helper/corpus.pickle')
    
    #corpus = corpus['concat']
#    window_size = 3
    model = Word2Vec(corpus['concat'], size=embedding_size, window=3, min_count=1, workers=4)
    #model.train(corpus,len(corpus),epochs = 10)
    
    model.save("helper/word2vec_"+str(embedding_size)+".model")
    
    features = []
    
    for i in range(len(corpus)):    
        sentence = corpus.loc[i,'concat']
        ave_vector = 0
        for s in sentence:
    #        print(s)
            ave_vector += model.wv[s]
        
        ave_vector = ave_vector / model.vector_size
        
        features.append(ave_vector)
        
    columns = ['feature'+str(i) for i in range(model.vector_size)]
    df_word2vec = pd.DataFrame.from_records(features,columns=columns)
    df_word2vec = df_word2vec.join(corpus['project'])
    
    print("Check Null")
    assert(df_word2vec.isnull().any().any() == False)
    
    df_word2vec.to_csv("features/word2vec_ave_"+str(embedding_size)+".csv")


def main():
#    embedding_size = [10,50,100,300,500,1000,2000]
#    embedding_size = [10,50,100,300]
    embedding_size = [100,300]
    print("Embedding Size: ",embedding_size)
#    proc = 16
    print("Extracture Features")
    with Pool(proc) as p:
    #    length = len(embedding_size)
        feature = p.map(Word2VecFeature, embedding_size)

    
if __name__ == "__main__":
    main()
    
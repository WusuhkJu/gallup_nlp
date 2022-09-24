# -*- coding: utf-8 -*-

"""
Gallup_NLP_v1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jn7fbw0LBqPb4SyQC_1Ac7O6ZwjmnjLL

[Gallup Open-answer Autocoding] - built by JWS
release_1: 20220923

Data
- Using answer data from question-answer dataset
- As the shape of input data is likely to be excel sheet and for usability of non-pythonic people, 
  this code is designed to encompass from preprocessing of raw excel data to getting a result by a few of clicks

Objective
- Clustering sentences(answer) having similar meanings

Method
- Embedding -> Clustering

Method Details
- Embedding
 : Transformating setences to vectors by SentenceTransformer_KoBERT
 : SentenceTransformer_BERT_large is also availiable

- Clustering 
 : Drawing euclidean distance between maxtrix pairs(=embedded sentence pairs)
 : Clustering sentences having distance above a given threshold, e.g. 0.75
"""

!pip3 install kobert-transformers 
!pip3 install ko-sentence-transformers
!pip3 install transformers
!pip3 install sentence_transformers
!pip3 install fairseq

import numpy as np
import pandas as pd
import torch
import time
import re

from google.colab import files
from statistics import mean
from sentence_transformers import SentenceTransformer, models
from ko_sentence_transformers.models import KoBertTransformer
from sklearn.cluster import AgglomerativeClustering

# Uploading datasets
upload = files.upload()

df = pd.read_excel(r'')

class Preprocessing:        
    def __init__(self, df, answer_col_name = '응답'):
        self.answer_col_name = answer_col_name        
        self.raw = df[answer_col_name].values
        
        self.raw_keys = np.array([i for i in range(len(df))])
        self.key_sentence_dic = None

        self.cleaned_sentences_short = None

    def _get_key_sentence_dic(self):
        key_sentence_dic = {}
        for k,v in zip(self.raw_keys,self.raw):
            key_sentence_dic[k] = v
        self.key_sentence_dic = key_sentence_dic
                                    
    def get_cleaned(self):
        def sub(x):
            return re.sub("[^A-Za-z0-9가-힣 ]",'',x)                                    

        self.cleaned_sentences = [ sub(sentence) for sentence in self.raw ]
        print('########## Sentences have been cleansed ##########')
    
    def get_short_sentences(self, cut_criterion='iqr'):
        # cut_criterion = 'iqr' OR dictionary of two numbers, upper and lower ; {'upper':n, 'lower':n}
        cleaned_sentences = self.cleaned_sentences
        raw_keys = self.raw_keys

        list_length = [len(chunk) for chunk in cleaned_sentences]
        
        if cut_criterion == 'iqr':
            q1 = np.quantile(list_length,0.25)
            q3 = np.quantile(list_length,0.75)
            iqr = q3 - q1        
            upper = q3 + 1.5*iqr
            lower = q1 - 1.5*iqr
            if lower < 0:
                lower = 0

            normal_idx = np.where((np.array(list_length) < upper) & (np.array(list_length) > lower))[0]
            normal_keys = raw_keys[normal_idx]
            
        else:
            uplow_dic = cut_criterion            
                        
            normal_idx = np.where((np.array(list_length) < uplow_dic['upper']) & (np.array(list_length) > uplow_dic['lower']))[0]
            normal_keys = raw_keys[normal_idx]
        
        short_sentences = []
        for idx in normal_idx:
            short_sentences.append(cleaned_sentences[idx])

        list_short_length = [len(chunk) for chunk in short_sentences]
        max_short_len = int(max(list_short_length))
                
        self.max_short_len = max_short_len
        self.normal_keys = normal_keys
        self.cleaned_sentences_short = short_sentences

class Autocoding:
    def __init__(self,preprocessing):
        self.sentences = np.array(preprocessing.cleaned_sentences_short)

        self.model = None
        self.model_selection = None
        self.embeddings = None
        self.sentence_embedding = None

        self.result = None

    def get_model(self,model_selection='kobert',device='cpu'):
        if model_selection == 'kobert':
            word_embedding_model = KoBertTransformer("monologg/kobert", max_seq_length=100)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        elif model_selection == 'bert':
            model = SentenceTransformer('sentence-transformers/stsb-bert-large')
        model = model.to(device)
        self.model = model
        self.model_selection = model_selection

    def get_embedding_vectors(self):
        start_time = time.time()
        
        sentences = self.sentences
        model = self.model
        model_selection = self.model_selection

        def embedding(x):
            return model.encode(x)
        embeddings = [ embedding(sentence) for sentence in sentences ]
        
        sentence_embedding = {}
        for sen, emb in zip(sentences, embeddings):
            sentence_embedding[sen] = emb

        if model_selection == 'kobert':
            dim = 768
        elif model_selection == 'bert':
            dim = 1024        
        embeddings_array = np.zeros(shape=(len(embeddings),dim))
        def put_in(i,x,embeddings_array,dim=dim):
            z = np.array(x).reshape(1,dim)
            embeddings_array[i] = z

        [ put_in(i,x,embeddings_array) for i,x in enumerate(embeddings) ]
        
        self.embeddings = embeddings_array
        self.sentence_embedding = sentence_embedding

        end_time = time.time()
        time_taking = round(end_time-start_time)
        print('{} sec has taken'.format(time_taking))
        
    def clustering(self, distance='cosine', threshold=0.173):
        
        """
        묶을 대상이 
         : 일반 문장인 경우 threshold=0.173 추천
         : 명사형인 경우 threshold=0.075 추천
        
        """
        
        embeddings = self.embeddings
        sentences = self.sentences

        agglo = AgglomerativeClustering(n_clusters=None,affinity=distance,linkage='average',compute_full_tree=True,distance_threshold=threshold)
        agglo.fit(embeddings)

        self.result = pd.DataFrame({'응답':sentences, '코딩':agglo.labels_})

# Preprocessing
pre = Preprocessing(df)
pre.get_cleaned()
pre.get_short_sentences()

# Loading a model
auto = Autocoding(pre)
auto.get_model()
auto.get_embedding_vectors()

auto.clustering(threshold=0.174)

auto.result

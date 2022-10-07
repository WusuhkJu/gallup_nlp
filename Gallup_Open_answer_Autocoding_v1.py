# -*- coding: utf-8 -*-

"""
Gallup_NLP_v1

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
 : SentenceTransformer_BERT_large or RoBERTa is also availiable

- Clustering 
 : Drawing euclidean distance between maxtrix pairs(=embedded sentence pairs)
 : Clustering sentences having distance above a given threshold, e.g. 0.174
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

        self.normal_keys = None
        self.cleaned_sentences_short = None

    def _get_key_sentence_dic(self):
        key_sentence_dic = {}
        for k,v in zip(self.raw_keys,self.raw):
            key_sentence_dic[k] = v
        self.key_sentence_dic = key_sentence_dic
                                    
    def get_cleaned(self):
        def sub(x):
            return re.sub("[^A-Za-z0-9가-힣 ]",'',str(x))                                    

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
        self.pre = preprocessing
        self.sentences = np.array(preprocessing.cleaned_sentences_short)

        self.model = None
        self.model_selection = None
        self.dim = None
        self.embeddings = None
        self.sentence_embedding = None

        self.codebook = None
        self.result = None

    def get_model(self,model_selection='kobert',device='cpu'):
        if model_selection == 'kobert':
            word_embedding_model = KoBertTransformer("monologg/kobert", max_seq_length=100)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        elif model_selection == 'bert':
            model = SentenceTransformer('sentence-transformers/stsb-bert-large')
        elif model_selection == 'roberta':
            model = SentenceTransformer('sentence-transformers/stsb-roberta-large')
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
        elif (model_selection == 'bert') or (model_selection == 'roberta'):
            dim = 1024

        embeddings_array = np.zeros(shape=(len(embeddings),dim))
        def put_in(i,x,embeddings_array,dim=dim):
            z = np.array(x).reshape(1,dim)
            embeddings_array[i] = z

        [ put_in(i,x,embeddings_array) for i,x in enumerate(embeddings) ]
        
        self.dim = dim
        self.embeddings = embeddings_array
        self.sentence_embedding = sentence_embedding

        end_time = time.time()
        time_taking = round(end_time-start_time)
        print('{} sec has taken'.format(time_taking))
        
    def clustering(self, distance='cosine', threshold=0.174):
        
        """
        묶을 대상이 
         : 일반 문장인 경우 threshold=0.174 추천
         : 명사형인 경우 threshold=0.075 추천
        
        """
        
        pre = self.pre
        embeddings = self.embeddings
        sentences = self.sentences
        dim = self.dim
                
        agglo = AgglomerativeClustering(n_clusters=None,affinity=distance,linkage='average',compute_full_tree=True,distance_threshold=threshold)
        agglo.fit(embeddings)

        nk_group = {}
        for k, g in zip(pre.normal_keys, agglo.labels_):
            nk_group[k] = g
        group_container = np.array( [99998 for _ in range(len(pre.raw_keys))] )
        def inserting_group(k,g,group_container=group_container):
            group_container[k] = g
        [ inserting_group(k,g) for k,g in nk_group.items() ]

        raw_sentences = pre.raw

        group_meaning = {}   
        group_meaning[str(99998)] = 'NA'
        def get_representitive_vect(uniq_g,labels=agglo.labels_,embeddings=embeddings,cleaned_sentences_short=pre.cleaned_sentences_short,dim=dim,group_meaning=group_meaning):
            idxs = np.where(labels == uniq_g)[0]
            temp_matrix = np.zeros(shape=(len(idxs),dim))
            def insert_to_matrix(i,idx,temp_matrix=temp_matrix,embeddings=embeddings):
                temp_matrix[i] = embeddings[idx]
            [ insert_to_matrix(i,idx) for i,idx in enumerate(idxs) ]            
            vector_mean = np.mean(temp_matrix,axis=0)

            distance_list = []
            def get_distance(x,vector_mean=vector_mean,distance_list=distance_list):
                d = np.linalg.norm(vector_mean-x)
                distance_list.append(d)
            [ get_distance(x) for x in embeddings[idxs] ]

            distance_array = np.array(distance_list)
            closest_idx = idxs[np.where(distance_array == np.min(distance_array))[0][0]]
            closest_sentence = cleaned_sentences_short[closest_idx]

            group_meaning[str(uniq_g)] = closest_sentence
        
        uniq_groups = np.unique(agglo.labels_)
        [ get_representitive_vect(uniq_g) for uniq_g in uniq_groups ]

        keys = np.array([])
        values = np.array([])
        for k,v in group_meaning.items():
            keys = np.append(keys,str(int(k)+1))
            values = np.append(values,v)
        
        group_container = group_container+1
        
        self.codebook = pd.DataFrame({'코드':keys,'의미':values})
        self.result = pd.DataFrame({'응답':raw_sentences, '코드':group_container})

    def save_excel(self,file_name='test'):
        result = self.result
        codebook = self.codebook

        result_name = '/content/' + file_name + '.xlsx'
        result.to_excel(result_name)

        codebook_name = '/content/' + file_name + '_' + 'codebook' + '.xlsx'
        codebook.to_excel(codebook_name)

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

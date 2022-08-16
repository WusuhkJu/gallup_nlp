# -*- coding: utf-8 -*-
"""Gallup_NLP_github

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1__z9Z5tkmINANQNTixUdePphaM9vmu38

[NLP project for Gallup] - built by JWS

Objective
- Clustering sentences having similar meanings

Method
- Embedding -> Fine Tuning -> Clustering

Method Details
- Embedding
 : Transformating setences to vectors by KoBERT
- Fine Tuning
 : Target values are not fixed but vary
 : Due to unfixed target values, it's hard to directly implement supervised learning
 : Therefore, some tricks as below are used for supervised learning
   > Firstly, clustering Xs (assigning an identical group number to Xs having similar meanings, let Z the group number vector)
   > e.g. (x1, x2, x3) = ('맛있다', '맛있네요', '맛있습니다') then  (z1, z2, z3) = (1, 1, 1)
   > Randomly select Xs having similar meanings(=identical group number) and get the average value of them(embedded vectors of Xs)
   > Using the average vector above as (y1, y2, y3)
- Clustering
 : Using K-Means for clustering embedded vectors(Xs)
 : Choosing the number of cluster by representing the largest average silhouette value
"""

# Installing kobert

!pip3 install kobert-transformers
!pip3 install transformers

# Importing packages

from google.colab import files

import numpy as np
import pandas as pd
import re

from statistics import mean, median
from tracemalloc import stop

import torch

from statistics import mean, median
from tracemalloc import stop

from transformers import BertModel, BertTokenizer
from kobert_transformers import tokenization_kobert

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional

# Checking the availability of CUDA

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

# Uploading Datasets
"""
upload = files.upload()
"""

# Loading Datasets
"""
ids = np.load(r'')
attention_masks = np.load(r'')
"""

df = pd.read_csv(r'')

# Class for setting input data(ids, attention masks)

class Gallup_NLP:
    # made by Wusuhk_Ju @ Gallup, 2022
    
    def __init__(self, df, answer_col_name = '응답', group_col_name = None, tokenizer_selection = 'kobert'):
        self.df = df
        self.answer_col_name = answer_col_name
        self.raw_sentences = df[answer_col_name].values
        self.group_col_name = group_col_name
        self.tokenizer_selection = tokenizer_selection        
        
        self.sentence_group = self._get_group()
        self.cleaned_sentences = self._get_cleaned_sentences()                
        self.tokenizer = self._get_tokenizer()
        
        self.tokenized = None
        self.token_length = None
        self.short_tokenized = None
        self.short_group = None
        self.input_for_tokens = None
        self.attention_masks = None
        self.ids = None
        
    def _get_group(self):
        df = self.df
        group_col_name = self.group_col_name
        try:
            if group_col_name is not None:
                return df[group_col_name].values
            elif group_col_name is None:
                return None
        except:
            print('????????????? Please make sure the column name! ?????????????')
            return None            
                    
    def _get_cleaned_sentences(self):
        array_raw_sentences = self.raw_sentences
        array_cleaned_sentences = np.array([])
        s = 0
        for sentence in array_raw_sentences:
            s += 1
            array_cleaned_sentences = np.append(array_cleaned_sentences, re.sub("[^A-Za-z0-9가-힣 ]",'',sentence))
            print('Cleansing {}/{} has been done'.format(s, len(array_raw_sentences)))
        print('########## Sentences for analysis has been set up ##########')
        return array_cleaned_sentences 
        
    def _get_tokenizer(self):
        if self.tokenizer_selection == 'kobert':
            tz = tokenization_kobert.KoBertTokenizer.from_pretrained('monologg/kobert')
        elif self.tokenizer_selection == 'bert':
            tz = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            raise ValueError('????? Please select a tokenizer between "kobert" and "bert" for tokenizer_selection ?????')
        print('########## Tokenizer({}) has been set up ##########'.format(self.tokenizer_selection))
        return tz
    
    def get_tokenized(self, cleaned_or_raw_sentences = 'cleaned'):
        tz = self.tokenizer        
        if cleaned_or_raw_sentences == 'cleaned':
            array_sentences = self.cleaned_sentences
        elif cleaned_or_raw_sentences == 'raw':
            array_sentences = self.raw_sentences
        else:
            raise ValueError('????? Please select target setences to be tokenized between "cleaned" and "raw" ?????')
        
        # 1. Creating tokens
        lists_tokenized = []
        t = 0
        n_of_t = len(array_sentences)
        for sentence in array_sentences:
            t += 1
            tokenized = tz.tokenize(sentence)
            lists_tokenized.append(tokenized)
            print('Tokenization for {}/{} has been done'.format(t, n_of_t))
        
        self.tokenized = lists_tokenized
        
    def get_short_tokenized(self, cut_criteron = 'iqr'):
        # cut_criteron = 'iqr' OR dictionary of two numbers, upper and lower ; {'upper':n, 'lower':n}
        lists_tokenized = self.tokenized
        sentence_group = self.sentence_group
        
        list_length = [len(inner_list) for inner_list in lists_tokenized]                        
        
        if cut_criteron == 'iqr':
            q1 = np.quantile(list_length,0.25)
            q3 = np.quantile(list_length,0.75)
            iqr = q3 - q1        
            upper = q3 + 1.5*iqr
            lower = q1 - 1.5*iqr
            if lower < 0:
                lower = 0
            
            normal_idx = np.where((np.array(list_length) < upper) & (np.array(list_length) > lower))[0]
            n_normal = [1 if i >= 0 else 0 for i in normal_idx]
            n_outliers = len(lists_tokenized) - sum(n_normal)        
            
            lists_short_tokenized = []
            for idx in normal_idx:
                lists_short_tokenized.append(lists_tokenized[idx])
                
            list_short_tokenized_group = []
            for idx in normal_idx:
                list_short_tokenized_group.append(sentence_group[idx])
                
        else:
            uplow_dic = cut_criteron
            
            normal_idx = np.where((np.array(list_length) < uplow_dic['upper']) & (np.array(list_length) > uplow_dic['lower']))[0]
            n_normal = [1 if i >= 0 else 0 for i in normal_idx]     
            n_outliers = len(lists_tokenized) - sum(normal_idx)
            
            lists_short_tokenized = []
            for idx in normal_idx:
                lists_short_tokenized.append(lists_tokenized[idx])
                
            list_short_tokenized_group = []
            for idx in normal_idx:
                list_short_tokenized_group.append(sentence_group[idx])
                        
        self.token_length = {'min': np.min(list_length), 'med': np.median(list_length), 'mean': np.mean(list_length), 'max': np.max(list_length), 
                             'n_of_outliers': n_outliers}
        self.short_tokenized = lists_short_tokenized
        self.short_group = list_short_tokenized_group
                                   
    def get_input_for_tokens(self):     
        lists_tokenized = self.short_tokenized
             
        # 2. [PAD]                    
        temp_for_len = [len(inner_list) for inner_list in lists_tokenized]
        max_len = max(temp_for_len)            
        lists_pad_tokenized = []
        p = 0
        n_of_p = len(lists_tokenized)
        for inner_list in lists_tokenized:
            p += 1
            if len(inner_list) < max_len:
                while len(inner_list) < max_len:
                    inner_list.append('[PAD]')
            else:
                pass
            lists_pad_tokenized.append(inner_list)
            print('Padding for {}/{} has been done'.format(p, n_of_p))

        # 3. [CLS]
        lists_clspad_tokenized = []
        cls = 0
        n_of_cls = len(lists_pad_tokenized)                    
        for inner_list in lists_pad_tokenized:
            cls += 1
            inner_list.insert(0, '[CLS]')
            lists_clspad_tokenized.append(inner_list)
            print('Adding [CLS] for {}/{} has been done'.format(cls, n_of_cls))                        
            
        # 4. [SEP]
        lists_clspadsep_tokenized = []
        sep = 0
        n_of_sep = len(lists_clspad_tokenized)          
        for inner_list in lists_clspad_tokenized:
            sep += 1
            inner_list.append('[SEP]')
            lists_clspadsep_tokenized.append(inner_list)
            print('Adding [SEP] for {}/{} has been done'.format(sep, n_of_sep))    

        # 5. Attention mask
        lists_attention_masks = []
        a = 0
        n_of_a = len(lists_clspadsep_tokenized)
        for inner_list in lists_clspadsep_tokenized:
            a += 1
            attention_mask = [0 if token == '[PAD]' else 1 for token in inner_list]
            lists_attention_masks.append(attention_mask)
            print('Creating an attention mask for {}/{} has been done'.format(a, n_of_a))
                            
        self.input_for_tokens = lists_clspadsep_tokenized
        self.attention_masks = lists_attention_masks
        
        print('########## Input tokens have been set up. Please check by "instance.input_for_tokens" ##########')
        print('########## Input attention masks have been set up. Please check by "instance.attention_mask" ##########')
        
    def change_tokens_to_ids(self):
        tz = self.tokenizer
        lists_clspadsep_tokenized = self.input_for_tokens
        
        lists_ids = []
        for inner_list in lists_clspadsep_tokenized:
            list_ids = tz.convert_tokens_to_ids(inner_list)
            lists_ids.append(list_ids)
        
        self.ids = lists_ids

# Function for checking errors

def error_test(NLP_inst):
    lists_tokenized = NLP_inst.input_for_tokens
    len_tk = len(lists_tokenized)

    # Do all tokens have the same length? (Does padding )
    n = 0
    for _ in range(10):
        n += 1
        random_int_1 = np.random.randint(len_tk)
        random_int_2 = np.random.randint(len_tk)
        len_1 = len(NLP_inst.input_for_tokens[random_int_1])
        len_2 = len(NLP_inst.input_for_tokens[random_int_2])
        if len_1 == len_2:
            print('test...{}/{}'.format(n,10))
        else:
            stop
            print('????????? {} and {} unmatched! ?????????'.format(random_int_1, random_int_2))
    print('########## No errors have been found :) ##########')

# Gallup_NLP instance

answer_inst = Gallup_NLP(df, answer_col_name = '응답', group_col_name = '응답그룹', tokenizer_selection = 'kobert')
answer_inst.get_tokenized(cleaned_or_raw_sentences = 'cleaned')
answer_inst.get_short_tokenized(cut_criteron = 'iqr')
answer_inst.get_input_for_tokens()
answer_inst.change_tokens_to_ids()

error_test(answer_inst)

# !!!!!!!!!!!!!!!!!! GPU MUST !!!!!!!!!!!!!!!!!!!!!!!!
# Embedding ids and attention masks by KoBERT
# This should be implemented in several times

class Embedding:
    def __init__(self, gallup_nlp_instance, device_setting = 'cuda', model_selection = 'kobert'):
        self.nlp_instance = gallup_nlp_instance
        self.device_setting = device_setting
        self.model_selection = model_selection
        
        self.device = self._set_device()
        self.model = None            
        self.last_hidden_layers = None
        self.cls_tanh_layers = None
        self.all_hidden_layers = None
        self.cnn_dataset = None
        self.y = None
        
    def _set_device(self):
        if self.device_setting == 'cuda':
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        return device
        
    def get_model(self):        
        if self.model_selection == 'kobert':
            model = BertModel.from_pretrained('monologg/kobert')
        elif self.model_selection == 'bert':
            model = BertModel.from_pretrained('bert-base-uncased')
        self.model = model
        print('BertModel.from_pretrained({}) has been set up'.format(self.model_selection))
                    
    def get_embedded_tensors(self):
        nlp_instance = self.nlp_instance
        model = self.model
        device = self.device
        
        lists_ids = nlp_instance.ids
        lists_attention_masks = nlp_instance.attention_masks
        
        tensor_ids = torch.Tensor(lists_ids).type(torch.int32)
        tensor_masks = torch.Tensor(lists_attention_masks).type(torch.int32)
        
        model_device = model.to(device)
        tensor_last_hidden_layers = torch.Tensor()
        tensor_cls_tanh_layers = torch.Tensor()
        list_all_hidden_layers = []
        
        for i in range(len(tensor_ids)):            
            len_sequence = tensor_ids[i].shape[0]
            id = tensor_ids[i].reshape(1, len_sequence).to(device)
            mask = tensor_masks[i].reshape(1, len_sequence).to(device)
                                                                        
            with torch.no_grad():
                output = model_device(input_ids = id, attention_mask = mask, output_hidden_states = True)
            last = output[0].to('cpu')
            cls = output[1].to('cpu')
            
            tensor_last_hidden_layers = torch.concat([ tensor_last_hidden_layers, last ])
            tensor_cls_tanh_layers = torch.concat([ tensor_cls_tanh_layers, cls ])            
            list_all_hidden_layers.append(output[2])
                                   
            print('Embedding {}/{} has been done'.format(i+1, len(tensor_ids)))
        
        self.last_hidden_layers = tensor_last_hidden_layers
        self.cls_tanh_layers = tensor_cls_tanh_layers
        
        sample_size = tensor_cls_tanh_layers.shape[0]
        token_length = 1
        feature_size = tensor_cls_tanh_layers.shape[1]
        self.cls_tanh_layers = tensor_cls_tanh_layers.reshape(shape = (sample_size, token_length, feature_size))
        
        self.all_hidden_layers = list_all_hidden_layers
        self.model = model_device
        
    def get_cnn_dataset(self, n_of_hidden_layers = 4, cls_adding = True, stack_way = 1, erase_all_hidden_layers = False):
        all_hidden_layers = self.all_hidden_layers
        cls_tanh_layers = self.cls_tanh_layers
        temp_range = 12 - n_of_hidden_layers
        
        tensor_cnn_dataset = torch.Tensor()
        for i in range(len(all_hidden_layers)):
            inner_ten = torch.Tensor()
            if cls_adding:
                inner_ten = torch.concat( [ inner_ten, cls_tanh_layers[i].reshape( shape = (1,1,768) ) ] )
            for j in range(12,temp_range,-1):
                inner_ten = torch.concat( [ inner_ten, all_hidden_layers[i][j].to('cpu') ], dim=stack_way )
            tensor_cnn_dataset = torch.concat( [ tensor_cnn_dataset, inner_ten ] ,dim=0 )
            
        if erase_all_hidden_layers == True:
            self.all_hidden_layers = '### Cleaned for memory. Please check out the value "inst.cnn_dataset" ###'
        
        self.cnn_dataset = tensor_cnn_dataset
        
    def get_y(self, type_of_x = 'cnn_dataset', num_of_x_for_mean = 10):
        nlp_instance = self.nlp_instance
        
        if type_of_x == 'cnn_dataset':
            tensor_x = self.cnn_dataset
        elif type_of_x == 'cls_tanh_layers':
            tensor_x = self.cls_tanh_layers
        else:
            print('Please choose the type of the dataset between cnn_dataset and cls_tanh_layers')
            
        tensor_y = torch.Tensor()
        s = 0
        for group_num in nlp_instance.short_group:
            s += 1
            array_idxs = np.where( np.array(nlp_instance.short_group) == group_num )[0] # Finding indices having the same group as x
    
            list_group_num_idxs = []
            for _ in range(num_of_x_for_mean):
                list_group_num_idxs.append( array_idxs[np.random.randint( 0, len(array_idxs) )] ) # Randomly select n of indices 

            tensor_size = tensor_x[0].shape
            tensor_mean = torch.zeros(size = tensor_size)
            for idx in list_group_num_idxs:
                tensor_mean += tensor_x[idx]
        
            tensor_mean = tensor_mean / num_of_x_for_mean
            tensor_mean = tensor_mean.reshape(shape = (1, tensor_size[0], tensor_size[1]))
    
            try:
                tensor_y = torch.concat( [tensor_y, tensor_mean] )
            except:
                tensor_y = tensor_mean
            
            print('Y for {}/{} has been created'.format(s, len(tensor_x)))
        
        self.y = tensor_y
        print('########## Y has been set up ##########')

# Getting final dataset for learning

embedding_inst = Embedding(nlp_inst, device_setting = 'cuda')
embedding_inst.get_model()
embedding_inst.get_embedded_tensors()
embedding_inst.get_cnn_dataset()
embedding_inst.get_cnn_dataset(cls_adding = False)
embedding_inst.get_y(type_of_x = 'cls_tanh_layers')

# Dataloader will be used here








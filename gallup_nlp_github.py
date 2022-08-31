
#_________________________________________________________________________________________#

# Installing kobert
!pip3 install kobert-transformers
!pip3 install transformers
!pip3 install fairseq

#_________________________________________________________________________________________#

# Importing packages
from google.colab import files

import re
from statistics import mean, median
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import BertModel, BertTokenizer
from transformers import XLMRobertaTokenizer
from kobert_transformers import tokenization_kobert

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#_________________________________________________________________________________________#

# Checking the availability of CUDA
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

#_________________________________________________________________________________________#

# Uploading Datasets
"""
upload = files.upload()
"""

#_________________________________________________________________________________________#

# Loading and checking a dataset
df = pd.read_excel(r'')
df.iloc[:5,:]

#_________________________________________________________________________________________#

# Class for setting input data(ids, attention masks)
class Gallup_NLP:        
    def __init__(self, df, question_answer_concat = True, question_col_name = '질문', answer_col_name = '응답', group_col_name = None, keys = 'keys', tokenizer_selection = 'kobert'):
        self.df = df.sort_values(by=keys)
        self.question_answer_concat = question_answer_concat
        self.question_col_name = question_col_name
        self.raw_question = df[question_col_name].values
        self.answer_col_name = answer_col_name        
        self.raw_answer = df[answer_col_name].values
        self.group_col_name = group_col_name
        self.keys = keys
        self.tokenizer_selection = tokenizer_selection        
        self.raw_keys = df[keys].values
        
        self.raw = self._get_raw()
        self.sentence_group = self._get_group()
        self.tokenizer = self._get_tokenizer()

        self.max_short_len = None
        self.cleaned_sentences_short = None          
        self.ids = None
        self.attention_masks = None

        self.last_hidden_layers = None
        self.cls_tanh_layers = None
        self.all_hidden_layers = None
        self.cnn_dataset = None
        
    def _get_raw(self):
        raw_question = self.raw_question
        raw_answer = self.raw_answer
        
        if self.question_answer_concat:            
            def shorten(x, word='그 다음'):
                idx = x.find(word)
                if idx < 0:
                    return x
                elif idx > 0:
                    return x[:idx]
            raw_question = np.array([ shorten(q) for q in raw_question ],dtype='object')
            blank = np.array([ ' ' for _ in range(len(raw_answer)) ])
            raw = raw_question + blank + raw_answer
        else:
            raw = raw_answer
        return raw
                
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
        
    def _get_tokenizer(self):
        if self.tokenizer_selection == 'kobert':
            tz = tokenization_kobert.KoBertTokenizer.from_pretrained('monologg/kobert')
        elif self.tokenizer_selection == 'roberta':
            tz = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        print('########## Tokenizer({}) has been set up ##########'.format(self.tokenizer_selection))
        return tz
                            
    def get_cleaned(self):
        def sub(x):
            return re.sub("[^A-Za-z0-9가-힣 ]",'',x)                                    
                    
        self.cleaned_sentences = [ sub(sentence) for sentence in self.raw ]
        print('########## Sentences has been cleansed ##########')
    
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
    
    def get_ids(self):
        max_len = self.max_short_len
        short_sentences = self.cleaned_sentences_short
        tz = self.tokenizer
        
        def encoding_plus(x):
            return tz.encode_plus(x, max_length=max_len, padding='max_length', return_tensors='pt', return_attention_mask=True)
        
        ids = [encoding_plus(sentence)['input_ids'] for sentence in short_sentences]
        attention_masks = [encoding_plus(sentence)['attention_mask'] for sentence in short_sentences]

        self.ids = ids
        self.attention_masks = attention_masks

    def get_features(self, n_of_hidden_layers = 4, model_selection='kobert', device_selection='cpu'):
        ids = self.ids
        attention_masks = self.attention_masks

        if self.tokenizer_selection == 'kobert':
            model = BertModel.from_pretrained('monologg/kobert').to(device_selection)            
            print('####### {} model has been set up #######'.format(model_selection))

            last_hidden_layers = torch.Tensor()
            cls_tanh_layers = torch.Tensor()
            all_hidden_layers = []
            
            for i in range(len(ids)):
                len_sequence = ids[i].shape[0]
                id = ids[i].to(device_selection)
                mask = attention_masks[i].to(device_selection)
                                                                        
                with torch.no_grad():
                    output = model(input_ids = id, attention_mask = mask, output_hidden_states = True)
                last = output[0].to('cpu')
                cls = output[1].to('cpu')

                last_hidden_layers = torch.concat([ last_hidden_layers, last ])
                cls_tanh_layers = torch.concat([ cls_tanh_layers, cls ])            

                len_hidden_layers = len(output[2])
                start_idx = len_hidden_layers - n_of_hidden_layers
                temp1 = output[2][start_idx:]
                temp2 = [ hl.to('cpu') for hl in temp1 ]
                all_hidden_layers.append(temp2)
                
                print('Embedding {}/{} has been done'.format(i+1, len(ids)))

            self.last_hidden_layers = last_hidden_layers
            self.cls_tanh_layers = cls_tanh_layers
            self.all_hidden_layers = all_hidden_layers
            
       
        else:
            # 'xlmr.base' 'xlmr.large', 'xlmr.xl', 'xlmr.xxl'
            model = torch.hub.load('pytorch/fairseq:main', model_selection).to(device_selection)
         
            last_hidden_layers = torch.Tensor()
            cls_tanh_layers = torch.Tensor()
            all_hidden_layers = []
            
            for i in range(len(ids)):
                id = ids[i].to(device_selection)

                with torch.no_grad():
                    last = model.extract_features(id, return_all_hiddens=False)
                    
                    all_hiddens = model.extract_features(id, return_all_hiddens=True)
                    len_all_hiddens = len(all_hiddens)-1
                    cls = all_hiddens[len_all_hiddens][:,0,:]

                last_hidden_layers = torch.concat([ last_hidden_layers, last.to('cpu') ])
                cls_tanh_layers = torch.concat([ cls_tanh_layers, cls.to('cpu') ])

                len_hidden_layers = len(all_hiddens)
                start_idx = len_hidden_layers - n_of_hidden_layers
                temp1 = all_hiddens[start_idx:]
                temp2 = [ hl.to('cpu') for hl in temp1 ]
                all_hidden_layers.append(temp2)
                
                print('Embedding {}/{} has been done'.format(i+1, len(ids)))

            self.last_hidden_layers = last_hidden_layers
            self.cls_tanh_layers = cls_tanh_layers
            self.all_hidden_layers = all_hidden_layers

#___________________________________________________________#

# Execution
nlp_inst_temp1 = Gallup_NLP(df = df, question_answer_concat = True, question_col_name = '질문', answer_col_name = '응답', group_col_name = None, keys = 'key', tokenizer_selection = 'kobert')
nlp_inst_temp1.get_cleaned()
nlp_inst_temp1.get_short_sentences(cut_criterion='iqr')
nlp_inst_temp1.get_ids()
nlp_inst_temp1.get_features(n_of_hidden_layers=4, model_selection='kobert', device_selection='cuda')

nlp_inst_temp2 = Gallup_NLP(df = df, question_answer_concat = True, question_col_name = '질문', answer_col_name = '응답', group_col_name = None, keys = 'key', tokenizer_selection = 'roberta')
nlp_inst_temp2.get_cleaned()
nlp_inst_temp2.get_short_sentences(cut_criterion='iqr')
nlp_inst_temp2.get_ids()
nlp_inst_temp2.get_features(n_of_hidden_layers=4, model_selection='xlmr.base', device_selection='cuda')

#_________________________________________________________________________________________#

# Trained model will be here
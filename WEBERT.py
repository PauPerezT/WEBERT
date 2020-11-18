
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Libraries
from torch.utils.data import Dataset
import torch
import pandas as pd
import csv

import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv
import os
import gc
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertModel

from tqdm import tqdm
from utils import noPunctuation, StopWordsRemoval,noPunctuationExtra
import copy

from transformers import AutoTokenizer, AutoModel
from scipy.stats import kurtosis, skew


#%%
# specify GPU device

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    
    torch.cuda.empty_cache()
    torch.cuda.get_device_name(0)
    n_gpu = torch.cuda.device_count()


class BERT:
    """
    WEBERT-BERT computes BERT to get static or dynamic embeddings. 
    BERT uses Transformers (https://github.com/huggingface/transformers). 
    It can be computed using english and spanish (multilingual) model.
    Also considers cased or uncased options, and stopword removal.
    
    :param inputs: input data
    :param file: name of the document.
    :param language: input language (By defalut: english).
    :param stopwords: boolean variable for removing stopwords (By defalut: False).
    :param model: base or large model (By defalut: base).
    :param cased: boolean variable to compute cased or lower-case model (By defalut: False).
    :param cuda: boolean value for using cuda to compute the embeddings, True for using it. (By defalut: False).
    :returns: WEBERT object
    """    
    
    def __init__(self,inputs, file, language='english', stopwords=False, model='base', cased=False, cuda=False):   
        


        
        
        self.data=inputs
        self.file_names=file
        self.words=[]
        self.word_counting=[] 

        
        self.stopwords=stopwords
        self.language=language
        self.neurons=768
        if model=='large':
            self.neurons=1024
        cased_str='cased'
        self.cased=cased
        if cased:
            cased_str='cased'
        
        self.model='bert-'+model+'-'+cased_str
        if self.language=='spanish':
            self.model='bert-'+model+'-multilingual-'+cased_str
            
        if cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device='cpu'

        
        
    def preprocessing(self,inputs):
        
        """
        Text Pre-processing
        
        :param inputs: input data
        :returns: proprocessed text
        """
        data=inputs
        
        docs=[]
        for j in range (len(data)):
            
            text =data[j]
            
    
            text_aux=copy.copy(text)
            text_aux=noPunctuationExtra(text_aux)
            text_aux=text_aux.replace('. '," [SEP]" )
            if text_aux[-5:]=="[SEP]":
                text_aux=text_aux[0:-5]
            text_org=noPunctuationExtra(text.replace('.',' '))
            text_aux=noPunctuation(text_aux)
            text_org=noPunctuation(text_org)
            
            
            if self.stopwords:
                text=StopWordsRemoval(text_aux,self.Language)
            self.words.append(text_org.split())
            docs.append(text_aux)
        return docs
            
    
    def __data_preparation(self):
        
        """
        Data preparation and adaptation for BERT to work properly

        """
        
        # add special tokens for BERT to work properly
        data=self.preprocessing(self.data)
        
        sentences = ["[CLS] " + query + " [SEP]" for query in data]


        
        # Tokenize with BERT tokenizer
        
        tokenizer = BertTokenizer.from_pretrained(self.model, do_lower_case=self.cased)
       

        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        self.word_counting= [len(words)-1 for words in tokenized_texts]

        self.tokenized_texts=tokenized_texts
        

        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        self.indexed_tokens = [np.array(tokenizer.convert_tokens_to_ids(tk)) for tk in tokenized_texts]

        data_ids = [torch.tensor(tokenizer.convert_tokens_to_ids(x)).unsqueeze(0) for x in tokenized_texts]

        
        # Create an iterator of our data with torch DataLoader 
        
 
        self.data_dataloader = DataLoader(data_ids,  batch_size=1)
        
        
                      
        
    def get_bert_embeddings(self, path, dynamic=True, static=False):
        """
        Bert embeddings computation using Transformes. It store and transforms the texts into BERT embeddings. The embeddings are stored in csv files.
        
        :param path: path to save the embeddings
        :param dynamic: boolean variable to compute the dynamic embeddings (By defalut: True).
        :param static: boolean variable to compute the static embeddings (By defalut: False).
        :returns: static embeddings if static=True
        
        """  
        
        self.__data_preparation()
        
        data_stat=[]

        bert = BertModel.from_pretrained(self.model).embeddings
        bert=bert.to(self.device)



        for idx_batch, sequence in enumerate(self.data_dataloader,1):
            sequence=sequence.to(self.device)

            ids_tokens=np.where((self.indexed_tokens[idx_batch-1]!=101) &(self.indexed_tokens[idx_batch-1]!=102) & (self.indexed_tokens[idx_batch-1]!=112))[0]
            tokens=np.array(self.tokenized_texts[idx_batch-1])[ids_tokens]
            index=[]
            index_num=[]
            for i in range(len(tokens)):
                if [idx for idx, x in enumerate(tokens[i]) if x=='#'] ==[]:
                    index.append(i)
                else:
                    index_num.append(i)
                
            

            bert_embeddings=bert(sequence)[0][:,ids_tokens].cpu().detach()

            embeddings=torch.tensor(np.zeros((bert_embeddings.shape[1]-len(index_num),bert_embeddings.shape[2])))
            count=0
            if index_num!=[]:
                for idx in range (len(ids_tokens)):
                     if np.where(index_num==np.array([idx]))[0].size!=0:
                         nums=bert_embeddings[0][idx]*bert_embeddings[0][idx-1]
                         embeddings[idx-count-1]=nums.cpu().detach()
                         count+=1
                     else:
                         embeddings[idx-count]=bert_embeddings[0][idx].cpu().detach()
            else:
                
                embeddings=bert_embeddings[0]
            
            if static:
                for emb in embeddings:
                    data_stat.append(emb)
                    
   
 

            if dynamic: 
                i=1
                data_csv=[]
                labelstf= []
                labelstf.append('Word')   
                for n in range (self.neurons):
                    labelstf.append('Neuron'+str(n+1))  
                for emb in embeddings:
                    data_csv.append(np.hstack((self.words[idx_batch-1][i-1], emb)))
                    i+=1
                with open(path+self.file_names+'.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                
                    writer.writerow(labelstf)
                    writer.writerows(data_csv)
                
        if static:        
            wordar=np.vstack(data_stat)
            del data_stat
            meanBERT=np.mean(wordar, axis=0)
            stdBERT=np.std(wordar, axis=0)
            kurtosisBERT=kurtosis(wordar, axis=0)
            skewnessBERT=skew(wordar, axis=0)
            skewnessBERT=skew(wordar, axis=0)
            minBERT=np.min(wordar, axis=0)
            maxBERT=np.max(wordar, axis=0)
            statisticalMeasures=np.hstack((meanBERT, stdBERT, kurtosisBERT, skewnessBERT,minBERT, maxBERT))
            
            
                   
            return statisticalMeasures
        
        else:
            del embeddings
            #del bert_embeddings
            del bert
            del self.data_dataloader
            del self.tokenized_texts
            del self.data
            
            gc.collect()
        
        

                
#%%
class BETO:
    """
    WEBERT-BETO computes BETO to get static or dynamic embeddings. 
    BETO is a pretrained BERT model from spanish corpus (https://github.com/dccuchile/beto).
    BETO uses Transformers (https://github.com/huggingface/transformers). 
    It can be computed using only spanish model.
    Also considers cased or uncased options, and stopword removal.
    
    :param inputs: input data
    :param file: name of the document.
    :param stopwords: boolean variable for removing stopwords (By defalut: False).
    :param model: base or large model (By defalut: base).
    :param cased: boolean variable to compute cased or lower-case model (By defalut: False).
    :param cuda: boolean value for using cuda to compute the embeddings, True for using it. (By defalut: False).
    :returns: WEBERT object
    """    
    
    def __init__(self,inputs,file, stopwords=False, model='base', cased=False, cuda=False):   
        


        
        
        self.data=inputs
        self.file_names=file
        self.words=[]
        self.word_counting=[] 

        
        self.stopwords=stopwords

        self.neurons=768
        if model=='large':
            self.neurons=1024
        cased_str='cased'
        self.cased=cased
        if cased:
            cased_str='cased'
        
        self.model='dccuchile/bert-'+model+'-spanish-wwm'+'-'+cased_str
        if cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device='cpu'
        
        
        
    def preprocessing(self,inputs):
        
        """
        Text Pre-processing
        
        :param inputs: input data
        :returns: proprocessed text
        """
        data=inputs
        
        docs=[]
        for j in range (len(data)):
            
            text =data[j]
            
    
            text_aux=copy.copy(text)
            text_aux=noPunctuationExtra(text_aux)
            text_aux=text_aux.replace('. '," [SEP]" )
            if text_aux[-5:]=="[SEP]":
                text_aux=text_aux[0:-5]
            text_org=noPunctuationExtra(text.replace('.',' '))
            text_aux=noPunctuation(text_aux)

            text_org=noPunctuation(text_org)
            
            
            if self.stopwords:
                text=StopWordsRemoval(text_aux,self.Language)
            self.words.append(text_org.split())
            docs.append(text_aux)
        return docs
            
    
    def __data_preparation(self):
        
        """
        Data preparation and adaptation for BETO to work properly

        """
        
        # add special tokens for BERT to work properly
        data=self.preprocessing(self.data)
        
        sentences = ["[CLS] " + query + " [SEP]" for query in data]


        
        # Tokenize with BERT tokenizer
        
        tokenizer = BertTokenizer.from_pretrained(self.model, do_lower_case=self.cased)
       

        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        self.word_counting= [len(words)-1 for words in tokenized_texts]

        self.tokenized_texts=tokenized_texts
        

        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        self.indexed_tokens = [np.array(tokenizer.convert_tokens_to_ids(tk)) for tk in tokenized_texts]

        data_ids = [torch.tensor(tokenizer.convert_tokens_to_ids(x)).unsqueeze(0) for x in tokenized_texts]

        
        # Create an iterator of our data with torch DataLoader 
        
 
        self.data_dataloader = DataLoader(data_ids,  batch_size=1)
        
        
                      
        
    def get_bert_embeddings(self, path, dynamic=True, static=False):
        """
        BETO embeddings computation using Transformes. It store and transforms the texts into BETO embeddings. The embeddings are stored in csv files.
        
        :param path: path to save the embeddings
        :param dynamic: boolean variable to compute the dynamic embeddings (By defalut: True).
        :param static: boolean variable to compute the static embeddings (By defalut: False).
        :returns: static embeddings if static=True
        
        """  
        
        self.__data_preparation()
        
        data_stat=[]

        bert = BertModel.from_pretrained(self.model).embeddings
        bert=bert.to(self.device)



        for idx_batch, sequence in enumerate(self.data_dataloader,1):
            sequence=sequence.to(self.device)

            ids_tokens=np.where((self.indexed_tokens[idx_batch-1]!=3) &(self.indexed_tokens[idx_batch-1]!=5) & (self.indexed_tokens[idx_batch-1]!=4))[0]
            tokens=np.array(self.tokenized_texts[idx_batch-1])[ids_tokens]
            index=[]
            index_num=[]
            for i in range(len(tokens)):
                if [idx for idx, x in enumerate(tokens[i]) if x=='#'] ==[]:
                    index.append(i)
                else:
                    index_num.append(i)
                
            

            bert_embeddings=bert(sequence)[0][:,ids_tokens].cpu().detach()

            embeddings=torch.tensor(np.zeros((bert_embeddings.shape[1]-len(index_num),bert_embeddings.shape[2])))
            count=0
            if index_num!=[]:
                for idx in range (len(ids_tokens)):
                     if np.where(index_num==np.array([idx]))[0].size!=0:
                         nums=bert_embeddings[0][idx]*bert_embeddings[0][idx-1]
                         embeddings[idx-count-1]=nums.cpu().detach()
                         count+=1
                     else:
                         embeddings[idx-count]=bert_embeddings[0][idx].cpu().detach()
            else:
                
                embeddings=bert_embeddings[0]
            
            if static:
                for emb in embeddings:
                    data_stat.append(emb)
                    
   
 

            if dynamic: 
                i=1
                data_csv=[]
                labelstf= []
                labelstf.append('Word')   
                for n in range (self.neurons):
                    labelstf.append('Neuron'+str(n+1))  
                for emb in embeddings:
                    data_csv.append(np.hstack((self.words[idx_batch-1][i-1], emb)))
                    i+=1
                with open(path+self.file_names+'.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                
                    writer.writerow(labelstf)
                    writer.writerows(data_csv)
                
        if static:        
            wordar=np.vstack(data_stat)
            del data_stat
            meanBERT=np.mean(wordar, axis=0)
            stdBERT=np.std(wordar, axis=0)
            kurtosisBERT=kurtosis(wordar, axis=0)
            skewnessBERT=skew(wordar, axis=0)
            skewnessBERT=skew(wordar, axis=0)
            minBERT=np.min(wordar, axis=0)
            maxBERT=np.max(wordar, axis=0)
            statisticalMeasures=np.hstack((meanBERT, stdBERT, kurtosisBERT, skewnessBERT,minBERT, maxBERT))
            
            
                   
            return statisticalMeasures
        else:
            del embeddings
            #del bert_embeddings
            del bert
            del self.data_dataloader
            del self.tokenized_texts
            del self.data
            
            gc.collect()
                        
            


#%%
class SciBERT:
    """
    WEBERT-SCIBERT computes BERT to get static or dynamic embeddings. 
    SCIBERT is a pre-trained model on english scientific text (https://github.com/allenai/scibert).
    BERT uses Transformers (https://github.com/huggingface/transformers). 
    This toolkit only considered the scivocab model.
    Also considers cased or uncased options, and stopword removal.
    
    :param inputs: input data
    :param file: name of the document.
    :param stopwords: boolean variable for removing stopwords (By defalut: False).
    :param cased: boolean variable to compute cased or lower-case model (By defalut: False).
    :param cuda: boolean value for using cuda to compute the embeddings, True for using it. (By defalut: False).
    :returns: WEBERT object
    """    
    
    def __init__(self,inputs, file, stopwords=False, cased=False, cuda=False):   
        


        
        
        self.data=inputs
        self.file_names=file
        self.words=[]
        self.word_counting=[] 

        
        self.stopwords=stopwords
        

        self.neurons=768

            
        cased_str='uncased'
        self.cased=cased
        if cased:
            cased_str='cased'
        
        self.model='allenai/scibert_scivocab_'+cased_str

        if cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device='cpu'
        
        
    def preprocessing(self,inputs):
        
        """
        Text Pre-processing
        
        :param inputs: input data
        :returns: proprocessed text
        """
        data=inputs
        
        docs=[]
        for j in range (len(data)):
            
            text =data[j]
            
    
            text_aux=copy.copy(text)
            text_aux=noPunctuationExtra(text_aux)
            text_aux=text_aux.replace('. '," [SEP]" )
            if text_aux[-5:]=="[SEP]":
                text_aux=text_aux[0:-5]
            text_org=noPunctuationExtra(text.replace('.',' '))
            text_aux=noPunctuation(text_aux)

            text_org=noPunctuation(text_org)
            
            
            if self.stopwords:
                text=StopWordsRemoval(text_aux,self.Language)
            self.words.append(text_org.split())
            docs.append(text_aux)
        return docs
            
    
    def __data_preparation(self):
        
        """
        Data preparation and adaptation for SciBERT to work properly

        """
        
        # add special tokens for BERT to work properly
        data=self.preprocessing(self.data)
        
        sentences = ["[CLS] " + query + " [SEP]" for query in data]


        
        # Tokenize with BERT tokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(self.model, do_lower_case=self.cased)
       

        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        self.word_counting= [len(words)-1 for words in tokenized_texts]

        self.tokenized_texts=tokenized_texts
        

        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        self.indexed_tokens = [np.array(tokenizer.convert_tokens_to_ids(tk)) for tk in tokenized_texts]

        data_ids = [torch.tensor(tokenizer.convert_tokens_to_ids(x)).unsqueeze(0) for x in tokenized_texts]

        
        # Create an iterator of our data with torch DataLoader 
        
 
        self.data_dataloader = DataLoader(data_ids,  batch_size=1)
        
        
                      
        
    def get_bert_embeddings(self, path, dynamic=True, static=False):
        """
        SciBert embeddings computation using Transformes. It store and transforms the texts into SciBERT embeddings. The embeddings are stored in csv files.
        
        :param path: path to save the embeddings
        :param dynamic: boolean variable to compute the dynamic embeddings (By defalut: True).
        :param static: boolean variable to compute the static embeddings (By defalut: False).
        :returns: static embeddings if static=True
        
        """  
        
        self.__data_preparation()
        
        data_stat=[]

        bert = AutoModel.from_pretrained(self.model).embeddings
        bert=bert.to(self.device)



        for idx_batch, sequence in enumerate(self.data_dataloader,1):
            sequence=sequence.to(self.device)

            ids_tokens=np.where((self.indexed_tokens[idx_batch-1]!=102) &(self.indexed_tokens[idx_batch-1]!=103) &(self.indexed_tokens[idx_batch-1]!=101) )[0]
            tokens=np.array(self.tokenized_texts[idx_batch-1])[ids_tokens]
            index=[]
            index_num=[]
            for i in range(len(tokens)):
                if [idx for idx, x in enumerate(tokens[i]) if x=='#'] ==[]:
                    index.append(i)
                else:
                    index_num.append(i)
                
            

            bert_embeddings=bert(sequence)[0][:,ids_tokens].cpu().detach()

            embeddings=torch.tensor(np.zeros((bert_embeddings.shape[1]-len(index_num),bert_embeddings.shape[2])))
            count=0
            if index_num!=[]:
                for idx in range (len(ids_tokens)):
                     if np.where(index_num==np.array([idx]))[0].size!=0:
                         nums=bert_embeddings[0][idx]*bert_embeddings[0][idx-1]
                         embeddings[idx-count-1]=nums.cpu().detach()
                         count+=1
                     else:
                         embeddings[idx-count]=bert_embeddings[0][idx].cpu().detach()
            else:
                
                embeddings=bert_embeddings[0]
            
            if static:
                for emb in embeddings:
                    data_stat.append(emb)
                    
   
 

            if dynamic: 
                i=1
                data_csv=[]
                labelstf= []
                labelstf.append('Word')   
                for n in range (self.neurons):
                    labelstf.append('Neuron'+str(n+1))  
                for emb in embeddings:
                    data_csv.append(np.hstack((self.words[idx_batch-1][i-1], emb)))
                    i+=1
                with open(path+self.file_names+'.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                
                    writer.writerow(labelstf)
                    writer.writerows(data_csv)
                
        if static:        
            wordar=np.vstack(data_stat)
            del data_stat
            meanBERT=np.mean(wordar, axis=0)
            stdBERT=np.std(wordar, axis=0)
            kurtosisBERT=kurtosis(wordar, axis=0)
            skewnessBERT=skew(wordar, axis=0)
            skewnessBERT=skew(wordar, axis=0)
            minBERT=np.min(wordar, axis=0)
            maxBERT=np.max(wordar, axis=0)
            statisticalMeasures=np.hstack((meanBERT, stdBERT, kurtosisBERT, skewnessBERT,minBERT, maxBERT))
            
            del embeddings
            #del bert_embeddings
            del bert
            del self.data_dataloader
            del self.tokenized_texts
            del self.data
            
            

            
            
                   
            return statisticalMeasures
        else:
            del embeddings
            #del bert_embeddings
            del bert
            del self.data_dataloader
            del self.tokenized_texts
            del self.data
            
            gc.collect()
            
        
        
            


        

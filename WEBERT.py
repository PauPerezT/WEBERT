
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT Embeddings

->BERT.py

Created on Fri Feb  7 19:19:38 2020

@author: P.A. Perez-Toro
@email:paula.perezt@udea.edu.co


"""
#%% Libraries
from torch.utils.data import Dataset
import torch
import pandas as pd
import csv

import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv
import os
import argparse
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertModel

from tqdm import tqdm
from utils import noPunctuation, StopWordsRemoval,noPunctuationExtra
import copy



from scipy.stats import kurtosis, skew


def create_fold(new_folder):
    if os.path.isdir(new_folder)==False:
        os.makedirs(new_folder)


#%%
# specify GPU device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    
    torch.cuda.empty_cache()
    torch.cuda.get_device_name(0)
    n_gpu = torch.cuda.device_count()


print(device)
class BERT:
    
    def __init__(self,inputs,file_names, language='english', stopwords=False, model='base', cased=False):
        """
        It computes BERT to get static or dynamic embeddings. BERT uses Transformers (https://github.com/huggingface/transformers). It can be computed using english and spanish (multilingual) model. Also considers cased or uncased options, and remotion of stopwords.
        :param data: input data
        :param file names: name of the document.
        :param language: input language (english).
        :param stopwords: boolean variable for the stopword remotion (False).
        :param model: base or large model (base).
        :param cased: boolean variable to compute cased or lower-case model (False).
        :param words: words after pre-processing
        :param word_counting: number of words after pre-processing.
        :param neurons: output dimension according to the model (768).
        
        """    
        
        
        
        self.data=inputs
        self.file_names=file_names
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
            self.model='bert-'+model+'-multilingual-'+cased
        
        
        
    def __preprocessing(self):
        """
        Text Pre-processing

        """
        docs=[]
        for j in range (len(self.data)):
            
            text =self.data[j]
            
    
            text_aux=copy.copy(text)
            text_aux=noPunctuationExtra(text_aux)
            text_aux=text_aux.replace('.'," [SEP]" )
            if text_aux[-5:]=="[SEP]":
                text_aux=text_aux[0:-5]
            text_org=noPunctuationExtra(text.replace('.',' '))

            text_org=noPunctuation(text_org)
            
            
            if self.stopwords:
                text=StopWordsRemoval(text_aux,self.Language)
            self.words.append(text_org.split())
            docs.append(text_aux)
        self.data=docs
            
    
    def __data_preparation(self):
        """
        Data preparation and adaptation for BERT to work properly

        """
        
        # add special tokens for BERT to work properly
        self.__preprocessing()
        
        sentences = ["[CLS] " + query + " [SEP]" for query in self.data]


        
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
        Bert embeddings computation using Transformes. It store and transforms the texts into BERT embeddings. The embedings are stored in csv files.
        :param path: path to save the embeddings
        :param dynamic: boolean variable to compute the dynamic embeddings (True).
        :param static: boolean variable to compute the static embeddings (False).
        :returns: static embeddings if static=True
        """  
        
        self.__data_preparation()
        
        data_stat=[]

        bert = BertModel.from_pretrained(self.model).embeddings
        bert=bert.to(device)



        for idx_batch, sequence in enumerate(self.data_dataloader,1):
            sequence=sequence.to(device)

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
        
        

                
                
            


            

#%%

if __name__ == '__main__':
    
    
    
    
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('-f','--files_path', default='./texts/',help='File folder of the set of documents', action="store")
    parser.add_argument('-s','--save_path', default='./bert_embeddings/',help='Path to save the embeddings', action="store")
    parser.add_argument('-d','--dynamic', default=True, help='Boolean value to get dynamic features= True. By defaul True.', choices=(True, False))
    parser.add_argument('-st','--static', default=False, help='Boolean value to get static features= True from the embeddings such as mean, standard deviation, kurtosis, skeweness, min and max. By default False.', choices=(True, False))
    parser.add_argument('-l','--language', default='english',help='Chosen language. Here is available only english or spanish. By default english.', choices=('english', 'spanish'))
    parser.add_argument('-sw','--stopwords', default=False, help='Boolean value, set True if you want to remove stopwords, By default False.' , choices=(True, False))
    parser.add_argument('-m','--model', default='base', help='Bert models, two options base and large. By default base.', choices=('base', 'large'))
    parser.add_argument('-c','--cased', default=False, help='Boolean value for cased= True o lower-cased= False models. By defaul False.', choices=(True, False))
    #parser.print_help()
    args = parser.parse_args()

    files_path=args.files_path
    save_path=args.save_path 
    language=str(args.language)
    stopwords=args.stopwords
    model=str(args.model)
    cased=args.cased
    dynamic=args.dynamic
    static=args.static

    
    
    files=np.hstack(sorted([f for f in os.listdir(files_path) if f.endswith('.txt')]))
    file_names=np.hstack([".".join(f.split(".")[:-1]) for f in files ])
    folder_path_static=save_path+'/Static/'
    folder_path=save_path+'/Dynamic/'
    create_fold(folder_path)
    create_fold(folder_path_static)
    j=0
    
    
    neurons=768
    if model=='large':
        neurons=1024
    
    if static:
        labelstf=[]
        labelstf.append('File')   
        for n in range (neurons):
            labelstf.append('Avg Neuron'+str(n+1)) 
        for n in range (neurons):
            labelstf.append('STD Neuron'+str(n+1)) 
        for n in range (neurons):
            labelstf.append('Skew Neuron'+str(n+1)) 
        for n in range (neurons):
            labelstf.append('Kurt Neuron'+str(n+1)) 
        for n in range (neurons):
            labelstf.append('Min Neuron'+str(n+1)) 
            
        for n in range (neurons):
            labelstf.append('Max Neuron'+str(n+1)) 
    
        with open(folder_path_static+'Bert_Static_Features.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
        
            writer.writerow(labelstf)

    pbar=tqdm(files)
        
    
    for file in pbar:
        pbar.set_description("Processing %s" % file)
        data = open(files_path+'/'+file)

        file_name=file_names[j]
        data_input=list(data)
        
        bert=BERT(data_input,file_name, language=language, stopwords=stopwords, 
                  model=model, cased=cased)
        
        j+=1
 
        if static:
            data_stat=bert.get_bert_embeddings(folder_path, dynamic=dynamic, static=static)
            with open(folder_path_static+'Bert_Static_Features.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(np.hstack((file_name, data_stat)))
        else:
            bert.get_bert_embeddings(folder_path, dynamic=dynamic, static=static)
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:45:40 2020

@author: P.A. Perez-Toro
"""

#%%Libraries

import argparse

from utils import create_fold
import csv
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from WEBERT import BERT, BETO, SciBERT
#%%



if __name__ == '__main__':
    
    
    
    
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('-f','--files_path', default='./texts/',help='File folder of the set of documents', action="store")
    parser.add_argument('-s','--save_path', default='./bert_embeddings/',help='Path to save the embeddings', action="store")
    parser.add_argument('-bm','--bert_model', default='Bert',help='Choose between three different BERT models: Bert, Beto and SciBert. By default BERT', choices=('Bert','Beto', 'SciBert'))
    parser.add_argument('-d','--dynamic', default=True, help='Boolean value to get dynamic features= True. By default True.', choices=(True, False))
    parser.add_argument('-st','--static', default=False, help='Boolean value to get static features= True from the embeddings such as mean, standard deviation, kurtosis, skeweness, min and max. By default False.', choices=(True, False))
    parser.add_argument('-l','--language', default='english',help='Chosen language (only available for BERT model). Here is available only english or spanish. By default english.', choices=('english', 'spanish'))
    parser.add_argument('-sw','--stopwords', default=False, help='Boolean value, set True if you want to remove stopwords, By default False.' , choices=(True, False))
    parser.add_argument('-m','--model', default='base', help='Bert models, two options base and large. By default base.', choices=('base', 'large'))
    parser.add_argument('-c','--cased', default=False, help='Boolean value for cased= True o lower-cased= False models. By defaul False.', choices=(True, False))
    #parser.print_help()
    args = parser.parse_args()

    files_path=args.files_path
    save_path=args.save_path 
    bert_model=str(args.bert_model)
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
    if (model=='large') & (bert_model!='SciBert'):
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
    
        with open(folder_path_static+bert_model+'_Static_Features.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
        
            writer.writerow(labelstf)

    pbar=tqdm(files)
        
    
    for file in pbar:
        pbar.set_description("Processing %s" % file)
        data = pd.read_csv(files_path+'/'+file, sep='\t', header=None)

        file_name=file_names[j]
        data_input=list(data[0])
        if bert_model=='Bert':
            bert=BERT(data_input,file_name, language=language, stopwords=stopwords, 
                      model=model, cased=cased)
        elif bert_model=='Beto':
            bert=BETO(data_input,file_name, stopwords=stopwords, 
                      model=model, cased=cased)
        elif bert_model=='SciBert':
            bert=SciBERT(data_input,file_name, stopwords=stopwords,
                         cased=cased)
        
        j+=1
 
        if static:
            data_stat=bert.get_bert_embeddings(folder_path, dynamic=dynamic, static=static)
            with open(folder_path_static+bert_model+'_Static_Features.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(np.hstack((file_name, data_stat)))
        else:
            bert.get_bert_embeddings(folder_path, dynamic=dynamic, static=static)
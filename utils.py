"""
@Author: Paula Andrea Perez Toro <PauPerezT>
@Date:   2018-11-27T20:36:36-05:00
@Email:  paula.perezt@udea.edu.co
@Filename: utils.py
# @Last modified by:   Paula Andrea Perez Toro
# @Last modified time: 2018-12-03T21:45:52-05:00

"""
"""
#%%%%%%% Natural Language Basic preprocessing code %%%%%%%#

    #%% Contains %%#
        *noPunctuation: to eliminate the Punctuation
        *PunctuationExtra: to eliminate the Punctuation related to spanish symbols
        *StopWordsRemoval: to eliminate stopwords
        *HesitationsRemoval: to remove hesitation labels (for example: [h mm])
        *Createfolder: to create a folder
        

"""

#%% Import Libraries
import string
import spacy
from nltk.corpus import stopwords
import unicodedata
import re
import os
import argparse

#%% noPunctuation
def noPunctuation(text):
    #eliminate the punctuation in form of characters
    nopunctuation= [char for char in text if char not in string.punctuation]

    #Now eliminate the punctuation and convert into a whole sentence
    nopunctuation=''.join(nopunctuation)

    #Split each words present in the new sentence
    nopunctuation.split()

    return nopunctuation
#%%
def noPunctuationExtra(text):
    
    texttoP=text
    #Just for this app without '.'
    forbidden1 = ('‘','[',']','<','>','+','%','=','\'','°','—','\'','*','º','%', '|', '»','«','?', 'Â¿', 'Â¡', '!','/' ,',','(', ')' ,';', '$', ':', '&','…', '...','_', '”','"', '“', 'XXXX', '’', '¿', '¡','-', '#','‐')
    for i in range (len(forbidden1)):
    
        idx=[n for n in range(len(texttoP)) if text.find(forbidden1[i], n) == n]
        if len(idx)!=0:
            for j in range(len(idx)):
                #texttoP=texttoP.replace(texttoP[idx[j]],' ')
                if forbidden1[i]=='.':
                    texttoP=texttoP.replace(texttoP[idx[j]],'\n')
                else:
                    texttoP=texttoP.replace(texttoP[idx[j]],' ')
                
    #print(texttoP)
    return texttoP

 #%%   
def removeURL(text):

    text = re.sub('https*://[\w\.]+\.com[\w/\-]+|https*://[\w\.]+\.com|[\w\.]+\.com/[\w/\-]+', lambda x:re.findall('(?<=\://)[\w\.]+\.com|[\w\.]+\.com', x.group())[0], text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text=re.sub(r'http\S+', '', text)
    text= re.sub('[\w\.]+@+[\w\.]+', '', text)
    text= re.sub('@+[\w\.]+', '', text)
    text= re.sub('[\w\.]+@+', '', text)

    #print(text)
    return text

#%%
def removeNumbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    
    return text
#%%
def removeEmojis(text):
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text) # no emoji
#%% StopWordsRemoval
def StopWordsRemoval(text,language='spanish'):
    #Now eliminate stopwords
    clean_sentence= [word for word in text.split() if word.lower() not in stopwords.words('spanish')]
    clean_sentence=' '.join(clean_sentence)

    return clean_sentence

#%% Lemmatizer
def Lemmatizer(text,language='spanish'):
    #nlp = en_core_web_sm.load()
    nlp = spacy.load('es_core_news_sm')
    #nlp = spacy.load('es_core_news_md')
    doc = nlp(text)
    tokenLemma=[]

    for token in doc:
        #print(token, token.lemma, token.lemma_)
        tokenLemma.append(token.lemma_)
    tokenLemma=' '.join(tokenLemma)
    return tokenLemma


#%% Lemmatizer
def stemming(text,language='spanish'):
    
    #First Tokenaize
    words = [word for word in wordpunct_tokenize(text)]

    
    #Stemming
    porter_stemmer = PorterStemmer()
    stemmers = [porter_stemmer.stem(word) for word in words]
    textStemm = [stem for stem in stemmers if stem.isalpha() and len(stem) > 1]
    
    textStemm=' '.join(textStemm)

    
    return textStemm


#%%
def botonSel(text):
    idx1=[n for n in range(len(text)) if text.find('botones seleccion:', n) == n]
    idx2=[n for n in range(len(text)) if text.find('))', n) == n]
    idxlen=len(idx1)
    idxlen2=len(idx2)
    
    
    while idxlen!=0:
        
        if idx1[0]<idx2[0]:
                text=text.replace(text[idx1[0]:idx2[0]+1],"")
                
        else:
                text=text.replace(text[idx1[0]:idx2[1]+1],"")
                
        
        
    
        idx1=[n for n in range(len(text)) if text.find('boton enlace:', n) == n]
        idx2=[n for n in range(len(text)) if text.find('))', n) == n]
        idxlen=len(idx1)
        #print(idxlen)
        idxlen2=len(idx2)
        #print(idx1)


    #print(text)
    return text
        
        
#%% HesitationsRemoval

def HesitationsRemoval(text):

    idx1=[n for n in range(len(text)) if text.find('[', n) == n]
    idx2=[n for n in range(len(text)) if text.find(']', n) == n]
    idxlen=len(idx1)
    idxlen2=len(idx2)
 

    while idxlen>0 and idxlen2:
        
        
        
        text=text.replace(text[idx1[0]:idx2[0]+1],"")
        

        idx1=[n for n in range(len(text)) if text.find('[', n) == n]
        idx2=[n for n in range(len(text)) if text.find(']', n) == n]
        idxlen=len(idx1)
        idxlen2=len(idx2)
        
    idx1=[n for n in range(len(text)) if text.find('(', n) == n]
    idx2=[n for n in range(len(text)) if text.find(')', n) == n]
    idxlen=len(idx1)
    idxlen2=len(idx2)
 

    while idxlen>0 and idxlen2:
        
        
        
        text=text.replace(text[idx1[0]:idx2[0]+1],"")
        

        idx1=[n for n in range(len(text)) if text.find('(', n) == n]
        idx2=[n for n in range(len(text)) if text.find(')', n) == n]
        idxlen=len(idx1)
        idxlen2=len(idx2)
        



    text= re.sub(r'\*.*?\*', '', text)
    

        #print(idx1)


    #print(text)
    return text


#%% spk1Removal

def spk1Removal(text):



    text=text.replace('spk1',"")
    text=text.replace(' AH '," ")
    text=text.replace(' EH '," ")

    return text
#%% toLowerCase

def toLowerCase(text):
    text=text.lower()
    return text

#%% accentRemoval
    

def accentRemoval(df):
    trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
    data = unicodedata.normalize('NFKC', unicodedata.normalize('NFKD', df).translate(trans_tab))

    return data

#%%
def create_fold(new_folder):
    
    if os.path.isdir(new_folder)==False:
        os.makedirs(new_folder)
#%%
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'True','true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

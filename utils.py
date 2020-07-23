"""
@Author: Paula Andrea PÃ©rez Toro <PauPerezT>
@Date:   2018-11-27T20:36:36-05:00
@Email:  paulaperezt16@gmail.com
@Filename: NLP_PreProcessing.py
# @Last modified by:   Paula Andrea PÃ©rez Toro
# @Last modified time: 2018-12-03T21:45:52-05:00

"""
###English implementation in progress
"""
#%%%%%%% Natural Language Basic preprocessing code %%%%%%%#

    #%% Contains %%#
        *noPunctuation: to eliminate the Punctuation
	*noPunctuationExtra: to eliminate the Punctuation related to spanish symbols
        *StopWordsRemoval: to eliminate stopwords
        *Lemmatizer: transform the word to its root
        *HesitationsRemoval: to remove hesitation labels (for example: [h mm])

    #%% Variables %%#
        - Text: text to preprocess
        - Language: 'spanish' or 'english'

"""

#%% Import Libraries
import string
import spacy
from nltk.corpus import stopwords
import unicodedata
import re
#%% noPunctuation
def noPunctuation(text):
    #eliminate the punctuation in form of characters
    nopunctuation= [char for char in text if char not in string.punctuation]

    #Now eliminate the punctuation and convert into a whole sentence
    nopunctuation=''.join(nopunctuation)

    #Split each words present in the new sentence
    nopunctuation.split()
    #nopunctuation = re.sub('\W+', '', nopunctuation) 

    return nopunctuation

def noPunctuationExtra(text):
    
    texttoP=text
    #Just for this app without '.'
    forbidden1 = ('?','@', 'Â¿', 'Â¡', '!','/' ,',','(', ')' ,';', '$', ':', '&','…', '...','_','~','\\', '”','"', '“', 'XXXX', '’', '¿', '¡','-','&')
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

#%% StopWordsRemoval
def StopWordsRemoval(text,language='english'):
    #Now eliminate stopwords
    clean_sentence= [word for word in text.split() if word.lower() not in stopwords.words(language)]
    clean_sentence=' '.join(clean_sentence)

    return clean_sentence

#%% Lemmatizer
def Lemmatizer(text,language='spanish'):
    #nlp = spacy.load('es_core_news_sm')
    nlp = spacy.load('es_core_news_md')
    doc = nlp(text)
    tokenLemma=[]

    for token in doc:
        #print(token, token.lemma, token.lemma_)
        tokenLemma.append(token.lemma_)
    tokenLemma=' '.join(tokenLemma)
    return tokenLemma

#%% HesitationsRemoval

def HesitationsRemoval(text):

    idx1=[n for n in range(len(text)) if text.find('[', n) == n]
    idx2=[n for n in range(len(text)) if text.find(']', n) == n]
    idxlen=len(idx1)
    idxlen2=len(idx2)
 

    while idxlen>0 and idxlen2:
        
        
        
        text=text.replace(text[idx1[0]:idx2[0]+1],"")
        

        idx1=[n for n in range(len(text)) if text.find(' [', n) == n]
        idx2=[n for n in range(len(text)) if text.find(']', n) == n]
        idxlen=len(idx1)
        idxlen2=len(idx2)
        
    idx1=[n for n in range(len(text)) if text.find('(', n) == n]
    idx2=[n for n in range(len(text)) if text.find(')', n) == n]
    idxlen=len(idx1)
    idxlen2=len(idx2)
 

    while idxlen>0 and idxlen2:
        
        
        
        text=text.replace(text[idx1[0]:idx2[0]+1],"")
        

        idx1=[n for n in range(len(text)) if text.find(' (', n) == n]
        idx2=[n for n in range(len(text)) if text.find(')', n) == n]
        idxlen=len(idx1)
        idxlen2=len(idx2)
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

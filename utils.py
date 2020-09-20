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
    
    """
    Remove the punctuation
    :param text: input text
    :returns: text without punctuations given by the string function
    """    
    #
    nopunctuation= [char for char in text if char not in string.punctuation]

    #Now eliminate the punctuation and convert into a whole sentence
    nopunctuation=''.join(nopunctuation)

    #Split each words present in the new sentence
    nopunctuation.split()
    #nopunctuation = re.sub('\W+', '', nopunctuation) 

    return nopunctuation

def noPunctuationExtra(text):
        
    """
    Remove the other special punctuation characters
    :param text: input text
    :returns: text without customized punctuations 
    """  
    
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
                

    return texttoP

#%% StopWordsRemoval
def StopWordsRemoval(text,language='english'):
            
    """
    Remotion of stopwords
    :param text: input text
    :param language: input language (english).
    :returns: text without stopwords
    """  
    #Now eliminate stopwords
    clean_sentence= [word for word in text.split() if word.lower() not in stopwords.words(language)]
    clean_sentence=' '.join(clean_sentence)

    return clean_sentence



#%% HesitationsRemoval

def HesitationsRemoval(text):
    """
    Hesitation removal. Remove this parts of text between [] that are consider hesitation, labels, marks or indicators that not provide relevan information.
    :param text: input text

    :return text without hesitations
    """

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

    return text

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
Welcome to WEBERT's documentation!
============================================================
This toolkit computes word embeddings using Bidirectional Encoder Representations from Transformers (BERT) for cased and large models in spanish and english automatically.
BERT embeddings are computed using Transformers (https://github.com/huggingface/transformers). The project is ongoing.

The code for this project is available at https://github.com/PauPerezT/WEBERT

Guide
^^^^^

.. toctree::
   :maxdepth: 3
   
   license
   help

Installation
------------
From the source file::

    git clone https://github.com/PauPerezT/WEBERT
    cd WEBERT
    
To install the requeriments, please run::
    ./install.sh



Executing commands
^^^^^^^^^^^^^^^^^^
    
This are

Run it automatically from linux terminal
----------------------------------------

To compute Bert embeddings automatically





    ====================  ===================  =====================================================================================
    Optional arguments    Optional Values      Description
    ====================  ===================  =====================================================================================
    -h                                         Show this help message and exit
    -f                                         File folder of the set of txt documents. 
                                               
                                               By defaul './texts'
    -s                                         Path to save the embeddings. 
    
                                               By defaul './bert_embeddings'
    -bm                   Bert,Beto,SciBert    Choose between three different BERT models.
    
                                               By default BERT				             
    -d                    True,False           Boolean value to get dynamic features= True.
    
                                               By default True.                                         
    -st                   True,False           Boolean value to get static features= True from the
    
                                               embeddings such as mean, standard deviation, kurtosis,
                                               
                                               skeweness, min and max. By default False.                       
    -l                    english,spanish      Chosen language (only available for BERT model).
                                               By default english.                               
    -sw                   True,False           Boolean value, set True if you want to remove
    
                                               stopwords. By default False.                                         
    -m                    base,large           Bert models, two options base and large.
     
                                               By default base.                                   
    -c                    True,False           Boolean value for cased= True o lower-cased= False
    
                                               models. No avalaible for SciBert. By defaul False.                     
    ====================  ===================  =====================================================================================


        
Usage example:: 

        python get_embeddings.py -f ./texts/ -s ./bert_embs -bm Bert -d True -st True -l english -sw True -m base -c True

  
  
  
Methods
-------------

.. automodule:: WEBERT

   .. autoclass:: BERT
       :members:
   .. autoclass:: BETO
       :members:
   .. autoclass:: SciBERT
       :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



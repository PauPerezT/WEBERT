Welcome to WEBERT's documentation!
============================================================
This toolkit computes word embeddings using Bidirectional Encoder Representations from Transformers (BERT) for cased and large models in spanish and english automatically.
BERT embeddings are computed using Transformers (https://github.com/huggingface/transformers). The project is currently ongoing.

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

    install.sh



Executing commands
^^^^^^^^^^^^^^^^^^
    


Run it automatically from linux terminal
----------------------------------------

To compute Bert embeddings automatically





====================  ===================  =====================================================================================
Optional arguments    Optional Values      Description
====================  ===================  =====================================================================================
-h                                         Show this help message and exit
-f                                         Path folder of the txt documents (Only txt format). 
                                           
                                           By default './texts'
-sv                                         Path to save the embeddings. 

                                           By default './bert_embeddings'
-bm                   Bert, Beto, SciBert  Choose between three different BERT models.

                                           By default BERT				             
-d                    True, False          Boolean value to get dynamic features= True.

                                           By default True.                                         
-st                   True, False          Boolean value to get static features= True from the

                                           embeddings such as mean, standard deviation, kurtosis,
                                           
                                           skeweness, min and max. By default False.                       
-l                    english, spanish     Chosen language (only available for BERT model).

                                           By default english.                               
-sw                   True, False          Boolean value, set True if you want to remove

                                           stopwords. By default False.                                         
-m                    base, large          Bert models, two options base and large.
 
                                           By default base.                                   
-ca                    True, False          Boolean value for cased= True o lower-cased= False

                                           models. No avalaible for SciBert. By default False.
-cu                    True, False         Boolean value for using cuda to compute the 
                                            
                                           embeddings (True). By default False.                                                   
====================  ===================  =====================================================================================


        
Usage example:: 

        python get_embeddings.py -f ./texts/ -sv ./bert_embs -bm Bert -d True -st True -l english -sw True -m base -ca True -cu True

  
  
  
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



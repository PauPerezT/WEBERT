==========
WEBERT
==========

.. image:: https://github.com/PauPerezT/WEBERT/blob/master/logos/logo_web.png?raw=true

Getting BERT embeddings from Transformers.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3964244.svg
   :target: https://doi.org/10.5281/zenodo.3964244
   :alt: Citation


"WEBERT: Word Embedding using BERT"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

WEBERT is a python toolkit designed to help students to compute dynamic and static Bidirectional Encoder Representations from Transformers (BERT) embeddings (https://github.com/huggingface/transformers). WEBERT is available for english and spanish (multilingual) models, as well as for base and large models, and  cased and lower-cased options. BETO and SciBERT are also available here. BETO is a pretrained BERT model from a spanish corpus (https://github.com/dccuchile/beto). SciBERT is a pre-trained model on english scientific text (https://github.com/allenai/scibert). The static features are computed per each neuron based on the mean, standard deviation, kurtosis, skewness, min and max. The project is currently ongoing.
It was test on linux.

The code for this project is available at https://github.com/PauPerezT/WEBERT

   
.. image:: https://readthedocs.org/projects/webert/badge/?version=latest
:target: https://webert.readthedocs.io/en/latest/
:alt: Documentation Status

From this repository::

    git clone https://github.com/PauPerezT/WEBERT
    
Install
^^^^^^^

To install the requeriments, please run::

    install.sh


Quickstart
^^^^^^^^^^


Run Example::

    python get_embeddings.py 
    
    
Run it automatically from linux terminal
-----------------------------------------

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





    
Usage Example::

    python get_embeddings.py -f ./texts/ -s ./bert_embs -bm Bert -d True -st True -l english -sw True -m base -c True -cu True

For more information: `Project Documentation <https://webert.readthedocs.io/en/latest/>`_

Reference
^^^^^^^^^

If you use WEBERT, please cite as follows

@software{p_a_perez_toro_2020_3964244,
author = {P. A. Perez-Toro}, title = {PauPerezT/WEBERT: Word Embeddings using BERT}, month = jul, year = 2020, publisher = {Zenodo}, version = {V0.0.1}, doi = {10.5281/zenodo.3964244}, url = {https://doi.org/10.5281/zenodo.3964244} }


License
^^^^^^^^

Apache

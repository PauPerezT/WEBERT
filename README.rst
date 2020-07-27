
========
WEBERT
========


Getting BERT embeddings from Transformers.

"WEBERT: Word Embedding using BERT"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

WEBERT is a python framework designed to help students in order to compute dynamic and static BERT embeddings using Transformers (https://github.com/huggingface/transformers). WEBERT is avalable for english and spanish (multilingual) models, as well as for base and large models. It also considered cased and lower-cased cases. The sentences are choosen according to the punctuation '.', to form composed sentences. If this punctuation is not available in the document, the entire document will be a single sentence. The static features are computed per each neuron based on the mean, standar deviation, kurtosis, skewness, min and max. The project is ongoing.

From this repository::

    git clone https://github.com/PauPerezT/WEBERT
    
Install
^^^^^^^

To install the requeriments, please run::

    install.sh

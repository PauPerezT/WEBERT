<p align="leftr">
  <img src="https://github.com/PauPerezT/WEBERT/blob/master/logo_wb.png" width="50" title="logo">

</p> 

# WEBERT
Getting BERT embeddings from Transformers.

## "WEBERT: Word EmbeddingExtraction using BERT"

WEBERT is a python framework designed to help people in order to compute dynamic and static BERT embeddings using Transformers (https://github.com/huggingface/transformers). WEBERT is avalable for english and spanish (multilingual) models, as well as for base and large models. It also considered cased and lower-cased cases. The sentences are choosen according to the punctuation '.', to form composed sentences. If this punctuation is not available in the document, the entire document will be a single sentence. The static features are computed per each neuron based on the mean, standar deviation, kurtosis, skewness, min and max. The project is ongoing.

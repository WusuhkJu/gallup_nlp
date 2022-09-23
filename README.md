# [Gallup Open-answer Autocoding] - built by JWS
release_v1: 202209

Data
- Using answer data from question-answer dataset
- As the shape of input data is likely to be excel sheet and for usability of non-pythonic people, 
  this code is designed to encompass from preprocessing of raw excel data to getting a result excel sheet by a few of clicks
- This was originally built for clustering answers having similar meanings but any kind of Korean sentences could be used

Objective
- Clustering sentences(answer) having similar meanings

Method
- Embedding -> Clustering

Method Details
- Embedding
 : Transformating setences to vectors by SentenceTransformers_KoBERT

- Clustering 
 : Agglomerative clustering used
 : Clustering sentences having distance above a given threshold, e.g. 0.173, 0.075

- Installing packages
!pip3 install kobert-transformers 
!pip3 install ko-sentence-transformers
!pip3 install transformers
!pip3 install sentence_transformers
!pip3 install fairseq

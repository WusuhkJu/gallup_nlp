# gallup_nlp

"""
[NLP project for Gallup] - built by JWS
1. Data
 - Question-answer data from surveys
 - e.g. ('자연어처리에 대하여 어떻게 생각하십니까?','재밌는 일이다')
 - As the shape of input data is likely to be excel sheet and for usability of non-pythonic people, 
   this code is designed to encompass from preprocessing of raw excel data to getting a result by a few of clicks
2. Objective
 - Clustering sentences(answer) having similar meanings
3. Method
 - Embedding -> Fine Tuning -> Clustering
4. Method Details
 - Embedding
  : Transformating setences to vectors by KoBERT
  : Use both question and answer
 - Fine Tuning
  : Target values are not fixed but vary
  : Due to unfixed target values, it's hard to directly implement supervised learning
  : Therefore, some tricks as below are used for supervised learning
    > Firstly, clustering Xs (assigning an identical group number to Xs having similar meanings, let Zs the group)
    > e.g. (x1, x2, x3) = ('맛있다', '맛있네요', '맛있습니다') then  (z1, z2, z3) = (1, 1, 1)
    > Randomly select Xs having similar meanings(=identical group number) and get the average value of them(embedded vectors of Xs)
    > Calculating the average of (y1, y2, y3)
    > Use this average as a label for supervised learning
 - Clustering
  : Using K-Means for clustering embedded vectors(Xs)
  : Choosing the number of cluster by representing the largest average silhouette value
"""

# CAUTION
This may need a lot of space of GPU and CPU memories
Please use on Colab

# Installing packages
!pip3 install kobert-transformers
!pip3 install transformers
!pip3 install fairseq
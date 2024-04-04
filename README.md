# ML_Job-recommendation-system
The project is to develop a recommendation system that recommends job offers to users based on their skills.

## Our recommendation system 
This system is modeled to manipulate and explore a large dataset based on a recommendation sub-system that recommends jobs to users based on their skills. 

![image](https://github.com/ChaiouraMohammed/ML_Job-recommendation-system/assets/91562298/4968d5ad-768f-410e-b532-621493f8a788)

## Using NLP technic to extract Key-words from a texte 
In the skills column , we have description of skills , like sentences , so we needed to extract keywords using NLP .

![image](https://github.com/ChaiouraMohammed/ML_Job-recommendation-system/assets/91562298/1eb2392d-f7df-4654-b8ba-ae53bcc8c98b)

### Used function 
```
def extract_skills(phrase):
    tokens = word_tokenize(phrase)
    tags = pos_tag(tokens)
    competence_tags = ['NN', 'NNS', 'NNP', 'JJ']
    competences = []
    for token, tag in tags:
        if tag in competence_tags:
            competences.append(token)
    return competences
```
## Dataset 
Columns : 

![image](https://github.com/ChaiouraMohammed/ML_Job-recommendation-system/assets/91562298/f7dd123a-2872-464e-b6e3-483edf004afe)

## Libraries 
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

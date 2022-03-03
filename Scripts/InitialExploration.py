"""

Author: Tyler Zimmer -- tzimme9@gmail.com
Date: 2.23.22

-------
Use this script to do some quick and dirty text analytics on cleaned text data.

TO USE: 


## Steps ##

1. Import data after it has already been cleaned and perform token frequency analysis
2. Produce Horizontal Bar Chart
3. Word Cloud

-------
"""

# IMPORT PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# IMPORT DATA
filepath = "C:\\Users\\hlmq\\OneDrive - Chevron\\Data\\TextAnalytics\\Cleaned_Data\\"
df = pd.read_csv(str(filepath)+"DS_Books.csv")


"""
Step 1: Count Frequencies
"""

# Count tokens
token_count = []

for i in range(0,len(df)):
    token_count.append(len(df['pageContent'][i]))
    
df['tokenCount'] = token_count


# Create DTM
cv = CountVectorizer(ngram_range = (1,1))
dtm = cv.fit_transform(df['pageContent'])
words = np.array(cv.get_feature_names())

# Look at top 50 most frequent words
freqs=dtm.sum(axis=0).A.flatten() 
index=np.argsort(freqs)[-20:] 
print(list(zip(words[index], freqs[index])))

WordFreq = pd.DataFrame.from_records(list(zip(words[index], freqs[index]))) 
WordFreq.columns = ['Word', 'Freq']

data = dict(zip(WordFreq['Word'].tolist(), WordFreq['Freq'].tolist()))


"""
Step 2: Horizontal Bar Graph
"""

# PLOT A HORIZONTAL BAR GRAPH OF TOKEN FREQUENCIES

fig, ax = plt.subplots(figsize=(8, 8))
WordFreq.sort_values(by='Freq').plot.barh(
                      x='Word',
                      y='Freq',
                      ax=ax,
                      color="deepskyblue")

plt.title("Count of Most Common Words")


"""
Step 3: Word Cloud
"""

# CREATE WORD CLOUD

wordcloud = WordCloud().generate_from_frequencies(data)

plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
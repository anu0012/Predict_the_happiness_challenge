import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk import word_tokenize, ngrams
import re

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Target Mapping
mapping_target = {'happy':0, 'not happy':1}
train = train.replace({'Is_Response':mapping_target})

# Browser Mapping
mapping_browser = {'Firefox':0, 'Mozilla':0, 'Mozilla Firefox':0,
                  'Edge': 1, 'Internet Explorer': 1 , 'InternetExplorer': 1, 'IE':1,
                   'Google Chrome':2, 'Chrome':2,
                   'Safari': 3, 'Opera': 4
                  }
train = train.replace({'Browser_Used':mapping_browser})
test = test.replace({'Browser_Used':mapping_browser})
# Device mapping
mapping_device = {'Desktop':0, 'Mobile':1, 'Tablet':2}
train = train.replace({'Device_Used':mapping_device})
test = test.replace({'Device_Used':mapping_device})

test_id = test['User_ID']
target = train['Is_Response']

# function to clean data
import string
import itertools 
from nltk.stem import WordNetLemmatizer
stops = set(stopwords.words("english"))
# punct = list(string.punctuation)
# punct.append("''")
# punct.append(":")
# punct.append("...")
# punct.append("@")
# punct.append('""')
def cleanData(text, lowercase = False, remove_stops = False, stemming = False, lemmatization = False):
    txt = str(text)
    
    # Replace apostrophes with standard lexicons
    txt = txt.replace("isn't", "is not")
    txt = txt.replace("aren't", "are not")
    txt = txt.replace("ain't", "am not")
    txt = txt.replace("won't", "will not")
    txt = txt.replace("didn't", "did not")
    txt = txt.replace("shan't", "shall not")
    txt = txt.replace("haven't", "have not")
    txt = txt.replace("hadn't", "had not")
    txt = txt.replace("hasn't", "has not")
    txt = txt.replace("don't", "do not")
    txt = txt.replace("wasn't", "was not")
    txt = txt.replace("weren't", "were not")
    txt = txt.replace("doesn't", "does not")
    txt = txt.replace("'s", " is")
    txt = txt.replace("'re", " are")
    txt = txt.replace("'m", " am")
    txt = txt.replace("'d", " would")
    txt = txt.replace("'ll", " will")
    
#     APPOSTOPHES = {"'s" : " is", "'re" : " are", "'m": " am", "'d": " would", "'ll": " will"}
#     words = txt.split()
#     reformed = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in words]
#     txt = " ".join(reformed)

#     # Emoji replacement
    txt = re.sub(r':\)',r' Happy ',txt)
    txt = re.sub(r':D',r' Happy ',txt)
    txt = re.sub(r':P',r' Happy ',txt)
    txt = re.sub(r':\(',r' Sad ',txt)
    
    # Remove urls and emails
    txt = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', txt, flags=re.MULTILINE)
    txt = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', txt, flags=re.MULTILINE)
    
    # Remove punctuation
#     txt = "".join(ch for ch in txt if ch not in punct)
    txt = txt.replace(".", " ")
    txt = txt.replace(":", " ")
    txt = txt.replace("!", " ")
    txt = txt.replace("&", " ")
    txt = txt.replace("#", " ")
    
    # Remove all symbols
    txt = re.sub(r'[^A-Za-z0-9\s]',r' ',txt)
    txt = re.sub(r'\n',r' ',txt)
    
    # Replace words like sooooooo with so
    txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))
    
    # Split attached words
    #txt = " ".join(re.findall('[A-Z][^A-Z]*', txt))   
    
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
        
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])
    if stemming:
        st = PorterStemmer()
#         print (len(txt.split()))
#         print (txt)
        txt = " ".join([st.stem(w) for w in txt.split()])
    
    if lemmatization:
        wordnet_lemmatizer = WordNetLemmatizer()
        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])

    return txt

# string = str("I read thNe reviews, saw the location and price and decided that booking a room here was a good idea. My advice:don't do it people, just don't. Go for a brand name hotel, and pay a little more. That way, if something goes horribly wrong, you may have a shot at getting some kind of resolution.This is a true story, swear to God: my husband went to wipe himself off with a towel-twice while taking a shower and both of the towels had fe..s on them. Yes, fe..s. He flipped out and took the towels to the front desk. The manager was not there, and upon checking out, I explained to the girl at the front desk that not only was this appalling, but dangerous. Particularly to my one month old baby. I also told her, that I did not feel like I should pay for our stay there. I did pay though (Dumb. In retrospect, I should not have), because I just thought I would settle this with the manager later. My thought was the manager would listen to what happened to us, and act appropriately. That didn't happen. I emailed """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Mike"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""", the manager, then followed up with a phone call. He basically called m a liar, said that """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""never in my career have I ever heard of something like this"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""", to which I replied, """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Well I haven't heard of anything like this either in all my years of traveling, but this really did happen."""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" He then proceeded to give me the third degree about where the towels were located, when this happened blah blah, etc. It was very apparent that he felt that this was some creative attempt on my family's part to get a refund. God, what a jerk. I am POed all over again just writing this. I asked for a refund, he said no, then I told him that I would need to file a complaint with the BBB. He then said, """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Go ahead! I don't care, you can threaten me all you want!"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Uh, no Mike this isn't personal, just horrendous customer service. Like, on very level imaginable. Whatever. I told him, fine. If this is the way you choose to handle this. I filed a complaint w-the BBB. come to think of it, I think I'll contact The Health Department too.")
# print (cleanData(string, lowercase=True, remove_stops=True, stemming=False, lemmatization = True))

# print (cleanData(string, lowercase=True, remove_stops=True, stemming=True, lemmatization = False))

# clean description
train['Description'] = train['Description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=False, stemming=False, lemmatization = True))
test['Description'] = test['Description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=False, stemming=False, lemmatization = True))

# Stanford NLP
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://corenlp.run:80')
arr = []
for sentence in test['Description'][:1000]:
    res = nlp.annotate(sentence,properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json'
                       })
    if res['sentences'][0]['sentiment'] == 'Positive':
    	arr.append(0)
    else:
    	arr.append(1)
    

pred = np.array(arr) 


# Result prediction
result = pd.DataFrame()
#result['User_ID'] = test_id
result['Is_Response'] = pred
mapping = {0:'happy', 1:'not_happy'}
result = result.replace({'Is_Response':mapping})

result.to_csv("stanford_nlp_result_1000.csv", index=False)
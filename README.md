# Predict_the_happiness_challenge
HackerEarth Machine learning challenge

# Approach

1. I used one hot encoding to change categorical features i.e Browser_Used into numerical ones.
 
2. Then I cleaned the text by removing punctuations, stopwords and by using techniques like stemming, lemmatisation.

3. Then I generated handcrafted features like number of unique words, number of punctuations, number of words, mean word length etc.

4. After that I generated text based features using TfidfVectorizer. I used 8000 features from it. Then I merged it with previously generated features.

5. After that I fed those features into a VotingClassifier. In that voting classifier, I used two logistic regression classifiers with different hyper-parameters as basic classifiers. 
 

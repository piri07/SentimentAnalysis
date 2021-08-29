# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:11:08 2021

@author: PRIYANSHU
"""
import nltk
import re,string
import random

from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

#download punkt 
#pre trained model that helps to tokenize words and sentences
nltk.download('punkt')

#5k positive and negative tweets 
#2000 with no sentiments
positive_tweets = twitter_samples.strings("positive_tweets.json")
negative_tweets = twitter_samples.strings("negative_tweets.json")
text = twitter_samples.strings("tweets.20150430-223406.json")
#strings method prints tweet in a string format

tweet_tokens = twitter_samples.tokenized("positive_tweets.json")

#print(tweet_tokens[0])

#now normalizing the data 
#for ex we know that a word has many forms in present past or future so to avoid that we need to normalize it 
#Normalization helps group together words with the same meaning but different forms


#print(pos_tag(tweet_tokens[0]))
#NNP: Noun, proper, singular
#NN: Noun, common, singular or mass
#IN: Preposition or conjunction, subordinating
#VBG: Verb, gerund or present participle
#VBN: Verb, past participle

#lemmatizer function below
'''def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence= []
    for word,tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos='n'
        elif tag.startswith('VB'):
            pos='v'
        else:
            pos='a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word,pos))
    return lemmatized_sentence
print(lemmatize_sentence(tweet_tokens[0]))'''
#You will notice that the verb being changes to its root form, be, and the noun members changes to member



#REMOVING NOISE FROM THE DATA
#. Noise is any part of the text that does not add meaning or information to data.
#Some examples of stop words are “is”, “the”, and “a”. 
#They are generally irrelevant when processing language, unless a specific use case warrants their inclusion.


#process
#1.first convert all the hyperlinks to t.co which will add no meaning to sentence hence removed
#2.twitter handles will be removed i.e starting with @
#3.punctuation and special characters are also removed

...



def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens
#The code takes two arguments: the tweet tokens and the tuple of stop words.
#also lemmatize function is commented as it is included in removed_noise

stop_words = stopwords.words('english')
#print(remove_noise(tweet_tokens[0],stop_words))

#function removes all @ mentions, stop words, and converts the words to lowercase.


#print(remove_noise(tweet_tokens[0], stop_words))
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))


#print(positive_tweet_tokens[500])
#print(positive_cleaned_tokens_list[500])

#WORD DENSITY

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token
all_pos_words = get_all_words(positive_cleaned_tokens_list)


#now find out the most occuring words
freq_dist_pos = FreqDist(all_pos_words)
#print(freq_dist_pos.most_common(10))

#From this data, you can see that emoticon entities form some of the most common parts of positive tweets.



#To summarize, you extracted the tweets from nltk, tokenized, normalized, and cleaned up the tweets 
#for using in the model. 
#Finally, you also looked at the frequencies of tokens in the data and checked the 
#frequencies of the top ten tokens.


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

#This code attaches a Positive or Negative label to each tweet. 
#It then creates a dataset by joining the positive and negative tweets.
positive_dataset = [(tweet_dict,"Positive") for tweet_dict in positive_tokens_for_model ]

negative_dataset = [(tweet_dict,"Negative") for tweet_dict in negative_tokens_for_model ]

dataset = positive_dataset + negative_dataset
random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]


classifier = NaiveBayesClassifier.train(train_data)

#print("Accuracy is: ",classify.accuracy(classifier,test_data))
#print(classifier.show_most_informative_features(10))

#shows 99.63% of accuracy 



custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."

custom_tokens = remove_noise(word_tokenize(custom_tweet))

print(classifier.classify(dict([token, True] for token in custom_tokens)))



















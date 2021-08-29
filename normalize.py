# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:59:40 2021

@author: PRIYANSHU
"""

import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
#wordnet is a lexical database for the English language that helps the script determine the base word. 
#You need the averaged_perceptron_tagger resource to determine the context of a word in a sentence.

from nltk.tag import pos_tag
print(pos_tag(tweet_tokens[0]))
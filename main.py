#!/usr/bin/python
import nltk
from nltk.corpus import twitter_samples
import re
import json
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.metrics.scores import precision, recall
from nltk.probability import FreqDist, ConditionalFreqDist
import math

emoticons_str = r"""
	    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
        )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(
        r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)

emoticon_re = re.compile(
        r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

class DataCleaning:
	# regex_str = [emoticons_str, r'<[^>]+>', r'(?:@[\w_]+)', r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', r'(?:(?:\d+,?)+(?:\.?\d+)?)', r"(?:[a-z][a-z'\-_]+[a-z])", r'(?:[\w_]+)', r'(?:\S)']

    def __init__(self, some_list):
    	self.some_list = some_list


    def tokenize(self, s):
    	self.s = s
    	return tokens_re.findall(s)


    def preprocess(self, some_list, lowercase=False):
        
        output_list = []
        for s in self.some_list:
        	
        	tokens = self.tokenize(s)
        	

        	for token in tokens:

        		output_dict = {}

        		if token.encode('utf-8').startswith('@') or len(token.encode('utf-8'))<=3 or token.encode('utf-8').startswith('http'):
        			continue
        		else:

           		    output_dict[token.encode('utf-8').lower()]= ''
           		    values = (output_dict,) + ('positive',)
           		    output_list.append(values)

        return output_list


def tweets_in_json(tweets_list):

	temp_list = []

	for tweets in tweets_list:

		tweets = json.loads(tweets)
		text_of_tweet = tweets['text']
		temp_list.append(text_of_tweet)

	return temp_list


#----------File reading of all tweets---------
files = twitter_samples.fileids()


negative_tweets_list = twitter_samples.open(files[0]).readlines()

positive_tweets_list = twitter_samples.open(files[1]).readlines()

regular_tweets_list  = twitter_samples.open(files[2]).readlines()

#--------Read all positive and negative words-------------

positive_words = []
negative_words = []

reg = re.compile(r'\n')

with open('positive-words.txt', 'r') as fp:
	words = fp.readlines()

	for word in words:
		word = reg.sub('',word)
		positive_words.append(word)


with open('negative-words.txt') as fn:
	words = fn.readlines()

	for word in words:
		word = reg.sub('',word)
		negative_words.append(word)

#---------------------tweets in json--------------------------------

regular_tweets = tweets_in_json(regular_tweets_list)
positive_tweets = tweets_in_json(positive_tweets_list)
negative_tweets = tweets_in_json(negative_tweets_list)

dataCleanerPos = DataCleaning(positive_tweets)
dataCleanerNeg = DataCleaning(negative_tweets)
print 'length of positive tweets list', len(positive_tweets)
print 'length of negative tweets list', len(negative_tweets)


preprocess_pos_tweets = []
preprocess_neg_tweets = []

preprocess_pos_tweets = dataCleanerPos.preprocess(positive_tweets)
preprocess_neg_tweets = dataCleanerNeg.preprocess(negative_tweets)

#------------------------not required now -------------------------------------
# for tweets in positive_tweets:
	
# 	tweets = dataCleaner.preprocess(tweets)
# 	pos_tagged = (tweets,) + ('positive',)
# 	preprocess_pos_tweets.append(pos_tagged)

# for tweets in negative_tweets:

# 	tweets = dataCleaner.preprocess(tweets)
# 	neg_tagged = (tweets,) + ('negative',)
# 	preprocess_neg_tweets.append(neg_tagged)


#------------------check if in positive or negative word list-------------------

for tweets in preprocess_pos_tweets:
	for words in tweets[0]:
		
		if words in positive_words:
			tweets[0][words] = True
		
		else:
			tweets[0][words] = False


for tweets in preprocess_neg_tweets:
	for words in tweets[0]:
	
		if words in negative_words:
			tweets[0][words] = True
	
		else:
			tweets[0][words] = False

#-----------------------------classifier-------------------------------------------

print preprocess_pos_tweets

posTweets = int(math.floor(len(preprocess_pos_tweets)*3/4))
negTweets = int(math.floor(len(preprocess_neg_tweets)*3/4))

trainFeatures = preprocess_pos_tweets[:posTweets] + preprocess_neg_tweets[:negTweets]
testFeatures = preprocess_pos_tweets[posTweets:] + preprocess_neg_tweets[negTweets:]

classifier = NaiveBayesClassifier.train(trainFeatures)

referenceSets = {'positive': set(), 'negative':set()}
testSets = {'positive':set(), 'negative':set()}

for i , (features, label) in enumerate(testFeatures):
	
	referenceSets[label].add(i)
	predicted = classifier.classify(features)
	testSets[predicted].add(i)


print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
print 'pos precision:', precision(referenceSets['positive'], testSets['positive'])
print 'pos recall:', recall(referenceSets['positive'], testSets['positive'])
print 'neg precision:', precision(referenceSets['negative'], testSets['negative'])
print 'neg recall:', recall(referenceSets['negative'], testSets['negative'])

classifier.show_most_informative_features(10)	

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
from nltk.corpus import stopwords
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


    def preprocess(self, some_list, tag, lowercase=False):
        
        output_list = []
        for s in self.some_list:
        	
        	tokens = self.tokenize(s)

        	for token in tokens:
        		

        		output_dict = {}
        		
        		# token = filter(lambda x: x not in stopwords.words(), token)
        		# token = token.encode('utf-8')

        		if token.encode('utf-8').startswith('@') or len(token.encode('utf-8'))<=3 or token.encode('utf-8').startswith('http'):
        			continue
        		else:

        			values = (output_dict,) + (tag,)
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


#---------------------tweets in json--------------------------------

regular_tweets = tweets_in_json(regular_tweets_list)
positive_tweets = tweets_in_json(positive_tweets_list)
negative_tweets = tweets_in_json(negative_tweets_list)

dataCleanerPos = DataCleaning(positive_tweets)
dataCleanerNeg = DataCleaning(negative_tweets)



preprocess_pos_tweets = []
preprocess_neg_tweets = []

preprocess_pos_tweets = dataCleanerPos.preprocess(positive_tweets, 'positive')
preprocess_neg_tweets = dataCleanerNeg.preprocess(negative_tweets, 'negative')


# #-----------------------------classifier-------------------------------------------



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

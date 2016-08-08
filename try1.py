from nltk.corpus import twitter_samples
import nltk
import json
from bson import json_util
import re
import os
import glob
import string


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
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
 
# tweet = "RT @marcobonzanini: just an example! :D http://example.com #NLP"
# print(preprocess(tweet))
# ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']


class Splitter(object):

    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences


class POSTagger(object):
    def __init__(self):
        pass
        
    def pos_tag(self, sentences):
        """
         input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        #adapt format
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos

class tag_nature(object):

	def __init__(self):

		directory = '/home/tavleen/PycharmProjects/twitter/'
		
		self.dictionary_positive = {}
		self.dictionary_negative = {}

		key_list = list(string.ascii_lowercase)

		for keys in key_list:
			
			self.dictionary_positive[keys] = []
			self.dictionary_negative[keys] = []


		for file in glob.glob('*.txt'):

			self.file_name = file.split('-')[0]
				
			if self.file_name == 'positive':
				
				with open(directory + file, 'r') as f:
					content = f.readlines()

					for words in content:
						first_letter = words[0]

						if first_letter in self.dictionary_positive:
							self.dictionary_positive[first_letter].append(words.strip('\n'))

			else:
				
				with open(directory + file, 'r') as f:
					content = f.readlines()

					for words in content:
						first_letter = words[0]

						if first_letter in self.dictionary_negative:

							self.dictionary_negative[first_letter].append(words.strip('\n'))

						    

	def pos_or_neg(self, pos_tagged_sentences):

		postive_words_key = []
		negative_words_key = []

		for tupples in pos_tagged_sentences:

			temp_key = tupples[0][0].encode('utf-8').lower()

			try:
				postive_words_key = self.dictionary_positive[temp_key]
				found = True
			except KeyError:
					continue

			try:
				negative_words_key = self.dictionary_negative[temp_key]
				found = True
					
			except KeyError:
				continue

			if found == True:

				if tupples[0].encode('utf-8').lower() in postive_words_key:
				
					tupples[2].append('positive')

				elif tupples[0].encode('utf-8').lower() in negative_words_key:
					
					tupples[2].append('negative')

				else:
					continue

		return pos_tagged_sentences

	def frequency_count(self, pos_tagged_sentences):
		
		features = {'positive':{}, 'negative':{}}
		count = {}

		for sent in pos_tagged_sentences:

			for tweets in sent:

				if len(tweets[2]) == 2:
					
					# print tweets[0]
					word = tweets[0].encode('utf-8').lower()
					count[word] = 0

			for tweets in sent:

				if len(tweets[2]) == 2:

					# print tweets[0]
					tag = tweets[2][1]
					word = tweets[0].encode('utf-8').lower()

					count[word] = count[word] + 1
					features[tag][word] = count[word]
		
		# print features['positive']
		# print '\n'
		# print "BREAK HERE"
		# print '\n'
		# print features['negative']

		return features

def value_of_sentiment(sentiment):


	if sentiment == 'positive':
		return 1
	if sentiment == 'negative':
		return -1
	return 0

def measure_sentiment(review_sentences):
	sum = 0
	for sentences in review_sentences:
		for tokenized_words in sentences:
			try:
				tags = tokenized_words[2][1]
			except IndexError:
				tags= ''
			sum+= value_of_sentiment(tags)
	return sum



splitter = Splitter()
postagger = POSTagger()

files = twitter_samples.fileids()
negative_tweets_file = files[0]
positive_tweets_file = files[1]
regular_tweets_file = files[2]

negative_tweets_list = twitter_samples.open(negative_tweets_file).readlines()
positive_tweets_list = twitter_samples.open(positive_tweets_file).readlines()
regular_tweets_list  = twitter_samples.open(regular_tweets_file).readlines()

# print regular_tweets_list[0]
tokenized_sentences = []
regular_tweets_text = []
positive_tweets_text = []
negative_tweets_text = []

tagObject = tag_nature()

for tweets in positive_tweets_list:
	
	tweets = json.loads(tweets)
	text_of_tweet = tweets['text']
	positive_tweets_text.append(text_of_tweet)

for tweets in negative_tweets_list:
	
	tweets = json.loads(tweets)
	text_of_tweet = tweets['text']
	negative_tweets_text.append(text_of_tweet)
	
all_sentences = []
positive_sentences = []
negative_sentences = []


#for positive sentences
for text in positive_tweets_text:
	positive_sentences.append(preprocess(text))


positive_tagged_sentences = postagger.pos_tag(positive_sentences)

temp_pos_sentences = []

for sentences in positive_tagged_sentences:
	pos = tagObject.pos_or_neg(sentences)
	temp_pos_sentences.append(pos)


pos_frequency = tagObject.frequency_count(temp_pos_sentences)


print "BREAK HERE"

#for negative sentences
for text in negative_tweets_text:
	negative_sentences.append(preprocess(text))

negative_tagged_sentences = postagger.pos_tag(negative_sentences)


temp_neg_sentences = []

for sentences in negative_tagged_sentences:
	neg = tagObject.pos_or_neg(sentences)
	temp_neg_sentences.append(neg)

neg_frequency = tagObject.frequency_count(temp_neg_sentences)

review_sentences = temp_pos_sentences + temp_neg_sentences
score = measure_sentiment(review_sentences)
print score







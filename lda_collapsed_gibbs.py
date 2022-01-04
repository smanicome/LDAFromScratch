import functools
import operator

import nltk
import numpy as np
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from scipy.constants import psi
from sklearn.datasets import fetch_20newsgroups

nltk.download('stopwords')
STOPWORDS = list(stopwords.words('english'))
STOPWORDS.extend([
	"", "\t", "a", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across", "act",
	"actually", "added", "adj", "affected", "affecting", "affects", "after", "afterwards", "again",
	"against", "ah", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among",
	"amongst", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything",
	"anyway", "anyways", "anywhere", "apparently", "approximately", "are", "aren", "arent", "arise", "around", "as",
	"aside", "ask", "asking", "at", "auth", "available", "away", "awfully", "b", "back", "be", "became", "because",
	"become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins",
	"behind", "being", "believe", "below", "beside", "besides", "between", "beyond", "biol", "both", "brief", "briefly",
	"but", "by", "c", "ca", "came", "can", "cannot", "cant", "can't", "cause", "causes", "certain", "certainly", "co",
	"com",
	"come", "comes", "contain", "containing", "contains", "could", "couldnt", "d", "date", "did", "didn't", "different",
	"do", "does", "doesn't", "doing", "done", "dont", "don't", "down", "downwards", "due", "during", "e", "each", "ed",
	"edu",
	"effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et",
	"et-al", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f",
	"far", "few", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "for", "former",
	"formerly", "forth", "found", "four", "from", "further", "furthermore", "g", "gave", "get", "gets", "getting",
	"give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "had", "happens", "hardly", "has",
	"hasn't", "have", "haven't", "having", "he", "hed", "hence", "her", "here", "hereafter", "hereby", "herein",
	"heres", "hereupon", "hers", "herself", "hes", "hi", "hid", "him", "himself", "his", "hither", "home", "how",
	"howbeit", "however", "hundred", "i", "id", "ie", "if", "i'll", "im", "immediate", "immediately", "importance",
	"important", "in", "inc", "indeed", "index", "information", "instead", "into", "invention", "inward", "is", "isn't",
	"it", "itd", "it'll", "its", "itself", "i've", "j", "just", "k", "keep", "keeps", "kept", "kg", "km", "know",
	"known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let",
	"lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "m", "made",
	"mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg",
	"might", "million", "miss", "ml", "more", "moreover", "most", "mostly", "mr", "mrs", "much", "mug", "must", "my",
	"myself", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs",
	"neither", "never", "nevertheless", "new", "next", "nine", "ninety", "no", "nobody", "non", "none", "nonetheless",
	"noone", "nor", "normally", "nos", "not", "noted", "nothing", "now", "nowhere", "o", "obtain", "obtained",
	"obviously", "of", "off", "often", "oh", "ok", "okay", "old", "omitted", "on", "once", "one", "ones", "only",
	"onto", "or", "ord", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over",
	"overall", "owing", "own", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps",
	"placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present",
	"previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv",
	"r", "ran", "rather", "rd", "re", "readily", "really", "recent", "recently", "ref", "refs", "regarding",
	"regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results",
	"right", "run", "s", "said", "same", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem",
	"seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "she", "shed",
	"she'll", "shes", "should", "shouldn't", "show", "showed", "shown", "showns", "shows", "significant",
	"significantly", "similar", "similarly", "since", "six", "slightly", "so", "some", "somebody", "somehow", "someone",
	"somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically",
	"specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "such",
	"sufficiently", "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends", "th", "than", "thank",
	"thanks", "thanx", "that", "that'll", "thats", "that've", "the", "their", "theirs", "them", "themselves", "then",
	"thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere",
	"theres", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'll", "theyre", "they've", "think",
	"this", "those", "thou", "though", "thoughh", "thousand", "throug", "through", "throughout", "thru", "thus", "til",
	"tip", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts",
	"twice", "two", "u", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "up", "upon",
	"ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various",
	"'ve", "very", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "was", "wasnt", "we", "wed", "welcome",
	"we'll", "went", "were", "werent", "we've", "what", "whatever", "what'll", "whats", "when", "whence", "whenever",
	"where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which",
	"while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "whose", "why",
	"widely", "willing", "wish", "with", "within", "without", "wont", "would", "wouldnt", "www", "x", "y", "yes", "yet",
	"you", "youd", "you'll", "your", "youre", "yours", "yourself", "yourselves", "you've", "z", "zer"
])
STOPWORDS.extend(['article'])  # 20newsgroups specifics
STOPWORDS.extend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

categories = ['sci.crypt',
              'sci.electronics',
              'sci.med',
              'sci.space']


def get_vocabulary(corpus):
	return list(set([word for doc in corpus for word in doc]))


# remove useless words and punctuations
def preprocess(text):
	doc = simple_preprocess(text, min_len=3)
	return list(filter(lambda word: word not in STOPWORDS, doc))


class LDA:
	def __init__(self, topic_number):
		self.K = topic_number
		self.vocabulary = None
		self.n_d_k = None
		self.n_k_w = None
		self.n_k = None
		self.z = None
		self.alpha = [0.01 for _ in range(topic_number)]
		self.beta = 0.01
		self.theta = None
		self.phi = None

	def gibbs_sampling(self, doc_index, word_index, word, doc):
		topic = self.z[doc_index][word_index]
		self.n_d_k[doc_index][topic] -= 1
		self.n_k_w[topic][self.vocabulary.index(word)] -= 1
		self.n_k[topic] -= 1

		p_k = np.zeros(self.K)
		for t in range(self.K):
			p_k[t] = (self.n_d_k[doc_index][t] + self.alpha[t]) / (len(doc) - 1 + np.sum(self.alpha)) * (
					self.n_k_w[t][self.vocabulary.index(word)] + self.beta) / (self.n_k[t] + self.beta)
		p_k /= np.sum(p_k)
		# print("{: <20} {} {}".format(sum(p_k), doc_index, word_index))
		topic = np.random.multinomial(1, p_k).argmax()

		self.z[doc_index][word_index] = topic
		self.n_d_k[doc_index][topic] += 1
		self.n_k_w[topic][self.vocabulary.index(word)] += 1
		self.n_k[topic] += 1

	def train(self, corpus, iterations, burn_in, ):
		self.vocabulary = get_vocabulary(corpus)
		self.n_d_k = np.zeros([len(corpus), self.K])  # number of words assigned to topic k in document d
		self.n_k_w = np.zeros([self.K, len(self.vocabulary)])  # number of times word w is assigned to topic k
		self.n_k = np.zeros(self.K)  # total number of times any word is assigned to topic k
		self.z = {}  # current topic assignment for each of the N words in the corpus
		self.theta = np.zeros([len(corpus), self.K])
		self.phi = np.zeros([self.K, len(self.vocabulary)])

		# initialisation
		for i in range(len(corpus)):
			self.z[i] = {}

		for doc_index, doc in enumerate(corpus):
			for word_index, word in enumerate(doc):
				topic = np.random.randint(0, self.K)
				self.n_d_k[doc_index][topic] += 1
				self.n_k_w[topic][self.vocabulary.index(word)] += 1
				self.n_k[topic] += 1
				self.z[doc_index][word_index] = topic

		# training
		for i in range(iterations):
			print("---------- Iteration {} -----------".format(i))
			for doc_index, doc in enumerate(corpus):
				for word_index, word in enumerate(doc):
					self.gibbs_sampling(doc_index, word_index, word, doc)

			if i < burn_in:
				for doc_index in range(len(corpus)):
					for topic in range(self.K):
						self.theta[doc_index][topic] += \
							(self.n_d_k[doc_index][topic] + self.alpha[topic]) / \
							(np.sum(self.n_d_k[doc_index]) + np.sum(self.alpha))
				for topic in range(self.K):
					for word_index in range(len(self.vocabulary)):
						self.phi[topic][word_index] += \
							(self.n_k_w[topic][word_index] + self.beta) / \
							(self.n_k[topic] + self.beta)

	def print_topics(self):
		for t in range(self.K):
			word_distribution = []
			for word_index, word_occ in enumerate(self.n_k_w[t]):
				word_distribution.append((word_occ / self.n_k[t], self.vocabulary[word_index]))

			print(
				"Topic {}: {}".format(t + 1, sorted(word_distribution, key=operator.itemgetter(0), reverse=True)[:10]))

	def test_classify(self):
		testDataset = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)

		for doc_index, doc in enumerate([preprocess(doc) for doc in testDataset.data[:20]]):
			topic_distribution = np.zeros(self.K)

			for word in doc:
				if self.vocabulary.count(word) == 0:
					continue

				topic = np.argmax([word_distribution[self.vocabulary.index(word)] for word_distribution in self.n_k_w])
				topic_distribution[topic] += 1


			# for word in doc:
			# 	if self.vocabulary.count(word) == 0:
			# 		continue
			# 	topic = np.random.randint(0, self.K)
			# 	topic_distribution[topic] += 1
			#
			# for word in doc:
			# 	if self.vocabulary.count(word) == 0:
			# 		continue
			#
			# 	p_k = np.zeros(self.K)
			# 	for t in range(self.K):
			# 		p_k[t] = (topic_distribution[t] + self.alpha[t]) / (len(doc) - 1 + np.sum(self.alpha)) * (
			# 				self.n_k_w[t][self.vocabulary.index(word)] + self.beta) / (self.n_k[t] + self.beta)
			# 	p_k /= np.sum(p_k)
			# 	topic_distribution[np.random.multinomial(1, p_k).argmax()] += 1

			target = testDataset.target_names[testDataset.target[doc_index]]
			print(
				"Document {: <11} Classified {: <13} Actual {}".format(doc_index + 1, np.argmax(topic_distribution) + 1,
				                                                       target))


if __name__ == '__main__':
	lda = LDA(6)

	trainDataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories).data[:100]

	documents = [preprocess(doc) for doc in trainDataset]
	lda.train(documents, 20, 5)
	lda.print_topics()
	lda.test_classify()
# print("Theta: {}".format(lda.theta))
# print("Phi: {}".format(lda.phi))

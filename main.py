import functools
import math

import nltk
import numpy as np
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
import operator
import random

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

documents = fetch_20newsgroups(
	subset='train',
	remove=('headers', 'footers', 'quotes'),
	shuffle=True,
).data[:1000]

print(str(len(documents)) + " documents")
topics = [i for i in range(20)]
dictionary = {}


# remove useless words and punctuations
def preprocess(text):
	doc = simple_preprocess(text, min_len=3)
	return list(filter(lambda word: word not in STOPWORDS, doc))


# Fill dictionary which will be used for tf_idf
def create_dictionary(docs):
	for doc in docs:
		for word in set(doc):
			if word not in dictionary:
				dictionary[word] = 1
			else:
				dictionary[word] += 1


def bag_of_words(docs):
	list_of_bow = []
	for doc in docs:
		bow = {}
		for word in doc:
			if word in bow:
				bow[word] += 1
			else:
				bow[word] = 1
		list_of_bow.append(list(bow.items()))
	return list_of_bow


def tf_idf(word, number_of_words_in_doc, number_of_docs):
	tfidf = (word[1] / number_of_words_in_doc) * (np.log((number_of_docs + 1) / (dictionary[word[0]] + 1)) + 1)
	return tfidf


# Filter most relevant words of the document based on tf_idf
def compute_tf_idfs(docs):
	return [[word for word in doc if 0.8 < (tf_idf(word, functools.reduce(lambda a, b: a + b, [occ for (_, occ) in doc], 0), len(docs))) < 1.2] for doc in docs]


# Associates each word with a random topic
def randomize_topics_distribution(docs):
	doc_list = []

	for doc in docs:
		sublist = []
		for word in doc:
			sublist.append((word, random.choice(topics)))
		doc_list.append(sublist)

	return doc_list


# Returns the topic distribution
def get_topic_distribution_in_list(doc):
	topics_in_doc = {}
	for t in topics:
		topics_in_doc[t] = 0

	for w in doc:
		topics_in_doc[w[1]] += 1 * w[0][1]
	return topics_in_doc


# Sort word distribution through Gibbs sampling
def sort_topics(docs):
	flatlist = [item for sublist in docs for item in sublist]

	doc_list = []
	for doc in docs:
		sublist = []
		for word in doc:
			topic_distribution_in_doc = get_topic_distribution_in_list(doc)
			corpus_word = list(filter(lambda item: item[0][0] == word[0][0], flatlist))
			word_distribution_in_corpus = get_topic_distribution_in_list(corpus_word)

			topics_weight = []
			for t in topics:
				weight = (topic_distribution_in_doc[t] + 1) * (word_distribution_in_corpus[t] + 1)
				topics_weight.append((t, weight))
			topic_selected = max(topics_weight, key=operator.itemgetter(1))
			sublist.append((word[0], topic_selected[0]))

		doc_list.append(sublist)

	return doc_list


# Display word distribution of each topic - UNSURE OF THE RESULT
def print_topics():
	flatlist = [item for sublist in model for item in sublist]
	for t in topics:
		word_distribution = []
		words_in_topic = list(filter(lambda item: item[1] == t, flatlist))
		words_in_topic_occ = (functools.reduce(lambda a, b: a + b, [occ for ((_, occ), __) in words_in_topic], 0))

		marked_words = []
		for word in words_in_topic:
			if word[0][0] in marked_words:
				continue
			marked_words.append(word[0][0])

			word_in_topic = list(filter(lambda item: item[0][0] == word[0][0], flatlist))
			word_in_topic_occ = (functools.reduce(lambda a, b: a + b, [occ for ((_, occ), __) in word_in_topic], 0))

			word_distribution.append((word_in_topic_occ / words_in_topic_occ, word[0][0]))

		print('Topic ' + str(t) + ": " + str(sorted(word_distribution, key=operator.itemgetter(0), reverse=True)[:10]))


if __name__ == '__main__':
	processed_docs = [preprocess(doc) for doc in documents]
	create_dictionary(processed_docs)
	bows = bag_of_words(processed_docs)
	filtered = compute_tf_idfs(bows)

	model = randomize_topics_distribution(filtered)

	for i in range(100):
		model = sort_topics(model)

	print_topics()

# bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
# tfidf = TfidfModel(bow_corpus)
# corpus_tfidf = tfidf[bow_corpus]
# lda_model = LdaMulticore(corpus_tfidf, num_topics=6, id2word=dictionary, passes=2, workers=2)
#
# unseen_document = 'I hate everyone'
# bow_vector = dictionary.doc2bow(preprocess(unseen_document))
# for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
# 	print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
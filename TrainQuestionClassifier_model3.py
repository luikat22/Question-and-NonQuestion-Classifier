# ##############################################################################################
#
# This program is for training a classifier to classify whether a sentence is "Question" or
# "NonQuestion". The model used in this program is to extract the keywords-sets and each
# keywords-set contains three words.
#
# In NLP, this program does not remove stopwords but punctuations are removed.
#
# ##############################################################################################

import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.classify import ClassifierI
from statistics import mode
import pickle
import json


print('----- !!!!! Running the train_question_classifier_in_three_word !!!!! -----')


def get_n_keywords(input_data, n):
    vec = CountVectorizer(analyzer='word', ngram_range=(3, 3)).fit(input_data)
    bag_of_words = vec.transform(input_data)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:n]


def text_to_features(txt):
    """
    convert original text into feature set which is used for training
    """
    try:
        vec = CountVectorizer(analyzer='word', ngram_range=(2, 2)).fit([txt.lower()])
        words_in_text = vec.get_feature_names()
        feature = {}

        # find whether the words in the input text are existing in keywords too
        for word in keywords:
            feature[word] = (word in words_in_text)

        return feature

    except Exception as e:
        print(e)
        return False


# ###### Get the Training Data ###### #
print('Loading training data...')
with open('../training_data/train_v2.1.json') as json_file:
    data = json.load(json_file)
print('Loading finished.')

train_questions = []
train_nonquestions = []

for query in data['query'].values():
    train_questions.append(query)

for answers in data['answers'].values():
    train_nonquestions.append(answers[0])


# ###### Get the Testing Data ###### #
print('Loading testing data...')
with open('../training_data/dev_v2.1.json') as json_file:
    data = json.load(json_file)
print('Loading finished.')

test_questions = []
test_nonquestions = []

for query in data['query'].values():
    test_questions.append(query)

for answers in data['answers'].values():
    test_nonquestions.append(answers[0])


# ###### Get the Keywords with 2 ngram_range ###### #
print('Finding keywords...')
keywords = get_n_keywords(train_questions+train_nonquestions, n=400)
keywords = [keyword for (keyword, _) in keywords]
keywords = keywords[1:]


# ###### Define the Training and Test Data Format ###### #
print("Formatting the training data set...")
formatted_dataset = []
for question in train_questions:
    text_feature = (text_to_features(question), 'Question')
    if text_feature[0] is not False:
        formatted_dataset.append(text_feature)

for answer in train_nonquestions:
    text_feature = (text_to_features(answer), 'NonQuestion')
    if text_feature[0] is not False:
        formatted_dataset.append(text_feature)

train_set = formatted_dataset

print("Formatting the testing data set...")
formatted_dataset = []
for question in test_questions:
    text_feature = (text_to_features(question), 'Question')
    if text_feature[0] is not False:
        formatted_dataset.append(text_feature)

for answer in test_nonquestions:
    text_feature = (text_to_features(answer), 'NonQuestion')
    if text_feature[0] is not False:
        formatted_dataset.append(text_feature)

test_set = formatted_dataset


# ###### Train and Test the Models ###### #
print('Start training.', '\n')

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
print('training BernoulliNB_classifier...')
BernoulliNB_classifier.train(train_set)
print('accuracy of BernoulliNB_classifier:',
      nltk.classify.accuracy(BernoulliNB_classifier, test_set), '\n')

MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
print('training MultinomialNB_classifier...')
MultinomialNB_classifier.train(train_set)
print('accuracy of MultinomialNB_classifier:',
      nltk.classify.accuracy(MultinomialNB_classifier, test_set), '\n')

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
print('training LogisticRegression_classifier...')
LogisticRegression_classifier.train(train_set)
print('accuracy of LogisticRegression_classifier:',
      nltk.classify.accuracy(LogisticRegression_classifier, test_set), '\n')

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
print('training SGDClassifier_classifier...')
SGDClassifier_classifier.train(train_set)
print('accuracy of SGDClassifier_classifier:',
      nltk.classify.accuracy(SGDClassifier_classifier, test_set), '\n')

LinearSVC_classifier = SklearnClassifier(LinearSVC())
print('training LinearSVC_classifier...')
LinearSVC_classifier.train(train_set)
print('accuracy of LinearSVC_classifier:',
      nltk.classify.accuracy(LinearSVC_classifier, test_set), '\n')

KNeighbors_classifier = SklearnClassifier(KNeighborsClassifier())
print('training KNeighbors_classifier...')
KNeighbors_classifier.train(train_set)
print('accuracy of KNeighbors_classifier:',
      nltk.classify.accuracy(KNeighbors_classifier, test_set), '\n')

DecisionTree_classifier = SklearnClassifier(DecisionTreeClassifier())
print('training DecisionTree_classifier...')
DecisionTree_classifier.train(train_set)
print('accuracy of DecisionTree_classifier:',
      nltk.classify.accuracy(DecisionTree_classifier, test_set), '\n')


# ###### Build a Voting System ###### #
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        self.result = None
        self.confidence = 0

    def classify(self, feature_set):
        votes = []

        for classifier in self._classifiers:
            vote = classifier.classify(feature_set)
            votes.append(vote)

        self.result = mode(votes)
        self.confidence = votes.count(self.result) / len(votes)
        return self.result


print('building voting system')
question_classifier = VoteClassifier(BernoulliNB_classifier, MultinomialNB_classifier,
                                     LogisticRegression_classifier, SGDClassifier_classifier,
                                     LinearSVC_classifier, KNeighbors_classifier, DecisionTree_classifier)

accuracy = 0
for i in range(0, len(test_set)):
    prediction = question_classifier.classify(test_set[i][0])
    # print('Voting system classifying test case {}. (confidence {})'.format(i+1, question_classifier.confidence))
    if test_set[i][1] == prediction:
        accuracy += 1

accuracy /= len(test_set)

print('accuracy of VoteClassifier:', accuracy)


"""
Accuracy of Different Classifiers:

BernoulliNB:            0.
MultinomialNB:          0.
LogisticRegression:     0.
SGDClassifier:          0.
LinearSVC:              0.
KNeighbors_classifier:  0.
DecisionTree_classifier:0.
VotingSystem:           0.

"""


# ###### Save the Classifiers and Data ###### #

save_classifier = open('../pickle_files/BernoulliNB_classifier_model3.pickle', 'wb')
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/MultinomialNB_classifier_model3.pickle', 'wb')
pickle.dump(MultinomialNB_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/LogisticRegression_classifier_model3.pickle', 'wb')
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/SGDClassifier_classifier_model3.pickle', 'wb')
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/LinearSVC_classifier_model3.pickle', 'wb')
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/KNeighbors_classifier_model3.pickle', 'wb')
pickle.dump(KNeighbors_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/DecisionTree_classifier_model3.pickle', 'wb')
pickle.dump(DecisionTree_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/voting_classifier_model3.pickle', 'wb')
pickle.dump(question_classifier, save_classifier)
save_classifier.close()

save_data = open('../pickle_files/keywords_model3.pickle', 'wb')
pickle.dump(keywords, save_data)
save_data.close()


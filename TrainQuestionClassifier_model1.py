# ##############################################################################################
#
# This program is for training a classifier to classify whether a sentence is "Question" or
# "NonQuestion". The model used in this program is to extract the keywords-sets and each
# keywords-set contains only one word.
#
# In NLP, this program does not remove stopwords and punctuations.
#
# ##############################################################################################


import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.classify import ClassifierI
from statistics import mode
import pickle
import json


print('----- !!!!! Running the train_question_classifier_in_one_word !!!!! -----')


# ###### Define Some Helper Functions ###### #
def text_cleansing(txt):
    try:
        # tokenize the text
        output_txt = []

        # lemmatize the text
        lem = WordNetLemmatizer()
        for word, tag in pos_tag(word_tokenize(txt.lower())):
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            if not wntag:
                output_txt.append(word)
            else:
                output_txt.append(lem.lemmatize(word, wntag))

    except Exception as e:
        print(e)

    return word_tokenize(txt.lower())


def text_to_features(txt):
    """
    convert original text into feature set which is used for training
    """
    words_in_text = word_tokenize(txt.lower())
    feature = {}

    # find whether the words in the input text are existing in keywords too
    for word in keywords:
        feature[word] = (word in words_in_text)

    return feature


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


# ###### Find the Keywords ###### #
print('Finding the keywords')
all_words = []

for text in train_questions:
    text = text_cleansing(text)
    all_words += text

for text in train_nonquestions:
    text = text_cleansing(text)
    all_words += text

print('number of all words:', len(all_words))
all_words = nltk.FreqDist(all_words)
keywords = list(all_words.keys())[:300]  # get the most common 300 words in all words
keywords += ['yes', 'no', 'whether', 'what', 'which', 'who', 'whom', 'when',
             'where', 'why', 'how']
keywords = set(keywords)
print('keywords:', keywords)

# ###### Define the Training and Test Data Format ###### #
print("Formatting the training data set...")
formatted_dataset = []
for question in train_questions:
    text_feature = (text_to_features(question), 'Question')
    formatted_dataset.append(text_feature)

for answer in train_nonquestions:
    text_feature = (text_to_features(answer), 'NonQuestion')
    formatted_dataset.append(text_feature)

train_set = formatted_dataset

print("Formatting the testing data set...")
formatted_dataset = []
for question in test_questions:
    text_feature = (text_to_features(question), 'Question')
    formatted_dataset.append(text_feature)

for answer in test_nonquestions:
    text_feature = (text_to_features(answer), 'NonQuestion')
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

BernoulliNB:            0.958147448389107
MultinomialNB:          0.9479340805001335
LogisticRegression:     0.9546209925514131
SGDClassifier:          0.9475334592899607
LinearSVC:              0.9541412362873789
KNeighbors_classifier:  0.
DecisionTree_classifier:0.
VotingSystem:           0.

"""


# ###### Save the Classifiers and Data ###### #

save_classifier = open('../pickle_files/BernoulliNB_classifier_model1.pickle', 'wb')
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/MultinomialNB_classifier_model1.pickle', 'wb')
pickle.dump(MultinomialNB_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/LogisticRegression_classifier_model1.pickle', 'wb')
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/SGDClassifier_classifier_model1.pickle', 'wb')
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/LinearSVC_classifier_model1.pickle', 'wb')
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/KNeighbors_classifier_model1.pickle', 'wb')
pickle.dump(KNeighbors_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/DecisionTree_classifier_model1.pickle', 'wb')
pickle.dump(DecisionTree_classifier, save_classifier)
save_classifier.close()

save_classifier = open('../pickle_files/voting_classifier_model1.pickle', 'wb')
pickle.dump(question_classifier, save_classifier)
save_classifier.close()

save_data = open('../pickle_files/keywords_model1.pickle', 'wb')
pickle.dump(keywords, save_data)
save_data.close()


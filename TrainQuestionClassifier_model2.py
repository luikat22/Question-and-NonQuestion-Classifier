import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from nltk.classify import ClassifierI
from statistics import mode
import pickle
import json


def get_n_keywords(input_data, n):
    vec = CountVectorizer(analyzer='word', ngram_range=(2, 2)).fit(input_data)
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
        return False


# ###### Get the Training Data ###### #
print('Loading training data...')
with open('MS MARCO/train_v2.1.json') as json_file:
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
with open('MS MARCO/dev_v2.1.json') as json_file:
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
keywords = get_n_keywords(train_questions+train_nonquestions, 400)
keywords = [keyword for (keyword, _) in keywords]
keywords = keywords[1:]

# manually remove the noise or redundant data
words_to_remove = ['answer present', 'social security', 'customer service', 'age of', '000 to', 'net worth',
                   'population of', 'side effects', 'the largest', 'temperature of', 'responsible for', 'an american',
                   'associated with', 'name meaning', 'blood pressure', 'an average', 'calories in', 'the definition',
                   'average temperature', 'square foot', 'temperature in', 'the difference', 'symptoms of',
                   'difference between', 'phone number', 'united states', 'the united', 'weather in', 'new york',
                   'per square', 'average salary', 'the process', 'the state', 'state of', 'salary for', 'the skin',
                   'cost per', 'to install', 'the earth', 'of water', 'many calories', 'the purpose', 'average cost',
                   'definition of', 'is considered', 'per month', 'benefits of', 'number for', 'price of', 'is defined',
                   'the heart', 'is considered', 'effects of', 'the brain', 'the weather', 'process of', 'purpose of',
                   'per year', 'located in', 'group of', 'salary of', 'characterized by', 'ability to', 'the current',
                   'act of', 'health care', 'the water', 'and or', 'number is', 'code for', 'to 10', 'nervous system',
                   'loss of', 'designed to', 'live in', 'time to', 'city in', 'parts of', 'for your', 'different types',
                   'many people', 'size of', 'years of', 'the cell', 'for in', 'the temperature', 'in new', 'the other',
                   'to build', 'set of', 'order to', 'the top', 'degrees fahrenheit', 'to work', 'variety of',
                   'to create', 'in one', 'the highest', 'to create', 'in one', 'the number', 'in order', 'the last',
                   'the new', 'use of', 'to remove', 'as an', 'function of', 'is called', 'name for', 'within the',
                   'to change', 'with an', 'the right', 'your body', 'over the', 'the main', 'most common', 'come from',
                   'defined as', 'the us', 'to find', 'based on', 'the following', 'during the', 'to cook', 'per hour',
                   'name of', 'amount of', 'form of', 'the meaning', 'is located', 'per day', 'between the',
                   'the average', 'type of', 'the two', 'is caused', 'to pay', 'is usually', 'related to',
                   'depending on', 'the population', 'come out', 'of blood', 'pay for', 'the time', 'routing number',
                   'found in', 'the amount', 'according to', 'of time', 'year was', 'around the', 'through the',
                   'types of', 'meaning of', 'the first', 'with the']
keywords = [w for w in keywords if w not in words_to_remove]
print('keywords:', keywords)


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
print('Start training!', '\n')

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
                                     LinearSVC_classifier)

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

BernoulliNB:            0.7769288072803414
MultinomialNB:          0.6129348876540227
LogisticRegression:     0.781866594185391
SGDClassifier:          0.7798381251510027
LinearSVC:              0.7812122493355883
VotingSystem:           0.7813380848836273

"""


# ###### Save the Classifiers and Data ###### #

save_classifier = open('pickle_files/BernoulliNB_classifier_v5.pickle', 'wb')
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

save_classifier = open('pickle_files/MultinomialNB_classifier_v5.pickle', 'wb')
pickle.dump(MultinomialNB_classifier, save_classifier)
save_classifier.close()

save_classifier = open('pickle_files/LogisticRegression_classifier_v5.pickle', 'wb')
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

save_classifier = open('pickle_files/SGDClassifier_classifier_v5.pickle', 'wb')
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()

save_classifier = open('pickle_files/LinearSVC_classifier_v5.pickle', 'wb')
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

save_classifier = open('pickle_files/voting_classifier_v5.pickle', 'wb')
pickle.dump(question_classifier, save_classifier)
save_classifier.close()

save_data = open('pickle_files/keywords_v5.pickle', 'wb')
pickle.dump(keywords, save_data)
save_data.close()


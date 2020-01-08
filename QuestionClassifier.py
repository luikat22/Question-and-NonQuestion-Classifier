from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.classify import ClassifierI
from statistics import mode
import pickle


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


# ###### Load the pre-trained models and data ###### #

pickle_loader = open('pickle_files/keywords_v4.pickle', 'rb')
keywords_1 = pickle.load(pickle_loader)
pickle_loader.close()

pickle_loader = open('pickle_files/voting_classifier_v4.pickle', 'rb')
voting_classifier_1 = pickle.load(pickle_loader)
pickle_loader.close()

pickle_loader = open('pickle_files/keywords_v5.pickle', 'rb')
keywords_2 = pickle.load(pickle_loader)
pickle_loader.close()

pickle_loader = open('pickle_files/voting_classifier_v5.pickle', 'rb')
voting_classifier_2 = pickle.load(pickle_loader)
pickle_loader.close()


def text_to_features_model1(txt, keywords):
    """
    convert original text into feature set which is used for training
    """
    try:
        words_in_text = word_tokenize(txt.lower())
        feature = {}

        # find whether the words in the input text are existing in keywords too
        for word in keywords:
            feature[word] = (word in words_in_text)

        return feature

    except Exception as e:
        print(e)
        return False


def text_to_features_model2(txt, keywords):
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


def classify_question(text):
    try:
        keywords = keywords_1
        text_1 = text_to_features_model1(text, keywords)
        prediction_1 = voting_classifier_1.classify(text_1)

        keywords = keywords_2
        text_2 = text_to_features_model2(text, keywords)
        prediction_2 = voting_classifier_2.classify(text_2)

        if prediction_1 == 'Question' or prediction_2 == 'Question':
            return 'Question'
        elif prediction_1 is False or prediction_2 is False:
            return 'The input string is invalid.'
        else:
            return 'NonQuestion'

    except Exception as e:
        # print("Exception found!", e)
        return 'The input string is invalid. Maybe your input is less than two words.'


if __name__ == '__main__':

    while True:
        print("Enter empty string to end. Or enter your test sentence:")
        test_sentence = input()
        if test_sentence == '':
            break
        result = classify_question(test_sentence)
        print(result)
        print('-----------------------------------------------------------', '\n')


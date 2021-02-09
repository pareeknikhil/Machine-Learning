import os
import numpy as np
import pandas

print("Preparing Data for building Model")
TRAIN_SPAM_PATH = '\\train\\spam'
TRAIN_HAM_PATH = '\\train\\ham'
TEST_SPAM_PATH = '\\test\\spam'
TEST_HAM_PATH = '\\test\\ham'
BASE_DIR = os.getcwd()

f = open("stopwords.txt", "r")
text = f.readline()
stopwords = []
while text != '':
    stopwords.append(text.split("\n")[0])
    text = f.readline()

os.chdir(BASE_DIR + TRAIN_SPAM_PATH)
train_list_of_spam = os.listdir()
all_words = set()
for filename in train_list_of_spam:
    try:
        file = open(filename, mode='r')
        text = file.readline()
        while text != '':
            list_of_words = text.split(' ')
            for word in list_of_words:
                if word not in stopwords:
                    all_words.add(word)
            text = file.readline()
        file.close()
    except:
        print('Error in Spam Training file : ' + filename)
        file.close()

os.chdir(BASE_DIR + TRAIN_HAM_PATH)
train_list_of_ham = os.listdir()
for filename in train_list_of_ham:
    try:
        file = open(filename, mode='r')
        text = file.readline()
        while text != '':
            list_of_words = text.split(' ')
            for word in list_of_words:
                if word not in stopwords:
                    all_words.add(word)
            text = file.readline()
        file.close()
    except:
        print('Error in Ham Training file : ' + filename)
        file.close()

df_Logistic_X = pandas.DataFrame(columns=all_words)
df_Logistic_Y = []

os.chdir(BASE_DIR + TRAIN_SPAM_PATH)
train_list_of_spam = os.listdir()
train_spam_vocabulary = {}
data_row = {}
for filename in train_list_of_spam:
    try:
        data_row.clear()
        file = open(filename, mode='r')
        text = file.readline()
        while text != '':
            list_of_words = text.split(' ')
            for word in list_of_words:
                if word not in stopwords:
                    try:
                        train_spam_vocabulary[word] += 1
                    except KeyError:
                        train_spam_vocabulary[word] = 1
                    try:
                        data_row[word] += 1
                    except KeyError:
                        data_row[word] = 1
            text = file.readline()
        file.close()
    except:
        print('Error in Spam Training file : ' + filename)
        file.close()
    df_Logistic_X = df_Logistic_X.append(data_row, ignore_index=True)
    df_Logistic_Y.append(1)

train_spam_prob = {}
for word in train_spam_vocabulary.keys():
    train_spam_prob[word] = np.log((train_spam_vocabulary[word] + 1) / (sum(train_spam_vocabulary.values())
                                                                        + len(train_spam_vocabulary.keys())))

os.chdir(BASE_DIR + TRAIN_HAM_PATH)
train_list_of_ham = os.listdir()
train_ham_vocabulary = {}
for filename in train_list_of_ham:
    try:
        data_row = {}
        file = open(filename, mode='r')
        text = file.readline()
        while text != '':
            list_of_words = text.split(' ')
            for word in list_of_words:
                if word not in stopwords:
                    try:
                        train_ham_vocabulary[word] += 1
                    except KeyError:
                        train_ham_vocabulary[word] = 1
                    try:
                        data_row[word] += 1
                    except KeyError:
                        data_row[word] = 1
            text = file.readline()
        file.close()
    except:
        print('Error in Ham Training file : ' + filename)
        file.close()
    df_Logistic_X = df_Logistic_X.append(data_row, ignore_index=True)
    df_Logistic_Y.append(0)

train_ham_prob = {}
for word in train_ham_vocabulary.keys():
    train_ham_prob[word] = np.log((train_ham_vocabulary[word] + 1) / (sum(train_ham_vocabulary.values())
                                                                      + len(train_ham_vocabulary.keys())))

df_Logistic_X = df_Logistic_X.fillna(0)
df_Logistic_X = df_Logistic_X.astype('int32')

intercept = np.ones((df_Logistic_X.shape[0], 1))
df_Logistic_X = np.concatenate((intercept, df_Logistic_X), axis=1)
df_Logistic_Y = np.array(df_Logistic_Y)

print("Building Naive Bayes and Logistic Regression Model")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_train(X, y, lr=0.0001, num_iter=1000000, lamda=0.01):
    theta = np.zeros(X.shape[1])
    for i in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = (np.dot(X.T, (h - y)) + lamda * np.sum(theta * theta)) / y.size
        theta -= lr * gradient
    return theta


final_theta = logistic_train(df_Logistic_X, df_Logistic_Y)

#################################################################
df_Logistic_X = pandas.DataFrame(columns=all_words)
df_Logistic_Y = []
prob_spam = len(train_list_of_spam) / (len(train_list_of_spam) + len(train_list_of_ham))
prob_ham = 1 - prob_spam

print("Testing Naive Bayes and Logistic Regression Model for Spam Directory")
os.chdir(BASE_DIR + TEST_SPAM_PATH)
test_list_of_spam = os.listdir()
test_spam_vocabulary = {}
count = 0
for filename in test_list_of_spam:
    test_spam_vocabulary.clear()
    data_row = {}
    try:
        file = open(filename, 'r')
        text = file.readline()
        while text != '':
            list_of_words = text.split(' ')
            for word in list_of_words:
                if word not in stopwords:
                    try:
                        test_spam_vocabulary[word] += 1
                    except KeyError:
                        test_spam_vocabulary[word] = 1
                    try:
                        if word in all_words:
                            data_row[word] += 1
                    except:
                        if word in all_words:
                            data_row[word] = 1
            text = file.readline()
        file.close()
    except:
        print('Error in Spam Test file : ' + filename)
        file.close()
    df_Logistic_X = df_Logistic_X.append(data_row, ignore_index=True)
    df_Logistic_Y.append(1)

    probability_of_spam = np.log(prob_spam)
    for word in test_spam_vocabulary.keys():
        try:
            probability_of_spam += (test_spam_vocabulary[word] * train_spam_prob[word])
        except KeyError:
            probability_of_spam += (test_spam_vocabulary[word] * np.log(1 / (sum(train_spam_vocabulary.values())
                                                                             + len(train_spam_vocabulary.keys()))))
    probability_of_ham = np.log(prob_ham)
    for word in test_spam_vocabulary.keys():
        try:
            probability_of_ham += (test_spam_vocabulary[word] * train_ham_prob[word])
        except KeyError:
            probability_of_ham += (test_spam_vocabulary[word] * np.log(1 / (sum(train_ham_vocabulary.values())
                                                                            + len(train_ham_vocabulary.keys()))))
    if probability_of_spam > probability_of_ham:
        count += 1

spam_accuracy = (count / len(test_list_of_spam)) * 100
print('Accuracy for Test Data on Spam Files based on Naive Bayes Model with stopwords: ' + str(spam_accuracy))

df_Logistic_X = df_Logistic_X.fillna(0)
df_Logistic_X = df_Logistic_X.astype('int32')

intercept = np.ones((df_Logistic_X.shape[0], 1))
df_Logistic_X = np.concatenate((intercept, df_Logistic_X), axis=1)
df_Logistic_Y = np.array(df_Logistic_Y)


def predict(X, theta):
    prob = sigmoid(np.dot(X, theta))
    return prob >= 0.5


def evaluate(test_x, test_y, theta):
    y_predicted = predict(test_x, theta)
    correct = 0
    for i, y in enumerate(test_y):
        if y == 0:
            y = False
        else:
            y = True
        if y == y_predicted[i]:
            correct = correct + 1
    total = y_predicted.size
    return (correct / total) * 100


spam_accuracy = evaluate(df_Logistic_X, df_Logistic_Y, final_theta)
print('Accuracy for Test Data on Spam Files based on Logistic Regression with L2 Regularization and stopwords: ' + str(
    spam_accuracy))

#############################################################
print("Testing Naive Bayes and Logistic Regression Model for Ham Directory")
df_Logistic_X = pandas.DataFrame(columns=all_words)
df_Logistic_Y = []

os.chdir(BASE_DIR + TEST_HAM_PATH)
test_list_of_ham = os.listdir()
count = 0
test_ham_vocabulary = {}
for filename in test_list_of_ham:
    test_ham_vocabulary.clear()
    data_row = {}
    try:
        file = open(filename, mode='r')
        text = file.readline()
        while text != '':
            list_of_words = text.split(' ')
            for word in list_of_words:
                if word not in stopwords:
                    try:
                        test_ham_vocabulary[word] += 1
                    except KeyError:
                        test_ham_vocabulary[word] = 1
                    try:
                        if word in all_words:
                            data_row[word] += 1
                    except KeyError:
                        if word in all_words:
                            data_row[word] = 1
            text = file.readline()
        file.close()
    except:
        print('Error in Ham Test file : ' + filename)
        file.close()
    df_Logistic_X = df_Logistic_X.append(data_row, ignore_index=True)
    df_Logistic_Y.append(0)

    probability_of_spam = np.log(prob_spam)
    for word in test_ham_vocabulary.keys():
        try:
            probability_of_spam += (test_ham_vocabulary[word] * train_spam_prob[word])
        except KeyError:
            probability_of_spam += (test_ham_vocabulary[word] * np.log(1 / (sum(train_spam_vocabulary.values())
                                                                            + len(train_spam_vocabulary.keys()))))
    probability_of_ham = np.log(prob_ham)
    for word in test_ham_vocabulary.keys():
        try:
            probability_of_ham += (test_ham_vocabulary[word] * train_ham_prob[word])
        except KeyError:
            probability_of_ham += (test_ham_vocabulary[word] * np.log(1 / (sum(train_ham_vocabulary.values())
                                                                           + len(train_ham_vocabulary.keys()))))
    if probability_of_ham > probability_of_spam:
        count += 1

ham_accuracy = (count / len(test_list_of_ham)) * 100
print('Accuracy for Test Data on Ham Files based on Naive Bayes Model with stopwords: ' + str(ham_accuracy))

df_Logistic_X = df_Logistic_X.fillna(0)
df_Logistic_X = df_Logistic_X.astype('int32')

intercept = np.ones((df_Logistic_X.shape[0], 1))
df_Logistic_X = np.concatenate((intercept, df_Logistic_X), axis=1)
df_Logistic_Y = np.array(df_Logistic_Y)

spam_accuracy = evaluate(df_Logistic_X, df_Logistic_Y, final_theta)
print('Accuracy for Test Data on Ham Files based on Logistic Regression with L2 Regularization and stopwords: ' + str(
    spam_accuracy))

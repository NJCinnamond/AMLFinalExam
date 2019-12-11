import dataholder 
import word2vec_model
import numpy as np
import pandas as pd
import scipy as sp
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

stop_words = stopwords.words('english')

def lemmatize(word_string):
    wordnet_lemmatizer = WordNetLemmatizer()

    line_with_lemmas = []
    for word in word_string:
        line_with_lemmas.append(wordnet_lemmatizer.lemmatize(word))
    return line_with_lemmas

# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def getTrainTags(num_samples):
    tags = {}
    for sample in range(num_samples):
        with open('tags_train/'+str(sample)+'.txt', 'r') as file:
            data = file.read().replace(':', ' ').replace('\n', ' ')
            data = data.split()
            data = lemmatize(data)
            data = data[1::2]
        tags[sample] = data
    return tags

def getTestTags(num_samples):
    tags = {}
    for sample in range(num_samples):
        with open('tags_test/'+str(sample)+'.txt', 'r') as file:
            data = file.read().replace(':', ' ').replace('\n', ' ')
            data = data.split()
            data = lemmatize(data)
            data = data[1::2]
        tags[sample] = data
    return tags

def getTrainDescript(num_samples):
    descript = {}
    for sample in range(num_samples):
        with open('descriptions_train/'+str(sample)+'.txt', 'r') as file:
            data = file.read().replace('\n', ' ')
            data = data.split()
            data = lemmatize(data)
            clean_data = [i for i in data if i not in stop_words]
            clean_data = ' '.join(clean_data)
        descript[sample] = clean_data
    return descript

def getTestDescript(num_samples):
    descript = {}
    for sample in range(num_samples):
        with open('descriptions_test/'+str(sample)+'.txt', 'r') as file:
            data = file.read().replace('\n', ' ')
            data = data.split()
            data = lemmatize(data)
            clean_data = [i for i in data if i not in stop_words]
            clean_data = ' '.join(clean_data)
        descript[sample] = clean_data
    return descript

def getTrainResNet(num_samples):
    resnet = {}
    resnet_fets = pd.read_csv('features_train/features_resnet1000_train.csv', header=None)
    resnet_fets = np.array(resnet_fets)

    for sample in range(num_samples):
        sample_index = np.where(resnet_fets[:,0] == 'images_train/'+str(sample)+'.jpg')[0][0]
        resnet[sample] = resnet_fets[sample_index][1:]

    print("Returning resnet train dict")
    return resnet

def getTestResNet(num_samples):
    resnet = {}
    resnet_fets = pd.read_csv('features_test/features_resnet1000_test.csv', header=None)
    resnet_fets = np.array(resnet_fets)

    for sample in range(num_samples):
        sample_index = np.where(resnet_fets[:,0] == 'images_test/'+str(sample)+'.jpg')[0][0]
        resnet[sample] = resnet_fets[sample_index][1:]

    print("Returning resnet test dict")
    return resnet

def getTrainResNetIntermediate(num_samples):
    resnet = {}
    resnet_fets = pd.read_csv('features_train/features_resnet1000intermediate_train.csv', header=None)
    resnet_fets = np.array(resnet_fets)

    for sample in range(num_samples):
        sample_index = np.where(resnet_fets[:,0] == 'images_train/'+str(sample)+'.jpg')[0][0]
        resnet[sample] = resnet_fets[sample_index][1:]
    
    print("Returning resnet intermediate train dict")
    return resnet

def getTestResNetIntermediate(num_samples):
    resnet = {}
    resnet_fets = pd.read_csv('features_test/features_resnet1000intermediate_test.csv', header=None)
    resnet_fets = np.array(resnet_fets)

    for sample in range(num_samples):
        sample_index = np.where(resnet_fets[:,0] == 'images_test/'+str(sample)+'.jpg')[0][0]
        resnet[sample] = resnet_fets[sample_index][1:]

    print("Returning resnet intermediate test dict")
    return resnet

def featureVectorFromDict(dictVals):
    feat_vect = []
    for item in dictVals:
        feat_vect.append(np.array(item, dtype=float))
    return feat_vect

NUM_SAMPLES_TRAIN = 10000
NUM_SAMPLES_TEST = 2000

imgData = dataholder.DataHolder(getTrainTags(NUM_SAMPLES_TRAIN), getTestTags(NUM_SAMPLES_TEST), 
                                    getTrainResNet(NUM_SAMPLES_TRAIN), getTestResNet(NUM_SAMPLES_TEST),
                                    getTrainResNetIntermediate(NUM_SAMPLES_TRAIN), getTestResNetIntermediate(NUM_SAMPLES_TEST),
                                    getTrainDescript(NUM_SAMPLES_TRAIN), getTestDescript(NUM_SAMPLES_TEST))

#  using word2vec path similarity model
'''
accuracy = 0
for idx in range(len(imgData.test_descript)):
    pred = word2vec_model.predictUsingDescript(imgData, imgData.test_descript[idx])
    print(pred)
    print(idx, " predicted ", pred[-1])

    if (idx == pred[-1]):
        accuracy += 1

print("Final accuracy: ", float(accuracy/NUM_SAMPLES))
'''

# Get tag vocabs
train_tags_vocab = set([item for sublist in imgData.train_tags.values() for item in sublist])
test_tags_vocab = set([item for sublist in imgData.test_tags.values() for item in sublist])

train_tags = [item for sublist in imgData.train_tags.values() for item in sublist]
test_tags = [item for sublist in imgData.test_tags.values() for item in sublist]

cv = CountVectorizer(min_df = 3)
X_train_bow = cv.fit_transform(imgData.train_descript.values()).toarray()
vocab = np.array(cv.get_feature_names())
print(vocab)
transformer = TfidfTransformer()
X_train_tfidf = transformer.fit_transform(X_train_bow).toarray()

#cv = CountVectorizer(vocabulary = vocab)
X_test_bow = cv.transform(imgData.test_descript.values()).toarray()
#transformer = TfidfTransformer()
X_test_tfidf = transformer.transform(X_test_bow).toarray()

final_train_tag = []
cv = CountVectorizer(vocabulary = train_tags_vocab)
for item in imgData.train_tags.values():
    temp = cv.transform(item).toarray()
    temp_add = np.zeros(len(train_tags_vocab))
    for row in temp:
        temp_add = np.add(row, temp_add)
    final_train_tag.append(np.array(temp_add))

final_test_tag = []
for item in imgData.test_tags.values():
    temp = cv.transform(item).toarray()
    temp_add = np.zeros(len(train_tags_vocab))
    for row in temp:
        temp_add = np.add(row, temp_add)
    final_test_tag.append(np.array(temp_add))

resnet_train_cnn = featureVectorFromDict(imgData.train_res.values())
resnet_train_int = featureVectorFromDict(imgData.train_resint.values())

resnet_test_cnn = featureVectorFromDict(imgData.test_res.values())
resnet_test_int = featureVectorFromDict(imgData.test_resint.values())

resnet_train = np.concatenate((resnet_train_cnn, resnet_train_int), axis=1)
resnet_test = np.concatenate((resnet_test_cnn, resnet_test_int), axis=1)

X_train = X_train_bow
#y_train = np.array(np.concatenate((final_train_tag, resnet_train_int), axis=1))
X_test = X_test_bow
#y_test = np.array(np.concatenate((final_test_tag, resnet_test_int), axis=1))

y_train = np.array(final_train_tag)
y_test = np.array(final_test_tag)

print(X_train)
print(y_train)
print(X_train.shape)
print(y_train.shape)

'''
stdscl = StandardScaler()

X_train = stdscl.fit_transform(X_train)
X_test = stdscl.fit_transform(X_test)
'''
'''
clf = Ridge()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)


'''

def baseline_model():
    # define the keras model
    model = Sequential()
    model.add(Dense(3500, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(3500, activation='relu'))
    model.add(Dense(3500, activation='relu'))
    model.add(Dense(3500, activation='relu'))
    model.add(Dense(3500, activation='relu'))
    model.add(Dense(3500, activation='relu'))
    model.add(Dense(3500, activation='relu'))
    model.add(Dense(len(y_train[0]), activation='linear'))

    # compile the keras model
    model.compile(loss='mse', optimizer='adam')
    return model

estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=20, batch_size=1000, verbose=1)

# fit the keras model on the dataset
estimator.fit(X_train, y_train, epochs=20, batch_size=1000, verbose=1)

# make class predictions with the model\
predictions = estimator.predict(X_test)


top_20_count = 0
correct_count = 0
pred_list = np.zeros((len(predictions), 2))
for i in range(len(predictions)):
    pred_list[i][0] = i
    pred_list_temp = np.zeros((len(y_test), 2))

    lowest_mse_index = None
    lowest_cur_mse = 10000000000000

    for j in range(len(y_test)):
        cur_mse = np.linalg.norm(predictions[i]-y_test[j])
        
        if cur_mse < lowest_cur_mse:
            lowest_cur_mse = cur_mse
            lowest_mse_index = j

        pred_list_temp[j][0] = j
        pred_list_temp[j][1] = cur_mse

    pred_list_temp = pred_list_temp[pred_list_temp[:,1].argsort()]
        
    pred_list[i][1] = lowest_mse_index
    '''
    print("True value: ", i)
    print("Predicted value: ", pred_list[i][1])
    print("Index of true value: ", np.where(pred_list_temp[:,0] == i))
    '''
    if (np.where(pred_list_temp[:,0] == i)[0] < 20):
        top_20_count = top_20_count + 10
        if (np.where(pred_list_temp[:,0] == i)[0] == 0):
            correct_count = correct_count + 1
        
print(pred_list)
print(top_20_count)
print(correct_count)

        







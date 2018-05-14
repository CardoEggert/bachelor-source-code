
# coding: utf-8

# In[1]:


# Vajalikud importid
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score


# In[2]:


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[3]:


def generateColumns(isStationary, isNotStationary):
    columns = []
    for x in range(32):
        for y in range(32):
            columns.append('stationary-non-stationary' + str(x) + ' ' + str(y))
    if isStationary and isNotStationary:
        for x in range(32):
            for y in range(32):
                columns.append('move' + str(x) + ' ' + str(y))
    columns.append('label')
    return columns


# In[4]:


def readInFiles(dest, readStat, readNotStat, label, percent):
    dirs = os.listdir(dest)
    pics = []
    folderPath = dest
    index = 0
    if not readStat and not readNotStat:
        # Loe niisama pildid
        for i in range(len(dirs)):
            if os.path.isfile(os.path.join(folderPath,dirs[i])):
                pics.append([])
                pics[index] = readPic(os.path.join(folderPath,dirs[i]), label)
                pics[index].append(label)
                index += 1
    elif not readStat and readNotStat:
        # Loe mittestatsionaarseid pilte
        for i in range(len(dirs)):
            if dirs[i] == "moving"+str(percent):
                dirMoving = os.listdir(os.path.join(folderPath,dirs[i]))
                for move in dirMoving:
                    if "move" not in move:
                        pics.append([])
                        pics[index] = readPic(os.path.join(folderPath,dirs[i], move), label)
                        pics[index].append(label)
                        index += 1
    elif readStat and not readNotStat:
        # Loe statsionaarseid pilte
        for i in range(len(dirs)):
            if dirs[i] == "moving"+str(percent):
                dirMoving = os.listdir(os.path.join(folderPath,dirs[i]))
                for move in dirMoving:
                    if "stationary" not in move:
                        pics.append([])
                        pics[index] = readPic(os.path.join(folderPath,dirs[i], move), label)
                        pics[index].append(label)
                        index += 1
    elif readStat and readNotStat:
        # Loe statsionaarseid ja mittestatsionaarseid pilte
        for i in range(len(dirs)):
            if dirs[i] == "moving"+str(percent):
                dirMoving = os.listdir(os.path.join(folderPath,dirs[i]))
                newPic = []
                for move in dirMoving:
                    if "move" not in move:
                        newPic = readPic(os.path.join(folderPath,dirs[i], move), label)
                for move in dirMoving:
                    if "stationary" not in move:
                        movingParts = readPic(os.path.join(folderPath,dirs[i], move), label)
                        for moving in movingParts:
                            newPic.append(moving)
                newPic.append(label)
                pics.append([])
                pics[index] = newPic
                index += 1
    return pics


# In[5]:


def readPic(picPath, label):
    imgValues = []
    img = cv2.imread(picPath, cv2.IMREAD_COLOR)
    for x in range(32):
        imgValues.append([])
        for y in range(32):
            imgValues[x].append(0)
            if img[x][y][0] > 0:
                imgValues[x][y] = 1
    newData = []
    for val in imgValues:
        for v in val:
            newData.append(v)
    return newData


# In[6]:


def extractData(allowedToPass, isStationary, isNotStationary):
    trainData = []
    testData = []
    train = []
    test = []
    labels = ['stairs', 'bending', 'sitting', 'standing', 'walk']
    for label in labels:
        for picDir in os.listdir(os.path.join(".", label, 'train')):
            train = readInFiles(os.path.join(".", label,'train', picDir), isStationary, isNotStationary, label, allowedToPass)
            for t in train:
                trainData.append(t)
        if not isStationary and not isNotStationary:
            for picDir in os.listdir(os.path.join(".", label, 'test')):
                testData.append(readInFiles(os.path.join(".", label,'test', picDir), isStationary, isNotStationary, label, allowedToPass))
        else:
            for picDir in os.listdir(os.path.join(".", label, 'test')):
                test = readInFiles(os.path.join(".", label,'test', picDir), isStationary, isNotStationary, label, allowedToPass)
                for t in test:
                    testData.append(t)
    if not isStationary and not isNotStationary:
        return trainData, testData
    numpyArrayTrain = np.array(trainData)
    numpyArrayTest = np.array(testData)
    newColumns = generateColumns(isStationary, isNotStationary)
    train = pd.DataFrame(numpyArrayTrain, columns=newColumns)
    test = pd.DataFrame(numpyArrayTest, columns=newColumns)
    X_test = test.drop(["label"], axis=1).values
    y_test = test["label"].values
    X = train.drop(["label"], axis=1).values
    Y = train["label"].values
    return train, test, X_test, y_test, X, Y


# In[7]:


trainPics, testPics = extractData(0.4, False, False)


# In[8]:


numpyArrayTrain = np.array(trainPics)
newColumns = generateColumns(False, False)
train = pd.DataFrame(numpyArrayTrain, columns=newColumns)
X_pics = train.drop(["label"], axis=1).values
Y_pics = train["label"].values


# In[9]:


trainStationaryPics, testStationaryPics, X_testStationaryPics, y_testStationaryPics, X_Stationarypics, Y_Stationarypics = extractData(0.4, True, False)


# In[10]:


trainNotStationaryPics, testNotStationaryPics, X_testNotStationaryPics, y_testNotStationaryPics, X_NotStationarypics, Y_NotStationarypics = extractData(0.4, False, True)


# In[11]:


trainStationaryAndNotStationaryPics, testStationaryAndNotStationaryPics, X_StationaryAndNotStationarytestPics, y_StationaryAndNotStationarytestPics, X_StationaryAndNotStationarypics, Y_StationaryAndNotStationarypics = extractData(0.4, True, True)


# In[12]:


trainSets = [trainPics, trainStationaryPics, trainNotStationaryPics, trainStationaryAndNotStationaryPics]
testSets = [testPics, testStationaryPics, testNotStationaryPics, testStationaryAndNotStationaryPics]
X_test_sets = [[], X_testStationaryPics, X_testNotStationaryPics, X_StationaryAndNotStationarytestPics]
Y_test_sets = [[], y_testStationaryPics, y_testNotStationaryPics, y_StationaryAndNotStationarytestPics]
X_train_sets = [X_pics, X_Stationarypics, X_NotStationarypics, X_StationaryAndNotStationarypics]
Y_train_sets = [Y_pics, Y_Stationarypics, Y_NotStationarypics, Y_StationaryAndNotStationarypics]


# In[13]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
def calculate_acc(y_true, y_pred):
    countCorrect = 0
    countAll = 0
    setOfValues = {}
    labels = ['stairs', 'bending', 'sitting', 'standing', 'walk']
    for label in labels:
        setOfValues[label] = (0, 0)
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            countCorrect += 1
            countAll += 1
            setOfValues[y_true[i]] = (setOfValues[y_true[i]][0]+1, setOfValues[y_true[i]][1]+1)
        else:
            countAll += 1
            setOfValues[y_true[i]] = (setOfValues[y_true[i]][0], setOfValues[y_true[i]][1]+1)
    matrix = confusion_matrix(y_true, y_pred, labels = labels)
    fscore = f1_score(y_true, y_pred, average=None, labels = labels)
    pscore = precision_score(y_true, y_pred, average=None, labels = labels)
    rscore = recall_score(y_true, y_pred, average=None, labels = labels)
    tags = ['TR', 'KU', 'I', 'SE', 'KÕ']
    print('\\begin{tabular}{| c | c | c | c | c | c | c | c | c | c | c |} \n \\hline \n & \\multicolumn{6}{| c |}{Ennustatud märgend} & \\multicolumn{3}{| c |}{Skoorid} \\\\ \n \\hline')
    print('Tegelik & TR & KU & I & SE & KÕ & $\\sum$ & T & S & FS \\\\ \n \\hline')
    for i in range(len(labels)):
        rowString = tags[i] + ' & '
        for j in range(len(matrix[i])):
            rowString += str(str(matrix[i][j]) + ' & ')
        rowString += str(str(sum(matrix[i])) + ' & ')
        rowString += str(str(round(pscore[i],2)) + ' & ')
        rowString += str(str(round(rscore[i],2)) + ' & ')
        rowString += str(str(round(fscore[i],2)) + ' \\\\')
        rowString += '\n \\hline'
        print(rowString)
    downsums = []
    allSum = 0
    for i in range(len(labels)):
        downsums.append([])
        for j in range(len(matrix[i])):
            downsums[i].append(matrix[j][i])
            allSum += matrix[j][i]
    downString = '$\\sum$ & '
    for i in range(len(downsums)):
        downString += str(str(sum(downsums[i])) + ' & ')
    downString += str(str(allSum) + ' & ')
    downString += '\\multicolumn{3}{| c |}{Täpsus: ' + str(str(round(accuracy_score(y_true, y_pred),3) * 100) + '\\%} \\\\')
    print(downString)
    print('\\hline')
    print('\\end{tabular}')
    return countCorrect/countAll


# In[14]:


i = 0
clf = RandomForestClassifier(n_estimators = 1000)
# use a full grid over all parameters
param_grid = {"max_depth": [7, 8 ,9],
              "max_features": [100, 125, 150],
              "min_samples_split": [6],
              "min_samples_leaf": [3]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_train_sets[i], Y_train_sets[i])

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


# In[ ]:


i = 1
clf = RandomForestClassifier(n_estimators = 2000)
# use a full grid over all parameters
param_grid = {"max_depth": [3, 4, 5, 6, 7],
              "max_features": [50, 75, 100, 125, 150],
              "min_samples_split": [5, 10, 20],
              "min_samples_leaf": [3, 5, 7, 10]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_train_sets[i], Y_train_sets[i])

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


# In[ ]:


i = 2
clf = RandomForestClassifier(n_estimators = 2000)
# use a full grid over all parameters
param_grid = {"max_depth": [3, 4, 5, 6, 7],
              "max_features": [50, 75, 100, 125, 150],
              "min_samples_split": [5, 10, 20],
              "min_samples_leaf": [3, 5, 7, 10]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_train_sets[i], Y_train_sets[i])

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


# In[ ]:


i = 3
clf = RandomForestClassifier(n_estimators = 2000)
# use a full grid over all parameters
param_grid = {"max_depth": [3, 4, 5, 6, 7],
              "max_features": [50, 75, 100, 125, 150],
              "min_samples_split": [5, 10, 20],
              "min_samples_leaf": [3, 5, 7, 10]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_train_sets[i], Y_train_sets[i])

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


# In[ ]:


i = 3
clf = RandomForestClassifier(n_estimators = 2000)
# use a full grid over all parameters
param_grid = {"max_depth": [ 9, 10, 11, 12 ],
              "max_features": [130, 135, 140, 145, 150, 155],
              "min_samples_split": [6, 7],
              "min_samples_leaf": [3]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_train_sets[i], Y_train_sets[i])

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
i = 0
print(i, "\n")
clf = KNeighborsClassifier()
# use a full grid over all parameters
param_grid = {"n_neighbors": [10, 11, 12, 13 ,14]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_train_sets[i], Y_train_sets[i])

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
for i in range(0,4,1):
    print(i, "\n")
    clf = KNeighborsClassifier()
    # use a full grid over all parameters
    param_grid = {"n_neighbors": [3, 4, 5, 6, 7],
                  "leaf_size": [5,10, 15, 20]}

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(X_train_sets[i], Y_train_sets[i])

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)


# In[14]:


def collectLabels(pics):
    labels = []
    for pic in pics:
        labels.append(pic[0][len(pic[0])-1])
    return labels


# In[15]:


def collectAllLabels(pics):
    labels = []
    for pic in pics:
        for subpic in pic:
            labels.append(subpic[len(subpic)-1])
    return labels


# In[16]:


i = 0
clf = RandomForestClassifier(n_estimators = 1000,
                             max_depth = 7, max_features = 100,
                             min_samples_leaf = 3, min_samples_split = 6)
clf.fit(X_train_sets[i], Y_train_sets[i])


# In[17]:


def parseToNParr(picData):
    numpyArrayTest = np.array([picData])
    newColumns = generateColumns(False, False)
    test = pd.DataFrame(numpyArrayTest, columns=newColumns)
    X_test = test.drop(["label"], axis=1).values
    return X_test


# In[18]:


def findMostCommonIndx(arr):
    mostCommonIndex = -1
    biggestCount = -1
    for el in list(set(arr)):
        counts = arr.count(el)
        if counts >= biggestCount:
            biggestCount = counts
            mostCommonIndex = el
    return mostCommonIndex


# In[19]:


def predictForTestPics(pics, clf):
    labels = ['stairs', 'bending', 'sitting', 'standing', 'walk']
    predictedVals = []
    for subPics in pics:
        pred = []
        for subPic in subPics:
            pred.append(labels.index(clf.predict(parseToNParr(subPic))))
        predictedVals.append(labels[findMostCommonIndx(pred)])
    return predictedVals


# In[20]:


def predictAll(pics, clf):
    pred = []
    for subPics in pics:
        for subPic in subPics:
            pred.append(clf.predict(parseToNParr(subPic)))
    return pred


# In[21]:


accArrRR = []
accArrKNN = []
f_scores_RR = []
f_scores_KNN = []
labels = ['stairs', 'bending', 'sitting', 'standing', 'walk']


# In[47]:


i = 0
clf = RandomForestClassifier(n_estimators = 1000,
                             max_depth = 7, max_features = 100,
                             min_samples_leaf = 3, min_samples_split = 6)
clf.fit(X_train_sets[i], Y_train_sets[i])
y_true = collectLabels(testPics)
y_pred = predictForTestPics(testPics, clf)
accArrRR.append((i, calculate_acc(y_true, y_pred)))
f_scores_RR.append(f1_score(y_true, y_pred, average=None, labels = labels))


# In[51]:


# seeria 157 võimalikud ennustused
for pic in testPics[157]:
    print(clf.predict(parseToNParr(pic)))


# In[44]:


showPics(testPics, 'walk', 'tüüp-1-rr', y_true, y_pred)


# In[45]:


i = 0
clf = KNeighborsClassifier(n_neighbors = 12)
clf.fit(X_train_sets[i], Y_train_sets[i])
y_true = collectLabels(testPics)
y_pred = predictForTestPics(testPics, clf)
accArrKNN.append((i, calculate_acc(y_true, y_pred)))
f_scores_KNN.append(f1_score(y_true, y_pred, average=None, labels = labels))


# In[104]:


def turnToPic(data):
    emptyPic = np.zeros([32,32,3])
    for i in range(32):
        subArr = data[i*32:(i+1)*32]
        for j in range(32):
            if subArr[j] == 1:
                emptyPic[i, j] = [255,255,255]
            else:
                emptyPic[i, j] = [0,0,0]
    return emptyPic        


# In[105]:


def generateMovingPic(pics):
    endPic = []
    for x in range(32*32):
        decisionArr = []
        for i in range(len(pics)):
            if int(pics[i][x]) == 1:
                decisionArr.append(1)
            else:
                decisionArr.append(0)
        if float(sum(decisionArr)/len(pics)) < 0.6 and float(sum(decisionArr)/len(pics)) > 0.4:
            endPic.append(1)
        else:
            endPic.append(0)
    return endPic
    


# In[106]:


def generateNotMovingPic(pics):
    endPic = []
    for x in range(32*32):
        decisionArr = []
        for i in range(len(pics)):
            if int(pics[i][x]) == 1:
                decisionArr.append(1)
            else:
                decisionArr.append(0)
        if float(sum(decisionArr)/len(pics)) > 0.6:
            endPic.append(1)
        else:
            endPic.append(0)
    return endPic
    


# In[127]:


import os
import shutil
def showImage(pics, indx, label, dir):
    directory = 'example_' + str(indx) + '_' + label
    os.makedirs(os.path.join(dir, directory))
    for i in range(len(pics[indx])):
        cv2.imwrite(str('image_predicted_' + label + str(i) + '.png'),turnToPic(pics[indx][i]))
        os.rename('./' + str('image_predicted_' + label + str(i) + '.png'), os.path.join('./', dir, directory,  str('image_predicted_' + label + str(i) + '.png')))
    cv2.imwrite(str('image_predicted_' + label + str(i) + 'moving.png'),turnToPic(generateMovingPic(pics[indx])))
    os.rename('./' + str('image_predicted_' + label + str(i) + 'moving.png'), os.path.join('./', dir, directory,  str('image_predicted_' + label + str(i) + 'moving.png')))
    cv2.imwrite(str('image_predicted_' + label + str(i) + 'notmoving.png'),turnToPic(generateNotMovingPic(pics[indx])))
    os.rename('./' + str('image_predicted_' + label + str(i) + 'notmoving.png'), os.path.join('./', dir, directory,  str('image_predicted_' + label + str(i) + 'notmoving.png')))


# In[129]:


# Otsida pilt, kus tuvastab pildile kõndimine
def showPics(pics, label, dir, y_true, y_pred):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    for i in range(len(y_true)):
        if y_true[i] == label or y_pred[i] == label:
            showImage(pics, i, str('predicted_' + y_pred[i]  + '_but_actual_' + y_true[i]), dir)


# In[46]:


showPics(testPics, 'walk', 'tüüp-1-knn', y_true, y_pred)


# In[69]:


i = 0
clf = RandomForestClassifier(n_estimators = 1000,
                             max_depth = 7, max_features = 100,
                             min_samples_leaf = 3, min_samples_split = 6)
clf.fit(X_train_sets[i], Y_train_sets[i])
y_true = collectAllLabels(testPics)
y_pred = predictAll(testPics, clf)
accArrRR.append((i, calculate_acc(y_true, y_pred)))
f_scores_RR.append(f1_score(y_true, y_pred, average=None, labels = labels))


# In[84]:


def showAllPics(pics, label, dir, y_true, y_pred):
    newPics = []
    for pic in pics:
        for subPic in pic:
            newPics.append(subPic)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    for i in range(len(y_true)):
        if y_true[i] == label:
            directory = str('example_' + str(i) + '_' + label + '_actual_' + y_pred[i])
            os.makedirs(os.path.join(dir, directory))
            cv2.imwrite(str('image_predicted_' + label + '_but_actually_' + y_pred[i] + str(i) + '.png'),turnToPic(newPics[i]))
            os.rename('./' + str('image_predicted_' + label + '_but_actually_' + y_pred[i] + str(i) + '.png'), os.path.join('./', dir, directory,  str('image_predicted_' + label + '_but_actually_' + y_pred[i] + str(i) + '.png')))


# In[85]:


showAllPics(testPics, 'walk', 'tüüp-1-1-rr-walks', y_true, y_pred)


# In[86]:


showAllPics(testPics, 'stairs', 'tüüp-1-1-rr-stairs', y_true, y_pred)


# In[87]:


i = 0
clf = KNeighborsClassifier(n_neighbors = 12)
clf.fit(X_train_sets[i], Y_train_sets[i])
y_true = collectAllLabels(testPics)
y_pred = predictAll(testPics, clf)
accArrKNN.append((i, calculate_acc(y_true, y_pred)))
f_scores_KNN.append(f1_score(y_true, y_pred, average=None, labels = labels))


# In[88]:


showAllPics(testPics, 'walk', 'tüüp-1-1-KNN-walks', y_true, y_pred)


# In[89]:


showAllPics(testPics, 'stairs', 'tüüp-1-1-KNN-stairs', y_true, y_pred)


# In[163]:


i = 1
clf = RandomForestClassifier(n_estimators = 1000,
                             max_depth = 7, max_features = 50,
                             min_samples_leaf = 3, min_samples_split = 5)
clf.fit(X_train_sets[i], Y_train_sets[i])
y_true = Y_test_sets[i]
y_pred = clf.predict(X_test_sets[i])
accArrRR.append((i, calculate_acc(y_true, y_pred)))
f_scores_RR.append(f1_score(y_true, y_pred, average=None, labels = labels))


# In[148]:


showPics(testPics, 'walk', 'tüüp-2-rr-walk', y_true, y_pred)


# In[149]:


showPics(testPics, 'stairs', 'tüüp-2-rr-stairs', y_true, y_pred)


# In[164]:


showPics(testPics, 'standing', 'tüüp-2-rr-standing', y_true, y_pred)


# In[165]:


i = 1
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train_sets[i], Y_train_sets[i])
y_true = Y_test_sets[i]
y_pred = clf.predict(X_test_sets[i])
accArrKNN.append((i, calculate_acc(y_true, y_pred)))
f_scores_KNN.append(f1_score(y_true, y_pred, average=None, labels = labels))


# In[166]:


showPics(testPics, 'walk', 'tüüp-2-knn-walk', y_true, y_pred)


# In[167]:


showPics(testPics, 'stairs', 'tüüp-2-knn-stairs', y_true, y_pred)


# In[168]:


showPics(testPics, 'bending', 'tüüp-2-knn-bending', y_true, y_pred)


# In[154]:


i = 2
clf = RandomForestClassifier(n_estimators = 1000,
                             max_depth = 7, max_features = 75,
                             min_samples_leaf = 3, min_samples_split = 5)
clf.fit(X_train_sets[i], Y_train_sets[i])
y_true = Y_test_sets[i]
y_pred = clf.predict(X_test_sets[i])
accArrRR.append((i, calculate_acc(y_true, y_pred)))
f_scores_RR.append(f1_score(y_true, y_pred, average=None, labels = labels))


# In[155]:


showPics(testPics, 'walk', 'tüüp-3-rr-walk', y_true, y_pred)


# In[156]:


showPics(testPics, 'stairs', 'tüüp-3-rr-stairs', y_true, y_pred)


# In[157]:


i = 2
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train_sets[i], Y_train_sets[i])
y_true = Y_test_sets[i]
y_pred = clf.predict(X_test_sets[i])
accArrKNN.append((i, calculate_acc(y_true, y_pred)))
f_scores_KNN.append(f1_score(y_true, y_pred, average=None, labels = labels))


# In[158]:


showPics(testPics, 'walk', 'tüüp-3-knn-walk', y_true, y_pred)
showPics(testPics, 'stairs', 'tüüp-3-knn-stairs', y_true, y_pred)


# In[159]:


i = 3
clf = RandomForestClassifier(n_estimators = 1000,
                             max_depth = 12, max_features = 135,
                             min_samples_leaf = 3, min_samples_split = 6)
clf.fit(X_train_sets[i], Y_train_sets[i])
y_true = Y_test_sets[i]
y_pred = clf.predict(X_test_sets[i])
accArrRR.append((i, calculate_acc(y_true, y_pred)))
f_scores_RR.append(f1_score(y_true, y_pred, average=None, labels = labels))


# In[160]:


showPics(testPics, 'walk', 'tüüp-4-rr-walk', y_true, y_pred)
showPics(testPics, 'stairs', 'tüüp-4-rr-stairs', y_true, y_pred)


# In[161]:


i = 3
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train_sets[i], Y_train_sets[i])
y_true = Y_test_sets[i]
y_pred = clf.predict(X_test_sets[i])
accArrKNN.append((i, calculate_acc(y_true, y_pred)))
f_scores_KNN.append(f1_score(y_true, y_pred, average=None, labels = labels))


# In[162]:


showPics(testPics, 'walk', 'tüüp-4-knn-walk', y_true, y_pred)
showPics(testPics, 'stairs', 'tüüp-4-knn-stairs', y_true, y_pred)


# TODO: Tuvastada tulemused ( Precision & F-measure )

# TODO: Analüüsida

# 
# Täpsused

# In[143]:


# Täpsus üleüldse RR ja KNN
import matplotlib.pyplot as plt
plt.axis([-0.5,3.5,0.35,0.725])

a = plt.bar([str('Tüüp '+str(x[0]+1)) for x in accArrRR],[x[1] for x in accArrRR])
b = plt.bar([str('Tüüp '+str(x[0]+1)) for x in accArrKNN],[x[1] for x in accArrKNN])
plt.legend([a, b], ['Random forest', 'KNN'])
plt.ylabel('Täpsus')
plt.show()


# In[144]:


def getScores(f_scores, n):
    emptyArr = []
    for i in range(len(f_scores)):
        emptyArr.append(f_scores[i][n])
    return emptyArr


# Mõlemad  samas tabelis

# In[145]:


i = 0
scoresRR = getScores(f_scores_RR, i)
scoresKNN = getScores(f_scores_KNN, i)
plt.axis([-0.5,3.5,max(min(scoresRR + scoresKNN) -0.2, 0), max(scoresRR + scoresKNN) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
a = plt.bar(types, scoresRR)
b = plt.bar(types, scoresKNN)
plt.legend([a, b], ['Random forest', 'KNN'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


i = 1
scoresRR = getScores(f_scores_RR, i)
scoresKNN = getScores(f_scores_KNN, i)
plt.axis([-0.5,3.5,max(min(scoresRR + scoresKNN) -0.2, 0), max(scoresRR + scoresKNN) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
a = plt.bar(types, scoresRR)
b = plt.bar(types, scoresKNN)
plt.legend([a, b], ['Random forest', 'KNN'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


i = 2
scoresRR = getScores(f_scores_RR, i)
scoresKNN = getScores(f_scores_KNN, i)
plt.axis([-0.5,3.5,max(min(scoresRR + scoresKNN) -0.2, 0), max(scoresRR + scoresKNN) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
a = plt.bar(types, scoresRR)
b = plt.bar(types, scoresKNN)
plt.legend([a, b], ['Random forest', 'KNN'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


i = 3
scoresRR = getScores(f_scores_RR, i)
scoresKNN = getScores(f_scores_KNN, i)
plt.axis([-0.5,3.5,max(min(scoresRR + scoresKNN) -0.2, 0), max(scoresRR + scoresKNN) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
a = plt.bar(types, scoresRR)
b = plt.bar(types, scoresKNN)
plt.legend([a, b], ['Random forest', 'KNN'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


i = 4
scoresRR = getScores(f_scores_RR, i)
scoresKNN = getScores(f_scores_KNN, i)
plt.axis([-0.5,3.5,max(min(scoresRR + scoresKNN) -0.2, 0), max(scoresRR + scoresKNN) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
a = plt.bar(types, scoresRR)
b = plt.bar(types, scoresKNN)
plt.legend([a, b], ['Random forest', 'KNN'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# RR samas tabelis

# In[ ]:


i = 0
scores = getScores(f_scores_RR, i)
plt.axis([-0.5,3.5,max(min(scores) - 0.2, 0),max(scores) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
a = plt.bar(types, scores)
plt.legend([a], ['Random forest'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


i = 1
scores = getScores(f_scores_RR, i)
plt.axis([-0.5,3.5,max(min(scores) - 0.2, 0),max(scores) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
a = plt.bar(types, scores)
plt.legend([a], ['Random forest'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


i = 2
scores = getScores(f_scores_RR, i)
plt.axis([-0.5,3.5,max(min(scores) - 0.2, 0),max(scores) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
a = plt.bar(types, scores)
plt.legend([a], ['Random forest'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


i = 3
scores = getScores(f_scores_RR, i)
plt.axis([-0.5,3.5,max(min(scores) - 0.2, 0),max(scores) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
a = plt.bar(types, scores)
plt.legend([a], ['Random forest'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


i = 4
scores = getScores(f_scores_RR, i)
plt.axis([-0.5,3.5,max(min(scores) - 0.2, 0),max(scores) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
a = plt.bar(types, scores)
plt.legend([a], ['Random forest'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# KNN ühes tabelis

# In[ ]:


i = 0
scores = getScores(f_scores_KNN, i)
plt.axis([-0.5,3.5,max(min(scores) - 0.2, 0),max(scores) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
b = plt.bar(types, scores)
plt.legend([b], ['KNN'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


i = 1
scores = getScores(f_scores_KNN, i)
plt.axis([-0.5,3.5,max(min(scores) - 0.2, 0),max(scores) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
b = plt.bar(types, scores)
plt.legend([b], ['KNN'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


i = 2
scores = getScores(f_scores_KNN, i)
plt.axis([-0.5,3.5,max(min(scores) - 0.2, 0),max(scores) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
b = plt.bar(types, scores)
plt.legend([b], ['KNN'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


i = 3
scores = getScores(f_scores_KNN, i)
plt.axis([-0.5,3.5,max(min(scores) - 0.2, 0),max(scores) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
b = plt.bar(types, scores)
plt.legend([b], ['KNN'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


i = 4
scores = getScores(f_scores_KNN, i)
plt.axis([-0.5,3.5,max(min(scores) - 0.2, 0),max(scores) + 0.2])
types = [str('Tüüp '+str(x)) for x in range(1,5)]
b = plt.bar(types, scores)
plt.legend([b], ['KNN'])
plt.ylabel('F-skoor')
plt.xlabel('Märgend ' + labels[i])
plt.show()


# In[ ]:


import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

def readPic(picPath):
    imgValues = []
    #picPath = picPath.replace('/',"\\")
    img = None
    if os.path.isfile(picPath):
        print(picPath)
        img = cv2.imread(picPath, cv2.IMREAD_COLOR)
        print(img)
        if img is None:
            img = cv2.imread(picPath.replace("/", "\\"), cv2.IMREAD_COLOR)
            if img is None:
                img = cv2.imread(picPath.replace("/", "//"), cv2.IMREAD_COLOR)
                if img is None:
                    img = cv2.imread(picPath.replace("/", "\ "), cv2.IMREAD_COLOR)
    if img is not None:
        for x in range(32):
            imgValues.append([])
            for y in range(32):
                imgValues[x].append(0)
                if img[x][y][0] > 0:
                    imgValues[x][y] = 1
    return imgValues

def readAllPicsInFolder(folderPath):
    dirs = os.listdir(folderPath)
    pics = []
    indx = 0
    for i in range(len(dirs)):
        if "moving" not in dirs[i]:
            pics.append([])
            pics[indx] = readPic(folderPath + '/' + dirs[i])
            indx +=1
    return pics

def extractStationaryAndNonStationaryParts(picData, stationaryLimit):
    #stationary limit should be from 10%-40% prefferably 20%
    newPicData = []
    for x in range(32):
        newPicData.append([])
        for y in range(32):
            decisionArr = []
            print(picData)
            for i in range(len(picData)):
                if int(picData[i][x][y]) == 1:
                    decisionArr.append(1)
                else:
                    decisionArr.append(0)
            if len(picData) > 0:
                if float(sum(decisionArr)/len(picData)) > (1-stationaryLimit):
                    newPicData[x].append(1)
                else:
                    newPicData[x].append(0)
    return newPicData

def extractMovingParts(picData, movingLimit):
    # Moving limit should be from 20 % to 80% prefferably 60%
    newPicData = []
    for x in range(32):
        newPicData.append([])
        for y in range(32):
            decisionArr = []
            print(picData)
            for i in range(len(picData)):
                if int(picData[i][x][y]) == 1:
                    decisionArr.append(1)
                else:
                    decisionArr.append(0)
            if float(sum(decisionArr)/len(picData)) < float(1- ((1-movingLimit)/2)) and float(sum(decisionArr)/len(picData)) > float(((1-movingLimit)/2)) and len(picData) > 0:
                newPicData[x].append(1)
            else:
                newPicData[x].append(0)
    return newPicData

def writePic(extractedData, writeWhere):
    blank_image = np.zeros((32,32,3), np.uint8)
    for i in range(32):
        for y in range(32):
            if extractedData[i][y] == 1:
                blank_image[i][y] = [255, 255, 255]
    cv2.imwrite(writeWhere, blank_image)

import os
labels = ['stairs', 'bending', 'kneeling', 'lying', 'sitting', 'standing', 'walk']
folderName = 'test'
# for allowedToPass in np.arange(0.1, 0.41, 0.05):
#     for label in labels:
#         for picDir in os.listdir(os.path.join(".", label, folderName)):
#             picData = readAllPicsInFolder(os.path.join(".", label, folderName, picDir))
#             extractedMovingParts = extractMovingParts(picData, allowedToPass)
#             extractedStationaryAndNonStationaryParts = extractStationaryAndNonStationaryParts(picData, 1 - (allowedToPass * 2))
#             if not os.path.exists(os.path.join(".", label, folderName, picDir, 'moving' + str(allowedToPass))):
#                 os.makedirs(os.path.join(".", label, folderName, picDir, 'moving' + str(allowedToPass)))
#             writePic(extractedMovingParts, os.path.join(".", label, folderName, picDir, 'moving' + str(allowedToPass), "result_move.png"))
#             writePic(extractedStationaryAndNonStationaryParts, os.path.join(".", label, folderName, picDir, 'moving' + str(allowedToPass), "result-stationary.png"))
allowedToPass = 0.4
label = 'walk'
picDir = "./tüüp-2-rr/example_612_walkpredictedwalk_but_actual_walk"
picData = readAllPicsInFolder(picDir)
extractedMovingParts = extractMovingParts(picData, allowedToPass)
extractedStationaryAndNonStationaryParts = extractStationaryAndNonStationaryParts(picData, 1 - (allowedToPass * 2))
if not os.path.exists(os.path.join(".", label, folderName, picDir, 'moving' + str(allowedToPass))):
    os.makedirs(os.path.join(".", label, folderName, picDir, 'moving' + str(allowedToPass)))
writePic(extractedMovingParts, os.path.join(".", label, folderName, picDir, 'moving' + str(allowedToPass), "result_move.png"))
writePic(extractedStationaryAndNonStationaryParts, os.path.join(".", label, folderName, picDir, 'moving' + str(allowedToPass), "result-stationary.png"))


# In[32]:


import os
seriesLength = []
seriesPerLabel = []
labels = ['stairs', 'bending', 'sitting', 'standing', 'walk']
for label in labels:
    seriesPerLabel.append([])
    for picDir in os.listdir(os.path.join(".", label, 'train')):
        train = os.listdir(os.path.join(".", label,'train', picDir))
        series = 0
        for t in train:
            if os.path.isfile(os.path.join(".", label,'train', picDir, t)):
                series += 1
        seriesLength.append(series)
        seriesPerLabel[labels.index(label)].append(series)
    for picDir in os.listdir(os.path.join(".", label, 'test')):
        test = os.listdir(os.path.join(".", label,'test', picDir))
        series = 0
        for t in test:
            if os.path.isfile(os.path.join(".", label,'test', picDir, t)):
                series += 1
        seriesLength.append(series)
        seriesPerLabel[labels.index(label)].append(series)


# In[33]:


print('Seeriad kokku ' + str(len(seriesLength)))
print('Keskmine seeria pikkus ' + str(sum(seriesLength)/len(seriesLength)))
labels = ['stairs', 'bending', 'sitting', 'standing', 'walk']
labelEst = ['trepp', 'kummardamine', 'istumine', 'seismine', 'kõndimine']
avg = 0
for i in range(5):
    avg += int(round(sum(seriesPerLabel[i])/len(seriesPerLabel[i]), 1))
    print(labelEst[i] + ' & ' + str(len(seriesPerLabel[i])) + ' & ' + str(round(sum(seriesPerLabel[i])/len(seriesPerLabel[i]), 1)) + '\\\\')
    print('\\hline')
print('\\sum & ' + str(len(seriesLength)) + ' & ' + str(round(sum(seriesLength)/len(seriesLength),1)) + '\\\\')    


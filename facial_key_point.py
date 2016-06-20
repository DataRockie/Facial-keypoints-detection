import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn import svm


face_data = pd.read_csv("training.csv")
print "facial data read successfully!"
test_data = pd.read_csv("test.csv")
print "test data read successfully:"
lookup = pd.read_csv("IdLookupTable.csv")
for d in [face_data]:
    d['Image'] = face_data['Image'].apply(lambda im: np.fromstring(im, sep=' '))
####face_data = face_data[:100]
####CLIENT_FEATURES = [66.3643896806, 35.5310589143, 30.1018867925, 45.0022641509, 59.0371197537, 38.416372738, 74.6130150172, 34.5560261842, 39.2175587217, 43.807470158, 23.5879861377, 47.943781286, 54.5453985369, 32.3357720445, 74.4970350403, 27.0521370812, 36.4469772815, 37.8559876782, 19.7286099346, 39.7092029266, 50.6448595404, 54.3533305925, 78.3442433579, 69.1016557567, 37.4670773968, 78.6226415094, 57.6203311512, 73.7870619944, 58.9418559876, 80.0643049671]

####face_data_1['Image'] = face_data_1['Image'].apply(lambda im: np.fromstring(im, sep=' '))
face_data = face_data.dropna()

####test_data = test_data.fillna(face_data.mean())
####face_data =face_data.astype(np.float)
#### convert from [0, 255] => [0.0, 1.0]
####face_data = np.dtype('f8')
####face_data[np.all(face_data != 0, axis=1)]

 
total_samples = face_data.shape[0]
total_features = face_data.shape[1]
print"total_features", total_features
###print face_data.head()
####feature_columns = list(face_data.columns[:-1])
####feature_columns = (feature_columns - 48) / 48  # scale target coordinates to [-1, 1]
####feature_columns = feature_columns.astype(np.float32)
####target_columns = face_data.columns[-1]
x_all = np.vstack(face_data.ix[:, 'Image']).astype(np.float)
print "x_all",x_all
y_all = face_data.drop('Image', axis=1).values
print "y_all", y_all

####print "test_x_all",test_x_all
####test_y_all = test_data.drop('Image', axis=1).values
#####print "test_y_all", test_y_all
##print "feature columns:", feature_columns
##print "target columns:", target_columns


##x_all = face_data[feature_columns]
##t=x_all.shape[1]
##print "features:", t
##y_all = face_data[target_columns]

##print "features:", x_all.head()
##print "tagets:", y_all.head()


####test = 200
####x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = test, random_state = 5)
####print x_train.shape[0]
####print y_train.shape[0]

##x_train = x_train[:1800]

##y_train = y_train[:1800]
##print x_train.column[1]
##print y_train.column[1]
####y_train_300.astype(dtype=np.float64)
####y_train_300 = np.dtype('f8')

####reg = DecisionTreeRegressor(max_depth=5)
####reg.fit(x_train,y_train)
####predicts = reg.predict(x_test)
####print "total_error:", metrics.mean_squared_error(y_test, predicts)
####print "accuracy:", reg.score(x_train, y_train)
reg2 = KNR()
reg2.fit(x_all, y_all)
####predicts2 = reg2.predict(x_test)
####print "total_error:", metrics.mean_squared_error(y_test, predicts2)
print "accuracy:", reg2.score(x_all, y_all)

####reg3 = svm.SVR()
####reg3.fit(x_train, y_train) 
####SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
####   kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
####predicts3 = reg3.predict(x_test)
####print "total_error:", metrics.mean_squared_error(y_test, predicts3)
####print "accuracy:", reg3.score(x_train, y_train)
##predictions = reg.predict(test_x_all)
##print predicts

###print "total error:", total_error
for d in [test_data]:
    d['Image'] = test_data['Image'].apply(lambda im: np.fromstring(im, sep=' '))
# stack all test images into one numpy array
test_x_all = np.vstack(test_data.ix[:, 'Image']).astype(np.float)
####test_x_all = test_x_all.dropna()
# predict all keypoints for the images in Y
predictions = reg2.predict(test_x_all)

# now create the result data and write to csv
preddf = pd.DataFrame(predictions, columns=face_data.columns[:-1])
results = pd.DataFrame(columns=['RowId', 'Location'])

for i in range(lookup.shape[0]):
    d = lookup.ix[i, :]
    r = pd.Series([d['RowId'], preddf.ix[d['ImageId']-1, :][d['FeatureName']]],
                  index=results.columns)
    results = results.append(r, ignore_index=True)

results['RowId'] = results['RowId'].astype(int)
results.to_csv('predictions.csv', index=False)









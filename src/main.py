import os
import datetime
import pprint

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

from one_class_classification_via_neural_networks import OneClassClassificationViaNeuralNetworks

DIR_PATH = 'data'
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, '1070106.csv')
DATA_INDEX = [1, 2, 5, 6, 7, 13, 60, 62, 64, 66, 67, 69, 71, 73, 77, 87, 91, 95, 97, 99]
NOT_ENCODING_INDEX = [0, 1, 2, 3, 4, 11]

NOT_ENCODING_INDEX_LIST = [
    ' 發生日期',
    ' 發生時間',
    ' 發生星期',
    ' GPS經度',
    ' GPS緯度',
    ' 速限(第1當事者)'
]

# ' 發生地址_路街代碼',
# ' 發生市區鄉鎮代碼',
ENCODING_INDEX_LIST = [
    ' 地址類型代碼',
    ' 天候代碼',
    ' 光線代碼',
    ' 道路類別(第1當事者)代碼',
    ' 道路型態大類別代碼',
    ' 道路型態子類別代碼',
    ' 事故位置大類別代碼',
    ' 事故位置子類別代碼',
    ' 路面狀況-路面狀態代碼',
    ' 號誌-號誌種類代碼',
    ' 車道劃分設施-分向設施大類別代碼',
    ' 車道劃分設施-分道設施-快車道或一般車道間代碼',
    ' 車道劃分設施-分道設施-快慢車道間代碼',
    ' 車道劃分設施-分道設施-路面邊線代碼'
	]
# ENCODING_INDEX :13, 17, 22, 60, 62, 64, 67, 69, 71, 73, 77, 87, 91, 95, 97, 99]

# 1 = 發生日期
# 2 = 發生時間
# 5 = 發生星期
# 6 = GPS經度
# 7 = GPS緯度
# 13 = 地址類型代碼
# 17 = 發生市區鄉鎮代碼
# 22 = 發生地址_路街代碼
# 60 = 天候代碼
# 62 = 光線代碼
# 64 = 道路類別(第1當事者)代碼
# 66 = 速限(第1當事者)
# 67 = 道路型態大類別代碼
# 69 = 道路型態子類別代碼
# 71 = 事故位置大類別代碼
# 73 = 事故位置子類別代碼
# 77 = 路面狀況-路面狀態代碼
# 87 = 號誌-號誌種類代碼
# 91 = 車道劃分設施-分向設施大類別代碼
# 95 = 車道劃分設施-分道設施-快車道或一般車道間代碼
# 97 = 車道劃分設施-分道設施-快慢車道間代碼
# 99 = 車道劃分設施-分道設施-路面邊線代碼

def preprocess():
    # Read data
    
    data_csv = pd.read_csv(DATA_FILE)
    data_frame = data_csv.iloc[ : , DATA_INDEX]

    # Delete the missing data
    # data_frame = data_frame[data_frame[' 發生地址_路街代碼'] != ' ']
    data_frame = data_frame[(data_frame[' 地址類型代碼'] == 1) | (data_frame[' 地址類型代碼'] == 2)]

    # integer
    # data_frame[' 發生時間'] = data_frame[' 發生時間'].astype('int')

    # replace 發生星期
    replace_week_dict = {
        '一' : 0,
        '二' : 1,
        '三' : 2,
        '四' : 3,
        '五' : 4,
        '六' : 5,
        '日' : 6,
    }
    data_frame = data_frame.replace(replace_week_dict)
    
    # Encoding categorical data
    for encoding_index in ENCODING_INDEX_LIST:
        label_encoder_x = LabelEncoder()
        data_frame[encoding_index] = label_encoder_x.fit_transform(data_frame[encoding_index])

        dummy_df = pd.get_dummies(data_frame[encoding_index])
        index = data_frame.columns.get_loc(encoding_index)
        data_frame = data_frame.drop(encoding_index, axis=1)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(dummy_df)
        # print('='*20)
        # os.system('pause')
        for dummy_index in range(dummy_df.columns.size):
            # print(dummy_df.loc[: , dummy_index])
            column_name = '{0}_{1}'.format(encoding_index, dummy_index)
            data_frame.insert(index, column_name, dummy_df.loc[: , dummy_index])
            index += 1
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(data_frame)
        # print('='*20)
        # os.system('pause')
    
    
    # Standard Scaler
    scaler = MinMaxScaler()
    data_frame[NOT_ENCODING_INDEX_LIST] = scaler.fit_transform(data_frame[NOT_ENCODING_INDEX_LIST])

    # Show Feature
    column_list = data_frame.columns.tolist()
    pprint.pprint(column_list)
    print('Feature number', len(column_list))
    # print(data_frame)
    # os.system('pause')
    return data_frame




def train(data_frame):
    data_set = data_frame.values
    column_list = data_frame.columns.tolist()
    feature_number = len(column_list)
    x_train, x_test = train_test_split(data_set)
    print('Training data size', len(x_train))
    print('Testing data size', len(x_test))

    # Train

    # GaussianMixture
    # model = GaussianMixture()
    # model.fit(x_train)
    # y_pred = model.predict(x_test)

    # OneClassSVM
    # model = OneClassSVM(kernel='linear',nu=0.1)
    # model.fit(x_train)
    # y_pred = model.predict(x_test)
    # print(y_pred)



    # OneClassClassificationViaNeuralNetworks
    model = OneClassClassificationViaNeuralNetworks(feature_number)
    model.fit(x_train)
    predict_x_test = model.predict(x_test)
    y_pred = model.compute_score(x_test, predict_x_test)
    # print(y_pred)


    # Result
    y_test = np.ones(len(x_test))
    cm = confusion_matrix(y_test, y_pred)
    print('confusion matrix', cm)



    (true_negative, false_positive, false_negative, true_positive) = np.ravel(cm)
    total = true_negative + false_positive + false_negative + true_positive
    accuracy = (true_negative + true_positive)/total
    precision = true_positive / (false_positive + true_positive)
    recall = true_positive / (false_negative + true_positive)
    f_score = 2 * precision * recall / ( precision + recall )
    print('Accuracy =', accuracy)
    print('Precision =', precision)
    print('Recall =',recall)
    print('f_score =',f_score)


if __name__ == '__main__':
    time_a = datetime.datetime.now()
    print('='*20, 'Preprocessing Time', '='*20)
    data_frame = preprocess()
    print('='*20, 'Training Time', '='*20)
    time_b = datetime.datetime.now()
    train(data_frame)
    time_c = datetime.datetime.now()
    print('The preprocessing time is', time_b - time_a)
    print('The training time is', time_c - time_b)

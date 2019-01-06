import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics.pairwise import cosine_similarity

class OneClassClassificationViaNeuralNetworks():
    def __init__(self, feature_number):
        self.model = Sequential()
        self.model.add(Dense(input_dim=feature_number, units=50, activation='sigmoid'))
        self.model.add(Dense(units=25, activation='sigmoid'))
        self.model.add(Dense(units=50, activation='sigmoid'))
        self.model.add(Dense(units=feature_number, activation='sigmoid'))
        self.model.compile(loss='mean_squared_error',
            optimizer='adam',
        )

    def fit(self, train_x_data):
        self.model.fit(train_x_data, train_x_data, batch_size=100, epochs=20)

    def predict(self, test_x_data):
        return self.model.predict(test_x_data)

    def compute_score(self, original_data_x, predict_data_x):
        # print(original_data_x)
        # print(predict_data_x)
        # os.system('pause')
        min_similarity = 0.8
        y_pred_array = []
        for index, original_vector in enumerate(original_data_x):
            predict_vector = predict_data_x[index]

            #reshape
            predict_vector = predict_vector.reshape((1, len(predict_vector)))
            original_vector = original_vector.reshape((1, len(original_vector)))
            # print(predict_vector)
            # print(original_vector)
            # os.system('pause')
            similarity = cosine_similarity(original_vector, predict_vector)
            # print(similarity)
            result = -1
            if similarity > min_similarity:
                result = 1
            y_pred_array.append(result)
        return np.array(y_pred_array)


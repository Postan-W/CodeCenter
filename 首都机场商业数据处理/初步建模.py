"""
@Time : 2021/11/9 15:37
@Author : wmingzhu
@Annotation : 
"""
import keras
from keras import models,regularizers,layers
import pandas as pd
import numpy as np
from keras.optimizers import Adam

class ModelOne:
    def __init__(self,csvpath):
        self.csvpath = csvpath

    def data_process(self,validation_split=0.9):
        data = pd.read_csv(self.csvpath)
        features_pd = data.drop(columns=["totalsales"])
        targets = np.array(list(data["totalsales"]))
        indexes = len(list(data.index))
        features = []
        for i in range(indexes):
            features.append(list(features_pd.loc[i].values))

        features = np.array(features)
        # rand = np.random.randint(0,len(targets))
        # np.random.seed(rand)
        # np.random.shuffle(features)
        # np.random.seed(rand)
        # np.random.shuffle(targets)
        #注意把Python列表转为numpy。不然Keras会解析错误
        self.train_data = features[:int(indexes*validation_split)]
        self.train_label = targets[:int(indexes*validation_split)]
        self.validation_data = features[int(indexes*validation_split):]
        self.validation_label = targets[int(indexes*validation_split):]

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=(np.array(self.train_data).shape[1],)))
        model.add(layers.Dense(128, activation='relu'))#, kernel_regularizer=regularizers.l2(0.001)
        # model.add(layers.Dropout(0.2))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer= Adam(lr=0.001,beta_1=0.9,beta_2=0.999),loss='mse',metrics=['mae'])
        self.model = model

    def fit(self,epochs=100,batch_size=10,verbose=True):#一个epoch验证一次
        tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard',histogram_freq=0, write_grads=False,write_images=False,write_graph=False,update_freq='epoch')
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.01,patience=3)
        checkpoint = keras.callbacks.ModelCheckpoint(filepath='./modelfile/model1.h5',
                                                     monitor='mean_absolute_error',save_best_only=True)
        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=30)
        history = self.model.fit(self.train_data, self.train_label,epochs=epochs,
                                 validation_data=(self.validation_data, self.validation_label), batch_size=batch_size, verbose=True,callbacks=[tensorboard])
        # ['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error']
        # print(history.history["mean_absolute_error"])
        # print(history.history["val_mean_absolute_error"])

    def predict(self,predict_data:[[]]):
        return self.model.predict(predict_data)

def use_modelone():
    modelone = ModelOne("version5.csv")
    modelone.data_process()
    modelone.build_model()
    modelone.fit(batch_size=10, epochs=20)
    print(modelone.predict(modelone.validation_data[20:24]))
    print(modelone.validation_label[20:24])
    print(modelone.predict(modelone.train_data[20:24]))
    print(modelone.train_label[20:24])
use_modelone()

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
class ModelTwo:
    def __init__(self, csvpath):
        self.csvpath = csvpath

    def data_process(self, test_size=0.1):
        data = pd.read_csv(self.csvpath)
        features_pd = data.drop(columns=["totalsales"])
        targets = np.array(list(data["totalsales"]))
        indexes = len(list(data.index))
        features = []
        for i in range(indexes):
            features.append(list(features_pd.loc[i].values))
        features = np.array(features)

        self.train_data,self.test_data,self.train_label,self.test_label = train_test_split(features,targets,test_size=test_size)

    def build_svm(self):
        self.model = SVR(kernel="poly",degree=2, C=5, epsilon=1.5)

    def build_decision_tree(self):
        self.model = DecisionTreeRegressor()

    def fit(self):
        self.model.fit(self.train_data,self.train_label)


def use_modeltwo():
    modeltwo = ModelTwo("version4.csv")
    modeltwo.data_process()
    modeltwo.build_svm()
    modeltwo.fit()
    print(modeltwo.test_label[:3])
    print(modeltwo.model.predict(modeltwo.test_data[:3]))


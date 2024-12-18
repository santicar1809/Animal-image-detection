import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from tensorflow.keras.metrics import Precision,Recall,BinaryAccuracy
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

def label(pred):
    if pred > 0.5:
        return 'Dog'
    else:
        return 'Cat'
        
def neural_network():
    ## Deep learning

    model=Sequential()

    # Neural Network
    # Input layer
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
    
    return model

def eval(test,model):
    ## Evaluation

    pre=Precision()
    re=Recall()
    acc=BinaryAccuracy()

    for batch in test.as_numpy_iterator():
        features,target= batch
        pred =model.predict(features)
        pre.update_state(target,pred)
        re.update_state(target,pred)
        acc.update_state(target,pred)
        
    return pre,re,acc

def built_model(data):
    ## Split data
    train_size=int(len(data)*.7)
    val_size=int(len(data)*.2)+1


    train=data.take(train_size)
    val=data.skip(train_size).take(val_size)
    test=data.skip(train_size+val_size)

    model=neural_network()

    model.summary()

    logdir='logs'
    tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist=model.fit(train,epochs=20,validation_data=val,callbacks=[tensorboard_callback])
    ## Plot performance
    if not os.path.exists('./files/figs/'):
        os.makedirs('./files/figs/')

    outpt_path='./files/figs/'

    # Loss
    fig=plt.figure()
    plt.plot(hist.history['loss'],color='teal',label='loss')
    plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
    fig.suptitle('Loss',fontsize=20)
    plt.legend(loc='upper left')
    fig.savefig(outpt_path+'loss.png')

    # Accuracy
    fig1=plt.figure()
    plt.plot(hist.history['accuracy'],color='teal',label='accuracy')
    plt.plot(hist.history['val_accuracy'],color='orange',label='val_accuracy')
    fig1.suptitle('Accuracy',fontsize=20)
    plt.legend(loc='upper left')
    fig1.savefig(outpt_path+'accuracy.png')

    pre,re,acc=eval(test,model)
    result=pd.DataFrame({'Precision':[pre.result().numpy()], 'Recall':[re.result().numpy()], 'Accuracy':[acc.result().numpy()]})    
    
    result.to_csv(outpt_path+'result_report.csv',index=False)
    
    # Guardamos el modelo
    model.save('./models/catdogmodel.h5')

    return model
    
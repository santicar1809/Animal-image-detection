import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from src.models.model import label
from tensorflow.keras.models import load_model

def test():
    
    if not os.path.exists('./files/figs/'):
        os.makedirs('./files/figs/')

    outpt_path='./files/figs/'
    model=load_model(os.path.join('models','catdogmodel.h5'))
    # Test
    img= cv2.imread('./cat_test.jpg')

    resize=tf.image.resize(img,(256,256))
    plt.imshow(resize.numpy().astype(int))
    pred=model.predict(np.expand_dims(resize/255,0))

    result_test_list=[]

    result_test_list.append(['cat',pred])

    img2= cv2.imread('./dog_test.jpg')
    resize2=tf.image.resize(img2,(256,256))
    plt.imshow(resize2.numpy().astype(int))
    pred2=model.predict(np.expand_dims(resize2/255,0))

    result_test_list.append(['dog',pred2])
    
    result_test=pd.DataFrame(result_test_list,columns=['image','target'])
    result_test['animal']=result_test['target'].apply(label)
    result_test.to_csv(outpt_path+'result_test.csv',index=False)
    
    return result_test

# coding: utf-8

# In[1]:

import cv2
import numpy as np
import os
from keras.utils import np_utils


# In[14]:

from keras.models import load_model

y_c = [1,2,3,4,5,6,7,8]
classes=np.array(y_c)
images=[]

model = load_model("2015A7PS0058G.h5")


# In[29]:

def preprocess(image):
    im=cv2.imread(image,cv2.IMREAD_COLOR)
    #print(im.shape)
    im=np.asarray(cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_CUBIC))
    images.append(im)
    

output_all=[]

def predict(image):
    preds = model.predict(image)

    preds=np.array(preds)
    maxval=np.argmax(preds)
    preds*=0
    preds[argmax]=1

    output=preds*classes

    output_all.append(output)

def predict_all(x_image,y_output):

    y_output = np_utils.to_categorical(y_output, num_classes=8)
    y_output = np.array(y_output)

    for image in x_image:
        if(image[:2]=='._' or image is None):
            continue    
        preprocess(image)

    images=np.asarray(images)

    for image in images:
        predict(image)

    tp=0
    tn=0
    fp=0
    fn=0

    for i in range(len(y_output)):
        for j in range(8):         
            if(output_all[i][j]==1 and y_output[i][j]==1):
                tp=tp+1;
            elif(output_all[i][j]==0 and y_output[i][j]==0):  
                tn=tn+1
            elif(output_all[i][j]==1 and y_output[i][j]==0):  
                fp=fp+1
            elif(output_all[i][j]==0 and y_output[i][j]==1):  
                fn=fn+1    

    print('True positives :'+str(tp))             
    print('True negatives :'+str(tn))     
    print('False positives :'+str(fp))             
    print('False negatives :'+str(fn)) 

    denom=fp+fn+tp
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    print('Recall : '+str(recall))
    print('Precision: '+str(precision))
    print('F-measure: '+str((2*precision*recall)/(precision+recall)))
    print('Accuracy :'+str((tp+tn)/denom)) 


# In[ ]:




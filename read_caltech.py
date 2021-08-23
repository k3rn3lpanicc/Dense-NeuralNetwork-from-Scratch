import cv2
import cupy as np
import os
import random



def get_data_mat(dir):
    X = []
    y = []
    filenames=[]
    k=0
    files = os.listdir(dir)
    random.shuffle(files)
    for file in files:
        if file[-3:] == 'jpg':
            img = cv2.imread(os.path.join(dir, file))
            #print(img.size)
            #img = cv2.resize(img, (44, 36))
            #th, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
            img=img[:,:,0]
            #img = img.reshape((50, 28, 1))
            img = img.reshape((img.shape[0]*img.shape[1],))
            img = img / 255.
            X.append(img)
            if(file[0]=='f'):
                y.append([0,1])

            else :
                y.append([1,0])
            
            #y.append(0 if file[0] == 'f' else 1)

            filenames.append(file)
    print(len(y))
    return filenames,np.array(X,dtype=np.float32), np.array(y)#,dtype=object)

#files,X,y=get_data_mat("test/")

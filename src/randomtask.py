import json
import cv2
import os
import  numpy as np
import string
import shutil
import os

# duplicate = ['ck9hjr7a4nqze0758pwm1h69k','ck9hkx25djwrq08894nsn9wj5','ck9hkybb2nwqw0758yq84xx1f','ck9hlpdxgo4br07589pxwxno1']
# classdict = {'concrete bucket':0,'cement mixer':1,'mixer truck':2, 'worker':3, 'Crane':4}
basedir = '/data/Guha/construction/Data/'
srcdir = '/data/Guha/construction/data/'
destdir = '/data/Guha/construction/mydata/'
#testdir = basedir+'mydata/testimages/'
# with open('/data/Guha/construction/Data/annotation.json') as f:
#   data = json.load(f)
#
# with open(basedir+'annotations.txt','w') as annotfile:
#     for itemdict in data:
#         name = itemdict['External ID']
#         if name == 'zon-9143283.jpg': continue
#         # if name in duplicate:
#         #     continue
#         print(name)
#         img = cv2.imread(os.path.join(srcdir,name))
#         id = itemdict['ID']
#         annot = [id]
#         for object in itemdict['Label']['objects']:
#             title = object['title']
#             bbox = object['bbox']
#             x1 = bbox['left']
#             y1 = bbox['top']
#             x2 = x1+bbox['width']
#             y2 = y1+bbox['height']
#             annot.extend([classdict[title]])
#             #annot.extend([bbox['top'],bbox['left'],bbox['height'],bbox['width']])
#             annot.extend([x1,y1,x2,y2])
#
#             ############# visualization ############
#             #cv2.rectangle(img, (bbox['left'], bbox['top']), (bbox['left']+bbox['width'], bbox['top']+bbox['height']), (255, 0, 0), 1)
#             #cv2.rectangle(img, (x1,y1),(x2,y2), (255, 0, 0), 1)
#
#         # cv2.imshow('frame', img)
#         # key = cv2.waitKey(0)
#         # if key & 0xFF == ord('q'):
#         #     continue
#         annot = [str(i) for i in annot]
#         annotfile.write(' '.join(annot).lstrip())
#         annotfile.write('\n')
#
# print("Done")
#print(data)

# for itemdict in data:
#     name = itemdict['External ID']
#     if name == 'zon-9143283.jpg': continue
#     id = itemdict['ID']
#     destimg = id+'.jpg'
#     if id in duplicate:
#         name = id+'.jpg'
#     #print(name)
#     shutil.copyfile(os.path.join(srcdir,name),os.path.join(destdir,destimg))



# import random
# import sklearn
# from sklearn.model_selection import train_test_split
# filesize = len(os.listdir(srcdir))
# files = os.listdir(srcdir)
# random.shuffle(files)
# training_dataset,test_dataset = train_test_split(files, train_size=0.9)
#
# with open(basedir+'train.txt','w') as trainfile:
#     for f in training_dataset:
#         trainfile.write(os.path.join(destdir,f.replace(".txt",".jpg"))+'\n')
#         shutil.move(os.path.join(destdir,f))
# with open(basedir+'test.txt','w') as testfile:
#     for f in test_dataset:
#         shutil.copyfile(os.path.join(destdir,f), os.path.join(testdir,f.replace(".txt",".jpg")))
#         testfile.write(os.path.join(destdir,f.replace(".txt",".jpg"))+'\n')

import cv2
cv2.imread('/data/Guha/construction/mydata/train')
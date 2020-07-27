import os
import numpy as np
import cv2

# path = '/home/guha/Downloads/labels/yieldsign/'
# for f in os.listdir(path):
#     labels = np.loadtxt(os.path.join(path,f))
#     if(len(labels)==0):
#         print(f)
#         continue
#     labels[0] = 6
#     np.savetxt(os.path.join(path,f),labels.reshape(1,-1),fmt='%f')

# path = '/data/Guha/construction/Data/mydata/newimages/'
# for f in os.listdir(path):
#     src = os.path.join(path,f)
#     f = f.replace('JPEG','jpg')
#     dst = os.path.join(path,f)
#     os.rename(src,dst)

path = '/data/Guha/construction/data/'
destpath = '/data/Guha/construction/mydata/'
# with open('/data/Guha/construction/Data/mydata/CBtrain.txt','w') as trainfile:
#     for f in os.listdir(path):
#         if(f.split('.')[1]=='jpg'):
#             trainfile.write(path+f+"\n")


import shutil,random
# if __name__ == "__main__":
#     print("helli")
#     classdict = {'concrete bucket': 0, 'cement mixer': 1, 'mixer truck': 2, 'worker': 3, 'Crane': 4}
#     with open('/data/Guha/construction/Data/annotation2.json') as f:
#         data = json.load(f)
#
#     dataPath = '/data/Guha/construction/Data/mydata/'
#     src = dataPath + 'images/'
#     dest = dataPath + 'CB/'
#     for itemdict in data:
#         id = itemdict['ID']
#         annot = [id]
#         for object in itemdict['Label']['objects']:
#             title = object['title']
#             bbox = object['bbox']
#             x1 = bbox['left']
#             y1 = bbox['top']
#             x2 = x1 + bbox['width']
#             y2 = y1 + bbox['height']
#             annot.extend([classdict[title]])
#             annot.extend([x1, y1, x2, y2])

# size = int(len(os.listdir(path))/2)
# count = 0
# files = os.listdir(path)
# random.shuffle(files)
#
# id = set()
# for f in files:
#     id.add(f.split('.')[0])
#
# for len,file in enumerate(id):
#
#     if(len<size*0.95):
#         if(os.path.exists(os.path.join(path,file+'.jpg'))):
#             imgpath = os.path.join(path,file+'.jpg')
#         else:
#             imgpath = os.path.join(path,file+'.png')
#         shutil.copyfile(imgpath,imgpath.replace(path,destpath+'/train/'))
#         shutil.copyfile(os.path.join(path,file+'.txt'), os.path.join(destpath, 'train', file + '.txt'))
#     else:
#         if (os.path.exists(os.path.join(path, file + '.jpg'))):
#             imgpath = os.path.join(path, file + '.jpg')
#         else:
#             imgpath = os.path.join(path, file + '.png')
#         shutil.copyfile(imgpath, imgpath.replace(path,destpath+'/test/'))
#         shutil.copyfile(os.path.join(path, file + '.txt'), os.path.join(destpath, 'test', file + '.txt'))

# for f in os.listdir(path):
#     if(f.split('.')[1]=='png'):
#         os.rename(os.path.join(path,f),os.path.join(path,f.replace('png','jpg')))

# with open(destpath+'test.txt', 'w') as fhandler:
#     for f in os.listdir(destpath+'test/'):
#         if(f.split('.')[1]=='txt'): continue
#         fhandler.write(destpath+'test/'+f+'\n')



for f in os.listdir(path):
    if(f.split('.')[1]=='txt'): continue
    #img = cv2.imread(os.path.join(path,f))
    img = cv2.imread('/data/Guha/construction/6_frames_labeled/train/cka84ymadeqar0854qi6le1cd.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (np.any(img) == None):
        print(f)
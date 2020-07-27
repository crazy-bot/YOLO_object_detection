import cv2
import os
import numpy as np

def convert_labels(path, x1, y1, x2, y2):
    """
    Definition: Parses label files to extract label and bounding box
        coordinates.  Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
    """
    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin

    img = cv2.imread(path)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    size = img.shape
    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    dw = 1./size[1]
    dh = 1./size[0]
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

# if __name__ == "__main__":
#     print("helli")
#     dataPath = '/data/Guha/construction/Data/mydata/'
#     with open(os.path.join(dataPath,'annotations.txt'),'r') as annotfile:
#         for line in annotfile:
#             print(line)
#             imgName = line.split()[0]
#             values = line.split()[1:]
#             values = [float(e) for e in values]
#             noOfobjects = int(len(values)/5)
#             print(noOfobjects)
#             # with open(os.path.join(dataPath,'labels',imgName+'.txt'),'w') as labelfile:
#             imgPath = os.path.join(dataPath, 'images', imgName + '.jpg')
#             labels = []
#             for i in range(noOfobjects):
#                 i = i * 5
#                 (x, y, w, h) = convert_labels(imgPath, values[i + 1], values[i + 2], values[i + 3], values[i + 4])
#                 labels.append([values[i], x, y, w, h])
#                 # labelfile.write(','.join([values[i], x, y, w, h]))
#                 # labelfile.write('\n')
#             labels = np.asarray(labels)
#             np.savetxt(os.path.join(dataPath,'labels',imgName+'.txt'),labels,fmt='%f')
#             #break

import shutil,json
if __name__ == "__main__":
    print("helli")
    classdict = {'concrete bucket': 0, 'cement mixer': 1, 'mixer truck': 2, 'worker': 3, 'Crane': 4}
    with open('/data/Guha/construction/Data/annotation2.json') as f:
        data = json.load(f)

    dataPath = '/data/Guha/construction/Data/mydata/'
    src = dataPath + 'Concrete bucket/'
    dest = dataPath + 'CB/'
    for itemdict in data:
        id = itemdict['ID']

        name = itemdict['External ID']
        if (name=='1'):continue
        imgPath = os.path.join('/data/Guha/construction/Data/Concrete bucket',name)
        #img = cv2.imread(imgPath)
        labels = []
        for object in itemdict['Label']:
            bbox = itemdict['Label'][object][0]['geometry']
            x1 = bbox[0]['x']
            y1 = bbox[0]['y']
            x2 = bbox[2]['x']
            y2 = bbox[2]['y']

            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # cv2.imshow('frame', img)
            # key = cv2.waitKey(0)

            (x, y, w, h) = convert_labels(imgPath, x1,y1,x2,y2)
            labels.append([classdict[object], x, y, w, h])

        labels = np.asarray(labels)
        np.savetxt(os.path.join(dest, id + '.txt'), labels, fmt='%f')
        shutil.copyfile(imgPath, os.path.join(dest, id+'.jpg'))
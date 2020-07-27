import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

classdict = {0: 'concrete bucket', 1:'cement mixer',2:'mixer truck',3:'worker',4: 'Crane'}

def from_yolo_to_cor(box, shape):
    img_h, img_w, _ = shape
    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    x1, y1 = int((box[0] + box[2] / 2) * img_w), int((box[1] + box[3] / 2) * img_h)
    x2, y2 = int((box[0] - box[2] / 2) * img_w), int((box[1] - box[3] / 2) * img_h)
    return x1, y1, x2, y2


def draw_boxes(path, labels):
    img = cv2.imread(path)

    for label in labels:
        x1, y1, x2, y2 = from_yolo_to_cor(label[1:], img.shape)
        cv2.rectangle(img, (x2, y2), (x1, y1), (0, 255, 0), 3)
        plt.scatter(x1,y1,c='m',s=20)
        plt.text(x1, y1 - 30, s=classdict[label[0]], color='white',horizontalalignment ='center',
                 verticalalignment='top',
                 bbox={'color':'r', 'pad': 0})
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    print("helli")
    dataPath = '/data/Guha/construction/Data/mydata/'
    for file in os.listdir(os.path.join(dataPath,'images')):
        imgfile = os.path.join(dataPath,'images',file)
        labelfile = os.path.join(dataPath,'labels',file.split('.jpg')[0]+'.txt')
        labels = np.loadtxt(labelfile)
        if(len(labels.shape)==1):
            labels = labels.reshape(1,-1)
        draw_boxes(imgfile,labels)


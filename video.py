from __future__ import division
import cv2
from utils.utils import *
from models import *
import random
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from skimage.transform import resize

import PIL.Image


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def resize_img(img, img_size=416):
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
    padded_h, padded_w, _ = input_img.shape
    # Resize and normalize
    input_img = resize(input_img, (img_size, img_size, 3), mode='reflect')
    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float().unsqueeze(0)
    return input_img, img


def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/mydata_yolov4.cfg', help='path to model config file')
    parser.add_argument('--weights_path', type=str,
                        default="/data/Guha/construction/checkpoints/pytorch/mydata_best.weights",
                        help='path to weights file')
    parser.add_argument('--class_path', type=str, default='config/mydata.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.9, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.5, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--use_cuda', type=bool, default=False, help='whether to use cuda if available')
    parser.add_argument("--output", default='/data/Guha/construction/output/6.check.mp4')
    parser.add_argument("--input", default='/data/Guha/construction/Videos/6.mp4')

    return parser.parse_args()


if __name__ == '__main__':

    cvfont = cv2.FONT_HERSHEY_PLAIN
    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    # cmap = plt.get_cmap('Vega20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    opt = arg_parse()
    cuda = torch.cuda.is_available() and opt.use_cuda
    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)
    model.load_weights(opt.weights_path)
    print('model path: ' + opt.weights_path)
    if cuda:
        model.cuda()
        print("using cuda model")
    model.eval()
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    ############# video file extraction ###############
    # videofile = '/data/Guha/construction/Data/mydata/Pouring Concrete.mp4'

    cap = cv2.VideoCapture(opt.input)
    # cap = cv2.VideoCapture(0)

    assert cap.isOpened(), 'Cannot capture source'
    writer = None
    frames = 0
    rendering = 1
    ################## save all detections ##############
    framecounts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_objects = 10
    out_detections = np.zeros((max_objects, framecounts, 5))

    for i in range(framecounts):

        ret, frame = cap.read()
        prev_time = time.time()
        if ret:
            # resized_img, img, dim = prep_image(frame, inp_dim)
            resized_img, img = resize_img(frame, opt.img_size)
            # print((resized_img.size(),type(resized_img)))
            # cv2.imshow('resized_img',resized_img)
            input_imgs = Variable(resized_img.type(Tensor))
            # print((resized_img.size(),type(resized_img)))
            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                # print(detections)
                alldetections = non_max_suppression(detections, 3, opt.conf_thres, opt.nms_thres)
                # print('detections get')
                # print((alldetections[0]))
                # print(type(alldetections[0]))
            detection = alldetections[0]
            current_time = time.time()
            inference_time = current_time - prev_time
            fps = int(1 / (inference_time))
            # print('current fps is : %d'%fps)

            if (rendering):
                # The amount of padding that was added
                pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
                pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
                # Image height and width after padding is removed
                unpad_h = opt.img_size - pad_y
                unpad_w = opt.img_size - pad_x

                # Draw bounding boxes and labels of detections
                if detection is not None:
                    # print(img.shape)
                    unique_labels = detection[:, -1].cpu().unique()

                    n_cls_preds = min(len(unique_labels), 20)
                    bbox_colors = random.sample(colors, n_cls_preds)
                    for j, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detection):
                        cls_pred = min(cls_pred, max(unique_labels))
                        # print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                        # Rescale coordinates to original dimensions
                        box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
                        box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]))
                        y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
                        x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))
                        x2 = int(x1 + box_w)
                        y2 = int(y1 + box_h)
                        # print(x1,y1,x2,y2)
                        ################ save detections #######
                        # out_detections[j,i,:]=np.array([x1,y1,x2,y2,int(cls_pred)])

                        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        # print(color)
                        # cv2.line(img,(int(x1), int(y1-5)),(int(x2), int(y1-5)),(255,255,255),14)
                        cv2.putText(img, classes[int(cls_pred)], (int(x1), int(y1)), cvfont, 1.5,
                                    (color[0] * 255, color[1] * 255, color[2] * 255), 2)
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                                      (color[0] * 255, color[1] * 255, color[2] * 255), 1)

                if writer is None:
                    # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(opt.output, fourcc, 30,
                                             (frame.shape[1], frame.shape[0]), True)
                ############# write the output frame to disk
                writer.write(frame)
                # cv2.imshow('frame', img)
                # free buffer
                # cv2.imshow('kitti detecting window',plt)
                # plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

        # no ret berak
        else:
            break
    writer.release()
    cap.release()
    # np.save('/data/Guha/construction/output/6.6.npy',out_detections)





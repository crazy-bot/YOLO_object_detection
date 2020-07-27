#Inference on Google cloud VM
Run the following script as below format:

Detection on images -->
python3 detect.py --weights_path <checkpoint-path>  --image_folder <folder-conraining-list-of-images>  --output <output-folder-to-save-the-result>
python3 detect.py --weights_path '~/checkpoints/best_ckpt/mydata_yolo4.weights'  --image_folder '~/data/test/'  --output '~/output/'

Detection video -->
python3 detect.py --weights_path <checkpoint-path>  --input <input-video-file-path>  --output <output-video-file-path>
python3 video.py --weights_path '/home/suparna_guha/checkpoints/best_ckpt/mydata_yolo4.weights' --input '/home/suparna_guha/6.mp4' --output '/home/suparna_guha/output/6_detect.mp4'

In the current VM paths are set as follows:
<checkpoint-path> - '~/checkpoints/best_ckpt/mydata_yolo4.weights'
<folder conraining list of images> - '~/data/test/'
<output folder to save the result> - '~/output/'
<input-video-file-path> - '/home/suparna_guha/6.mp4'
<output-video-file-path> - '/home/suparna_guha/output/6_detect.mp4'

#training on custom dataset -->
##configuration-- config/mydata.data
classes= no of classes on your dataset
train=/data/Guha/construction/6_frames_labeled/train.txt (contains all the training images path)
valid=/data/Guha/construction/6_frames_labeled/test.txt (contains all the validation images path)

##configuration-- config/mydata.names - list all the labels for each class index
for e.g if classes=3 we have following 3 class labels
concrete bucket
mixer truck
person

## run training script
YOLOV3 - python3 trainV3.py 
YOLOV4 - python3 trainV4.py
for detailed command line arguments please check the relevant file


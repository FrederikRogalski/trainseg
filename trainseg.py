import numpy as np
import os
import cv2
from threading import Thread
import time
from matplotlib import pyplot as plt
from pycoral.utils import edgetpu
from pycoral.adapters import common
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Inference Script for detection train rails.')
parser.add_argument("--threshold", type=int, default=20)
parser.add_argument("--stream", default="http://192.168.2.166/mjpeg/1")
parser.add_argument("--model", default="./trainseg_quantv1.0_edgetpu.tflite")
parser.add_argument("--view", default="segmentation")
parser.add_argument("--single", action='store_true')
parser.add_argument("--stopAt", type=int, default=140)
args = parser.parse_args()
threshold = args.threshold
stream_ip = args.stream
model_file = args.model
view = args.view
single = args.single
stopAt = ergs.stopAt
class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [x for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 1:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years
    
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def boundary_fill(img, start, boundary=0, fill=1):
  image = np.array(img)
  new = np.zeros_like(image)
  stack = []
  stack.append(start)
  while len(stack)>0:
    current = stack.pop()
    if True in (current >= image.shape):
      continue
    if image[current[0], current[1]] != boundary:
      image[current[0], current[1]] = fill
      new[current[0], current[1]] = 1
      stack.append(np.array([current[0]-1,current[1]]))
      stack.append(np.array([current[0]+1,current[1]]))
      stack.append(np.array([current[0],current[1]-1]))
      stack.append(np.array([current[0],current[1]+1]))
  return new


stream = LoadStreams(stream_ip)
dl = iter(stream)
print("Stream intitialized")
cv2.namedWindow("camtest")
cv2.startWindowThread()


interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()
size = common.input_size(interpreter)
print(f"Model input size: {size}")
output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
zeros = np.zeros((224,224,1))

for path, in0, img, vid_cap in tqdm(dl):
    in0 = np.moveaxis(in0[0], 0, -1)
    common.set_input(interpreter, cv2.resize(in0, dsize=size))
    interpreter.invoke()
    zeros[:,:] = 0
    zeros[output()[0]>threshold]=1
    if single:
      zeros = np.expand_dims(boundary_fill(zeros[:,:,0], np.array([200, 110]), boundary=0, fill=0),axis=-1)
    if zeros[:stopAt].max()>0:
      text="Freie Fahrt"
      color=(0,255,0)
    else:
      text="STOPP!"
      color=(0,0,255)
    if view=="overlay":
      img = np.array(img[0])
      out = cv2.resize(zeros, dsize=(img.shape[1],img.shape[0]))
      red = img[:,:,2]
      red[out>0]=255
      img[:,:,2]=red
      out = img/255.0
    else:
      out = zeros
    #mask = np.array(output()[0],dtype=np.float32) # create Mask with threshold
    #mask = cv2.resize(mask, dsize=masked_img.shape[:2])
    #mask= np.moveaxis(mask > 1,0,1)
    #red = masked_img[:,:,2] # get red component
    #red[mask[:,:]] = 255.0 # push red to 255 where mask is True
    #masked_img[:,:,2] = red # add red component
    cv2.putText(out,text, (100,100), 1, fontScale=2, color=color)
    cv2.imshow('camtest', out)
    key = cv2.waitKey(1) #pauses for 3 seconds before fetching next image
    if key == 27:#if ESC is pressed, exit loop
        break

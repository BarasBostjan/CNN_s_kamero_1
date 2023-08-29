import torch
from torch import nn
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm
import numpy as np
import cv2
import math
import sys
from pathlib import Path
import pyrealsense2 as rs
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def imageToTensor(image, device):
    return torch.from_numpy(image).permute(2 ,0, 1).unsqueeze(dim=0).to(device)

def tensorToImage(tensor):
    return tensor.squeeze(dim=0).permute(1, 2, 0).cpu().detach().numpy()

class needleModelV1(nn.Module):         #AlexNet (https://towardsdatascience.com/the-w3h-of-alexnet-vggnet-resnet-and-inception-7baaaecccc96)
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, stride=4, kernel_size=11),                #1
            nn.ReLU(),                                                                          #1
            nn.MaxPool2d(stride=2, kernel_size=3),                                              #2
            nn.Conv2d(in_channels=96, out_channels=256, stride=1, kernel_size=5, padding=2),    #3
            nn.ReLU(),                                                                          #3
            nn.MaxPool2d(stride=2, kernel_size=3),                                              #4
            nn.Conv2d(in_channels=256, out_channels=384, stride=1, padding=1, kernel_size=3),   #5
            nn.ReLU(),                                                                          #5
            nn.Conv2d(in_channels=384, out_channels=384, stride=1, padding=1, kernel_size=3),   #6
            nn.ReLU(),                                                                          #6
            nn.Conv2d(in_channels=384, out_channels=256, stride=1, padding=1, kernel_size=3),   #7
            nn.ReLU(),                                                                          #7
            nn.MaxPool2d(stride=2, kernel_size=3),                                              #8
            nn.Flatten(),                                                                       #9
            nn.Linear(in_features=9216, out_features=4096),                                     #9
            nn.ReLU(),                                                                          #9
            nn.Linear(in_features=4096, out_features=4096),                                     #10
            nn.ReLU(),                                                                          #10
            nn.Linear(in_features=4096, out_features=1000),                                     #11
            nn.ReLU(),                                                                          #12
            nn.Linear(in_features=1000, out_features=1),                                        #13
        )

    def forward(self, x):
        return self.block(x)
    
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def cutObject(mask, n, image):
    anns = mask[n:n+1]
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        img[m] = 1

    return img * image / 255

def frameObjects(image, outputCoordinates = False):
    gray = image[:,:,0] + image[:,:,1] + image[:,:,2]
    a = np.sum(gray, axis=0)
    b = np.sum(gray, axis=1)
    i = 0
    while b[i] == 0 and i != b.size - 1:
        i += 1
    h0 = i
    i = b.size - 1
    while b[i] == 0 and i != 0:
        i -= 1
    h1 = i
    i = 0
    while a[i] == 0 and i != a.size - 1:
        i += 1
    w0 = i
    i = a.size - 1
    while a[i] == 0 and i != 0:
        i -= 1
    w1 = i
    if h1 - h0 >= w1 - w0:
        oImage = np.zeros((h1 - h0, h1 - h0, 3))
        oImage[0 : h1 - h0, int((h1 - h0 - w1 + w0) / 2) : w1 - w0 + int((h1 - h0 - w1 + w0) / 2), :] = image[h0 : h1, w0 : w1, :]
    else:
        oImage = np.zeros((w1 - w0, w1 - w0, 3))
        oImage[int((w1 - w0 - h1 + h0) / 2) : h1 - h0 + int((w1 - w0 - h1 + h0) / 2), 0 : w1 - w0, :] = image[h0 : h1, w0 : w1, :]
    
    if outputCoordinates:
        return oImage, h0, h1, w0, w1
    else:
        return oImage
    
def showImageandLabel(X, y):
    b, c, h, w = X.shape
    for i in range(b):
        image = tensorToImage(X[i])
        print("Label:", y[i])
        while True:
            cv2.imshow('Slika', image / 255)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('o'):
                break
    
if __name__ =='__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    st_igel = 7474
    st_ostalo = 5423

    MODEL_PATH = Path("models")
    MODEL_NAME = "model_0.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    model_0 = needleModelV1().to(device)
    model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))

    #Priprava kamere

    pipeline = rs.pipeline()

    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device_camera = pipeline_profile.get_device()
    device_product_line = str(device_camera.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device_camera.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)

    #Inicializacija segmentacije

    print("inicializacija")

    sys.path.append("..")

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    while True:

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        masks = mask_generator.generate(color_image)

        print("Št mask:", len(masks))
        for n in range(len(masks)):
            cut_object = cutObject(masks, n, color_image)
            framed_object, h0, h1, w0, w1 = frameObjects(cut_object, outputCoordinates=True)
            #print(framed_object)
            #print(framed_object.shape)
            tensor = F.resize(imageToTensor(framed_object * 255, device), (227, 227)).float()
            model_0.eval()
            with torch.inference_mode():
                y_logits = model_0(tensor).squeeze()
                y_pred = torch.round(torch.sigmoid(y_logits))
                #print(torch.sigmoid(y_logits))
                #print(y_pred)
            if y_pred:
                print(h0, h1, w0, w1)
                color_image = cv2.rectangle(color_image, (w0, h0), (w1, h1), (255, 0, 0), 2)        

        cv2.namedWindow('Slika', cv2.WINDOW_NORMAL)
        cv2.imshow('Slika', color_image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break




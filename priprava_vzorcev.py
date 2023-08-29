import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from timeit import default_timer as timer
import math
import kornia

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    total_time = end - start
    print("Train time on", device, total_time, "seconds")
    return total_time

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

def imageToTensor(image, device):
    return torch.from_numpy(image).permute(2 ,0, 1).unsqueeze(dim=0).to(device)

def tensorToImage(tensor):
    return tensor.squeeze(dim=0).permute(1, 2, 0).cpu().detach().numpy()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)   

    steviloTreningSlik = 69

    hflipper = T.RandomHorizontalFlip(p=0.5)

    rotater = T.RandomRotation(degrees=180, expand=True)
    rotater_times_ostalo = 10
    rotater_times_igle = 100

    print("inicializacija")

    sys.path.append("..")

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    st_igle = 0
    st_ostalo = 0
    for i in range(steviloTreningSlik):
        image = cv2.imread('Trening slike\slika_' + str(i) + '.jpg')
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print("Začetek segmentacije")
        time_start = timer()
        masks = mask_generator.generate(image)
        time_end= timer()
        total_time = print_train_time(start=time_start, end=time_end, device=device)

        print("slika", i)

        for n in range(len(masks)):
            cut_object = cutObject(masks, n, image)
            framed_object = frameObjects(cut_object)
            combined_image = np.hstack((image / 255, cut_object))
            while True:
                cv2.imshow('Slika', combined_image)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('i'):
                    cv2.imwrite('Igle\igla_' + str(st_igle) + '.jpg', framed_object * 255)
                    st_igle += 1
                    for j in range(rotater_times_igle):
                        rotated_object = frameObjects(tensorToImage(rotater(hflipper(imageToTensor(framed_object, device)))))
                        try:
                            cv2.imwrite('Igle\igla_' + str(st_igle) + '.jpg', rotated_object * 255)
                            st_igle += 1
                        except:
                            print("Slike ni bilo mogoče shraniti")
                    break

                if key & 0xFF == ord('o'):
                    cv2.imwrite('Ostalo\ostalo_' + str(st_ostalo) + '.jpg', framed_object * 255)
                    st_ostalo += 1
                    for j in range(rotater_times_ostalo):
                        rotated_object = frameObjects(tensorToImage(rotater(hflipper(imageToTensor(framed_object, device)))))
                        try:
                            cv2.imwrite('Ostalo\ostalo_' + str(st_ostalo) + '.jpg', rotated_object * 255)
                            st_ostalo += 1
                        except:
                            print("Slike ni bilo mogoče shraniti")
                    break

                if key & 0xFF == ord('a'):
                    sys.exit(0)
            cv2.destroyAllWindows()



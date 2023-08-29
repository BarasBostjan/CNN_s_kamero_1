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

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    total_time = end - start
    print("Train time on", device, total_time, "seconds")
    return total_time

def imageToTensor(image, device):
    return torch.from_numpy(image).permute(2 ,0, 1).unsqueeze(dim=0).to(device)

def tensorToImage(tensor):
    return tensor.squeeze(dim=0).permute(1, 2, 0).cpu().detach().numpy()

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

class needleModelV0(nn.Module):         #VGGNet(VGG16)  (https://towardsdatascience.com/the-w3h-of-alexnet-vggnet-resnet-and-inception-7baaaecccc96)
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),      #1
            nn.ReLU(),                                                                          #1
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),     #2
            nn.ReLU(),                                                                          #2   
            nn.MaxPool2d(kernel_size=2, stride=2),                                              #2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),    #3
            nn.ReLU(),                                                                          #3
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),   #4
            nn.ReLU(),                                                                          #4
            nn.MaxPool2d(kernel_size=2, stride=2),                                              #4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),   #5
            nn.ReLU(),                                                                          #5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),   #6
            nn.ReLU(),                                                                          #6
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),   #7
            nn.ReLU(),                                                                          #7
            nn.MaxPool2d(kernel_size=2, stride=2),                                              #7
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),   #8
            nn.ReLU(),                                                                          #8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),   #9
            nn.ReLU(),                                                                          #9
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),   #10
            nn.ReLU(),                                                                          #10
            nn.MaxPool2d(kernel_size=2,stride=2),                                               #10
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),   #11
            nn.ReLU(),                                                                          #11
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),   #12
            nn.ReLU(),                                                                          #12   
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),   #13
            nn.ReLU(),                                                                          #13
            nn.MaxPool2d(kernel_size=2,stride=2),                                               #13
            nn.Flatten(),                                                                       #14
            nn.Linear(in_features=25088, out_features=4096),                                    #14
            nn.ReLU(),                                                                          #14
            nn.Linear(in_features=4096, out_features=4096),                                     #15
            nn.ReLU(),                                                                          #15
            nn.Linear(in_features=4096, out_features=1000),                                     #16
            nn.ReLU(),                                                                          #16
            nn.Linear(in_features=1000, out_features=1)                                         #17

        )

    def forward(self, x):
        return self.block(x)
    
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
    
def batchLoader(batch_num, batch_size, X_train, Y_train, imSize):
    if (batch_num + 1) * batch_size > (X_train.size):
        thisBatchSize = X_train.size - batch_num * batch_size
    else:
        thisBatchSize = batch_size

    startThisBatchSize = batch_num * batch_size
    endThisBatchSize = startThisBatchSize + batch_size

    X_batch = torch.zeros((thisBatchSize, 3, imSize, imSize)).to(device)
    for i in range(thisBatchSize):
        j = X_train[i + startThisBatchSize]
        if j >= st_ostalo:
            image = cv2.imread('Igle\igla_' + str(j - st_ostalo) + '.jpg')
        else:
            image = cv2.imread('Ostalo\ostalo_' + str(j) + '.jpg')

        plt.imshow(image)
        plt.show()
        imageTensor = imageToTensor(image, device)
        X_batch[i] = F.resize(imageTensor, (imSize, imSize))
    Y_batch = Y_train[startThisBatchSize : endThisBatchSize]
    return X_batch, Y_batch

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

    return

if __name__ =='__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    #Parametri--------------------------------------------------------------------------------

    st_igel = 7474
    st_ostalo = 5423
    split = 0.8
    batch_size = 256
    epochs = 60
    image_size = 227

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    #-----------------------------------------------------------------------------------------

    acc = np.zeros((1))
    lossarray = np.zeros((1))

    X_all = np.arange(st_igel + st_ostalo)
    np.random.shuffle(X_all)
    Y_all = torch.zeros(st_igel + st_ostalo).to(device)
    Y_all[X_all >= st_ostalo] = 1

    X_train = X_all[0 : int((st_igel + st_ostalo) * 0.8)]
    Y_train = Y_all[0 : int((st_igel + st_ostalo) * 0.8)]
    X_test = X_all[int((st_igel + st_ostalo) * 0.8) : ]
    Y_test = Y_all[int((st_igel + st_ostalo) * 0.8) : ]

    MODEL_PATH = Path("models")
    MODEL_NAME = "model_0.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    #model_0 = needleModelV0().to(device)
    model_0 = needleModelV1().to(device)
    model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.0001)

    st_batch_train = int(X_train.size / batch_size)
    ostanek_batch_train = X_train.size % batch_size
    if ostanek_batch_train > 0:
        k = 1
    else:
        k = 0

    st_batch_test = int(X_test.size / batch_size)
    ostanek_batch_test = X_test.size % batch_size
    if ostanek_batch_test > 0:
        m = 1
    else:
        m = 0

    print("St_batch:", st_batch_train)
    print("Ostanek_batch:", ostanek_batch_train)

    #trening loop

    
    for epoch in tqdm(range(epochs)):
        print("")
        print("Epoch:", epoch)

        train_loss, train_acc = 0, 0

        model_0.train()
        for batch_num in range(st_batch_train + k):

            #print("Številka batcha:", batch_num)

            X, y = batchLoader(batch_num, batch_size, X_train, Y_train, image_size)

            #showImageandLabel(X, y)

                    #1. forward pass
            y_logits = model_0(X).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))

            #print(y)
            #print(y_pred)



                #2 Calculate loss (per batch)
            loss = loss_fn(y_logits, y)
            train_loss += loss #accumulate train loss
            train_acc += accuracy_fn(y_true=y, y_pred=y_pred)

            if math.isnan(loss):
                print(y)
                print(y_pred)
                sys.exit(0)

            #print(loss)

                    #3 Optimizer zero grad
            optimizer.zero_grad()

                    #4 loss backward
            loss.backward()

                    #5 optimizer step
            optimizer.step()
        
        train_loss /= st_batch_train + k
        train_acc /= st_batch_train + k
        print("Train loss:", train_loss, "Train acc:", train_acc, "%")
        acc = np.append(acc, train_acc)
        lossarray = np.append(lossarray, train_loss.cpu().detach().numpy())
        #print(acc)
        #print(lossarray)
        
        test_loss, test_acc = 0, 0

        model_0.eval()

        with torch.inference_mode():
            for batch_num in range(st_batch_test + m):

                #print("Številka batcha:", batch_num)

                X, y = batchLoader(batch_num, batch_size, X_test, Y_test, image_size)

                y_logits = model_0(X).squeeze()
                y_pred = torch.round(torch.sigmoid(y_logits))

                test_loss += loss_fn(y_logits, y)
                test_acc += accuracy_fn(y_true=y, y_pred=y_pred)


            test_loss /= st_batch_test + m
            test_acc /= st_batch_test + m
            print("Test loss:", test_loss, "Test acc:", test_acc, "%")

    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    print("Saving model to:", MODEL_SAVE_PATH)
    torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

    plt.plot(acc)
    plt.show()
    plt.plot(lossarray)
    plt.show()

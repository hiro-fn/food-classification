from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3, 229, 229
        self.conv1 = nn.Conv2d(3, 128, 3, 2) # 114
        self.drop1 = nn.Dropout2d(0.25)        
        self.conv2 = nn.Conv2d(128, 64, 4, 2) # 56
        self.conv3 = nn.Conv2d(64, 32, 4, 2) # 27
        self.drop2 = nn.Dropout2d(0.25)
        self.pool1 = nn.MaxPool2d(3, 3) # 9
        self.conv4 = nn.Conv2d(32, 16, 3, 2) # 4
        self.conv5 = nn.Conv2d(16, 8, 1) # 4

        self.fc1 = nn.Linear(8 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 101)
        

    def forward(self, x):
        x = F.relu(self.drop1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool1(self.drop1(self.conv3(x))))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.view(-1, 8 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class VGG16Net(nn.Module):
    def __init__(self):
        super(VGG16Net, self).__init__()
        # 3, 224, 224
        self.conv1 = nn.Conv2d(3, 64, 1) # 224
        self.conv2 = nn.Conv2d(64, 64, 1) # 224
        self.pool1 = nn.MaxPool2d(2, 2) # 112

        self.conv3 = nn.Conv2d(64, 128, 1) # 112
        self.conv4 = nn.Conv2d(128, 128, 1) # 112
        self.pool2 = nn.MaxPool2d(2, 2) # 56

        self.conv5 = nn.Conv2d(128, 256, 1) # 56
        self.conv6 = nn.Conv2d(256, 256, 1) # 56
        self.conv7 = nn.Conv2d(256, 256, 1) # 56        
        self.pool3 = nn.MaxPool2d(2, 2) # 28

        self.conv8 = nn.Conv2d(256, 512, 1) # 28
        self.conv9 = nn.Conv2d(512, 512, 1) # 28
        self.conv10 = nn.Conv2d(512, 512, 1) # 28
        self.pool4 = nn.MaxPool2d(2, 2) # 14

        self.conv11 = nn.Conv2d(512, 512, 1) # 14
        self.conv12 = nn.Conv2d(512, 512, 1) # 14
        self.conv13 = nn.Conv2d(512, 512, 1) # 14
        self.pool5 = nn.MaxPool2d(2, 2) # 7

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 101)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv2(self.conv1(x))))
        x = F.relu(self.pool2(self.conv4(self.conv3(x))))
        x = F.relu(self.pool3(self.conv7(self.conv6(self.conv5(x)))))
        x = F.relu(self.pool4(self.conv10(self.conv9(self.conv8(x)))))
        x = F.relu(self.pool5(self.conv13(self.conv12(self.conv11(x)))))

        x = x.view(-1, 512 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
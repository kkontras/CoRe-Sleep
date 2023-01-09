import torch.nn as nn
import torch.nn.functional as F
from ..weights_initializer import init_model_weights


class Mnist_Dropout(nn.Module):
    def __init__(self,):
        super(Mnist_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=0)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(11*11*32, 128)
        self.fc2 = nn.Linear(128, 10)

        self.train_p1 =  0.25
        self.train_p2 =  0.5
        self.test_p1 =  0.25
        self.test_p2 = 0.5

        # self.apply(init_model_weights)

    def forward(self, x, training_time, traineval):
        if training_time:
            drop_prob1, drop_prob2 = self.train_p1, self.train_p2
        else:
            drop_prob1, drop_prob2 = self.test_p1, self.test_p2
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=drop_prob1, training= traineval)
        x = x.view(-1, 11*11*32)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=drop_prob2, training= traineval)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    # def test_drop(self, x, traineval):
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #
    #     x = F.max_pool2d(x, 5)
    #     x = F.dropout(x, p=self.test_p1, training= traineval)
    #     # print(x.size())
    #
    #     x = x.view(-1, 1024)
    #
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, p=self.test_p2, training= traineval)
    #     x = self.fc2(x)
    #     return F.softmax(x, dim=1)


# class Mnist_Dropout(nn.Module):
#     def __init__(self,):
#         super(Mnist_Dropout, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(512, 128)
#         self.fc2 = nn.Linear(128, 10)
#
#         self.train_p1 =  0.2
#         self.train_p2 =  0.4
#         self.test_p1 =  0.01
#         self.test_p2 = 0.05
#
#         self.apply(init_model_weights)
#
#     def forward(self, x, traineval):
#         # print(x.size())
#         x = F.relu(self.conv1(x))
#         # print(x.size())
#
#         x = F.relu(self.conv2(x))
#         # print(x.size())
#
#         x = F.max_pool2d(x, 5)
#         x = F.dropout(x, p=self.train_p1, training= traineval)
#         # print(x.size())
#
#         x = x.view(-1, 512)
#
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, p=self.train_p2, training= traineval)
#         x = self.fc2(x)
#         return F.softmax(x, dim=1)
#
#     def test_drop(self, x, traineval):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#
#         x = F.max_pool2d(x, 4)
#         x = F.dropout(x, p=self.test_p1, training= traineval)
#
#         x = x.view(-1, 512)
#
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, p=self.test_p2, training= traineval)
#         x = self.fc2(x)
#         return F.softmax(x, dim=1)
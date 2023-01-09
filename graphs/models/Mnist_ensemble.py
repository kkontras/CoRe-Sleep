import torch.nn as nn
import torch.nn.functional as F
from ..weights_initializer import init_model_weights


class Mnist_Ensemble(nn.Module):
    def __init__(self):
        super(Mnist_Ensemble, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

        self.train_p1 = 0.25
        self.train_p2 = 0.5
        self.test_p1 = 0.25
        self.test_p2 = 0.5

        self.apply(init_model_weights)

    def forward(self, x, dropout_enable):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 5)
        x = F.dropout(x, p=self.train_p1, training=dropout_enable)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.train_p2, training=dropout_enable)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
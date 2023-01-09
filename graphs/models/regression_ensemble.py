import torch.nn as nn
import torch.nn.functional as F
from ..weights_initializer import weights_init


class Reg(nn.Module):
   def __init__(self,learning_rate,num_hidden,dropout_prob,dropout_prob_test):
       super().__init__()
       self.learning_rate = learning_rate
       self.num_hidden = num_hidden
       self.fc1 = nn.Linear(1, num_hidden)
       self.fc2 = nn.Linear(num_hidden, 1)
       self.dropout = dropout_prob
       self.dropout_test = dropout_prob_test
       self.apply(weights_init)

   def forward(self, x, training_prob= True, drop= False):
       x = F.relu(self.fc1(x))
       if not(drop):
           drop_prob = self.dropout
       else:
           drop_prob =self.dropout_test
       x = self.fc2(F.dropout(x, p=drop_prob, training = drop))
       return x


class Reg_nll(nn.Module):
# This model returns two outputs the mean value and the variance of the function we want to predict.
# It also uses a softplus in the variance output according to [Lakshminarayanan 2017] with a summation of  1e-06 for numerical stability.

   def __init__(self,learning_rate,num_hidden,dropout_prob,dropout_prob_test):
       super().__init__()
       self.learning_rate = learning_rate
       self.num_hidden = num_hidden
       self.fc1 = nn.Linear(1, num_hidden)
       self.fc2 = nn.Linear(num_hidden, 2)
       self.dropout = dropout_prob
       self.dropout_test = dropout_prob_test
       self.apply(weights_init)

   def forward(self, x, training_prob= True, drop= False):
       if not(drop):
           drop_prob = self.dropout
       else:
           drop_prob =self.dropout_test

       x = F.relu(self.fc1(x))
       x = self.fc2(F.dropout(x, p=drop_prob, training = drop))
       mean = x[:, 1]
       variance = F.softplus(x[:, 0]) + 1e-06
       return variance, mean


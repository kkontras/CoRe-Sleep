import math
import torch

#Mean IOU metric. For this metric we are not using background as
# name2id = {v:k for k,v in enumerate(codes)}
# void_code = name2id['background']

def mean_iou(inputs, targets, num_classes,ignoreEval):
  '''
  Returns the Mean IoU (Intersection over Union), iterating over each image and each class.
  In the average between the mean_iou of the classes participate any that have tp+fp+fn > 0.
  Input:
    inputs: A vector with the network's predictions for the classes (N,21,img_size,img_size)
    targets: A vector with the target values for the corresponding predictions (N,1,img_size,img_size)
    num_classes: Number of classes.
    ignoreEval: Boolean list with size num_classes, indices correspond to classes.
  Output:
    torch_mean_iou: Torch cuda Tensor of the Mean IoU for all images and all classes.

  Note: We do not take into consideration classes that have IgnoreEval True.
  '''
  total_iou = []
  targets = targets.squeeze(1)
  for i in range(1,num_classes):
    if not ignoreEval[i]:
      mask = targets == i
      batch_iou = (inputs.argmax(dim=1)[mask]==targets[mask]).float().mean()
      if not (math.isnan(batch_iou)):
        total_iou.append(batch_iou)
  if(len(total_iou)>0):
    m_iou = torch.stack(total_iou).float().mean()
  else:
    m_iou = torch.FloatTensor([0])
  return m_iou

def get_mean_iou(inputs, targets, num_classes):
  '''
  Returns the vector which includes the Mean IoU (Intersection over Union) for each class, iterating over each image and each class.
  In the average between the mean_iou of the classes participate any that have tp+fp+fn > 0.
  Input:
    inputs: A vector with the network's predictions for the classes (N,21,img_size,img_size)
    targets: A vector with the target values for the corresponding predictions (N,1,img_size,img_size)
  Output:
    m_iou: Torch cuda Tensor of the Mean IoU for all inputs and all classes.
  '''
  total_iou = []
  targets = targets.squeeze(1)
  for i in range(num_classes):
    mask = targets == i
    batch_iou = (inputs.argmax(dim=1)[mask]==targets[mask]).float().mean()
    total_iou.append(batch_iou)
  if(len(total_iou)>0):
    m_iou = torch.stack(total_iou).float()
    m_iou[m_iou != m_iou] = 0
  else:
    m_iou = torch.zeros(1, num_classes).float()
  return m_iou

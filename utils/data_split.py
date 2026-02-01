import torch
from torch.utils.data import random_split


def test_train_split(dataset,train_p,val_p,random_seed=42):
  total_size = len(dataset)
  train_size = int(train_p * total_size)
  val_size   = int(val_p * total_size)
  test_size  = total_size - train_size - val_size

  generator = torch.Generator().manual_seed(42)
  train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
  )
  return train_dataset,val_dataset,test_dataset
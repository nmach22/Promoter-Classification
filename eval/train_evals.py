import torch

class TrainEvals:
  def __init__(self,
      accuracy:bool=True,
      sensitivity:bool=True,
      specificity:bool=True
      ) -> None:

      self.accuracy = accuracy
      self.sensitivity = sensitivity
      self.specificity = specificity

  # data is list of tuples: (probs, y)
  # probs and y is tensors  
  def evaluate(self,data):
    probs = [x[0] for x in data]
    y = [x[1] for x in data]
    evals = {}
    if self.accuracy:
      accuracy = self.calculate_accuracy(probs,y)
      evals['accuracy'] = accuracy

    if self.sensitivity:
      sensitivity = self.calucate_sensitivity(probs,y)
      evals['sensitivity'] = sensitivity

    if self.specificity:
      specificity = self.calculate_specificity(probs,y)
      evals['specificity'] = specificity

    return evals
  
  def calculate_accuracy(self,probs,y):
    probs = torch.cat(probs).view(-1)
    y = torch.cat(y).view(-1)

    preds = (probs >= 0.5).long()

    correct = (preds == y).sum().item()
    total = y.numel()

    return correct / total
  
  def calucate_sensitivity(self, probs,y):
    probs = torch.cat(probs).view(-1)
    y = torch.cat(y).view(-1)

    preds = (probs >= 0.5).long()

    TP = ((preds == 1) & (y == 1)).sum().item()
    FN = ((preds == 0) & (y == 1)).sum().item()

    if TP + FN == 0:
        return 0.0

    return TP / (TP + FN)

  def calculate_specificity(self,probs, y):
    probs = torch.cat(probs).view(-1)
    y = torch.cat(y).view(-1)

    preds = (probs >= 0.5).long()

    TN = ((preds == 0) & (y == 0)).sum().item()
    FP = ((preds == 1) & (y == 0)).sum().item()

    if TN + FP == 0:
        return 0.0

    return TN / (TN + FP)

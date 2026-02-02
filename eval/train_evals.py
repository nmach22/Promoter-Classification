import torch
import math

class TrainEvals:
  def __init__(self,
      accuracy:bool=True,
      sensitivity:bool=True,
      specificity:bool=True,
      correlation_coef:bool=True
      ) -> None:

      self.accuracy = accuracy
      self.sensitivity = sensitivity
      self.specificity = specificity
      self.correlation_coef = correlation_coef

  def evaluate(self,data):
    probs = [x[0] for x in data]
    y = [x[1] for x in data]

    probs = torch.cat(probs).view(-1)
    y = torch.cat(y).view(-1)

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

    if self.correlation_coef:
      correlation_coef = self.calulcate_cc(probs,y)
      evals['correlation_coef'] = correlation_coef

    return evals
  
  def calculate_accuracy(self,probs,y):
    preds = (probs >= 0.5).long()

    correct = (preds == y).sum().item()
    total = y.numel()

    return correct / total
  
  def calucate_sensitivity(self, probs,y):
    preds = (probs >= 0.5).long()

    TP = ((preds == 1) & (y == 1)).sum().item()
    FN = ((preds == 0) & (y == 1)).sum().item()

    if TP + FN == 0:
        return 0.0

    return TP / (TP + FN)

  def calculate_specificity(self,probs, y):
    preds = (probs >= 0.5).long()

    TN = ((preds == 0) & (y == 0)).sum().item()
    FP = ((preds == 1) & (y == 0)).sum().item()

    if TN + FP == 0:
        return 0.0

    return TN / (TN + FP)

  def calulcate_cc(self,probs,y):
    preds = (probs >= 0.5).long()

    TP = ((preds == 1) & (y == 1)).sum().item()
    TN = ((preds == 0) & (y == 0)).sum().item()
    FP = ((preds == 1) & (y == 0)).sum().item()
    FN = ((preds == 0) & (y == 1)).sum().item()

    numerator = TP * TN - FP * FN
    denominator = math.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator

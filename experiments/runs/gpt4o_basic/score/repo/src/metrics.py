import torch

def accuracy_at_first_attempt(y_pred, y_true):
    return torch.sum(y_pred == y_true).item() / y_true.shape[0]

def accuracy_at_second_attempt(y_pred, y_true):
    return torch.sum(y_pred == y_true).item() / y_true.shape[0]

def delta_t1_t2(y_pred_1, y_pred_2, y_true):
    acc_t1 = accuracy_at_first_attempt(y_pred_1, y_true)
    acc_t2 = accuracy_at_second_attempt(y_pred_2, y_true)
    return acc_t2 - acc_t1

def delta_i_to_c(y_pred_1, y_pred_2, y_true):
    incorrect_to_correct = torch.sum((y_pred_1 != y_true) & (y_pred_2 == y_true)).item()
    total_incorrect = torch.sum(y_pred_1 != y_true).item()
    return incorrect_to_correct / total_incorrect if total_incorrect > 0 else 0.0

def delta_c_to_i(y_pred_1, y_pred_2, y_true):
    correct_to_incorrect = torch.sum((y_pred_1 == y_true) & (y_pred_2 != y_true)).item()
    total_correct = torch.sum(y_pred_1 == y_true).item()
    return correct_to_incorrect / total_correct if total_correct > 0 else 0.0



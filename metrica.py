import numpy as np

confusion_matrix = np.array([[ 58,  26,  18,  53,  15,   2,  14,  17],
 [ 10,  81,  29,  24,  25,   9,  11,  29],
 [  5,  14, 105,  17,  14,   9,  18,   9],
 [ 21,  13,  10, 129,   5,   6,   8,   9],
 [ 11,  35,  43,  10,  78,   1,   9,  10],
 [ 16,  20,  51,  47,   3,  27,  20,  13],
 [ 26,  17,  45,  33,   5,  15,  26,  37],
 [  6,  17,  15,   8,   5,   6,  22, 109]])

precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
f_score = 2 * (precision * recall) / (precision + recall)

weighted_precision = np.sum(precision * np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix))
weighted_recall = np.sum(recall * np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix))
weighted_f_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)

print("Precision:", precision)
print("Recall:", recall)
print("F-score:", f_score)
print("\nWeighted Precision:", weighted_precision)
print("Weighted Recall:", weighted_recall)
print("Weighted F-score:", weighted_f_score)

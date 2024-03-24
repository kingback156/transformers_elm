import torch
import torch.nn as nn
import torch.nn.functional as F
def orth (m,n):   # generate an m x n orthogonal matrix (tensor), m>n
    if m<n:
        print("orthogonal matrix should be full rank")
        exit(1)
    raw_weight = torch.randn(m,n)
    q, _ = torch.linalg.qr(raw_weight)
    weight = q[:, :n]
    return weight

W = orth(9,6)

print(torch.dot(W[:,1],W[:,5]))
print(torch.dot(W[:,3],W[:,3]))

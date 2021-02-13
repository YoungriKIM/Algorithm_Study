# https://wikidocs.net/52460 참고

import torch

# 1차원
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])

print(t)                # tensor([0., 1., 2., 3., 4., 5., 6.])

print(t.dim())          # 차원: 1
print(t.shape)          # torch.Size([7])
print(t.size())         # torch.Size([7])

print(t[0], t[1], t[-1])    # tensor(0.) tensor(1.) tensor(6.)
print(t[2:5], t[4:-1])      # tensor([2., 3., 4.]) tensor([4., 5.])
print(t[:2], t[3:])         # tensor([0., 1.]) tensor([3., 4., 5., 6.])

# 2차원
t = torch.FloatTensor([[1.,2.,3.],
                      [4.,5.,6.],
                      [7.,8.,9.],
                      [10.,11.,12.]])


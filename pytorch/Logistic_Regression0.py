"""
import numpy as np
import matplotlib.pyplot as plt

#! sigmoid 함수 --> 참/거짓을 판별 할 때 쓰이는 함수 (s 모양을 하고있어서 참거짓에 유용)
#! 0~1사이 값을 가짐
#! 0.5를 기준으로 참거짓을 판별 할 수 있음
def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y= sigmoid(x)

plt.plot(x, y, 'g')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

#! 데이터 
x_data = [[1,2],[2,3],[3,1],[4,3],[5,4],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
#! 데이터를 pytorch 데이트로 변경
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
print(x_train.shape)
print(y_train.shape)

#! x_data의 크기가 6x2 이기 때문에 2x1 행렬 W 만듦
W = torch.zeros((2,1), requires_grad=True) 
#!
b = torch.zeros(1, requires_grad=True)

#! x에 대응하는 결과가 sigmoid 함수를 따른다고 가정함 (True or False)
#hypothesis = 1/(1+torch.exp(-(x_train.matmul(W)+b)))
#hypothesis = torch.sigmoid(x_train.matmul(W)+b)

#! 가정에 대한 실제값과의 오차 구하기
# losses = -(y_train*torch.log(hypothesis)+(1-y_train)*torch.log(1-hypothesis))
# cost = losses.mean()
# cost = F.binary_cross_entropy(hypothesis, y_train)
#! 최적화 알고리듬 정의
optimizer = optim.SGD([W,b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    #! y에 대한 가정
    hypothesis = torch.sigmoid(x_train.matmul(W)+b)
    #! 비용함수
    cost = F.binary_cross_entropy(hypothesis,y_train)
    
    #! 정의한 최적화 함수로 cost에 대한 최적화 적용
    optimizer.zero_grad() # optimizer 0으로 셋팅
    cost.backward() # cost를 미분함
    #! 미분을 하면 클래스에서 전역변수가 되어서(?) 인식하게됨
    optimizer.step() # 미분된 값을 적용 시킴
    
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

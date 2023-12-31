import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__() # 상속받은 nn.Module 클래스의 속성들을 가지고 초기화됨
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))

torch.manual_seed(1)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,4],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
#! 데이터를 pytorch 데이트로 변경
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(),lr=1)
nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    hypothesis = model(x_train)
    
    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    
    if epoch % 20 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        
        correct_prediction = prediction.float() == y_train #실제값과 일치하는 경우만 True로 간주
        #print(prediction, correct_prediction)
        accuracy = correct_prediction.sum().item()/len(correct_prediction) #정확도를 계산
        print('Epoch {:4d}/{} Cost:{:.6f} Accuracy {:2.2f}%'.format(epoch, nb_epochs, cost.item(),accuracy*100,))

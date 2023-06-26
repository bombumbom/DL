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

#! Sequential 을 모델을 여러가지 겹치에 만들어 준다.
#! --> 여러 함수를 연결 하게 해주는 역할을 한다.
#! W와 b는 랜덤 초기화상태
model = nn.Sequential(
    nn.Linear(2,1), #input_dim = 2, output_dim = 1
    nn.Sigmoid() # 출력은 시그모이드 함수를 거친다.
)

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    #! y에 대한 가정
    hypothesis = model(x_train)
    #! 비용함수
    cost = F.binary_cross_entropy(hypothesis,y_train)
    
    #! 정의한 최적화 함수로 cost에 대한 최적화 적용
    optimizer.zero_grad() # optimizer 0으로 셋팅
    cost.backward() # cost를 미분함
    #! 미분을 하면 클래스에서 전역변수가 되어서(?) 인식하게됨
    optimizer.step() # 미분된 값을 적용 시킴
    
    # 100번마다 로그 출력
    if epoch % 20 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        
        correct_prediction = prediction.float() == y_train #실제값과 일치하는 경우만 True로 간주
        #print(prediction, correct_prediction)
        accuracy = correct_prediction.sum().item()/len(correct_prediction) #정확도를 계산
        print('Epoch {:4d}/{} Cost:{:.6f} Accuracy {:2.2f}%'.format(epoch, nb_epochs, cost.item(),accuracy*100,))

print(list(model.parameters()))
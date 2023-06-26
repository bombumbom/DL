import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Dataset 상속
class CustomDataset(Dataset): 
  def __init__(self):
    #! 데이터 전처리
    self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
    self.y_data = [[152], [185], [180], [196], [142]]

  # 총 데이터의 개수를 리턴
  def __len__(self):
    #!데이터 길이
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx):
    #! 데이터 셋을 파이토치 tensor 형태로 리턴
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y

#! Customdataset을 이용하여 파이토치 형태로 만들기
dataset = CustomDataset()
#! Customd으로 만든 dataset를 DataLoader하기
#! Bath_size =2 --> 전체 데이터중 2개씩 작업 시작
#! shuffle=True 데이터셋의 순서를 무작위로 변경
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#! 선형 모형 불러오기 (3개의 x와, 1개의 결과(y))
model = torch.nn.Linear(3,1)
#! 최적화 방법으로 SGD 채택
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

nb_epochs = 20 #!학습을 수행할 전체 횟수
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    # print(batch_idx)
    # print(samples)
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))

new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 

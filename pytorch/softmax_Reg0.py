import torch
import torch.nn.functional as F
torch.manual_seed(1)

# z = torch.FloatTensor([1,2,3])
# hypothesis = F.softmax(z, dim=0)
# print(hypothesis)

#! 3x5 행렬 만들기
z = torch.rand(3,5,requires_grad=True)

#! z에 대응하는 softmax 모델 적용
#! 3개의 샘플중 5개의 클래스 중에서 어떤 클래스가 정답인지
hypothesis = F.softmax(z, dim=1)

y = torch.randint(5, (3,)).long()

#! 모든 요소가 0인 모형과 같은 차원의 행렬 생성
y_one_hot = torch.zeros_like(hypothesis)
#! 원-핫 인코딩 : 각 행에 1 하나씩 넣기
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

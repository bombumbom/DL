import torch
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])  # *공부한 시간
y_train = torch.FloatTensor([[2], [4], [6]])  # *공부한 시간에 대응하는 점수

#! 모형, 가설 --> y=Wx + b 를 따른다! 선형 회귀
# 가중치 초기 값 0, requires_grad=True,학습을 통해서 변화는 값 정의해줌
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)


#! 경사 하강법으로 비용함수 최소가 되는 것 찾기
# SGD -> 경사 하강법의 한 종류
# lr (learning rate) -> 학습률
optimizer = optim.SGD([W, b], lr=0.01)


nb_epochs = 1999  # 경사하강법 반복 회수
for epoch in range(nb_epochs + 1):
    hypothesis = x_train * W + b  #! 모형
    cost = torch.mean((hypothesis - y_train) ** 2)  # 비용, 손실함수

    #! pytorch는 미분값을 계속 누적하기 때문에 매 epoch마다 0으로 바꿔줘야 한다.
    optimizer.zero_grad()  # 기울기를 0으로 초기화
    cost.backward()  # 비용 함수를 미분하여 gradient 계산
    optimizer.step()  # W와 b를 업데이트

    if epoch % 100 == 0:
        print(
            "Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}".format(
                epoch, nb_epochs, W.item(), b.item(), cost.item()
            )
        )

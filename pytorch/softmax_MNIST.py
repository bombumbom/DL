import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import time

start = time.time()

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
# device = torch.device("cpu")
print("Current cuda device is", device)
random.seed(777)
torch.manual_seed(777)

#! 전체 훈련 회수
training_epochs = 15

#!데이터를 분활해서 볼 사이즈
#!전체 데이터 개수를 "batch_size" 으로 나눠서 본다.
batch_size = 50

#! train, test 용 데이터 다운 & pytorch tensor로 변형
mnist_train = dsets.MNIST(
    root="MNIST_data/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

mnist_test = dsets.MNIST(
    root="MNIST_data/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

print(f"the number of train data : {len(mnist_train)}")
print(f"the number of test data : {len(mnist_test)}")

#! 훈련용 데이터를 DataLoader를 이용해서 batch에 맞게 분활하기
#! shuffle --> 순서 섞기
#! drop_last --> batch_size로 나누고 마지막에 batch_size보다 작은 나머지 버릴지 말지 정하기
data_loader = DataLoader(
    dataset=mnist_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

#! MNIST data image of shape 28pixel*28pixel = 784pixel
#! 모델 설계 --> W * x + b 모형 채택
#! 784개의 정보를 가짐, 결과는 10개(0부터 9까지)
#! 28*28 2차원 행렬을 1차원으로 줄여서 보겠다는 것
#! to 함수는 이 모형의 연산을 어디서 수행할 지 알려주는 역할
#! bias --> 모델에서 b를 쓸지 말지 정하는것 defualt=True
linear = nn.Linear(784, 10, bias=True).to(device)

#! 비용함수, softmax를 포함하는 함수 채택
criterion = nn.CrossEntropyLoss().to(device)
#! 최적화 함수, linear 변수의 parameters를 최적화 해서 적용시킴
optimizer = torch.optim.SGD(linear.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda epoch: 0.95**epoch,
    last_epoch=-1,
    verbose=False,
)


for epoch in range(training_epochs):
    print("lr: ", optimizer.param_groups[0]["lr"])
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        #! 배치 크기가 100이므로 아래의 연산에서 X는 (100, 784)의 텐서가 됨
        #! 784의 이미지를 100개씩 본다
        X = X.view(-1, 28 * 28).to(device)
        #! Y -> 레이블은 원-핫 원코딩이 된 상태가 아니라 0~9의 정수를 가짐
        Y = Y.to(device)

        #! 최적화 함수 0으로 셋팅
        optimizer.zero_grad()

        #!위에서 정의한 linear 모형을 각각 배치에 맞게 다시 모형 만듦
        hypothesis = linear(X)

        #! 위에서 정의한 criteriond을 배치에 맞게 선언
        cost = criterion(hypothesis, Y)

        #! 미분!
        cost.backward()

        #! cost 미분을 반영하여 optimizer.step()으로 반영
        optimizer.step()

        #! 평균 손실 함수 보기위함 option
        avg_cost += cost / total_batch

    #! learning rate 조절
    scheduler.step()

    print(
        "Epoch:",
        "%04d" % (epoch + 1),
        "cost =",
        "{:.9f}".format(avg_cost),
    )
print("Learning finished")


# 테스트 데이터를 사용하여 모델을 테스트한다.
for step in range(10):
    with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
        X_test = (
            mnist_test.test_data.view(-1, 28 * 28).float().to(device)
        )
        Y_test = mnist_test.test_labels.to(device)

        prediction = linear(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print("Accuracy:", accuracy.item())

        # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
        r = random.randint(0, len(mnist_test) - 1)
        X_single_data = (
            mnist_test.test_data[r : r + 1]
            .view(-1, 28 * 28)
            .float()
            .to(device)
        )
        Y_single_data = mnist_test.test_labels[r : r + 1].to(device)

        print("Label: ", Y_single_data.item())
        single_prediction = linear(X_single_data)
        print(
            "Prediction: ", torch.argmax(single_prediction, 1).item()
        )

        plt.imshow(
            mnist_test.test_data[r : r + 1].view(28, 28),
            cmap="Greys",
            interpolation="nearest",
        )
        plt.show()

endt = time.time()
print((endt - start))

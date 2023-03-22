import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import random
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

random_seed = 1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

'사용자 데이터셋 인스턴스 생성 클래스'
class train_Dataset(Dataset): # torch의 Dataset 상속받음
    def __init__(self, data, transform = None):
        self.fashion_mnist = list(data.values)
        self.transform = transform
        label, img = [], [] 

        for one_line in self.fashion_mnist:
            label.append(one_line[0]) 
            img.append(one_line[1:]) # 이미지 1개 당 픽셀 784개

        self.label = np.asarray(label)
        self.img = np.asarray(img).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNEL).astype('float32')

    def __len__(self): # 학습 데이터 개수
        return len(self.label)

    def __getitem__(self, idx): # 앞서 만든 리스트의 인덱스 값을 참조해 데이터에 관한 여러 일처리를 진행한 후 그 결과를 반환
        label, img = self.label[idx], self.img[idx]
        if self.transform:
            img = self.transform(img)
        return label, img

class test_Dataset(Dataset):
    def __init__(self, data, transform = None):
        self.fashion_mnist = list(data.values)
        self.transform = transform
        img = []

        for one_line in self.fashion_mnist:
            img.append(one_line[:]) # 이미지 1개 당 픽셀 784개

        self.img = np.asarray(img).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNEL).astype('float32')

    def __len__(self):
        return len(self.fashion_mnist)

    def __getitem__(self, idx):
        img = self.img[idx]
        if self.transform:
            img = self.transform(img)
        return img

'하이퍼 파라미터'
BATCH_SIZE = 25
LR = 5e-3
NUM_CLASS = 10
IMAGE_SIZE = 28
CHANNEL = 1
Train_epoch = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'사용자 transform: 여기서는 augmentation 같은거 안 하고 ToTensor() 하나만 적용'
My_transform = transforms.Compose([
    transforms.ToTensor(), # default : range [0, 255] -> [0.0, 1.0] 스케일링
])

'데이터'
train = pd.read_csv('./data_fashion_mnist/train.csv', index_col='index') # len 60000
test = pd.read_csv('./data_fashion_mnist/test.csv', index_col='index') # len 10000

'valid set'
valid_size = 10000
indices = torch.randperm(len(train)) # shuffled indices from 0 to 59999
train_indices = indices[:len(indices) - valid_size]
valid_indices = indices[len(indices) - valid_size:] if valid_size else None

Train_data = train_Dataset(train, transform=My_transform)
Valid_data = train_Dataset(train, transform=My_transform)
Test_data = test_Dataset(test, transform=My_transform)

'torch DataLoader 함수: 데이터를 mini-batch로 처리해 주고, GPU를 통한 병렬처리, 학습효율의 향상'
Train_dataloader = DataLoader(dataset=Train_data,
                              batch_size = BATCH_SIZE,
                              shuffle=False,
                              sampler=SubsetRandomSampler(train_indices))
Valid_dataloader = DataLoader(dataset=Valid_data,
                              batch_size = BATCH_SIZE,
                              shuffle=False,
                              sampler=SubsetRandomSampler(valid_indices))
Test_dataloader = DataLoader(dataset=Test_data,
                             batch_size = 1,
                             shuffle=False)

'사용자 모델'
class My_model(nn.Module): # torch nn.Module 상속
    def __init__(self, num_of_class):
        super(My_model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # 28 * 28 * 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 14 * 14 * 16
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 14 * 14 * 32
            nn.BatchNorm2d(32),
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2, stride=2)) 7 * 7 * 32
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 14 * 14 * 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 7 * 7 * 64
        ) 
        
        self.fc = nn.Linear(7 * 7 * 64, num_of_class)

    def forward(self, x):
        out = self.layer1(x) # (28, 28, 1) -> (14, 14, 16)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1) # (7, 7, 64) -> flatten -> (7, 7*64) ??????????????????????
        out = self.fc(out)
        #out = F.softmax(out, dim=0) # shape (25,10)
        return out


'모델 학습'
def train():
    
    model = My_model(NUM_CLASS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    criterion = nn.CrossEntropyLoss() 
    valid_loss_min = np.inf # 초기화 (나중에 업데이트 함)

    for epoch in range(1, Train_epoch + 1): # epoch: 모든 데이터

        train_loss = 0.0
        valid_loss = 0.0

        for batch_id, (label, image) in enumerate(Train_dataloader): # iter: batch 데이터 (25개) 
            label, image = label.to(device), image.to(device) # shape: (25,)
            
            output = model(image)   # 1. 모델에 데이터 입력해 출력 얻기 # 10개 클래스에 대한 로짓 # shape: (25, 10)
            loss = criterion(output, label) # 2. loss 계산 # NLL loss 2.3078 # shape: () 
            optimizer.zero_grad() # 3. 기울기 초기화 (iter 끝날때마다 초기화)
            loss.backward() # 4. 역전파
            optimizer.step() # 5. 최적화
        
        for batch_id, (label, image) in enumerate(Valid_dataloader):
            label, image = label.to(device), image.to(device)

            output = model(image)
            loss = criterion(output, label)
            valid_loss += loss.item()
        
        # calculate avg losses
        train_loss = train_loss/len(Train_dataloader.dataset)
        valid_loss = valid_loss/len(Valid_dataloader.dataset)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model, './data_fashion_mnist/best_model.pt')
            torch.save(model.state_dict(), './data_fashion_mnist/best_model.pth')
            torch.save(valid_indices, './data_fashion_mnist/valid_indices.pth')
            valid_loss_min = valid_loss
    
    return model
    

'학습된 모델로 테스트'
def test(model):
    model = torch.load('./data_fashion_mnist/best_model.pt') # 모델 불러오기
    print('success load best_model')
    pred = []
    with torch.no_grad(): # 파라미터 업데이트 안 함
        correct = 0
        total = 0
        for image in Test_dataloader: 
            image = image.to(device)

            outputs = model(image)

            predicted = np.asarray(torch.argmax(outputs, dim=1).cpu()) # 예측 클래스
            pred.append(predicted)
    
    return np.array(pred).flatten() # flatten: 1차원으로 펴 줌 (submission 파일에 값을 대입하기 위해) # (10000,)

'메인문'
if __name__ == '__main__':
    model = train()
    pred = test(model)

'예측 라벨 저장'
submission = pd.read_csv('./data_fashion_mnist/sample_submission.csv')
submission['label'] = pred
submission.to_csv('./data_fashion_mnist/submission.csv', index=False)




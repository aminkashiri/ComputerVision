import torch, torchvision, logging, gc, time
from os import path
from torch import nn
from matplotlib import pyplot

start_time = time.time()

torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
number_of_workers = 4
learning_rate = 1e-3
# batch_sizes = [50,50,20,40,40] #1 affin, crop , 30 epochs
# batch_sizes = [10,10,10,10,10] #2 no affine only equalize, 40 epochs
batch_sizes = [16,16,16,16,16] # 3
epochs = 60


def get_data():
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomEqualize(),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.RandomAffine(1,translate=(0.01,0.01),scale=(0.99,1.01)),
        # torchvision.transforms.CenterCrop((224,224)),
        normalize
    ])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224,224)),
        # torchvision.transforms.CenterCrop((224,224)),
        normalize
    ])

    train_set = torchvision.datasets.ImageFolder(
        root=path.join('Data','Train'),
        transform=transform_train
    )
    test_set = torchvision.datasets.ImageFolder(
        root=path.join('Data','Test'),
        transform=transform_test
    )

    return train_set, test_set


def all_models():
    alexnet = AlexNet_1()
    yield (1, alexnet, alexnet.parameters())

    alexnet = AlexNet_2()
    yield (2, alexnet, alexnet.parameters())

    alexnet = AlexNetFull()
    yield (3, alexnet, alexnet.parameters())

    alexnet = torchvision.models.alexnet(pretrained=True, progress=True)
    for param in alexnet.parameters():
        param.requires_grad = False
    last_layer_inputs = alexnet.classifier[-1].in_features
    alexnet.classifier[-1] = nn.Linear(last_layer_inputs,15)
    yield (4,alexnet, alexnet.classifier[-1].parameters())

    alexnet = torchvision.models.alexnet(pretrained=True, progress=True)
    last_layer_inputs = alexnet.classifier[-1].in_features
    alexnet.classifier[-1] = nn.Linear(last_layer_inputs,15)
    yield (5,alexnet, alexnet.parameters())


def load_model(model_path):
    alexnet = torchvision.models.alexnet()
    last_layer_inputs = alexnet.classifier[-1].in_features
    alexnet.classifier[-1] = nn.Linear(last_layer_inputs,15)
    alexnet.load_state_dict(torch.load(model_path))
    alexnet = alexnet.to(device)
    return alexnet

train_set, test_set = get_data()

class AlexNet_1(nn.Module):
    def __init__(self):
        super(AlexNet_1, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.dropout7 = nn.Dropout()
        self.fc7 = nn.Linear(64 * 13 * 13, 4096)
        self.relu7 = nn.ReLU(inplace=True)

        self.output = nn.Linear(4096, 15)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = torch.flatten(x, 1)

        x = self.dropout7(x)
        x = self.fc7(x)
        x = self.relu7(x)

        x = self.output(x)
        return x

class AlexNet_2(nn.Module):
    def __init__(self):
        super(AlexNet_2, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.max_pool_extra = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.dropout6 = nn.Dropout()
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU(inplace=True)

        self.dropout7 = nn.Dropout()
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)

        self.output = nn.Linear(4096, 15)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.max_pool_extra(x)

        x = torch.flatten(x, 1)

        x = self.dropout6(x)
        x = self.fc6(x)
        x = self.relu6(x)

        x = self.dropout7(x)
        x = self.fc7(x)
        x = self.relu7(x)

        x = self.output(x)
        return x

class AlexNetFull(nn.Module):
    def __init__(self):
        super(AlexNetFull, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.reu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.max_pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.dropout6 = nn.Dropout()
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU(inplace=True)

        self.dropout7 = nn.Dropout()
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)

        self.output = nn.Linear(4096, 15)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.reu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.max_pool5(x)

        # x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout6(x)
        x = self.fc6(x)
        x = self.relu6(x)

        x = self.dropout7(x)
        x = self.fc7(x)
        x = self.relu7(x)

        x = self.output(x)
        return x


logger = logging.getLogger()  
logging.basicConfig(filename='log.log',format='%(message)s',filemode='w', encoding='utf-8', level=logging.INFO)


for (j, alexnet, params) in all_models():
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_sizes[j-1], shuffle=True, num_workers=number_of_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_sizes[j-1], shuffle=True, num_workers=number_of_workers)
    alexnet = alexnet.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)

    train_losses = torch.zeros((epochs))
    test_losses = torch.zeros((epochs))
    top1_accuracies = torch.zeros((epochs))
    top5_accuracies = torch.zeros((epochs))

    for epoch in range(epochs):
        logger.info(f'----------------------- epoch {epoch}---------------------------')
        print(f'----------------------- epoch {epoch}---------------------------')
        train_loss_sum = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            temp = time.time()
            outputs = alexnet(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        top1_correct = 0
        top5_correct = 0
        total = 0

        train_loss = train_loss_sum / train_set.__len__()
        logger.info(f'-> Train Loss: {train_loss}')
        print(f'-> Train Loss: {train_loss}')

        logger.info('-> Testing')
        print('-> Testing')
        test_loss_sum = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = alexnet(inputs)
                total += labels.size(0)

                loss = loss_function(outputs, labels)
                test_loss_sum += loss.item()

                _, results = torch.max(outputs.data, 1)
                top1_correct +=  ((results - labels) == 0).sum().item()

                _, results = torch.topk(outputs.data, 5, 1)
                top5_correct += torch.any( ((results - labels[:,None]) == 0), 1).sum().item()

        test_loss = test_loss_sum / test_set.__len__()
        logger.info(f'-> Test Loss: {test_loss}')
        print(f'-> Test Loss: {test_loss}')


        top1_accuracy = 100 * top1_correct / total
        top5_accuracy = 100 * top5_correct / total


        train_losses[epoch] = train_loss
        test_losses[epoch] = test_loss
        top1_accuracies[epoch] = top1_accuracy
        top5_accuracies[epoch] = top5_accuracy


        logger.info(top1_correct)
        logger.info(top5_correct)
        logger.info(total)
        logger.info('-> Accuracy of the top 1 ' + str( top1_accuracy ) + '%')
        logger.info('-> Accuracy of the top 5 ' + str( top5_accuracy ) + '%')
        print(top1_correct)
        print(top5_correct)
        print(total)
        print('-> Accuracy of the top 1',  top1_accuracy , '%')
        print('-> Accuracy of the top 5',  top5_accuracy , '%')

    model_path = f'models/alexnet_{j}_epochs={epochs}_lr={learning_rate}.pth'
    torch.save(alexnet.state_dict(), model_path)

    pyplot.plot(train_losses, label='train losses')
    pyplot.plot(test_losses, label='test losses')
    pyplot.xticks(range(0,epochs))
    pyplot.legend()
    pyplot.savefig(path.join('outputs',f'losses_{j}'))
    # pyplot.show()
    pyplot.close()

    pyplot.plot(top1_accuracies, label='top 1')
    pyplot.plot(top5_accuracies, label='top 5')
    pyplot.xticks(range(0,epochs))
    pyplot.legend()
    pyplot.savefig(path.join('outputs',f'accuracy{j}'))
    pyplot.close()
    # pyplot.show()


    gc.collect()
    torch.cuda.empty_cache()


print('Total Time: ', time.time() - start_time)
logger.info('Total Time: ' + str( time.time() - start_time) )

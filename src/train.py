import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets

from net import Net, VGG16Net

def create_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


def set_dataset(dataset_path, transform, batch_size):
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=True, num_workers=2)


def run_train(net, dataset, options):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=options['lr'], momentum=0.9)

    print(net)

    with open('log.txt', 'w') as f:
        f.write(f'{str(net)} \n')
        for epoch in range(options['epoch']):
            running_loss = 0.0

            print(epoch)
            for i, data in enumerate(dataset, 1):
                raw_inputs, raw_labels = data
                inputs = raw_inputs.to('cuda')
                labels = raw_labels.to('cuda')

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]

                if i % 10 == 0:
                    print(f'[{epoch}, {i}] loss: {running_loss}')
                    f.write(f'{epoch}: {running_loss}\n')
                    running_loss = 0.0

            torch.save(net, f'model\\{epoch}')


def main():
    train_dataset_path = 'D:\\project\\dataset\\food\\food-101\\train'
    # test_dataset_path = 'D:\\project\\dataset\\food\\food-101\\test'
    options = {
        'batch_size': 16,
        'epoch': 500,
        'lr': 1e-6
    }

    data_transform = create_transform()
    train_dataset = set_dataset(train_dataset_path, data_transform, options['batch_size'])
    # test_dataset = set_dataset(test_dataset_path, data_transform, batch_size)

    net = VGG16Net().to('cuda')
    run_train(net, train_dataset, options)

if __name__ == '__main__':
    main()
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import argparse
import logging
import boto3
from botocore.exceptions import NoCredentialsError

parser = argparse.ArgumentParser()

parser.add_argument('--s3-client-models-folder', help='S3 folder for client models', required=True)
parser.add_argument('--s3-main-models-folder', help='S3 folder for main models', required=True)
parser.add_argument('--client-models', help='Comma-separated list of client models to average', required=True)
parser.add_argument('--main-model', help='S3 folder for main models', required=True)
parser.add_argument('--main-bucket', help='Bucket name for main models', required=True)
parser.add_argument('--clients-bucket', help='Bucket name for client models', required=True)
parser.add_argument('--local-models-folder', help='Local folder for client models', required=True)
parser.add_argument('--s3-access-key', help='Credentials for AWS', required=False)
parser.add_argument('--s3-secret-key', help='Credentials for AWS', required=False)
parser.add_argument('-d', '--debug', help="Debug mode for the script")

args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def welcome():
    msg = """



 _______  _______  _______  _______  ___   __    _  _______ 
|       ||       ||       ||       ||   | |  |  | ||       |
|_     _||    ___||  _____||_     _||   | |   |_| ||    ___|
  |   |  |   |___ | |_____   |   |  |   | |       ||   | __ 
  |   |  |    ___||_____  |  |   |  |   | |  _    ||   ||  |
  |   |  |   |___  _____| |  |   |  |   | | | |   ||   |_| |
  |___|  |_______||_______|  |___|  |___| |_|  |__||_______|




"""
    print(msg)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def download_from_aws(bucket, remote_path, local_path):
    logging.info("Downloading from S3 bucket")

    s3 = boto3.client('s3', aws_access_key_id=args.s3_access_key,
                      aws_secret_access_key=args.s3_secret_key)

    try:
        logging.debug("Bucket: " + bucket)
        logging.debug("Remote Path: " + remote_path)
        logging.debug("Local Path: " + local_path)
        s3.download_file(bucket, remote_path, local_path)
        logging.info("Download Successful")
        return True
    except FileNotFoundError:
        logging.error("The file was not found")
        return False
    except NoCredentialsError:
        logging.error("Credentials not available")
        return False


def test_model(path):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net()
    net.load_state_dict(torch.load(path).state_dict())

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


welcome()
print('Testing with final model')
print('_________________________________________________')
download_from_aws(args.main_bucket, args.s3_main_models_folder + "/" + args.main_model,
                  args.local_models_folder + "/" + args.main_model)
test_model(args.local_models_folder + "/" + args.main_model)
print('')
print('')
print('')
print('')

dir_items = args.client_models.split(',')

counter = 0
for item in dir_items:
    download_from_aws(args.main_bucket, args.s3_client_models_folder + "/" + item,
                      args.local_models_folder + "/" + item)
    print('Testing with client' + item + ' model')
    print('_________________________________________________')
    test_model(args.local_models_folder + '/' + item)
    print('')
    print('')
    print('')
    print('')
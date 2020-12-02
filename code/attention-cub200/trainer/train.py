import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
import os
import random
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
import shutil
from trainer.model import AttnVGG_before
from trainer.utilities import *


ROOT = 'data'
MODEL = 'model'
data_dir = os.path.join(ROOT, 'CUB_200_2011')
images_dir = os.path.join(data_dir, 'images')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
TRAIN_RATIO = 0.6


class Args:
    # outf = os.path.join(os.curdir, "logs")
    outf = "gs://zifei_bucket/logs"
    num_aug = 3
    num_classes = 200
    image_size = 112
    batch_size = 16
    epochs = 150
    lr = 0.1
    log_images = True
    normalize_attn = True
    no_attention = False


opt = Args()


def main():
    print('working in directory: ', os.curdir)

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    download_cub(ROOT)

    download_model(MODEL)

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(train_dir)
    os.makedirs(test_dir)

    classes = os.listdir(images_dir)

    for c in classes:

        class_dir = os.path.join(images_dir, c)

        images = os.listdir(class_dir)

        n_train = int(len(images) * TRAIN_RATIO)

        train_images = images[:n_train]
        test_images = images[n_train:]

        os.makedirs(os.path.join(train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(test_dir, c), exist_ok=True)

        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, c, image)
            shutil.copyfile(image_src, image_dst)

        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, c, image)
            shutil.copyfile(image_src, image_dst)

    print('\nloading the dataset ...\n')

    transform_train = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(opt.image_size, padding=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    trainset = torchvision.datasets.ImageFolder(
        root=train_dir, transform=transform_train)
    testset = torchvision.datasets.ImageFolder(
        root=test_dir, transform=transform_test)
    # trainset = torchvision.datasets.CIFAR100(
    #     root='data', train=True, download=True, transform=transform_train)
    # testset = torchvision.datasets.CIFAR100(
    #     root='data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=opt.batch_size)

    print(f'Number of training examples: {len(trainset)}')
    print(f'Number of testing examples: {len(testset)}')

    net = AttnVGG_before(
        attention=not opt.no_attention, normalize_attn=opt.normalize_attn, init='xavierUniform')

    criterion = nn.CrossEntropyLoss()
    # move to GPU
    print('\nmoving to GPU ...\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    criterion.to(device)
    print('GPU setup done')

    # Load pretrained model
    pretrained_model_path = os.path.join(MODEL, 'model_net.pth')
    if os.path.isfile(pretrained_model_path):
        print('found pretrained model...')
        model.load_state_dict(torch.load(pretrained_model_path))
        model.eval()

    IN_FEATURES = model.module.classify.in_features

    final_fc = nn.Linear(IN_FEATURES, opt.num_classes)
    model.module.classify = final_fc
    new_dense = nn.Conv2d(in_channels=model.module.dense.in_channels, out_channels=model.module.dense.out_channels,
                          kernel_size=int(opt.image_size/32), padding=model.module.dense.padding, bias=True)
    model.module.dense = new_dense
    model = model.cuda()
    print(model)

    # load network
    print('\nloading the network ...\n')
    # use attention module?
    if not opt.no_attention:
        print('\nturn on attention ...\n')
    else:
        print('\nturn off attention ...\n')

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)

    def lr_lambda(epoch): return np.power(0.5, int(epoch/25))
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # training
    print('\nstart training ...\n')
    step = 0
    running_avg_accuracy = 0
    writer = SummaryWriter(opt.outf)
    for epoch in range(opt.epochs):
        images_disp = []
        # adjust learning rate
        writer.add_scalar('train/learning_rate',
                          optimizer.param_groups[0]['lr'], epoch)
        print("\nepoch %d learning rate %f\n" %
              (epoch, optimizer.param_groups[0]['lr']))
        # run for one epoch
        for aug in range(opt.num_aug):
            for i, data in enumerate(trainloader, 0):
                # warm up
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                if (aug == 0) and (i == 0):  # archive images in order to save to logs
                    images_disp.append(inputs[0:36, :, :, :])
                # forward
                pred, __, __, __ = model(inputs)
                # backward
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                # display results
                if i % 10 == 0:
                    model.eval()
                    pred, __, __, __ = model(inputs)
                    predict = torch.argmax(pred, 1)
                    total = labels.size(0)
                    correct = torch.eq(predict, labels).sum().double().item()
                    accuracy = correct / total
                    running_avg_accuracy = 0.9*running_avg_accuracy + 0.1*accuracy
                    writer.add_scalar('train/loss', loss.item(), step)
                    writer.add_scalar('train/accuracy', accuracy, step)
                    writer.add_scalar(
                        'train/running_avg_accuracy', running_avg_accuracy, step)
                    print("[epoch %d][aug %d/%d][%d/%d] loss %.4f accuracy %.2f%% running avg accuracy %.2f%%"
                          % (epoch, aug, opt.num_aug-1, i, len(trainloader)-1, loss.item(), (100*accuracy), (100*running_avg_accuracy)))
                step += 1
        # the end of each epoch: test & log
        print('\none epoch done, saving records ...\n')
        torch.save(model.state_dict(), os.path.join(os.curdir, 'net.pth'))
        upload_blob('model/net.pth', 'net.pth')
        if epoch == opt.epochs / 2:
            torch.save(model.state_dict(), os.path.join(
                os.curdir, 'net%d.pth' % epoch))
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            # log scalars
            for i, data in enumerate(testloader, 0):
                images_test, labels_test = data
                images_test, labels_test = images_test.to(
                    device), labels_test.to(device)
                if i == 0:  # archive images in order to save to logs
                    images_disp.append(inputs[0:36, :, :, :])
                pred_test, __, __, __ = model(images_test)
                predict = torch.argmax(pred_test, 1)
                total += labels_test.size(0)
                correct += torch.eq(predict, labels_test).sum().double().item()
            writer.add_scalar('test/accuracy', correct/total, epoch)
            print("\n[epoch %d] accuracy on test data: %.2f%%\n" %
                  (epoch, 100*correct/total))
            # log images
            if opt.log_images:
                print('\nlog images ...\n')
                I_train = utils.make_grid(
                    images_disp[0], nrow=6, normalize=True, scale_each=True)
                writer.add_image('train/image', I_train, epoch)
                if epoch == 0:
                    I_test = utils.make_grid(
                        images_disp[1], nrow=6, normalize=True, scale_each=True)
                    writer.add_image('test/image', I_test, epoch)
            if opt.log_images and (not opt.no_attention):
                print('\nlog attention maps ...\n')
                # base factor
                min_up_factor = 1
                # sigmoid or softmax
                if opt.normalize_attn:
                    vis_fun = visualize_attn_softmax
                else:
                    vis_fun = visualize_attn_sigmoid
                # training data
                __, c1, c2, c3 = model(images_disp[0])
                if c1 is not None:
                    attn1 = vis_fun(
                        I_train, c1, up_factor=min_up_factor, nrow=6)
                    writer.add_image('train/attention_map_1', attn1, epoch)
                if c2 is not None:
                    attn2 = vis_fun(
                        I_train, c2, up_factor=min_up_factor*2, nrow=6)
                    writer.add_image('train/attention_map_2', attn2, epoch)
                if c3 is not None:
                    attn3 = vis_fun(
                        I_train, c3, up_factor=min_up_factor*4, nrow=6)
                    writer.add_image('train/attention_map_3', attn3, epoch)
                # test data
                __, c1, c2, c3 = model(images_disp[1])
                if c1 is not None:
                    attn1 = vis_fun(
                        I_test, c1, up_factor=min_up_factor, nrow=6)
                    writer.add_image('test/attention_map_1', attn1, epoch)
                if c2 is not None:
                    attn2 = vis_fun(
                        I_test, c2, up_factor=min_up_factor*2, nrow=6)
                    writer.add_image('test/attention_map_2', attn2, epoch)
                if c3 is not None:
                    attn3 = vis_fun(
                        I_test, c3, up_factor=min_up_factor*4, nrow=6)
                    writer.add_image('test/attention_map_3', attn3, epoch)
        scheduler.step()


if __name__ == "__main__":
    main()

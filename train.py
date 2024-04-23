import warnings
warnings.filterwarnings("ignore")
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch, time, datetime, tqdm
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致

from dataset import DataGenerator
from models.modeling import deeplabv3plus_resnet101

from math import cos, pi
from sklearn.metrics import confusion_matrix

def metrice(y_true, y_pred):
    y_true, y_pred = y_true.to('cpu').detach().numpy(), np.argmax(y_pred.to('cpu').detach().numpy(), axis=1)
    y_true, y_pred = y_true.reshape((-1)), y_pred.reshape((-1))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    pa = np.diag(cm).sum() / (cm.sum() + 1e-7)

    mpa_arr = np.diag(cm) / (cm.sum(axis=1) + 1e-7)
    mpa = np.nanmean(mpa_arr)

    return pa, mpa

def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=True):
    warmup_epoch = 10 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min) * (
                    1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    BATCH_SIZE = 2

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    model = deeplabv3plus_resnet101(num_classes=2, pretrained_backbone=True)
    model.backbone.conv1 = torch.nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)

    train_dataset = DataGenerator('train.txt')
    train_generator = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=1,pin_memory=True)


    test_dataset = DataGenerator('test.txt', valid=True)
    valid_generator = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, shuffle=True, num_workers=1,pin_memory=True)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.0005)
    loss = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.1, 1.0])).float()).to(DEVICE)



    with open('train.log', 'w+') as f:
        f.write('epoch,train_loss,test_loss,train_pa,test_pa,train_mpa,test_mpa')

    best_mpa = 0
    print('{} begin train!'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    for epoch in range(200):
        model.to(DEVICE)
        model.train()
        train_loss = 0
        begin = time.time()
        num = 0
        train_pa, train_mpa = 0, 0
        adjust_learning_rate(optimizer, epoch, 200, lr_min=0.001, lr_max=0.001)
        for x, y in train_generator:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            pred = model(x.float())

            l = loss(pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss += float(l.data)
            train_pa_, train_mpa_ = metrice(y, pred)
            train_pa += train_pa_
            train_mpa += train_mpa_
            num += 1
        train_loss /= num
        train_pa, train_mpa = train_pa / num, train_mpa / num

        num = 0
        test_loss = 0
        model.eval()
        test_pa, test_mpa = 0, 0
        with torch.no_grad():
            for x, y in valid_generator:
                x, y = x.to(DEVICE), y.to(DEVICE).long()

                pred = model(x.float())
                l = loss(pred, y)
                num += 1
                test_loss += float(l.data)

                test_pa_, test_mpa_ = metrice(y, pred)
                test_pa += test_pa_
                test_mpa += test_mpa_

        test_loss /= num
        test_pa, test_mpa = test_pa / num, test_mpa / num
        if test_mpa > best_mpa:
            best_mpa = test_mpa
            model.to('cpu')
            torch.save(model, 'model.pkl')
            print('BestModel Save Success!')
        print(
            '{} epoch:{}, time:{:.2f}s, lr:{:.6f}, train_loss:{:.4f}, val_loss:{:.4f}, train_pa:{:.4f}, val_pa:{:.4f}, train_mpa:{:.4f}, test_mpa:{:.4f}'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch + 1, time.time() - begin, optimizer.state_dict()['param_groups'][0]['lr'], train_loss, test_loss, train_pa, test_pa, train_mpa, test_mpa
            ))
        with open('train.log', 'a+') as f:
            f.write('\n{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                epoch, train_loss, test_loss, train_pa, test_pa, train_mpa, test_mpa
            ))

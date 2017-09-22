import torch
from torch.autograd import Variable
import argparse
import copy
import pickle
from Datahelper2 import *
from Model import *
from gloable_parameter import *


from hyperboard import Agent


# train 36431  32384
# validate 4048  8095
# test 61191

def train(model_path,train_batch_size,validate_batch_size,validate_batch_num,resize,train_gpu,validate_gpu=-1):

    # train_gpu = 0
    # validate_gpu = 1
    # model_path = '../amazon2/alexnet'
    # train_batch_size = 256
    # validate_batch_size = 128
    # validate_batch_num = 8

    # parameters

    k=5
    epochs = 1
    lr = 1e-4
    weight_decay = 0
    momentum = 0.9


    criteria2metric = {
        'train loss': 'loss',
        'valid loss': 'loss'
    }
    hyperparameters_train = {
        'name':'train',
        'learning rate': lr,
        'batch size': train_batch_size,
        'optimizer': 'Adam',
        'momentum': 0,
        'net':model_path.split('/')[-1],
        'epoch':'No.1',
    }
    hyperparameters_validate = {
        'name':'validate',
        'learning rate': lr,
        'batch size': train_batch_size,
        'optimizer': 'Adam',
        'momentum': 0,
        'net':model_path.split('/')[-1],
        'epoch': 'No.1',
    }


    agent = Agent(username='jlb',password='1993610')
    train_loss_show = agent.register(hyperparameters_train, criteria2metric['train loss'])
    validate_loss_show = agent.register(hyperparameters_validate, criteria2metric['valid loss'])
    global_step = 0

    with open('kdf.pkl', 'rb') as f:
        kfold = pickle.load(f,encoding='latin1')

    loss_info = []    # 第i个记录了 fold i 的最小(train_loss,validate_loss)
    for fold in range(k):
        train_index = kfold[fold][0]
        validate_index = kfold[fold][1]


        model = AM_alex()
        if model.getname()!=model_path.split('/')[-1]:
            print('Wrong Model!')
            return
        model.cuda(device_id=train_gpu)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

        dset_train = AmazonDateset_train(train_index,IMG_TRAIN_PATH,IMG_EXT,LABEL_PATH,resize=resize)
        train_loader = DataLoader(dset_train, batch_size=train_batch_size, shuffle=True, num_workers=6)
        min_loss = [0.9,0.9]
        for epoch in range(epochs):
            print('--------------Epoch %d: train-----------' % epoch)
            model.train()
            for step, (data, target) in enumerate(train_loader):
                data, target = Variable(data), Variable(target)
                data = data.cuda(device_id=train_gpu)
                target = target.cuda(device_id=train_gpu)

                optimizer.zero_grad()
                output = model(data)
                # print(output.size())
                loss = F.binary_cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                agent.append(train_loss_show, global_step, loss.data[0])
                global_step += 1
                if step % 10 == 0:
                    model.eval()
                    if validate_gpu != -1:
                        model.cuda(validate_gpu)
                    dset_validate = AmazonDateset_validate(validate_index, IMG_TRAIN_PATH, IMG_EXT, LABEL_PATH,random_transform=True,resize=resize)
                    validate_loader = DataLoader(dset_validate, batch_size=validate_batch_size, shuffle=True, num_workers=6)
                    total_vloss = 0
                    for vstep, (vdata, vtarget) in enumerate(validate_loader):
                        vdata, vtarget = Variable(vdata), Variable(vtarget)
                        if validate_gpu != -1:
                            vdata = vdata.cuda(validate_gpu)
                            vtarget = vtarget.cuda(validate_gpu)
                        else:
                            vdata = vdata.cuda(train_gpu)
                            vtarget = vtarget.cuda(train_gpu)

                        voutput = model(vdata)
                        vloss = F.binary_cross_entropy(voutput, vtarget)
                        total_vloss += vloss.data[0]
                        if vstep == (validate_batch_num-1):
                            break
                    vloss = total_vloss / validate_batch_num
                    model.train()
                    if validate_gpu != -1:
                        model.cuda(train_gpu)

                    agent.append(validate_loss_show, global_step, vloss)

                    print('{} Fold{} Epoch{} Step{}: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tValidate Loss: {:.6f}'.format(model_path.split('/')[-1],fold, epoch,global_step, step * train_batch_size,
                                                                                   len(train_loader.dataset),
                                                                                   100. * step / len(train_loader),
                                                                                   loss.data[0],vloss))
                    if vloss<min_loss[1]:
                        min_loss[1] = vloss
                        min_loss[0] = loss.data[0]
        model_save = copy.deepcopy(model)
        torch.save(model_save.cpu(), os.path.join(model_path,'fold%d.mod'%(fold)))
        loss_info.append(min_loss)


    print('-----------------------------------------')
    print(model_path.split('/')[-1]+':')
    for i,l in enumerate(loss_info):
        print('Fold%d: Train loss:%f\tValidate loss:%f'%(i,l[0],l[1]))

    with open(os.path.join(model_path,'train_loss_info.pkl'),'wb') as f:
        pickle.dump(loss_info,f)







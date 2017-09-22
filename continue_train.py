import torch
from torch.autograd import Variable
import argparse
import copy
import pickle
from Datahelper2 import *
from Model import *
from gloable_parameter import *


# from hyperboard import Agent

# torch.backends.cudnn.benchmark=True


# train 36431  32384
# validate 4048  8095
# test 61191

def continue_train(model_path,epoch,lr,train_batch_size,validate_batch_size,validate_batch_num,resize,train_gpu,validate_gpu=-1):


    k=5
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
        'epoch': 'No.%d'%epoch,
    }
    hyperparameters_validate = {
        'name':'validate',
        'learning rate': lr,
        'batch size': train_batch_size,
        'optimizer': 'Adam',
        'momentum': 0,
        'net':model_path.split('/')[-1],
        'epoch': 'No.%d'%epoch,
    }


    # agent = Agent(username='jlb',password='1993610')
    # train_loss_show = agent.register(hyperparameters_train, criteria2metric['train loss'])
    # validate_loss_show = agent.register(hyperparameters_validate, criteria2metric['valid loss'])
    global_step = 0


    with open('kdf.pkl', 'rb') as f:
        kfold = pickle.load(f,encoding='latin1')


    lr = lr*train_batch_size/32
    for fold in range(k):
        train_index = kfold[fold][0]
        validate_index = kfold[fold][1]


        model = torch.load(os.path.join(model_path,'fold%d.mod'%fold))
        model.cuda(device_id=train_gpu)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

        dset_train = AmazonDateset_train(train_index,IMG_TRAIN_PATH,IMG_EXT,LABEL_PATH,resize=resize)
        train_loader = DataLoader(dset_train, batch_size=train_batch_size, shuffle=True, num_workers=4)
        print('--------------Epoch %d: train-----------' % epoch)
        model.train()
        for step, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            data = data.cuda(device_id=train_gpu)
            target = target.cuda(device_id=train_gpu)

            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            # agent.append(train_loss_show, global_step, loss.data[0])
            global_step += 1
            if step % 10 == 0:
                # model.eval()
                # if validate_gpu!=-1:
                #     model.cuda(validate_gpu)
                # dset_validate = AmazonDateset_validate(validate_index, IMG_TRAIN_PATH, IMG_EXT, LABEL_PATH,random_transform=True,resize=resize)
                # validate_loader = DataLoader(dset_validate, batch_size=validate_batch_size, shuffle=True, num_workers=4)
                # total_vloss = 0
                # for vstep, (vdata, vtarget) in enumerate(validate_loader):
                #     vdata, vtarget = Variable(vdata,volatile=True), Variable(vtarget,volatile=True)
                #     if validate_gpu!=-1:
                #         vdata = vdata.cuda(validate_gpu)
                #         vtarget = vtarget.cuda(validate_gpu)
                #     else:
                #         vdata = vdata.cuda(train_gpu)
                #         vtarget = vtarget.cuda(train_gpu)
                #
                #     voutput = model(vdata)
                #     vloss = F.binary_cross_entropy(voutput, vtarget)
                #     total_vloss += vloss.data[0]
                #     if vstep == (validate_batch_num-1):
                #         break
                # vloss = total_vloss / validate_batch_num
                # model.train()
                # if validate_gpu!=-1:
                #     model.cuda(train_gpu)

                # agent.append(validate_loss_show, global_step, vloss)
                vloss = 0

                print('{} Fold{} Epoch{} Step{}: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tValidate Loss: {:.6f}'.format(model_path.split('/')[-1],fold, epoch,global_step, step * train_batch_size,
                                                                               len(train_loader.dataset),
                                                                               100. * step / len(train_loader),
                                                                               loss.data[0],vloss))

        model_save = copy.deepcopy(model)
        torch.save(model_save.cpu(), os.path.join(model_path,'fold%d.mod'%(fold)))


    print('-----------------------------------------')
    print(model_path.split('/')[-1]+' Finished!')








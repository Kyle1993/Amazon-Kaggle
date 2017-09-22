from train import *
from continue_train import *
from eval import *
from predict import *
from train_2epoch import *
from train_3epoch import *

# model_path = '../amazon2/inception'
# train_gpu = 0
# validate_gpu = -1
# train_batch_size = 128
# validate_batch_size = 128
# validate_batch_num = 8
# resize = 224

# train(model_path,train_batch_size,validate_batch_size,validate_batch_num,resize,train_gpu,validate_gpu)
# continue_train(model_path,train_batch_size,validate_batch_size,validate_batch_num,train_gpu,validate_gpu)
# eval(model_path,train_batch_size,train_gpu)
# predict(model_path,train_batch_size,train_gpu)

# predict('../amazon2/resnet34',128,0)
# train_2epoch('../amazon2/pynet10',1e-4,24,8,8,256,0,1)
# eval('../amazon2/densenet161',24,3)
# predict('../amazon2/densenet161',24,3)
# train_2epoch('../amazon2/inception',1e-6,32,8,8,299,0,1)    #16 5g
# predict('../amazon2/_alexnet',128,2)
# eval('../amazon2/pynet10',14,256,0)
# predict('../amazon2/pynet10',14,256,0)

# train_2epoch('../amazon2/resnet101',1e-4,64,8,8,256,3,2)
# train_2epoch('../amazon2/squeezenet11',1e-4,256,128,4,224,0)
# eval('../amazon2/squeezenet11',256,224,0)
# predict('../amazon2/squeezenet11',256,224,0)
# eval('../amazon2/resnet101',32,256,3)
# predict('../amazon2/resnet101',32,256,3)
# continue_train('../amazon2/squeezenet11',3,1e-4/(50*10),256,128,4,224,0,1)
# eval('../amazon2/squeezenet11',128,224,1)
# predict('../amazon2/squeezenet11',128,224,1)
# train_2epoch('../amazon2/vgg13',1e-4,64,4,16,224,0,2)
# train_2epoch('../amazon2/resnet34',1e-4,128,1,1,256,0)
# eval('../amazon2/resnet34',32,256,2)
# predict('../amazon2/resnet34',32,256,2)
# train_2epoch('../amazon2/resnet152',1e-4,32,0,0,256,0)

# train_2epoch('../amazon2/densenet161',1e-4,32,32,2,256,3)
# train_2epoch('../amazon2/resnet50',1e-4,28,32,8,256,2)

# eval('../amazon2/resnet152',128,256,0)
# predict('../amazon2/resnet152',128,256,0)
# eval('../amazon2/resnet50',128,256,0)
# predict('../amazon2/resnet50',128,256,0)
# eval('../amazon2/densenet161',256,256,0)
# predict('../amazon2/densenet161',256,256,0)
# train_2epoch('../amazon2/resnet18',1e-4,64,64,4,256,1)
# eval('../amazon2/resnet18',64,256,1)'../amazon2/densenet161_new'
# predict('../amazon2/resnet18',64,256,1)
# train_2epoch('../amazon2/densenet161_new',1e-4,32,32,8,256,3)
# eval('../amazon2/densenet161_new',64,256,3)
# predict('../amazon2/densenet161_new',256,256,3)
# continue_train('../amazon2/densenet161_new',1,(1e-4/500),32,32,8,256,3)
# eval('../amazon2/densenet161_new',128,256,3)
# predict('../amazon2/densenet161_new',128,256,2)
# train_2epoch('../amazon2/densenet201_new',1e-4,32,32,4,256,1)
# predict('../amazon2/resnet18',64,256,2)
# eval('../amazon2/densenet201_new',128,256,2)
# predict('../amazon2/densenet201_new',128,256,2)
# continue_train('../amazon2/densenet201_3epoch',1,(1e-4/500),32,32,8,256,1)
# eval('../amazon2/densenet201_3epoch',128,256,1)
# predict('../amazon2/densenet201_3epoch',128,256,1)

# train_3epoch('../amazon2/resnet34_new',1e-4,128,64,8,256,1)
# eval('../amazon2/resnet34_new',128,256,1)
# predict('../amazon2/resnet34_new',128,256,1)

# train_3epoch('../amazon2/resnet50_new',1e-4,64,32,8,256,0)
# eval('../amazon2/resnet50_new',128,256,0)
# predict('../amazon2/resnet50_new',128,256,0)

# train_2epoch('../amazon2/resnet18_new',1e-4,256,128,8,256,0)
# eval('../amazon2/resnet18_new',256,256,0)
# predict('../amazon2/resnet18_new',256,256,0)

# predict('../amazon2/resnet101_new',256,256,3)

train_2epoch('../amazon2/vgg11',1e-4,32,32,1,224,2)
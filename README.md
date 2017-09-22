# 项目说明
## 任务    
Kaggle比赛： https://www.kaggle.com/c/planet-understanding-the-amazon-from-space  
图像多标签多分类问题，图像为亚马逊丛林高空俯视卫星图片，图像标签为图片中包含的气候、地貌（多云、河流、农田、矿井等17类），训练集4万，测试集6万。  
知乎文章：   

## 运行环境  
* 包依赖：torch, torchvision, PIL, pandas, (hyperboard)  
* OS: linux  
* 注意事项：  
1.项目文件夹下不包含数据  
2.省略了很多预处理  
3.不含ensemble部分代码  
4.训练时要新建与模型名字一样的文件夹 ，文件夹下保存着这个模型生成的所有文件  
5.可以使用hyperboard，不过在代码里已经被注释了  
  
## 主要py文件说明：  
 __Model.py__: 记录了使用的各种模型，及其修改版本  
 __Datahelper2.py__: 数据加载器，其中lables.pkl是预先处理好的label  
 __process.py__: 统一执行环境，其实只是把各个模块封装好放到这里执行  
 __train.py__: 训练模型，生成每个fold对应的模型  
 __train_2epoch.py__ : 训练两个epoch  
__eval.py__: 评估模型，保存中间变量，生成validation，best_threshold  
__predict.py__: 对数据进行预测，保存中间变量，生成预测文件*_result  
__process_data__: 将模型生成的中间变量保存成npy格式的概率矩阵  
__bagging.py__: 模型bagging  
  
## 主要数据文件说明：  
__kdf.pkl__: kfold的划分  
__kfold.pkl__: 另一种kfold的划分  
__labels.pkl__: 预处理的labels  
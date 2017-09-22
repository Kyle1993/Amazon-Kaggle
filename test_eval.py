import pickle
import os



model_path = '../amazon2/resnet101'
with open(os.path.join(model_path,model_path.split('/')[-1]+'_test_file'),'w') as f:
    f.write('adsas')
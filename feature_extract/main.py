import argparse
import os
import shutil
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from dataset import CUB_data
from model import inception_model

_IMAGE_ROOT_ = '../data/images/'
_IMAGE_LIST_ = '../data/images.txt'
_PKL_ROOT_ = '/home/data/CUB-FINE-GRAINED/inception_pkl/'
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
args = parser.parse_args()

def main():
    extract_data = CUB_data(_IMAGE_ROOT_, _IMAGE_LIST_,
                                  transforms.Compose([
                                  transforms.Resize((299,299)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))
    extract_loader = data.DataLoader(extract_data,
                                   batch_size=32,
                                   shuffle=False,
                                   num_workers=4)

    network = inception_model()
    network.cuda()

    extractor(extract_loader, network)
    '''
    for i in total_feature.keys():
        dir_path = i.split('/')[0]
        if not os.path.exists(_PKL_ROOT_ + dir_path):
            os.makedirs(_PKL_ROOT_ + dir_path)
        pickle.dump(total_feature[i], open(_PKL_ROOT_ + i[:-3] + 'pkl', 'wb'))
        '''
    #pickle.dump(total_feature, open('./4_19_data/'+args.load_model+'.pkl','wb'))


def extractor(eval_loader, model):
    model.eval()
    #total_result = {}
    count = 0
    for i, (input, target, image_name) in enumerate(eval_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        for j in range(len(image_name)):
            count += 1
            real_name = image_name[j][len(_IMAGE_ROOT_):]
            dir_path = real_name.split('/')[0]
            if not os.path.exists(_PKL_ROOT_ + dir_path):
                os.makedirs(_PKL_ROOT_ + dir_path)
            pickle.dump(output[1].data.cpu().numpy()[j,:], open(_PKL_ROOT_ + real_name[:-3] + 'pkl', 'wb'))
            #total_result[real_name] = output[1].data.cpu().numpy()[j,:]
            if count % 1000 == 0:
                print('image_feature_extract_process: ' + str(count))
    #return total_result

if __name__ == '__main__':
    main()
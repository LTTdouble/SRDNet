import os
import numpy as np
import torch
from os import listdir
import torch.nn as nn
from torch.autograd import Variable
from option import  opt
from data_utils import is_image_file
from architecture.SRDNet import SRDNet
import torch.nn.init as init
import scipy.io as scio  
from eval import PSNR, SSIM, SAM,compare_corr,compare_rmse,compare_ergas
# from architecture import *

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    wn = lambda x: torch.nn.utils.weight_norm(x)
    return wn(nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias))
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
def main( ):
    input_path = 'dataset/tests/' + opt.datasetName + '/' + str(opt.upscale_factor) + '/'
    out_path = 'result/' + opt.datasetName + '/' + str(opt.upscale_factor) + '/' + opt.method + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    PSNRs = []
    SSIMs = []
    SAMs = []
    CORs = []
    RMSEs=[]
    ERGASs=[]

    if not os.path.exists(out_path):
        os.makedirs(out_path)
                    
    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    model = SRDNet(opt)
        if opt.cuda:
     # model = nn.DataParallel(model).cuda()
       model = model.cuda()

    checkpoint  = torch.load(opt.model_name)
    model.load_state_dict(checkpoint["model"])
    nearest = nn.Upsample(scale_factor=opt.upscale_factor, mode='nearest')
    model.eval()
    # test_number = 0
    images_name = [x for x in listdir(input_path) if is_image_file(x)]           
    with torch.no_grad():
        for index in range(len(images_name)):

            mat = scio.loadmat(input_path + images_name[index])
            hyperLR = mat['LR'].transpose(2,0,1).astype(np.float32)
            # hyperLR = mat['lr'].astype(np.float32)
            input = Variable(torch.from_numpy(hyperLR).float(), volatile=True).contiguous().view(1, -1, hyperLR.shape[1], hyperLR.shape[2])
            if opt.cuda:
               input = input.cuda()
            lms = nearest(input)
            output = model(input)
            HR = mat['HR'].transpose(2,0,1).astype(np.float32)

            # print(input.shape)
            # print(HR.shape)
            # HR = mat['hr'].astype(np.float32)
            SR = output.cpu().data[0].numpy().astype(np.float32)
            SR[SR<0] = 0
            SR[SR>1.] = 1.
            psnr = PSNR(SR, HR)
            ssim = SSIM(SR, HR)
            sam = SAM(SR, HR)
            cor=compare_corr(SR, HR)
            rmse=compare_rmse(SR, HR)
            ergas=compare_ergas(SR, HR,4)

            PSNRs.append(psnr)
            SSIMs.append(ssim)
            SAMs.append(sam)
            CORs.append(cor)
            RMSEs.append(rmse)
            ERGASs.append(ergas)

            SR = SR.transpose(1,2,0)
            HR = HR.transpose(1,2,0)
            torch.cuda.empty_cache()

            scio.savemat(out_path + images_name[index], {'HR': HR, 'SR':SR})

            print("===The {}-th picture=====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}=====COR:{:.3f}=====RMSE:{:.3f}=====ERGAS:{:.3f}====Name:{}".format(index+1,  psnr, ssim, sam,cor,rmse,ergas,images_name[index]))
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}=====averCOR:{:.3f}=====averRMSE:{:.3f}=====averERGAS:{:.3f}".format(np.mean(PSNRs), np.mean(SSIMs), np.mean(SAMs),np.mean(CORs),np.mean(RMSEs),np.mean(ERGASs)))


def sum_dict(a, b):
    temp = dict()
    for key in a.keys() | b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

if __name__ == "__main__":
    main()

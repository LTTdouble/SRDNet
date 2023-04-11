import math
import numpy as np
from scipy.signal import convolve2d


def PSNR(pred, gt):
      
    valid = gt - pred
    rmse = math.sqrt(np.mean(valid ** 2))   
    
    if rmse == 0:
        return 100
    psnr = 20 * math.log10(1.0 / rmse)
    return psnr

def SSIM(pred, gt):
    ssim = 0
    for i in range(gt.shape[0]):
        ssim = ssim + compute_ssim(pred[i,:,:], gt[i,:,:])
    return ssim /gt.shape[0]    
    	
    
def SAM(pred, gt):

    eps = 2.2204e-16
    pred[np.where(pred==0)] = eps
    gt[np.where(gt==0)] = eps 
      
    nom = sum(pred*gt)
    denom1 = sum(pred*pred)**0.5
    denom2 = sum(gt*gt)**0.5
    sam = np.real(np.arccos(nom.astype(np.float32)/(denom1*denom2+eps)))
    sam[np.isnan(sam)]=0     
    sam_sum = np.mean(sam)*180/np.pi   	       
       	       
    return  sam_sum



def matlab_style_gauss2D(shape=np.array([11,11]),sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    siz = (shape-np.array([1,1]))/2
    std = sigma
    eps = 2.2204e-16
    x = np.arange(-siz[1], siz[1]+1, 1)
    y = np.arange(-siz[0], siz[1]+1, 1)
    m,n = np.meshgrid(x, y)
    
    h = np.exp(-(m*m + n*n).astype(np.float32) / (2.*sigma*sigma))    
    h[ h < eps*h.max() ] = 0    	
    sumh = h.sum()   	

    if sumh != 0:
        h = h.astype(np.float32) / sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
 
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):
 
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
 
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=np.array([win_size,win_size]), sigma=1.5)
    window = window.astype(np.float32)/np.sum(np.sum(window))
 
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
 
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)).astype(np.float32) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
 
    return np.mean(np.mean(ssim_map))


def compare_corr( x_pred,x_true):
    """
    Calculate the cross correlation between x_pred and x_true.
    求对应波段的相关系数，然后取均值
    CC is a spatial measure.
    """
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    x_true = x_true - np.mean(x_true, axis=1).reshape(-1, 1)
    x_pred = x_pred - np.mean(x_pred, axis=1).reshape(-1, 1)
    numerator = np.sum(x_true * x_pred, axis=1).reshape(-1, 1)
    denominator = np.sqrt(np.sum(x_true * x_true, axis=1) * np.sum(x_pred * x_pred, axis=1)).reshape(-1, 1)
    return (numerator / denominator).mean()

def img_2d_mat(x_true, x_pred):
    """
    # 将三维的多光谱图像转为2位矩阵
    :param x_true: (H, W, C)
    :param x_pred: (H, W, C)
    :return: a matrix which shape is (C, H * W)
    """
    h, w, c = x_true.shape
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    x_mat = np.zeros((c, h * w), dtype=np.float32)
    y_mat = np.zeros((c, h * w), dtype=np.float32)
    for i in range(c):
        x_mat[i] = x_true[:, :, i].reshape((1, -1))
        y_mat[i] = x_pred[:, :, i].reshape((1, -1))
    return x_mat, y_mat

def compare_rmse( x_pred,x_true):
    """
    Calculate Root mean squared error
    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    return np.linalg.norm(x_true - x_pred) / (np.sqrt(x_true.shape[0] * x_true.shape[1] * x_true.shape[2]))

def compare_ergas( x_pred,x_true, ratio):
    """
    Calculate ERGAS, ERGAS offers a global indication of the quality of fused image.The ideal value is 0.
    :param x_true:
    :param x_pred:
    :param ratio: 上采样系数
    :return:
    """

    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    sum_ergas = 0
    for i in range(x_true.shape[0]):
        vec_x = x_true[i]
        vec_y = x_pred[i]
        err = vec_x - vec_y
        r_mse = np.mean(np.power(err, 2))
        tmp = r_mse / (np.mean(vec_x)**2)
        sum_ergas += tmp
    return (100 / ratio) * np.sqrt(sum_ergas / x_true.shape[0])



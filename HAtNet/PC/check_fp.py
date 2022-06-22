from __future__ import division
import numpy as np

import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


## Recover {0,1} BIBD code for an User given his correlation with orthogonal basis matrix
def extract_BIBD_code(corr_score):
    recovered_coeff = np.sign(corr_score)         # close to {+1, -1}
    int_coeff = np.rint(recovered_coeff)
    recovered_coeff =  (int_coeff+1)/2            # change back to {0, 1}
    return recovered_coeff.astype(int)

def check_FP(model, layer):
    if layer == "FC" and model=="mnist":
        orthonormal_mats = np.loadtxt(f'{model}/{layer}_layer/orthonormalB.txt', delimiter=',' )
        # print('orthonormal_mats shape = ', orthonormal_mats.shape)
        projX = np.loadtxt(f'{model}/{layer}_layer/sharedProjectionMatrix.txt', delimiter=',')
        # print('projX shape: ', projX.shape)
        BIBD_AND_ACC = np.loadtxt(f'{model}/{layer}_layer/BIBD_AND_ACC_31_6_1.csv', dtype=int, delimiter=",")  # elements:{0,1}

        code_len = 31
        embed_dim = 512   ## const 
        i = 0             ## fp id 
        marked_FC_weights = np.load(f'invalid/{model}_marked{layer}_weights.npy')
        print('marked_FC_weights shape = ', marked_FC_weights.shape)

        marked_FC_weights_avg = np.mean(marked_FC_weights, axis=-1)
        W_flatten = np.reshape(marked_FC_weights_avg, (embed_dim, 1) )

        marked_Xw = np.dot(projX, W_flatten)        #original dim: <N=288, 1>
        marked_user_corr = np.dot(marked_Xw.transpose(), orthonormal_mats) 
        recovered_user_BIBD_code = extract_BIBD_code(marked_user_corr)
        user_true_code_vec = BIBD_AND_ACC[:,i]
        code_match = (recovered_user_BIBD_code==user_true_code_vec)*1
        BER = 1 - np.sum(code_match)/code_len
        print('BER = ', BER)
            
    elif layer == "conv" and model=="mnist":
        orthonormal_mats = np.loadtxt(f'{model}/{layer}_layer/orthonormal_B_Dim31_fnEpoch20.txt', delimiter=',' )
        # print('orthonormal_mats shape = ', orthonormal_mats.shape)
        projX = np.loadtxt(f'{model}/{layer}_layer/sharedProjectionMatrix_Wdim288_codeLen31_fnEpoch20.txt', delimiter=',')
        # print('projX shape: ', projX.shape)
        BIBD_AND_ACC = np.loadtxt(f'{model}/{layer}_layer/BIBD_AND_ACC_31_6_1.csv', dtype=int, delimiter=",")  # elements:{0,1}

        code_len = 31
        embed_dim = 288   #const 
        i = 0
        num_tests = 10000
        t1 = time.time()
        for cnt in range(num_tests):
            marked_conv_weights_4D = np.load(f'{model}/{layer}_layer/markedWeights4D_num_users31_UserIdx'+str(i)+'_Codelen31_scale0.5_fnEpoch20.npy')  
            marked_conv_weights_3D = np.mean(marked_conv_weights_4D, axis=-1)
            W_flatten = np.reshape(marked_conv_weights_3D, (embed_dim, 1) )
            Marked_Xw = np.dot(projX, W_flatten)         
            Marked_user_corr = np.dot(Marked_Xw.transpose(), orthonormal_mats)
            recovered_user_BIBD_code = extract_BIBD_code(Marked_user_corr)
            user_true_code_vec = BIBD_AND_ACC[:,i]
            code_match = (recovered_user_BIBD_code==user_true_code_vec)*1
            # print(' BER = ', 1 - np.sum(code_match)/code_len)
        t2 = time.time()
        t_avg = (t2-t1)/num_tests
        print('avg time = {} ms '.format(t_avg*1000))
        
    elif layer == "conv" and model == "cifar10":
        # print('Load orthonormal basis matrix')
        orthonormal_mats = np.loadtxt(f'{model}/{laer}_layer/orthonormal_B_Dim31.txt', delimiter=',' )
        # print('orthonormal_mats shape = ', orthonormal_mats.shape)
        projX = np.loadtxt(f'{model}/{layer}_layer/sharedProjectionMatrix_Wdim576_codeLen31.txt', delimiter=',')
        # print('projX shape: ', projX.shape)
        BIBD_AND_ACC = np.loadtxt(f'{model}/{layer}_layer/BIBD_AND_ACC_31_6_1.csv', dtype=int, delimiter=",")  # elements:{0,1}

        code_len = 31
        embed_dim = 576   #const, 3*3*64*64 
        i = 0
        marked_conv_weights_4D = np.load(f'{model}/{layer}_layer/markedWeights4D_num_users31_UserIdx0_Codelen31_scale0.1.npy')  
        print(f'{model}/{layer}_layer/marked_conv_weights_4D shape = ', marked_conv_weights_4D.shape)

        num_tests = 10000
        t1 = time.time()
        for cnt in range(num_tests):
            # print('marked_conv_weights_4D shape = ', marked_conv_weights_4D.shape)
            marked_conv_weights_3D = np.mean(marked_conv_weights_4D, axis=-1)
            W_flatten = np.reshape(marked_conv_weights_3D, (embed_dim, 1) )
            Marked_Xw = np.dot(projX, W_flatten)         
            Marked_user_corr = np.dot(Marked_Xw.transpose(), orthonormal_mats)
            recovered_user_BIBD_code = extract_BIBD_code(Marked_user_corr)
            user_true_code_vec = BIBD_AND_ACC[:,i]
            code_match = (recovered_user_BIBD_code==user_true_code_vec)*1
            # print(' BER = ', 1 - np.sum(code_match)/code_len)
        t2 = time.time()
        t_avg = (t2-t1)/num_tests
        print('avg time = {} ms '.format(t_avg*1000))
    else:
        print("layer or model not supported")



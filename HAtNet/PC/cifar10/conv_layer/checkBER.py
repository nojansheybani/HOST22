## Huili Chen, 03-02-2018
## Prove robustness of finerprints embedding
##  Updated: 2018-03-11  
from __future__ import division
import numpy as np
import pandas as pd

import time
import random

## Recover {0,1} BIBD code for an User given his correlation with orthogonal basis matrix
def extract_BIBD_code(corr_score):
    recovered_coeff = np.sign(corr_score)         # close to {+1, -1}
    int_coeff = np.rint(recovered_coeff)
    recovered_coeff =  (int_coeff+1)/2            # change back to {0, 1}
    return recovered_coeff.astype(int)


def check_FP():
    # print('Load orthonormal basis matrix')
    orthonormal_mats = np.loadtxt('orthonormal_B_Dim31.txt', delimiter=',' )
    # print('orthonormal_mats shape = ', orthonormal_mats.shape)
    projX = np.loadtxt('sharedProjectionMatrix_Wdim576_codeLen31.txt', delimiter=',')
    # print('projX shape: ', projX.shape)
    BIBD_AND_ACC = np.loadtxt('BIBD_AND_ACC_31_6_1.csv', dtype=int, delimiter=",")  # elements:{0,1}
    
    code_len = 31
    embed_dim = 576   #const, 3*3*64*64 
    i = 0
    marked_conv_weights_4D = np.load('markedWeights4D_num_users31_UserIdx0_Codelen31_scale0.1.npy')  
    print('marked_conv_weights_4D shape = ', marked_conv_weights_4D.shape)

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
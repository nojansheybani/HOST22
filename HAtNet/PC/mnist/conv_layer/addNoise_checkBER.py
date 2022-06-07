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


if __name__ == '__main__':
    # print('Load orthonormal basis matrix')
    orthonormal_mats = np.loadtxt('orthonormal_B_Dim31_fnEpoch20.txt', delimiter=',' )
    # print('orthonormal_mats shape = ', orthonormal_mats.shape)
    projX = np.loadtxt('sharedProjectionMatrix_Wdim288_codeLen31_fnEpoch20.txt', delimiter=',')
    # print('projX shape: ', projX.shape)
    BIBD_AND_ACC = np.loadtxt('BIBD_AND_ACC_31_6_1.csv', dtype=int, delimiter=",")  # elements:{0,1}
    
    code_len = 31
    embed_dim = 288   #const 
    i = 0
    marked_conv_weights_4D = np.load('markedWeights4D_num_users31_UserIdx'+str(i)+'_Codelen31_scale0.5_fnEpoch20.npy')  
    print('marked_conv_weights_4D shape = ', marked_conv_weights_4D.shape)
    noise_range = np.arange(0, 1, 0.01) ## granularity=100
    noise_std = np.arange(0, 10, 0.1)

    # noise_range = np.arange(0, 1, 0.05)  ## granularity = 20
    # noise_std = np.arange(0, 10, 0.5)

    num_tests = 10
    noise_mean = 0.

    Z = np.zeros((noise_range.size, noise_std.size))
    for cnt_x, x in enumerate(noise_range):
        for cnt_y, y in enumerate(noise_std):
            avg_BER = 0.
            for n in range(num_tests):
                noise_mask = np.zeros((1, marked_conv_weights_4D.size))
                idx = np.random.choice(marked_conv_weights_4D.size, int(marked_conv_weights_4D.size*x), replace=False)

                # --- assign random pertubation to mask ----#
                noise_val = np.random.normal(noise_mean, y, idx.size)
                noise_mask[0, idx] = noise_val
                noise_mask = np.reshape(noise_mask, marked_conv_weights_4D.shape)
                # print('noise_mask = ', noise_mask)

                W_marked_noised = marked_conv_weights_4D + noise_mask
                marked_FC_weights_avg = np.mean(W_marked_noised, axis=-1)
                W_flatten = np.reshape(marked_FC_weights_avg, (-1, 1) )
                marked_Xw = np.dot(projX, W_flatten)        #original dim: <N=288, 1>
                marked_user_corr = np.dot(marked_Xw.transpose(), orthonormal_mats) 
                recovered_user_BIBD_code = extract_BIBD_code(marked_user_corr)
                user_true_code_vec = BIBD_AND_ACC[:,i]
                code_match = (recovered_user_BIBD_code==user_true_code_vec)*1
                BER = 1 - np.sum(code_match)/code_len
                avg_BER += BER

            avg_BER = avg_BER/num_tests
            Z[cnt_x, cnt_y] = avg_BER
            
    print('Z = ', Z)
    np.savetxt('conv_robustBER_grid100.txt', Z, delimiter=',')
    print('num_tests={}, noise_range={}, noise_mean={}, noise_std={}.'.format(num_tests,noise_range,noise_mean,noise_std))



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

def check_FP():
    orthonormal_mats = np.loadtxt('orthonormalB.txt', delimiter=',' )
    # print('orthonormal_mats shape = ', orthonormal_mats.shape)
    projX = np.loadtxt('sharedProjectionMatrix.txt', delimiter=',')
    # print('projX shape: ', projX.shape)
    BIBD_AND_ACC = np.loadtxt('BIBD_AND_ACC_31_6_1.csv', dtype=int, delimiter=",")  # elements:{0,1}
    
    code_len = 31
    embed_dim = 512   ## const 
    i = 0             ## fp id 
    marked_FC_weights = np.load('markedFC_weights.npy')
    print('marked_FC_weights shape = ', marked_FC_weights.shape)

    check_marked = False
    check_robust = False  
    make_plot = True  

    if check_marked:
        print('== Check BER of marked model without attack. ')
        marked_FC_weights_avg = np.mean(marked_FC_weights, axis=-1)
        W_flatten = np.reshape(marked_FC_weights_avg, (embed_dim, 1) )

        marked_Xw = np.dot(projX, W_flatten)        #original dim: <N=288, 1>
        marked_user_corr = np.dot(marked_Xw.transpose(), orthonormal_mats) 
        recovered_user_BIBD_code = extract_BIBD_code(marked_user_corr)
        user_true_code_vec = BIBD_AND_ACC[:,i]
        code_match = (recovered_user_BIBD_code==user_true_code_vec)*1
        BER = 1 - np.sum(code_match)/code_len
        print('BER = ', BER)

    if check_robust:
        print('===== Add noise to weight matrix, and check BER.======')
        noise_range = 0.5   # for mask, changed portion  
        noise_mean = 0
        noise_std = 10
        # noise_relative_std = 0.1   # relative magnitude/variance 
        # noise_std = noise_relative_std*np.mean(marked_FC_weights)

        num_tests = 10    # random simulation 
        avg_BER = 0 
        # ---- mask to add noise (random position) -----#
        for j in range(num_tests):
            noise_mask = np.zeros((1, marked_FC_weights.size))
            idx = np.random.choice(marked_FC_weights.size, int(marked_FC_weights.size*noise_range), replace=False)

            # --- assign random pertubation to mask ----#
            noise_val = np.random.normal(noise_mean, noise_std, idx.size)
            noise_mask[0, idx] = noise_val
            noise_mask = np.reshape(noise_mask, marked_FC_weights.shape)
            # print('noise_mask = ', noise_mask)

            W_marked_noised = marked_FC_weights + noise_mask
            marked_FC_weights_avg = np.mean(W_marked_noised, axis=-1)
            W_flatten = np.reshape(marked_FC_weights_avg, (embed_dim, 1) )
            marked_Xw = np.dot(projX, W_flatten)        #original dim: <N=288, 1>
            marked_user_corr = np.dot(marked_Xw.transpose(), orthonormal_mats) 
            recovered_user_BIBD_code = extract_BIBD_code(marked_user_corr)
            user_true_code_vec = BIBD_AND_ACC[:,i]
            code_match = (recovered_user_BIBD_code==user_true_code_vec)*1
            BER = 1 - np.sum(code_match)/code_len
            avg_BER += BER
        avg_BER = avg_BER/num_tests
        print('num_tests={}, noise_range={}, noise_mean={}, noise_std={}, avg_BER={}.'.format(num_tests,noise_range,noise_mean,noise_std,avg_BER))



    if make_plot:
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')

        # Make data.
        noise_range = np.arange(0, 1, 0.025)     ## grid=40
        noise_std = np.arange(0, 1, 0.025)

        # noise_relative_std = np.arange(0, 1, 0.25)
        # noise_std = noise_relative_std*np.mean(marked_FC_weights)

        # noise_range, noise_std = np.meshgrid(noise_range, noise_std)

        num_tests = 10
        noise_mean = 0.

        Z = np.zeros((noise_range.size, noise_std.size))
        for cnt_x, x in enumerate(noise_range):
            for cnt_y, y in enumerate(noise_std):
                avg_BER = 0.
                for n in range(num_tests):
                    noise_mask = np.zeros((1, marked_FC_weights.size))
                    idx = np.random.choice(marked_FC_weights.size, int(marked_FC_weights.size*x), replace=False)

                    # --- assign random pertubation to mask ----#
                    noise_val = np.random.normal(noise_mean, y, idx.size)
                    noise_mask[0, idx] = noise_val
                    noise_mask = np.reshape(noise_mask, marked_FC_weights.shape)
                    # print('noise_mask = ', noise_mask)

                    W_marked_noised = marked_FC_weights + noise_mask
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
        np.savetxt('FC_robustBER_grid40.txt', Z, delimiter=',')
        print('num_tests={}, noise_range={}, noise_mean={}, noise_std={}, avg_BER={}.'.format(num_tests,noise_range,noise_mean,noise_std,avg_BER))


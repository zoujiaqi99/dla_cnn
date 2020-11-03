import numpy as np
import math
import re, os, traceback, sys, json
import argparse
import tensorflow as tf
import timeit
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from linetools.spectra.xspectrum1d import XSpectrum1D
from pyigm.abssys.dla import DLASystem
from pyigm.abssys.lls import LLSSystem
from pyigm.abssys.utils import hi_model
ops.reset_default_graph()
#tf.compat.v1.disable_eager_execution()
#init = tf.compat.v1.global_variables_initializer()

from modell1 import build_model

tensor_regex = re.compile('.*:\d*')
# Get a tensor by name, convenience method
def t(tensor_name):
    tensor_name = tensor_name+":0" if not tensor_regex.match(tensor_name) else tensor_name
    return tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)

def plot_dla(p,resulttype,sight_id,zabs,NHI,matrix_flux,matrix_lam,wvoff=60.):
    # Get spectrum,dla
    #spec = XSpectrum1D.from_tuple((10**sightline.loglam,sightline.flux))#generate xspectrum1d
    spec = XSpectrum1D.from_tuple((matrix_lam,matrix_flux))
    if NHI<20.3:
        dla = LLSSystem((0,0), zabs, None, NHI=NHI)      
    else:
        dla = DLASystem((0,0), zabs, None, NHI)
    #all_spec, all_meta = igmsp.allspec_at_coord(dla.coord, groups=['SDSS_DR7'])
    #spec = all_spec[0]
    # Get wavelength limits
    wvcen = (1+zabs)*1215.67
    
    gd_wv = (spec.wavelength.value > wvcen-wvoff) & (spec.wavelength.value < wvcen+wvoff)
    # Continuum?
    co = np.amax(spec.flux[gd_wv])
    # Create DLA
    lya, lines = hi_model(dla, spec, lya_only=True)
    # Plot
    plt.clf()
    plt.figure(figsize=(12,8))
    ax = plt.gca()
    ax.plot(spec.wavelength,spec.flux, 'k', drawstyle='steps-mid')
    #ax.plot(spec.wavelength, sightline.error, 'r:')
    # Model
    ax.plot(lya.wavelength, lya.flux*co, color='red') # model
    #if use_co:
        #ax.plot(spec.wavelength, spec.co, '--', color='gray')
    # Axes
    #ax.set_xlim(wvcen-wvoff, wvcen+wvoff)
    plt.axhline(y=0,ls="--",c="green",linewidth=2)
    plt.ylabel('Relative Flux',fontsize=20)
    plt.xlabel('Wavelength'+'['+'$\AA$'+']',fontsize=20)
    plt.title('spec-%s(%s)'%(sight_id,resulttype),fontdict=None,loc='center',pad='20',fontsize=30,color='blue')
    ax.set_xlim(np.amin(matrix_lam), np.amax(matrix_lam))
    #ax.set_ylim(-0.1*co, co*1.1)
    
    plt.axvline(x=wvcen,ls="--",c="red",linewidth=2)
    ab=max(spec.flux)
    #plt.text(wvcen+20,ab,'actual:'+'$\Delta$'+'z=%s'%(a_offset),fontsize=18,color='blue')
    #plt.text(wvcen+20,ab-0.2,'pred:'+'$\Delta$'+'z=%.2f'%(p_offset),fontsize=18,color='red')
    plt.text(wvcen+5,ab*0.8,'z=%.2f'%(zabs),fontsize=18,color='blue',rotation=90)
    plt.text(wvcen+20,ab-0.4,'log${N_{\mathregular{HI}}}$'+'=%.2f'%(NHI),fontsize=18,color='blue')
    plt.savefig('result_plot/6-2/%s/spec-%s-%s.png'%(resulttype,sight_id,p),dpi=200)
    print('%s saved'%(p))
    plt.show()


def predictions_ann(hyperparameters, flux, checkpoint_filename, TF_DEVICE=''):
    timer = timeit.default_timer()
    BATCH_SIZE = 4000
    n_samples = flux.shape[0]
    pred = np.zeros((n_samples,), dtype=np.float32)
    conf = np.copy(pred)
    offset = np.copy(pred)
    coldensity = np.copy(pred)

    with tf.Graph().as_default():
        build_model(hyperparameters)

        with tf.device(TF_DEVICE), tf.compat.v1.Session() as sess:
            tf.compat.v1.train.Saver().restore(sess, checkpoint_filename+".ckpt")
            for i in range(0,n_samples,BATCH_SIZE):
                pred[i:i+BATCH_SIZE], conf[i:i+BATCH_SIZE], offset[i:i+BATCH_SIZE], coldensity[i:i+BATCH_SIZE] = \
                    sess.run([t('prediction'), t('output_classifier'), t('y_nn_offset'), t('y_nn_coldensity')],
                             feed_dict={t('x'):                 flux[i:i+BATCH_SIZE,:],
                                        t('keep_prob'):         1.0})

    print("Localize Model processed {:d} samples in chunks of {:d} in {:0.1f} seconds".format(
          n_samples, BATCH_SIZE, timeit.default_timer() - timer))

    # coldensity_rescaled = coldensity * COL_DENSITY_STD + COL_DENSITY_MEAN
    return pred, conf, offset, coldensity


# Called from train_ann to perform a test of the train or test data, needs to separate pos/neg to get accurate #'s




if __name__ == '__main__':


    parameter_names = ["learning_rate", "training_iters", "batch_size", "l2_regularization_penalty", "dropout_keep_prob",
                       "fc1_n_neurons", "fc2_1_n_neurons", "fc2_2_n_neurons", "fc2_3_n_neurons",
                       "conv1_kernel", "conv2_kernel", "conv3_kernel",
                       "conv1_filters", "conv2_filters", "conv3_filters",
                       "conv1_stride", "conv2_stride", "conv3_stride",
                       "pool1_kernel", "pool2_kernel", "pool3_kernel",
                       "pool1_stride", "pool2_stride", "pool3_stride"]
    parameters = [
        # First column: Keeps the best best parameter based on accuracy score
        # Other columns contain the parameter options to try


        # learning_rate
        [0.00002,         0.0005, 0.0007, 0.0010, 0.0030, 0.0050, 0.0070],
        # training_iters
        [300000],
        # batch_size
        [700,           400, 500, 600, 700, 850, 1000],
        # l2_regularization_penalty
        [0.005,         0.01, 0.008, 0.005, 0.003],
        # dropout_keep_prob
        [0.98,          0.75, 0.9, 0.95, 0.98, 1],
        # fc1_n_neurons
        [350,           200, 350, 500, 700, 900, 1500],
        # fc2_1_n_neurons
        [200,           200, 350, 500, 700, 900, 1500],
        # fc2_2_n_neurons
        [350,           200, 350, 500, 700, 900, 1500],
        # fc2_3_n_neurons
        [150,           200, 350, 500, 700, 900, 1500],
        # conv1_kernel
        [32,            20, 22, 24, 26, 28, 32, 40, 48, 54],
        # conv2_kernel
        [16,            10, 14, 16, 20, 24, 28, 32, 34],
        # conv3_kernel
        [16,            10, 14, 16, 20, 24, 28, 32, 34],
        # conv1_filters
        [100,           64, 80, 90, 100, 110, 120, 140, 160, 200],
        # conv2_filters
        [96,            80, 96, 128, 192, 256],
        # conv3_filters
        [96,            80, 96, 128, 192, 256],
        # conv1_stride
        [3,             2, 3, 4, 5, 6, 8],
        # conv2_stride
        [1,             1, 2, 3, 4, 5, 6],
        # conv3_stride
        [1,             1, 2, 3, 4, 5, 6],
        # pool1_kernel
        [7,             3, 4, 5, 6, 7, 8, 9],
        # pool2_kernel
        [6,             4, 5, 6, 7, 8, 9, 10],
        # pool3_kernel
        [6,             4, 5, 6, 7, 8, 9, 10],
        # pool1_stride
        [4,             1, 2, 4, 5, 6],
        # pool2_stride
        [4,             1, 2, 3, 4, 5, 6, 7, 8],
        # pool3_stride
        [4,             1, 2, 3, 4, 5, 6, 7, 8]
    ]

    # Random permutation of parameters out some artibrarily long distance
    #r = np.random.permutation(1000)

    # Write out CSV header
    


    #hyperparameters = [parameters[i][0] for i in range(len(parameters))]
    hyperparameters = {}

    for k in range(0,len(parameter_names)):
        hyperparameters[parameter_names[k]] = parameters[k][0]

    pred_dataset='/home/bwang/dataset/61/695-dataset.npy'
    r=np.load(pred_dataset,allow_pickle = True,encoding='latin1').item()

    checkpoint_filename='/home/bwang/CNN/train_528/current_99999'
    #spe_id=[110096384,110096520,110096533]


    TP=[]
    TN=[]
    FP=[]
    FN=[]
    for sight_id in r.keys():
    #for m in range(0,3):

        #sight_id=spe_id[m]
        flux=r[sight_id]['FLUX']


        (pred, conf, offset, coldensity)=predictions_ann(hyperparameters, flux, checkpoint_filename, TF_DEVICE='')
    
        print(pred)
        print(conf)
        print(offset)
        print(coldensity)

    
        for p in range(0,len(pred)):
            if (r[sight_id]['labels_classifier'][p]==1) & (pred[p]==1):
                TP.append(p)

            if (r[sight_id]['labels_classifier'][p]==1) & (pred[p]==0):
                FN.append(p)
                wave_dla=200+r[sight_id]['labels_offset'][p]
                zabs=(r[sight_id]['lam'][p][int(wave_dla)]/1215.67)-1
                NHI=r[sight_id]['col_density'][p]
                matrix_lam=r[sight_id]['lam'][p]
                matrix_flux=r[sight_id]['FLUX'][p]
                resulttype='FN'
                plot_dla(p,resulttype,sight_id,zabs,NHI,matrix_flux,matrix_lam,wvoff=60.)



            if (r[sight_id]['labels_classifier'][p]==0) & (pred[p]==0):
                TN.append(p)

            if (r[sight_id]['labels_classifier'][p]==0) & (pred[p]==1):
                FP.append(p)
                wave_dla=200+offset[p]
                zabs=(r[sight_id]['lam'][p][int(wave_dla)]/1215.67)-1
                NHI=coldensity[p]
                matrix_lam=r[sight_id]['lam'][p]
                matrix_flux=r[sight_id]['FLUX'][p]
                resulttype='FP'
                plot_dla(p,resulttype,sight_id,zabs,NHI,matrix_flux,matrix_lam,wvoff=60.)

    print('samples of TP is %s'%(len(TP)))
    print('samples of TN is %s'%(len(TN)))
    print('samples of FP is %s'%(len(FP)))
    print('samples of FN is %s'%(len(FN)))
            




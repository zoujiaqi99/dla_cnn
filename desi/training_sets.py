""" Code to build/load/write DESI Training sets"""

'''
1. Load up the Sightlines
2. Split into samples of kernel length
3. Grab DLAs and non-DLA samples
4. Hold in memory or write to disk??
5. Convert to TF Dataset
'''

import multiprocessing
from multiprocessing import Pool
import itertools

import numpy as np

from dla_cnn.Timer import Timer
#from dla_cnn.desi.DesiMock import DesiMock
from dla_cnn.spectra_utils import get_lam_data
from dla_cnn.training_set import select_samples_50p_pos_neg
from dla_cnn.desi.defs import REST_RANGE,kernel,best_v

def pad_sightline(sightline, lam, lam_rest, ix_dla_range,kernelrangepx,v=best_v['all']):
    c = 2.9979246e8
    dlnlambda = np.log(1+v/c)
    #pad left side
    if np.nonzero(ix_dla_range)[0][0]<kernelrangepx:
        pixel_num_left=kernelrangepx-np.nonzero(ix_dla_range)[0][0]
        pad_lam_left= lam[0]*np.exp(dlnlambda*np.array(range(-pixel_num_left,0)))
        pad_value_left = np.mean(sightline.flux[0:50])
    else:
        pixel_num_left=0
        pad_lam_left=[]
        pad_value_left=[] 
    #pad right side
    if np.nonzero(ix_dla_range)[0][-1]>len(lam)-kernelrangepx:
        pixel_num_right=kernelrangepx-(len(lam)-np.nonzero(ix_dla_range)[0][-1])
        pad_lam_right= lam[0]*np.exp(dlnlambda*np.array(range(len(lam),len(lam)+pixel_num_right)))
        pad_value_right = np.mean(sightline.flux[-50:])
    else:
        pixel_num_right=0
        pad_lam_right=[]
        pad_value_right=[]
    flux_padded = np.hstack((pad_lam_left*0+pad_value_left, sightline.flux,pad_lam_right*0+pad_value_right))
    lam_padded = np.hstack((pad_lam_left,lam,pad_lam_right))
    return flux_padded,lam_padded,pixel_num_left
    
    
def split_sightline_into_samples(sightline, REST_RANGE=REST_RANGE, kernel=kernel, v=best_v['all']):
    """
    Split the sightline into a series of snippets, each with length kernel

    Parameters
    ----------
    sightline: dla_cnn.data_model.Sightline
    REST_RANGE: list
    kernel: int, optional

    Returns
    -------

    """
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    kernelrangepx = int(kernel/2) #200
    #samplerangepx = int(kernel*pos_sample_kernel_percent/2) #60
    #padding the sightline:
    flux_padded,lam_padded,pixel_num_left=pad_sightline(sightline,lam,lam_rest,ix_dla_range,kernelrangepx,v=v)
     
    #ix_dlas = [(np.abs(lam[ix_dla_range]-dla.central_wavelength).argmin()) for dla in sightline.dlas]
    #coldensity_dlas = [dla.col_density for dla in sightline.dlas]       # column densities matching ix_dlas

    # FLUXES - Produce a 1748x400 matrix of flux values
    #fluxes_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0])))
    fluxes_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(flux_padded), np.nonzero(ix_dla_range)[0]+pixel_num_left)))
    lam_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(lam_padded), np.nonzero(ix_dla_range)[0]+pixel_num_left)))
    #using cut will lose side information,so we use padding instead of cutting 
    #fluxes_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0][cut])))
    #lam_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(lam), np.nonzero(ix_dla_range)[0][cut])))
    #the wavelength and flux array we input:
    input_lam=lam_padded[np.nonzero(ix_dla_range)[0]+pixel_num_left]
    input_flux=flux_padded[np.nonzero(ix_dla_range)[0]+pixel_num_left]
    # Return
    return fluxes_matrix, sightline.classification, sightline.offsets, sightline.column_density,lam_matrix,input_lam,input_flux
    #return fluxes_matrix, sightline.classification, sightline.offsets, sightline.column_density

def prepare_training_test_set(ids_train, ids_test,
                                      train_save_file="../data/localize_train.npy",
                                      test_save_file="../data/localize_test.npy",
                                      ignore_sightline_markers={},
                                      save=False):
    """
    Build a Training set for DESI

    and a test set, as desired

    Args:
        ids_train: list
        ids_test: list (can be empty)
        train_save_file: str
        test_save_file: str
        ignore_sightline_markers:
        save: bool

    Returns:

    """
    num_cores = multiprocessing.cpu_count() - 1
    p = Pool(num_cores, maxtasksperchild=10)  # a thread pool we'll reuse

    # Training data
    with Timer(disp="read_sightlines"):
        sightlines_train=[]
        for ii in ids_train:
            sightlines_train.append(specs.get_sightline(ids_train[ii],'all',True,True))
        # add the ignore markers to the sightline
        for s in sightlines_train:
            if hasattr(s.id, 'sightlineid') and s.id.sightlineid >= 0:
                s.data_markers = ignore_sightline_markers[s.id.sightlineid] if ignore_sightline_markers.has_key(
                    s.id.sightlineid) else []
    with Timer(disp="split_sightlines_into_samples"):
        data_split = p.map(split_sightline_into_samples, sightlines_train)
    with Timer(disp="select_samples_50p_pos_neg"):
        sample_masks = p.map(select_samples_50p_pos_neg, data_split[1])
    with Timer(disp="zip and stack"):
        zip_data_masks = zip(data_split, sample_masks)
        data_train = {}
        data_train['flux'] = np.vstack([d[0][m] for d, m in zip_data_masks])
        data_train['labels_classifier'] = np.hstack([d[1][m] for d, m in zip_data_masks])
        data_train['labels_offset'] = np.hstack([d[2][m] for d, m in zip_data_masks])
        data_train['col_density'] = np.hstack([d[3][m] for d, m in zip_data_masks])
    if save:
        with Timer(disp="save train data files"):
            save_tf_dataset(train_save_file, data_train)

    # Same for test data if it exists
    data_test = {}
    if len(ids_test) > 0:
        sightlines_test = p.map(read_sightline, ids_test)
        data_split = map(split_sightline_into_samples, sightlines_test)
        sample_masks = map(select_samples_50p_pos_neg, data_split)
        zip_data_masks = zip(data_split, sample_masks)
        data_test['flux'] = np.vstack([d[0][m] for d, m in zip_data_masks])
        data_test['labels_classifier'] = np.hstack([d[1][m] for d, m in zip_data_masks])
        data_test['labels_offset'] = np.hstack([d[2][m] for d, m in zip_data_masks])
        data_test['col_density'] = np.hstack([d[3][m] for d, m in zip_data_masks])
        if save:
            save_tf_dataset(test_save_file, data_test)

    # Return
    return data_train, data_test

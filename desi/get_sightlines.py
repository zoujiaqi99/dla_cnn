from dla_cnn.desi.preprocess import estimate_s2n,normalize,rebin
from dla_cnn.desi.DesiMock import DesiMock
#from dla_cnn.desi.insert_dlas import insert_dlas
from dla_cnn.desi.defs import best_v
import numpy as np
import os
from os.path import join
def get_sightlines(input='/desi-0.2-100/spectra-16', output='MOCK_spectra/processed/pre_sightlines.npy'):
    """
    Insert DLAs manually into sightlines without DLAs or only choose sightlines with DLAs
    
    Return
    ---------
    sightlines:list of `dla_cnn.data_model.sightline.Sightline` object
    
    """
    sightlines=[]
    #get folder list
    item1 = os.listdir(path)
    for k in item1:
        #get subfolder list
        item = os.listdir(path+'/'+str(k))
        for j in item:
            file_path = path+'/'+str(k)+'/'+str(j)
            spectra = join(file_path,"spectra-16-%s.fits"%j)
            truth = join(file_path,"truth-16-%s.fits"%j)
            zbest = join(file_path,"zbest-16-%s.fits"%j)
            specs = DesiMock()
            specs.read_fits_file(spectra,truth,zbest)
            keys = list(specs.data.keys())
            for jj in keys:
                sightline = specs.get_sightline(jj,camera = 'all', rebin=False, normalize=True)
                #calculate S/N
                sightline.s2n=estimate_s2n(sightline)
                if (sightline.z_qso >= 2.33)&(sightline.s2n >= 1):#apply filtering
                        #only use blue band data
                        sightline.flux = sightline.flux[0:sightline.split_point_br]
                        sightline.error = sightline.error[0:sightline.split_point_br]
                        sightline.loglam = sightline.loglam[0:sightline.split_point_br]
                        rebin(sightline, best_v['b'])
                        sightlines.append(sightline)
                
    np.save(output,sightlines)
    return sightlines

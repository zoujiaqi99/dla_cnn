#make individual sightlines
import numpy as np
import os
from os.path import join
from multiprocessing import Process
from astropy.table import Table,vstack
from dla_finder.datasets.DesiMock import DesiMock
from dla_finder.datasets import preprocess 
from tqdm import tqdm
import scipy.io as scio
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool, Process
import time
'''
deal with desi spectra:
1. extract qso catalog (fuji_guadalupe.ipynb, iron.ipynb)
2. extract sightlines from fits file based on qso catalog(make_desi_sightlines)
3. preprocess sightlines:rebin+normalize+s2n (preprocess_sightlines)
Note1: some sightlines do not have enough pixels in norm_range because of pixel mask & flux=0, only select s2n>0 and targetid>0(if targetid<0 bad sightline)
'''
def preprocess_sightlines(sightlines,v=44735,out_path=None, ind_path=None):
    '''
    output:
    pre_sightlines: sightlines after preprocessing
    process_ind: 1 if preprocessed, 0 if can not be processed (S/N<0, z_qso>5.8, id<0,zwarn>0)
    '''
    pre_sightlines=[]
    process_ind=[]
    snr=[]
    for sightline in tqdm(sightlines):
        if sightline!=[]:
            sightline.s2n = preprocess.estimate_s2n(sightline)
            #filters: S/N!=nan, right id, zwarn=0
            if (sightline.s2n >0):
                preprocess.normalize(sightline, 10**sightline.loglam, sightline.flux)
                preprocess.rebin(sightline, v)
                pre_sightlines.append(sightline)
                process_ind.append(1)
                snr.append(sightline.s2n)
            else:
                #filter flag=1
                pre_sightlines.append([])
                process_ind.append(0)
                snr.append(sightline.s2n)
        else:
            pre_sightlines.append([])
            process_ind.append(0)
            snr.append(0)
    if out_path:
        np.save(out_path,pre_sightlines)
    if ind_path:
        np.save(ind_path,process_ind)
    
    return pre_sightlines,process_ind,snr

def preprocess_forgp(sightlines,indlist, output_path=None):
    '''
    output:
    preload_qso.mat: matlab file for gp model to use
    directly use preprocessed sightline data, so do not need pixel mask 
    '''
    all_wavelengths    =  []
    all_flux           =  []
    all_noise_variance =  []
    all_pixel_mask     =  []
    all_normalizers    = []
    sightline_ids=[]
    loading_min_lambda = 910
    loading_max_lambda = 1217
    for i in tqdm(range(0,len(sightlines))):
        sightline=sightlines[i]
        if (indlist[i]==1)&(sightline != []):
            this_wavelength=10**sightline.loglam
            this_pixel_mask=sightline.pixel_mask
            flux=sightline.flux
            var=sightline.error**2
            rest_wavelength=this_wavelength/(1+sightline.z_qso)
            sightline_ids.append(sightline.id)
            ind = (rest_wavelength >= loading_min_lambda) & (rest_wavelength <= loading_max_lambda)
            if sum(ind) !=0:
                ind[max(0,np.nonzero(ind)[0][0]-1)]=True
                ind[min(np.nonzero(ind)[0][-1]+1,len(np.nonzero(ind)[0]-1))]=True
                all_wavelengths.append(list(this_wavelength[ind]))
                all_flux.append(list(flux[ind]))
                all_noise_variance.append(list(var[ind]))
            #no pixel mask
            else:
                sightline_ids.append([])
                all_wavelengths.append([])
                all_flux.append([])
                all_noise_variance.append([])
                print(sightline.id)
                indlist[i]=0
        else:
            sightline_ids.append([])
            all_wavelengths.append([])
            all_flux.append([])
            all_noise_variance.append([])
            #all_pixel_mask.append([])
    scio.savemat(output_path, {'sightline_ids':sightline_ids, 'wavelengths':all_wavelengths,'flux':all_flux,'noise_variance':all_noise_variance})
    print('make preload_qsos done')
    return indlist
'''
def make_sightline(kk, qsocat, specs):
    sightline = specs.get_sightline(kk,camera = 'all', rebin=False, normalize=False)
    assert sightline.id==kk
    sightline.z_qso=float(qsocat[qsocat['TARGETID']==kk]['Z'])
    sightline.spectype=str(qsocat[qsocat['TARGETID']==kk]['SPECTYPE'])
    sightline.zwarn=int(qsocat[qsocat['TARGETID']==kk]['ZWARN'])
    return sightline

def process_multi(data, qsocat, specs):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(lambda x: make_sightline(x, qsocat, specs), data)
    return results
'''            
def make_desi_sightlines(release,survey,program,dir,d, qsocat, bal=True):
    '''
    This module is used to process afterburner qso spectra using GP and CNN
    
    Output files:
    a)indlist: to validate which qso can be applied to the models
    b)For CNN model:
    i)raw-sightlines: extract spectra directly from fits file
    ii)pre-sightlines: preprocess for raw sightlines, and will be the input of CNN model
    c)For GP model:
    i)catalog.mat& fits: add BAL filter and filter flags to the QSO catalog
    ii)preload_qsos.mat: the input for GP model, directly transform from pre-sightlines
    
    Future modification:
    1)afterburner qso catalog may change 'afterburn_qsocatalog'
    2)bal catalog may change 'balpath'
    3)spectra path may change 'path'
    4) the speed of extracting spectra, 16min/5w, without multiprocess, may need 2-3 hour to process desi spectra
    '''
    #first, define output files
    rawsightline_path='/global/cfs/cdirs/desi/users/jqzou/%s/sightlines/%s/%s/%s/%s/raw-sightlines.npy'%(release,survey,program,dir,d)
    presightline_path='/global/cfs/cdirs/desi/users/jqzou/%s/sightlines/%s/%s/%s/%s/pre-sightlines.npy'%(release,survey,program,dir,d)
    premat_path='/global/cfs/cdirs/desi/users/jqzou/%s/sightlines/%s/%s/%s/%s/preload_qsos.mat'%(release,survey,program,dir,d)
    catmat_path='/global/cfs/cdirs/desi/users/jqzou/%s/sightlines/%s/%s/%s/%s/catalog.mat'%(release,survey,program,dir,d)
    #catfits_path='/global/cfs/cdirs/desi/users/jqzou/%s/sightlines/%s-%s-catalog.fits'%(release,survey,program)
    start_time = time.time()
    sightlines=[]
    path='/global/cfs/cdirs/desi/spectro/redux/%s/healpix/%s/%s'%(release,survey,program)
    healpix_loc=path+'/'+str(dir)+'/'+str(d)
    spectra=healpix_loc+'/coadd-%s-%s-%s.fits'%(survey,program,d)
    zbest=healpix_loc+'/redrock-%s-%s-%s.fits'%(survey,program,d)
    specs = DesiMock()
    specs.read_fits_file(spectra,[],[])
    this_qso_keys=qsocat[(qsocat['HPXPIXEL']==int(d))&(qsocat['z_ind']==1)&(qsocat['bal_ind']==1)&(qsocat['target_ind']==1)]['TARGETID']
    print('%s %s %s %s %s, %s spectra'%(release,survey,program,dir,d,len(this_qso_keys)))
    sightlines=[]
    for kk in this_qso_keys:
        sightline = specs.get_sightline(kk,camera = 'all', rebin=False, normalize=False)
        assert sightline.id==kk
        sightline.z_qso=float(qsocat[qsocat['TARGETID']==kk]['Z'])
        sightline.spectype=str(qsocat[qsocat['TARGETID']==kk]['SPECTYPE'])
        sightline.zwarn=int(qsocat[qsocat['TARGETID']==kk]['ZWARN'])
        sightlines.append(sightline)
    
    #sightlines = process_multi(this_qso_keys, qsocat, specs)
    
    #sightlines = [sl_pix for sl_pix in results]
    end_time = time.time()
    print("Extracting sightlines time: {} seconds".format(end_time - start_time))
    np.save(rawsightline_path,sightlines)
    #preprocess
    pre_sightlines,process_ind,snr=preprocess_sightlines(sightlines,v=44735,out_path=presightline_path)
    
    #make preload_qsos.mat:
    process_ind=preprocess_forgp(pre_sightlines,process_ind,output_path=premat_path)
    assert len(process_ind)==len(this_qso_keys)
    
    #make catalog mat and fits:
    this_qsocat=qsocat[(qsocat['HPXPIXEL']==int(d))&(qsocat['z_ind']==1)&(qsocat['bal_ind']==1)&(qsocat['target_ind']==1)]
    this_qsocat['S/N']=snr
    this_qsocat['process_ind']=process_ind
    scio.savemat(catmat_path, {'ras':list(this_qsocat['TARGET_RA']),'decs':list(this_qsocat['TARGET_DEC']),
                               'target_ids':list(this_qsocat['TARGETID']),'z_qsos':list(this_qsocat['Z']),
                               'snrs':list(this_qsocat['S/N']),
                               'bal_visual_flags':list(np.ones(len(this_qsocat))-this_qsocat['bal_ind']),
                               'filter_flags':list(np.ones(len(this_qsocat))-this_qsocat['process_ind']),
                               'zwarning':list(this_qsocat['ZWARN'])})

    
release='himalayas'
survey='main'
program='dark'
'''
os.system('mkdir /global/cfs/cdirs/desi/users/jqzou/%s'%release)
os.chdir('/global/cfs/cdirs/desi/users/jqzou/%s'%release)
os.system('mkdir sightlines')
os.chdir('sightlines')
surveys=os.listdir('/global/cfs/cdirs/desi/spectro/redux/%s/healpix'%release)
for survey in surveys:
    os.system('mkdir %s'%survey)
    os.chdir('%s'%survey)
    programs=os.listdir('/global/cfs/cdirs/desi/spectro/redux/%s/healpix/%s'%(release,survey))
    for program in programs:
        os.system('mkdir %s'%program)
        os.chdir('%s'%program)
        dirs=os.listdir('/global/cfs/cdirs/desi/spectro/redux/%s/healpix/%s/%s'%(release,survey,program))
        for dir in tqdm(dirs):
            os.system('mkdir %s'%dir)
            os.chdir('%s'%dir)
            ds=os.listdir('/global/cfs/cdirs/desi/spectro/redux/himalayas/%s/%s/%s/%s'%(release,survey,program,dir))
            for d in tqdm(ds):
                os.system('mkdir %s'%d)
                make_desi_sightlines(release,survey,program,dir,d)
'''
qsocat_path='/global/cfs/cdirs/desi/users/jqzou/himalayas/QSO_cat_himalayas_main_dark_healpix_v1-bal-ind.fits'
qsocat=Table.read(qsocat_path)#this cat added bal ind
dirs=os.listdir('/global/cfs/cdirs/desi/spectro/redux/%s/healpix/%s/%s'%(release,survey,program))
os.chdir('/global/cfs/cdirs/desi/users/jqzou/%s/sightlines/%s/%s'%(release,survey,program))
def multi_sightlines(dir):
    os.system('mkdir %s'%dir)
    os.chdir('%s'%dir)
    ds=os.listdir('/global/cfs/cdirs/desi/spectro/redux/%s/healpix/%s/%s/%s'%(release,survey,program,dir))
    for d in tqdm(ds):
        os.system('mkdir %s'%d)
        print('running: %s %s %s %s %s'%(release,survey,program,dir,d))
        make_desi_sightlines(release,survey,program,dir,d,qsocat)
p = Pool(multiprocessing.cpu_count())
p.map(multi_sightlines,tqdm(dirs[1:]))

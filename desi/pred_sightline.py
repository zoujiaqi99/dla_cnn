import numpy as np 
from dla_cnn.data_model.Sightline import Sightline
from dla_cnn.data_model.Prediction import Prediction
from dla_cnn.spectra_utils import get_lam_data
import matplotlib.pyplot as plt
from linetools.spectra.xspectrum1d import XSpectrum1D
from matplotlib.pyplot import MultipleLocator
import scipy.signal as signal
from dla_cnn.desi.analyze_prediction import analyze_pred
from dla_cnn.desi.training_sets import split_sightline_into_samples
from astropy.table import Table, vstack

def save_pred(sightlines,pred,PEAK_THRESH,level,filename=None):
    """
    Using prediction for windows to get prediction for sightlines, get a pred DLA catalog.
    
    Parameters
    ---------------
    sightlines: data_model.Sightline object list
    pred: dict
    PEAK_THRESH: float
    level: float
    filename: str, use it to save DLA catalog
    
    Returns
    ---------------
    pred_abs: astropy.Table 
    
    """
    pred_abs = Table(names=('TARGET_RA','TARGET_DEC', 'ZQSO','Z','TARGETID','S/N','DLAID','NHI','DLA_CONFIDENCE','NHI_STD','ABSORBER_TYPE'),dtype=('float','float','float','float','int','float','str','float','float','float','str'),meta={'EXTNAME': 'DLACAT'})
    for ii in range(0,len(sightlines)):
        sightline=sightlines[ii]
        conf=pred[sightline.id]['conf']
        #classifier=[]
        #for ii in range(0,len(conf)):
            #if conf[ii]>level:
                #classifier.append(1)
            #else:
                #classifier.append(0)
        #classifier=np.array(classifier)
        #real_classifier=real_claasifiers[ii]
        classifier=pred[sightline.id]['pred']
        offset=pred[sightline.id]['offset']
        coldensity=pred[sightline.id]['coldensity']
        pred_abs=vstack((pred_abs,analyze_pred(sightline,classifier,conf,offset,coldensity,PEAK_THRESH)))
    pred_abs.write(filename,overwrite=True)
    return pred_abs

def label_catalog(real_catalog,pred_catalog,realname=None,dlaname=None):
    """
    Compare real absorbers and predicted absorbers to add TP, FN, FP informations to DLA catalogs, calculate numbers of TP,FN,FP.
    
    Parameters
    ---------------
    real_catalog:astropy.Table class 
    pred_catalog:astropy.Table class
    realname: str, use it to save the new real DLA catalog
    predname: str, use it to save the new pred DLA catalog
    
    Reutrns
    ----------------
    tp_preds: list
    fn_num: int
    fp_num: int
   
    """
    tp_pred=[]
    fn_num=0
    fp_num=0
    pred_catalog.add_column('str',name='label')
    pred_catalog.add_index('DLAID')
    real_catalog.add_column('str',name='label')
    for real_dla in real_catalog:
        pred_dlas=pred_catalog[pred_catalog['TARGETID']==real_dla['TARGETID']]
        central_wave=1215.67*(1+real_dla['Z'])
        pred_wave=1215.67*(1+pred_dlas['Z'])
        col_density=real_dla['NHI']
        pred_coldensity=pred_dlas['NHI']
        lam_difference=np.abs(pred_wave-central_wave)
        if len(lam_difference) != 0:
            nearest_ix = np.argmin(lam_difference) 
            if (lam_difference[nearest_ix]<=10)&(pred_dlas[nearest_ix]['ABSORBER_TYPE']!='LYB')&(pred_dlas[nearest_ix]['label']=='str'):#距离小于10且不是lyb
                real_dla['label']='tp'
                dlaid=pred_dlas[nearest_ix]['DLAID']
                pred_catalog.loc[dlaid]['label']='tp'
                tp_pred.append([central_wave,col_density,pred_wave[nearest_ix],pred_coldensity[nearest_ix]])
            else:
                real_dla['label']='fn'
                fn_num=fn_num+1
        else:
            real_dla['label']='fn'
            fn_num=fn_num+1  
    for pred_dla in pred_catalog:
        if pred_dla['ABSORBER_TYPE']=='LYB':
            pred_dla['label']='LYB'
        else:
            if pred_dla['label']=='str':
                pred_dla['label']='fp'
                fp_num=fp_num+1
    
    real_catalog.write(realname,overwrite=True)
    pred_catalog.write(dlaname,overwrite=True)
    return tp_pred, fn_num, fp_num

def get_results(real_catalog,pred_catalog,realname=None,predname=None,path=None):
    """
    Compare real absorbers and predicted absorbers to add TP, FN, FP informations to DLA catalogs, calculate numbers of TP,FN,FP and draw histogram.
    
    Parameters
    ---------------
    real_catalog:astropy.Table class 
    pred_catalog:astropy.Table class
    realname: str, use it to save the new real DLA catalog
    predname: str, use it to save the new pred DLA catalog
    path: str, use it to save histograms.
    
    """
    tp_pred, fn_num, fp_num=label_catalog(real_catalog,pred_catalog,realname=realname,predname=predname)
    print('true_positive=%s,false_negative=%s,false_positive=%s'%(len(tp_pred),fn_num,fp_num))
    #draw hist
    delta_z=[]
    delta_NHI=[]
    for pred in tp_pred:
        pred_z=pred[2]/1215.67-1
        real_z=pred[0]/1215.67-1
        delta_z.append(pred_z-real_z)
        delta_NHI.append(pred[3]-pred[1])
    arr_mean = np.mean(delta_z)
    arr_var = np.var(delta_z)
    arr_std = np.std(delta_z,ddof=1)

    arr_mean_2 = np.mean(delta_NHI)
    arr_var_2 = np.var(delta_NHI)
    arr_std_2 = np.std(delta_NHI,ddof=1)
    plt.figure(figsize=(10,10))
    plt.title('stddev=%.4f mean=%.5f'%(arr_std,arr_mean),fontdict=None,loc='center',pad='20',fontsize=20,color='red')
    plt.hist(delta_z,bins=50,density=False)#,edgecolor='black')
    plt.ylabel('N',fontsize=20)
    plt.xlabel('$\Delta$'+'z',fontsize=20)
    plt.tick_params(labelsize=18)
    #plt.savefig('%s/delta_z.pdf'%(path))

    plt.figure(figsize=(10,10))
    plt.title('stddev=%.4f mean=%.5f'%(arr_std_2,arr_mean_2),fontdict=None,loc='center',pad='20',fontsize=20,color='red')
    plt.hist(delta_NHI,bins=100,density=False)#,edgecolor='black')
    plt.ylabel('N',fontsize=20)
    plt.xlabel('$\Delta$'+'log${N_{\mathregular{HI}}}$',fontsize=20)
    plt.tick_params(labelsize=18)
    #plt.savefig('%s/delta_NHI.pdf'%(path))


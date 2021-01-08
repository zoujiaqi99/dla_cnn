# dla_cnn
---------------------
It's a DLA finder for DESI QSO spectra based on Park's code.

Damped  Lyαsystems  (DLAs)  play  an  important  role  in  research  of  quasar  absorption  lines.We have updated and applied a convolutional neural network(CNN) machine learning model to discoverand characterize DLAs based on DESI mock spectra.We have optimized the training process and madethe CNN model performing well on low signal-to-noise (SNR≈1) mock spectra.This CNN model cangive  a  classification  accuracy  above  97%  and  make  excellent  estimates  for  redshift  and  HI  columndensity.We used different mock spectra to develop the algorithm and tested it on the realistic DESImock spectra.Besides, our model can find overlapping DLAs and sub-DLAs as well.Our work providesDLA catalogs which are very close to the real mock DLA catalogs.

We also tested the impact of differentDLA  catalogs  on  the  measurement  of  Baryon  Acoustic  Oscillation(BAO).It  confirms  that  our  DLAcatalogs can give an accurate fitting result compared to the real mock catalogs.And masking metalabsorptions and high column density absorbers can make the BAO fitting much better.Our DLA finderwill be mainly used in the future DESI quasar spectra.

The preprocess and training set steps are in  desi dir. The structure of CNN model is located in training_model dir.

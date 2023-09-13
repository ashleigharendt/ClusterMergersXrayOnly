import tensorflow as tf
import numpy as np
from astropy.io import fits
import os
from PIL import Image
import cv2

class BaseModel():
    """Methods associated with predicting whether galaxy clusters are merging."""
    
    def __init__(self):
        self.model = tf.keras.models.load_model('./model')
        
    
    def predict_merger_prob(self, X, normalise=True):
        """
        Function to return predictions on a set of images.
        
        Inputs
        --------
        X: numpy array - dimensions (N, 96, 96)
            
        normalise: bool, optional
            True (default) - images are normalised using model training data within the prediction process
            False - images are pre-normalised outside of model prediction, recommended if simulation data 
            is not generated in the pipeline used for the study, can use normalise function if needed
        
        Output
        --------
        numpy array with each element corresponding to the input cluster merger probability, dimensions (N,1)
        """
        
        X = X.reshape((-1,96,96,1))
        
        if not normalise:
            norm_layer_xr = self.model.get_layer("normalization") # Xray normalisation layer
            norm_model = tf.keras.Model(self.model.inputs, norm_layer_xr.output)
            new_model = tf.keras.Model(inputs=norm_layer_xr.output, outputs=self.model.layers[-1].output)
            y_prob = new_model.predict([X])
        
        else:
            y_prob = self.model.predict(X)
        
        return y_prob
    
    def normalise_img(self, X):
        """
        Inputs
        --------
        X: numpy array - dimensions (N, 96, 96, 1)
            
        Output
        --------
        As input, but with each channel independently normalised.
        
        """
        
        mean_xr = np.mean(X)
        std_xr = np.std(X)
        
        X = (X - mean_xr) / std_xr
       
        return X
    

    def read_fits(self, filename, resize=True):
        """Function to read in sz and xray fits files and convert to numpy array.
        Inputs
        --------
        filename: str
            Name of file which contains the fits file to read e.g. 'Xray.fits'
            
        resize: bool, optional
            True: resizes to 96x96 (use if original image size is different to 96x96)
            False: keeps original image shape
            
        Output
        --------
        numpy array - dimensions (N, 96, 96)
        
        """
        
        hdu = fits.open(filename, ignore_missing_simple=True)
        xr_img = hdu[0].data
        xr_img = np.array(Image.fromarray(xr_img))
        hdu.close()
        
        if resize:
            xr_img = cv2.resize(xr_img, dsize=(96, 96), interpolation=cv2.INTER_AREA)
                     
        npx = xr_img.shape[1]
        X = xr_img.reshape((-1,npx,npx))
        
        return X

        
        
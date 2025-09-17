
"""

@author: Dr. Ruslan Sherstyukov, Sodankyla Geophysical Observatory, 2025

"""


import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def set_path():
    
    models_directory = os.path.join(os.getcwd(), "Models")
    ionograms_directory = os.path.join(os.getcwd(), "2021P")
    return models_directory,ionograms_directory


def dice_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (2 * intersection + 1e-7) / (denominator + 1e-7)

def load_models():
    
    models_directory,ionograms_directory = set_path()


    # # Segmentation models
    ModelTraceE=tf.keras.models.load_model(os.path.join(models_directory,"E_trace_S0_5.h5"),custom_objects={'dice_loss': dice_loss})
    ModelTraceF1=tf.keras.models.load_model(os.path.join(models_directory,"F1_trace_S0_5.h5"),custom_objects={'dice_loss': dice_loss})
    ModelTraceF2=tf.keras.models.load_model(os.path.join(models_directory,"F2_trace_S0_5.h5"),custom_objects={'dice_loss': dice_loss})

    Models = {"TraceF2": ModelTraceF2,
              "TraceF1": ModelTraceF1,
              "TraceE": ModelTraceE} 
    return Models


def TraceShow(models = None, model_name = "TraceF2", ionogram_time = "2021-7-29-23-0"):
    
    # Models = load_models()
    
    models_directory,ionograms_directory = set_path()
    ionogram_name = os.path.join(ionograms_directory, f"{ionogram_time}.png")
    ionogram = load_img(ionogram_name, target_size=(256, 256),color_mode='grayscale')
    ionogram = img_to_array(ionogram)
    ionogram = ionogram.astype('float32') / 255.
    ionogram = np.expand_dims(ionogram, axis=0)
    trace = models[model_name].predict(ionogram)
    
    fig, axs1 = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    axs1.pcolor(ionogram[0,:,:,0] + trace[0,:,:,0]/5)
    axs1.set_xticks(ticks=np.arange(0, 256, 1 / 0.0262 * 0.4339))
    axs1.set_xticklabels(np.arange(0.5, 256 * 0.0262 / 0.4339 + 0.5262, 1))
    axs1.set_yticks(ticks=np.arange(0, 256, 100 / 2.8617 * 0.4876))
    axs1.set_yticklabels(np.arange(0, 256 * 2.8617 / 0.4876, 100))
    axs1.tick_params(axis='x', labelsize=30)
    axs1.tick_params(axis='y', labelsize=30)
    axs1.set_xlabel('frequency (MHz)', fontsize=30)
    axs1.set_ylabel('virtual height (km)', fontsize=30)
    axs1.set_title(ionogram_time, fontsize=40)
    plt.show()   
    plt.show()
    fig.savefig(f"{ionogram_time}.png")
    return trace
   
 
if __name__ == "__main__":
    print("ModelsEvaluation.py is running directly.")


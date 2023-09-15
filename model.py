import pandas as pd
import numpy as np
from numpy.linalg import norm
from IPython.display import display
import os

from tensorflow.keras.applications import vgg16, resnet50
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from load import DataLoad
from setting import SETTINGS


class SimModel():
    
    def __init__(self, poke_name, model_name):
        self._df = DataLoad()._df_load()
        self._input_name_idx = self._df[self._df['Name'] == poke_name].index[0]
        self._model_name = model_name
        
        
    def _model_select(self):
        
        if self._model_name == 'vgg16':
            model = vgg16.VGG16(weights = 'imagenet')
            return model, 'fc1'
        
        elif self._model_name == 'resnet50':
            model = resnet50.ResNet50(weights = 'imagenet')
            return model, 'avg_pool'
        
        
    def _extract(self):
        
        folder_dir = './data/extract/'
        os.makedirs(folder_dir, exist_ok = True)
        
        try:
            extract = np.load(folder_dir + self._model_name + '_extracted_array.npy')
            
        except:
            model, layer = self._model_select()

            extract_model = Model(inputs = model.input,
                                  outputs = model.get_layer(layer).output)
            
            img_array = DataLoad().loader()

            extract = extract_model.predict(img_array)
            
            np.save(folder_dir + self._model_name + '_extracted_array', extract)
            
            extract = np.load(folder_dir + self._model_name + '_extracted_array.npy')
        
        return extract
        
        
    def _cos_sin(self, A, B):
        return np.dot(A, B) / (norm(A) * norm(B))
    
    
    def _similarity(self):
        
        extract = self._extract()
        
        df_result = pd.DataFrame(columns = ['Name','similarity'])

        predicted1 = extract[self._input_name_idx].reshape(-1)



        for i in range(len(extract)):
            
            result_list = []
            
            predicted2 = extract[i].reshape(-1)
            
            result = self._cos_sin(predicted1, predicted2)
            
            result_list.append(self._df['Name'][i])
            result_list.append(result)
            
            df_result.loc[i] = result_list
            
            df_result = df_result.sort_values(by = 'similarity', ascending = False)
            result = df_result.iloc[2:7].reset_index(drop = True)
            
        display(result)
        return result
        
        
    def showimage(self):
        
        df = self._df
        sim_df = self._similarity()
        
        path = SETTINGS['path']
        fig,((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(7, 5))
        ax = [ax1, ax2, ax3, ax4, ax5, ax6]
        
        try:
            img = mpimg.imread(path + df['Name'][self._input_name_idx] + '.png')
        except:
            img = mpimg.imread(path + df['Name'][self._input_name_idx] + '.jpg')

        ax1.imshow(img)
        ax1.set_title(df['Name'][self._input_name_idx], color = 'r')
        ax1.axis('off')
        for i in range(5):
            similarity = str(round(sim_df['similarity'][i], 3))
            try:
                img = mpimg.imread(path + sim_df['Name'][i] + '.png')
            except:
                img = mpimg.imread(path + sim_df['Name'][i] + '.jpg')
            ax[i+1].imshow(img)
            ax[i+1].set_title(sim_df['Name'][i] + f'\nSimilarity: {similarity}')
            ax[i+1].axis('off')
        plt.tight_layout()
        plt.show()
import pandas as pd
import numpy as np
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

from setting import SETTINGS



path = SETTINGS['path']

class DataLoad:

    def __init__(self):
        pass


    def _df_load(self):
        df = pd.read_csv('./data/pokemon.csv')
        
        return df


    def loader(self):
        img_list = []

        for i in range(len(self._df_load().index)):
            try:
                img = Image.open(path + self._df_load()['Name'][i] + '.png').convert('RGB')
            except:
                img = Image.open(path + self._df_load()['Name'][i] + '.jpg').convert('RGB')
                
            resized_img = img.resize((224,224), resample = 1)
            
            x = np.array(resized_img)
            x[x==0] = 255
            
            img_list.append(x)
            
        img_array = np.array(img_list)
        
        return img_array
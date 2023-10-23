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
        '''
        1) poke_name => 유사도를 구하고 싶은 포켓몬 이름
        2) model_name => layer를 구하고 싶은 모델 이름
        '''
        
        self._df = DataLoad()._df_load()
        self._input_name_idx = self._df[self._df['Name'] == poke_name].index[0]  # 포켓몬 리스트 df 중에서 입력받은 포켓몬 이름의 인덱스를 반환
        self._model_name = model_name
        
        
    def _model_select(self):
        
        if self._model_name == 'vgg16':
            model = vgg16.VGG16(weights = 'imagenet')
            return model, 'fc1'
        
        elif self._model_name == 'resnet50':
            model = resnet50.ResNet50(weights = 'imagenet')
            return model, 'avg_pool'
        
        
    def _extract(self):
        '''
        1) folder_dir 경로를 생성. 이미 있을 경우에는 넘어감.
        2) FC layer 반환 과정
            a. forder_dir 경로에 model_name_extracted_array.npy 파일이 있으면 불러와서 사용.
            b. 파일이 없으면 입력받은 모델을 통해 fc layer에서 1d array를 반환.
        3) 반환받은 array를 저장하고 return.
        '''
        
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
        '''
        1) 입력받은 포켓몬과 모든 포켓몬과의 코사인 유사도를 계산하여 데이터프레임 생성
        2) 유사도 기준 상위 5개의 데이터를 출력 및 반환
        '''
        
        extract = self._extract()
        
        df_result = pd.DataFrame(columns = ['Name','similarity'])

        array_input = extract[self._input_name_idx].reshape(-1)

        for i in range(len(extract)):
            
            result_list = []
            
            array_all = extract[i].reshape(-1)
            
            result = self._cos_sin(array_input, array_all)
            
            result_list.append(self._df['Name'][i])
            result_list.append(result)
            
            df_result.loc[i] = result_list
            
        df_result = df_result.sort_values(by = 'similarity', ascending = False)
        result = df_result.iloc[2:7].reset_index(drop = True)
            
        display(result)
        return result
        
        
    def showimage(self):
        '''
        1) 포켓몬의 이미지와 유사도를 2*3 형태로 출력
        2) 좌상단에 입력받은 포켓몬 위치        
        '''
        
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
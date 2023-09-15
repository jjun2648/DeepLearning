import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names





class DataLoad():
    
    def __init__(self):
        pass
    
    
    def _df_load(self):
        print('data load')
        df = pd.read_csv('./data/movielens.csv')
        return df
    
    
    def _fixlen_feature_columns(self, df, sparse_features, dense_features):
        
        sparse_feat = [SparseFeat(feat, df[feat].max() + 1, embedding_dim = 4) for feat in sparse_features]
        dense_feat = [DenseFeat(feat, 1, ) for feat in dense_features]
        return sparse_feat + dense_feat
    
    
    def _feature_names(self, df, sparse_features, dense_features):
        
        fixlen_feature_columns = self._fixlen_feature_columns(df, sparse_features, dense_features)
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

        return feature_names, (linear_feature_columns, dnn_feature_columns)


    def preprocess(self):
        
        data = self._df_load()
        
        sparse_features = ['userId', 'title', 'genres', 'tag']
        dense_features = ['rating']
        
        target_col = ['target']
        
        encoders = []

        for i in range(len(sparse_features)):
            encoders.append(i)
            encoders[i] = LabelEncoder()
            data[sparse_features[i]] = encoders[i].fit_transform(data[sparse_features[i]])
            
        scaler = MinMaxScaler()
        data['rating'] = scaler.fit_transform((np.array(data['rating'])).reshape(-1, 1))
        
        
        feature_names, feature_columns = \
            self._feature_names(data, sparse_features, dense_features)
        
        
        train, test = train_test_split(data, test_size=0.2, random_state=777)
        train_model_input = {name:train[name].values for name in feature_names}
        test_model_input = {name:test[name].values for name in feature_names}
        
        return feature_columns, train_model_input, test_model_input, train, test, encoders
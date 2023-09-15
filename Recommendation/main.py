from IPython.display import display

from load import DataLoad
from model import DeepFM_Model



if __name__=="__main__":
    
    sparse_features = ['userId', 'title', 'genres', 'tag']
    target_col = ['target']
    

    feature_columns, train_model_input, test_model_input, train, test, encoders = DataLoad().preprocess()

    model = DeepFM_Model(feature_columns)
    
    pred = model.predict(train_model_input, train[target_col].values, test_model_input)

    model.evaluate(pred, test[target_col].values)
    
    test['pred'] = pred
    
    for i in range(len(sparse_features)):
        test[sparse_features[i]] = encoders[i].inverse_transform(test[sparse_features[i]])
        
    result = test[['userId', 'title', 'pred']].sort_values(by = ['userId', 'title']).reset_index(drop = True)    
    
    result.to_csv('./data/result.csv', index = False)
    display(result)
from sklearn.metrics import mean_squared_error

from deepctr.models import DeepFM




class DeepFM_Model():
    
    def __init__(self, feature_columns):
        self._feature_columns = feature_columns
    
    
    def _build_model(self):
        
        model = DeepFM(*self._feature_columns, task='regression')
        model.compile("adam", "mse", metrics=['mse'], )
        
        return model
        
        
    def train(self, input, target):
        
        model = self._build_model()
        
        history = model.fit(input, target,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
        
        print(history)
        return model
        
        
    def predict(self, train_input, target, test_input):
        
        pred_ans = self.train(train_input, target).predict(test_input, batch_size=256)
        
        return pred_ans
    
    
    def evaluate(self, pred, test_target_value):
        
        print("test MSE", round(mean_squared_error(test_target_value, pred), 4))
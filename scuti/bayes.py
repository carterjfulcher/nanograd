'''
cfulcher

A recursive algorithm to filter out noise 
'''

import numpy as np 

class Bayes(Object): 
    def __init__(self, model: np.array, measurements: np.array) -> None:
        self._model = model
        self._measurements = measurements 
    
    @classmethod 
    def forward(self): #predict the next position
        pass 
    
    @classmethod
    def feed(self, position: np.array, measurement: np.array) -> None: 
        ''' Feeds next step and calls self.forward() '''
        pass 
        
        



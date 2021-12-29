# Scuti 
A production ready filtering and prediction library

## How to use 
```python
from scuti.models import HiddenMarkov 
from scuti.filters import Kalman 

import numpy 


model = HiddenMarkov(data: list or np.array)
filter = Kalman()


pred = filter.predict(model.states, model.measurements) 
print(pred) 

```

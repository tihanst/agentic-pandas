import config
from pathlib import Path
from datetime import datetime
path_settings = config.FilePathConfig() #type: ignore

path = Path(path_settings.output_path).resolve().as_posix()
# Pandas


SYSTEM_PROMPT = """
You are an expert pandas coding agent. When given a task, respond with ONLY a python code block and nothing else. 
No explanation, no commentary, no markdown prose before or after. Always import pandas, numpy, datetime, and time.
Import whatever other python libraries you need, depending on the user's request.
You will be told what the current kernel state of the jupyter environment is when needed.
You will consider the kernel state and utilize this knowledge when you find it necessary in your coding task.

Your entire response must be a single fenced code block:

```python
# your code here
```

If you cannot complete the task, still respond only with a code block creating a 1 by 1 data frame containing a comment explaining why as a Python string.
If your code produces one or more final DataFrame results (NOT intermediate ones), you will save the DataFrames by naming the dataframe(s) with a logical name using the pl.DataFrame.name method. For each final DataFrame always save it at the end using the following code to define the file name:
timestamp = datetime.datetime.strptime(time.ctime(), "%a %b %d %H:%M:%S %Y").strftime("%Y-%m-%d_%H-%M-%S")
filename_variable = name + '_' + timestamp + '.csv'
Then append the filename to the below funciton call where indicated
result_df.to_csv(r"{path}/" + filename_variable", index=True)
"""


SYSTEM_PROMPT_WITH_STEPS = """
You are an expert in pandas coding agent. When given a task, respond with ONLY a python code block and nothing else. 
No explanation, no commentary, no markdown prose before or after. Always import pandas, numpy, datetime, and time.
Import whatever other python libraries you need, depending on the user's request.
You will be told what the current kernel state of the jupyter environment is when needed.
You will consider the kernel state and utilize this knowledge when you find it necessary in your coding task.

Your entire response must be a single fenced code block:

```python
# your code here
```

If you cannot complete the task, still respond only with a code block containing a comment explaining why as a Python comment.
If your code produces one or more final DataFrame results, you will save the DataFrames by naming the dataframe(s) with a logical name using the pd.DataFrame.name method. For each final DataFrame always save it at the end using the following code to define the file name:
timestamp = datetime.datetime.strptime(time.ctime(), "%a %b %d %H:%M:%S %Y").strftime("%Y-%m-%d_%H-%M-%S")
filename_variable = name + '_' + timestamp + '.csv'
Then append the filename to the below funciton call where indicated
result_df.to_csv(r"{path}/" + filename_variable, index=True)
If you produce intermediate DataFrames and are asked to save them by the user, save them with the prefix 'STEP_X' appended, where X is the step number.
"""


LOAD_STATE = r"""
import pandas as pd
import numpy as np
import time
import datetime

initial_data_frame = pd.read_csv('{filepath}/{file}')
"""

STATE_PROBE = r"""
import pandas as pd, json

class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, datetime):
            return str(obj)
        return super().default(obj)

_state = {}
for _name, _obj in list(globals().items()):
    if _name.startswith('_'): continue
    if isinstance(_obj, pd.DataFrame):
        _state[_name] = {
            'type': 'DataFrame',
            'shape': list(_obj.shape),
            'columns': list(_obj.columns),
            'dtypes': {c: str(t) for c, t in _obj.dtypes.items()},
            'head': _obj.head(1).to_dict(orient='records'),
            'nulls': _obj.isnull().sum().to_dict()
        }
    elif isinstance(_obj, pd.Series):
        _state[_name] = {'type': 'Series', 'len': len(_obj), 'dtype': str(_obj.dtype)}
    elif isinstance(_obj, (int, float, str, bool, list, dict)):
        _state[_name] = {'type': type(_obj).__name__, 'value': str(_obj)[:200]}

print(json.dumps(_state, cls=_Encoder))
"""

# ----------------------------------------------

# # Polars

# SYSTEM_PROMPT = f"""
# You are an expert polars coding agent. When given a task, respond with ONLY a python code block and nothing else. 
# No explanation, no commentary, no markdown prose before or after. Always import polars, numpy, datetime, and time.
# Import whatever other python libraries you need, depending on the user's request.
# You will be told what the current kernel state of the jupyter environment is when needed.
# You will consider the kernel state and utilize this knowledge when you find it necessary in your coding task.

# Your entire response must be a single fenced code block:

# ```python
# # your code here
# ```

# If you cannot complete the task, still respond only with a code block containing a comment explaining why as a Python comment.
# If your code produces one or more final DataFrame results (NOT intermediate ones), you will save the DataFrames by naming the dataframe(s) with a logical name using the pl.DataFrame.name method. For each final DataFrame always save it at the end using the following code to define the file name:
# timestamp = datetime.datetime.strptime(time.ctime(), "%a %b %d %H:%M:%S %Y").strftime("%Y-%m-%d_%H-%M-%S")
# filename_variable = name + _ + timestamp + '.csv'
# Then append the filename to the below funciton call where indicated
# result_df.to_csv(r"{settings.output_path}+filename_variable", index=True)
# """


# LOAD_STATE = r"""
# import polars as pl
# import numpy as np
# import time
# import datetime

# initial_data_frame = pl.read_csv('{filepath}/{file}')
# """

# STATE_PROBE = r"""
# import polars as pl, json

# class _Encoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, pl.Timestamp):
#             return str(obj)
#         if isinstance(obj, datetime):
#             return str(obj)
#         return super().default(obj)

# _state = {}
# for _name, _obj in list(globals().items()):
#     if _name.startswith('_'): continue
#     if isinstance(_obj, pl.DataFrame):
#         _state[_name] = {
#             'type': 'DataFrame',
#             'shape': list(_obj.shape),
#             'columns': list(_obj.columns),
#             'dtypes': {c: str(t) for c, t in _obj.dtypes.items()},
#             'head': _obj.head(1).to_dict(orient='records'),
#             'nulls': _obj.isnull().sum().to_dict()
#         }
#     elif isinstance(_obj, pl.Series):
#         _state[_name] = {'type': 'Series', 'len': len(_obj), 'dtype': str(_obj.dtype)}
#     elif isinstance(_obj, (int, float, str, bool, list, dict)):
#         _state[_name] = {'type': type(_obj).__name__, 'value': str(_obj)[:200]}

# print(json.dumps(_state, cls=_Encoder))
# """
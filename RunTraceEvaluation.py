
"""

@author: Dr. Ruslan Sherstyukov, Sodankyla Geophysical Observatory, 2025

"""

import TraceModelEvaluation

# Load avaliable models
Models = TraceModelEvaluation.load_models()

# Specify the model for evaluation
# model_names: "TraceF2","TraceF1","TraceE"
model_name = "TraceF2"

# Specify the ionogram time
# ionogram_time: "2021-1-1-1-0"
ionogram_time = "2021-7-29-23-0"

TraceModelEvaluation.TraceShow(Models, model_name, ionogram_time)




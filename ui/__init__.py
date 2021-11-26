from .list import run_list
from .predict import run_predict
from .training import run_training

pages = {
    'Training': run_training,
    'Predict': run_predict,
    'List': run_list
}

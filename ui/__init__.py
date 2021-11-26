from .predict import run_predict
from .training import run_training
from .list import run_list

pages = {
    'Training': run_training,
    'Predict': run_predict,
    'List': run_list
}

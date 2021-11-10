from .explore import run_explore
from .predict import run_predict
from .training import run_training

pages = {
    'Training': run_training,
    'Predict': run_predict,
    'Explore': run_explore,
}

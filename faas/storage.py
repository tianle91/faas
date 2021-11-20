import logging
import random
import string
from datetime import datetime

from sqlitedict import SqliteDict

from faas.wrapper.lightgbm import ETLWrapperForLGBM

logger = logging.getLogger(__name__)

MODEL_STORE = 'model_store.db'
KEY_LENGTH = 24


def id_gen():
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(KEY_LENGTH))


def create_key() -> str:
    with SqliteDict(MODEL_STORE) as d:
        k = id_gen()
        while k in d:
            k = id_gen()
        return k


def list_models() -> dict:
    d_print = {}
    with SqliteDict(MODEL_STORE) as d:
        for k, v in d.items():
            dt, e = v
            e: ETLWrapperForLGBM = e
            d_print[k] = {'timestamp': dt, 'config': e.config}
    return d_print


def write_model(model: ETLWrapperForLGBM) -> str:
    with SqliteDict(MODEL_STORE) as d:
        key = create_key()
        d[key] = (datetime.now(), model)
        d.commit()
        logger.info(f'Model store at: {MODEL_STORE} is now {len(d)}')
    return key


def read_model(key: str) -> ETLWrapperForLGBM:
    with SqliteDict(MODEL_STORE) as d:
        if key not in d:
            raise KeyError(f'Key: {key} not found!')
        else:
            _, model = d[key]
            return model

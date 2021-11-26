import logging
import pprint as pp
import random
import string
from datetime import datetime
from typing import Tuple

from sqlitedict import SqliteDict

from faas.config import Config
from faas.lightgbm import ETLWrapperForLGBM

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


def write_model(model: ETLWrapperForLGBM, conf: Config) -> str:
    with SqliteDict(MODEL_STORE) as d:
        key = create_key()
        dt = datetime.now()
        d[key] = (dt, model, conf)
        d.commit()
        logger.info(f'Wrote model with key: {key} at {dt}')
        logger.info(f'Model store at: {MODEL_STORE} is now {len(d)}')
    return key


def read_model(key: str) -> Tuple[ETLWrapperForLGBM, Config]:
    with SqliteDict(MODEL_STORE) as d:
        if key not in d:
            raise KeyError(f'Key: {key} not found!')
        else:
            dt, model, conf = d[key]
            logger.info(f'Read model with key: {key} created at {dt}')
            return model, conf


def list_models() -> dict:
    l = []
    with SqliteDict(MODEL_STORE) as d:
        for key in d:
            dt, _, conf = d[key]
            l.append(f'dt: {dt} key: {key}\n{pp.pformat(conf.__dict)}')
    sep = '\n' + '-' * 1000 + '\n'
    return sep.join(l)

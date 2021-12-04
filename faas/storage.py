import logging
import random
import string
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

from sqlitedict import SqliteDict

from faas.config import Config
from faas.transformer.lightgbm import ETLWrapperForLGBM

logger = logging.getLogger(__name__)

MODEL_STORE = 'model_store.db'
KEY_LENGTH = 24


@dataclass
class StoredModel:
    dt: datetime
    m: ETLWrapperForLGBM
    config: Config
    num_calls_remaining: int = 100


def id_gen():
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(KEY_LENGTH))


def create_key() -> str:
    with SqliteDict(MODEL_STORE) as d:
        k = id_gen()
        while k in d:
            k = id_gen()
        return k


def write_model(stored_model: StoredModel) -> str:
    with SqliteDict(MODEL_STORE) as d:
        key = create_key()
        d[key] = stored_model
        d.commit()
        logger.info(f'Wrote model with key: {key}')
        logger.info(f'Model store at: {MODEL_STORE} is now {len(d)}')
    return key


def read_model(key: str) -> StoredModel:
    with SqliteDict(MODEL_STORE) as d:
        if key not in d:
            raise KeyError(f'Key: {key} not found!')
        else:
            logger.info(f'Read model with key: {key}')
            return d[key]


def set_num_calls_remaining(key: str, n: int):
    with SqliteDict(MODEL_STORE) as d:
        if key not in d:
            raise KeyError(f'Key: {key} not found!')
        else:
            stored_model: StoredModel = d[key]
            n_prev = stored_model.num_calls_remaining
            stored_model.num_calls_remaining = n
            d[key] = stored_model
            d.commit()
            logger.info(f'Updated num_calls_remaining for {key}: {n_prev} -> {n}')
            return d[key]


def list_models() -> Dict[str, StoredModel]:
    with SqliteDict(MODEL_STORE) as d:
        return {k: v for k, v in d.items()}

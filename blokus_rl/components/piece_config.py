import enum
from typing import (
    Callable, Dict
)

import numpy as np


class PieceConfig:
    __slots__ = ['name', 'shape', 'operations']

    def __init__(self, name: str, shape: np.ndarray, operations: Dict[str, Callable[[np.ndarray], np.ndarray]]):
        self.name = name
        self.shape = shape
        self.operations = operations


class PieceOperations(str, enum.Enum):
    ROT90 = 'ROT90'
    ROT180 = 'ROT180'
    ROT270 = 'ROT270'
    ROT90_FLIPH = 'ROT90_FLIPH'
    ROT180_FLIPH = 'ROT180_FLIPH'
    ROT90_FLIPV = 'ROT90_FLIPV'
    ROT180_FLIPV = 'ROT180_FLIPV'


OPERATION_CONFIG = {
    PieceOperations.ROT90: lambda x: np.rot90(x, k=1, axes=(0, 1)),
    PieceOperations.ROT180: lambda x: np.rot90(x, k=2, axes=(0, 1)),
    PieceOperations.ROT270: lambda x: np.rot90(x, k=3, axes=(0, 1)),
    PieceOperations.ROT90_FLIPH: lambda x: np.rot90(np.flipud(x), k=1, axes=(0, 1)),
    PieceOperations.ROT180_FLIPH: lambda x: np.rot90(np.flipud(x), k=2, axes=(0, 1)),
    PieceOperations.ROT90_FLIPV: lambda x: np.rot90(np.fliplr(x), k=1, axes=(0, 1)),
    PieceOperations.ROT180_FLIPV: lambda x: np.rot90(np.fliplr(x), k=2, axes=(0, 1))
}

PIECE_CONFIG = {
    '1': PieceConfig(
        name='1',
        shape=np.array([
            [1]
        ], dtype=np.dtype('u1')),
        operations={}
    ),
    '2': PieceConfig(
        name='2',
        shape=np.array([
            [1, 1]
        ], dtype=np.dtype('u1')),
        operations={
            PieceOperations.ROT90: OPERATION_CONFIG[PieceOperations.ROT90]
        }
    ),
    'V3': PieceConfig(
        name='V3',
        shape=np.array([
            [1, 1],
            [0, 1]
        ], dtype=np.dtype('u1')),
        operations={
            PieceOperations.ROT90: OPERATION_CONFIG[PieceOperations.ROT90],
            PieceOperations.ROT180: OPERATION_CONFIG[PieceOperations.ROT180],
            PieceOperations.ROT270: OPERATION_CONFIG[PieceOperations.ROT270]
        }
    ),
    'I3': PieceConfig(
        name='I3',
        shape=np.array([
            [1, 1, 1]
        ], dtype=np.dtype('u1')),
        operations={
            PieceOperations.ROT90: OPERATION_CONFIG[PieceOperations.ROT90]
        }
    ),
    'T4': PieceConfig(
        name='T4',
        shape=np.array([
            [0, 1, 0],
            [1, 1, 1]
        ], dtype=np.dtype('u1')),
        operations={
            PieceOperations.ROT90: OPERATION_CONFIG[PieceOperations.ROT90],
            PieceOperations.ROT180: OPERATION_CONFIG[PieceOperations.ROT180],
            PieceOperations.ROT270: OPERATION_CONFIG[PieceOperations.ROT270]
        }
    ),
    'O': PieceConfig(
        name='O',
        shape=np.array([
            [1, 1],
            [1, 1]
        ], dtype=np.dtype('u1')),
        operations={}
    ),
    'L4': PieceConfig(
        name='L4',
        shape=np.array([
            [1, 0, 0],
            [1, 1, 1]
        ], dtype=np.dtype('u1')),
        operations=OPERATION_CONFIG
    ),
    'I4': PieceConfig(
        name='I4',
        shape=np.array([
            [1, 1, 1, 1]
        ], dtype=np.dtype('u1')),
        operations={
            PieceOperations.ROT90: OPERATION_CONFIG[PieceOperations.ROT90]
        }
    ),
    'Z4': PieceConfig(
        name='Z4',
        shape=np.array([
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.dtype('u1')),
        operations={
            PieceOperations.ROT90: OPERATION_CONFIG[PieceOperations.ROT90],
            PieceOperations.ROT90_FLIPH: OPERATION_CONFIG[PieceOperations.ROT90_FLIPH],
            PieceOperations.ROT180_FLIPH: OPERATION_CONFIG[PieceOperations.ROT180_FLIPH]
        }
    ),
    'F': PieceConfig(
        name='F',
        shape=np.array([
            [0, 1, 0],
            [1, 1, 1],
            [1, 0, 0]
        ], dtype=np.dtype('u1')),
        operations=OPERATION_CONFIG
    ),
    'X': PieceConfig(
        name='X',
        shape=np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=np.dtype('u1')),
        operations={}
    ),
    'P': PieceConfig(
        name='P',
        shape=np.array([
            [1, 1],
            [1, 1],
            [1, 0]
        ], dtype=np.dtype('u1')),
        operations=OPERATION_CONFIG
    ),
    'W': PieceConfig(
        name='W',
        shape=np.array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1]
        ], dtype=np.dtype('u1')),
        operations={
            PieceOperations.ROT90: OPERATION_CONFIG[PieceOperations.ROT90],
            PieceOperations.ROT180: OPERATION_CONFIG[PieceOperations.ROT180],
            PieceOperations.ROT270: OPERATION_CONFIG[PieceOperations.ROT270]
        }
    ),
    'Z5': PieceConfig(
        name='Z5',
        shape=np.array([
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 1]
        ], dtype=np.dtype('u1')),
        operations={
            PieceOperations.ROT90: OPERATION_CONFIG[PieceOperations.ROT90],
            PieceOperations.ROT90_FLIPH: OPERATION_CONFIG[PieceOperations.ROT90_FLIPH],
            PieceOperations.ROT180_FLIPH: OPERATION_CONFIG[PieceOperations.ROT180_FLIPH]
        }
    ),
    'Y': PieceConfig(
        name='Y',
        shape=np.array([
            [1, 1, 1, 1],
            [0, 1, 0, 0]
        ], dtype=np.dtype('u1')),
        operations=OPERATION_CONFIG
    ),
    'L5': PieceConfig(
        name='L5',
        shape=np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 0]
        ], dtype=np.dtype('u1')),
        operations=OPERATION_CONFIG
    ),
    'U': PieceConfig(
        name='U',
        shape=np.array([
            [1, 1, 1],
            [1, 0, 1]
        ], dtype=np.dtype('u1')),
        operations={
            PieceOperations.ROT90: OPERATION_CONFIG[PieceOperations.ROT90],
            PieceOperations.ROT180: OPERATION_CONFIG[PieceOperations.ROT180],
            PieceOperations.ROT270: OPERATION_CONFIG[PieceOperations.ROT270]
        }
    ),
    'T5': PieceConfig(
        name='T5',
        shape=np.array([
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1]
        ], dtype=np.dtype('u1')),
        operations={
            PieceOperations.ROT90: OPERATION_CONFIG[PieceOperations.ROT90],
            PieceOperations.ROT180: OPERATION_CONFIG[PieceOperations.ROT180],
            PieceOperations.ROT270: OPERATION_CONFIG[PieceOperations.ROT270]
        }
    ),
    'V5': PieceConfig(
        name='V5',
        shape=np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1]
        ], dtype=np.dtype('u1')),
        operations={
            PieceOperations.ROT90: OPERATION_CONFIG[PieceOperations.ROT90],
            PieceOperations.ROT180: OPERATION_CONFIG[PieceOperations.ROT180],
            PieceOperations.ROT270: OPERATION_CONFIG[PieceOperations.ROT270]
        }
    ),
    'N': PieceConfig(
        name='N',
        shape=np.array([
            [0, 1, 1, 1],
            [1, 1, 0, 0]
        ], dtype=np.dtype('u1')),
        operations=OPERATION_CONFIG
    ),
    'I5': PieceConfig(
        name='I5',
        shape=np.array([
            [1, 1, 1, 1, 1]
        ], dtype=np.dtype('u1')),
        operations={
            PieceOperations.ROT90: OPERATION_CONFIG[PieceOperations.ROT90],
        }
    )
}

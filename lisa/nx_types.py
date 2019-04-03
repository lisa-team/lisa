import numpy as np
from typing import Tuple, Iterable, Dict

NodeExpd = Tuple[int, int, str]
NodesExpd = Iterable[NodeExpd]
XYDict = Dict[str, float]
EdgeExpd = Tuple[NodeExpd, NodeExpd, XYDict]
EdgesExpd = Iterable[EdgeExpd]
NodeExpdToXYNP = Dict[NodeExpd, np.ndarray]
NodeID = int

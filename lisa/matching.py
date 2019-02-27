# -*- coding: utf-8 -*-
""" Functions used for map matching

This module contains functions for matching long/lat coordinates to our graph
representations. There are two main functions here, one for matching a single
long/lat (for example a stop sign) to a node, and one for matching multiple
coordinates to a path through the graph.

Example:
    TODO insert some examples if needed

"""

from lisa.lisagraph import LISAGraph
from typing import Tuple, List

# TODO this file is mostly Uma and Nathan's I believe?


def match_single(g: LISAGraph, coord: Tuple[float, float]):
    """This is a single line describing function.

    This is your more detailed breakdown of the functions inner workings if
    necessary. This can be multiple lines, and is separated from the one liner
    with a blank line.

    Args:
        g (lisa.expansion.Graph): the graph to match onto
        coord (Tuple[float, float]): the latitude and longitude to match

    Returns:
        int: the best-match node ID

    """
    pass


def match_trace(g: LISAGraph, trace: List[Tuple[float, float]]):
    """This is a single line describing function.

    This is your more detailed breakdown of the functions inner workings if
    necessary. This can be multiple lines, and is separated from the one liner
    with a blank line.

    Args:
        g (lisa.expansion.Graph): the graph to match onto
        trace (List[Tuple[float, float]]): a series of (lat, long) points to
            match

    Returns:
        [int]: the list of best match node IDs

    """
    pass


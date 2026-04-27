"""Memory layer — cross-question learning.

- `traces`: store every forecast with its full trace, queryable.
- `failures`: cluster losing trades into named patterns. The next iteration
  targets the largest cluster.
"""

from clio.memory.traces import TraceStore, Trace
from clio.memory.failures import FailureClusterer, FailureCluster, FailureLabel

__all__ = [
    "TraceStore",
    "Trace",
    "FailureClusterer",
    "FailureCluster",
    "FailureLabel",
]

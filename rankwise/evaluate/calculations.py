# Copyright 2024 Banco Bilbao Vizcaya Argentaria, S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from warnings import warn

from rankwise.evaluate.data import metrics_statistics


def _calculate_evaluation_metrics_statistics(metrics):
    def _calculate_metric_statistic(values):
        statistics = dict()
        for metric_name, metric_fn in metrics_statistics.items():
            try:
                statistics[metric_name] = metric_fn(values)
            except Exception as exc:
                warn(f"Error calculating metric {metric_name!r}, namely {exc!r}")
                statistics[metric_name] = float("nan")
        return statistics
    return {metric: _calculate_metric_statistic(values) for metric, values in metrics.items()}


def build_evaluation_report(metrics, model_name, top_k, nodes, input_queries):
    return {
        "model": model_name,
        "top_k": top_k,
        "nodes_count": len(nodes),
        "queries_count": len(input_queries),
        "metrics": _calculate_evaluation_metrics_statistics(metrics),
    }


class Appender(defaultdict):
    """
    A dictionary-like collection where values are lists, with a monoidal
    `accumulate` operation that appends elements from any given dictionary-like
    object.
    """

    def __init__(self, *args, **kwargs):
        super(Appender, self).__init__(*args, **kwargs)
        self.default_factory = list

    def accumulate(self, other):
        for key, value in other.items():
            self[key].append(value)

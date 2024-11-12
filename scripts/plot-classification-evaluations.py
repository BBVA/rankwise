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

import argparse
import json
from os.path import basename

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def read_jsonl_file(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def plot_metrics(metrics_list, metrics_names):
    metrics_list.sort(key=lambda x: (x["f1"], x["precision"]))
    filenames = [basename(metrics["filename"]) for metrics in metrics_list]
    fig, ax = plt.subplots()

    for metric in metrics_names:
        values = [metrics[metric] for metrics in metrics_list]
        ax.plot(filenames, values, marker=".", label=metric)

    fig.subplots_adjust(bottom=0.55, left=0.15, right=0.98, top=0.96)
    ax.set_title("Classifications")
    ax.grid(True, linestyle="--")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.legend()
    ax.set_xticks(filenames, filenames, rotation=45, ha="right")
    fig.tight_layout()
    fig.set_size_inches(14, 12)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    args = parser.parse_args()

    metrics_list = read_jsonl_file(args.input)
    metrics_names = ["accuracy", "precision", "recall", "f1"]
    plot_metrics(metrics_list, metrics_names)


if __name__ == "__main__":
    main()

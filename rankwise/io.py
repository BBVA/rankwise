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

from functools import wraps
import json
import sys

from rankwise.data import InputData


def read_evaluate_input(file):
    accumulated_content = set()
    queries_with_expected_sorted_content = dict()
    for line in file:
        data = json.loads(line)
        accumulated_content.update(data["contents"])
        queries_with_expected_sorted_content[data["query"]] = data["contents"]

    return InputData(
        queries_with_expected_sorted_content=queries_with_expected_sorted_content,
        accumulated_content=accumulated_content,
    )


def read_generate_input(file):
    return set(file)


def as_jsonlines(fn):
    @wraps(fn)
    def _as_jsonlines(*args, **kwargs):
        for entry in fn(*args, **kwargs):
            sys.stdout.buffer.write(json.dumps(entry).encode("utf-8"))
            sys.stdout.buffer.write(b"\n")

    return _as_jsonlines

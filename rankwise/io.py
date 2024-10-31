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

import json
import sys
from functools import wraps

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
    return list(json.loads(line) for line in file)


def as_jsonlines(fn):
    @wraps(fn)
    def _as_jsonlines(cli_args, *args, **kwargs):
        output_file = sys.stdout.buffer if cli_args.output_file is None else cli_args.output_file
        for entry in fn(cli_args, *args, **kwargs):
            output_file.write(json.dumps(entry, ensure_ascii=False).encode("utf-8"))
            output_file.write(b"\n")

    return _as_jsonlines

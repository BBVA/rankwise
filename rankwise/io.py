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
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from tqdm import tqdm

from rankwise.data import ClassificationData, InputData


def read_json_lines_input(file):
    return list(json.loads(line) for line in file)


def read_evaluate_embedding_input(file):
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


def read_evaluate_classification_input(file):
    data = read_json_lines_input(file)
    good_questions_with_document = {
        good_question: item["document"]
        for item in data
        for good_question in item["questions"]["good"]
    }
    bad_questions = [bad_question for item in data for bad_question in item["questions"]["bad"]]

    return ClassificationData(
        good_questions_with_document=good_questions_with_document, bad_questions=bad_questions
    )


def as_jsonlines(fn):
    @wraps(fn)
    def _as_jsonlines(cli_args, *args, **kwargs):
        output_file = sys.stdout.buffer if cli_args.output_file is None else cli_args.output_file
        for entry in fn(cli_args, *args, **kwargs):
            output_file.write(json.dumps(entry, ensure_ascii=False).encode("utf-8"))
            output_file.write(b"\n")

    return _as_jsonlines


def get_document_embeddings(documents, embedding_function, max_workers=5, show_progress=False):
    counter_lock = threading.Lock()

    def _embedding_function_with_progress(document, pbar):
        embedding = embedding_function(document)
        with counter_lock:
            pbar.update(1)
        return embedding

    pbar = tqdm(total=len(documents), disable=not show_progress)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        embeddings = list(
            executor.map(lambda doc: _embedding_function_with_progress(doc, pbar), documents)
        )
    return embeddings


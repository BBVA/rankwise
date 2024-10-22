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

from llama_index.core.llama_dataset.generator import RagDatasetGenerator

from rankwise.calculations import content_to_node


def generate_dataset(model, contents, queries_count, question_gen_query):
    nodes = [content_to_node(c) for c in contents]
    dataset_generator = RagDatasetGenerator(
        llm=model,
        nodes=nodes,
        num_questions_per_chunk=queries_count,
        question_gen_query=question_gen_query,
    )
    return dataset_generator.generate_dataset_from_nodes()


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

import asyncio

from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import RetrieverEvaluator
from tqdm.asyncio import tqdm as tqdm_asyncio

from rankwise.calculations import content_id, content_to_node
from rankwise.evaluate.embedding_model.calculations import Appender


def build_evaluator(embed_model, top_k, contents, metrics):
    nodes = [content_to_node(c) for c in contents]
    index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=top_k)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=retriever)

    async def _evaluate(query, contents_in_expected_order):
        return await retriever_evaluator.aevaluate(
            query=query,
            expected_ids=[content_id(c) for c in contents_in_expected_order],
        )

    return _evaluate


def accumulate_evaluation_metrics(
    evaluate_fn, queries_with_expected_sorted_content, max_workers=5, show_progress=False
):
    metrics = Appender()
    evaluation_items = list(queries_with_expected_sorted_content.items())
    semaphore = asyncio.Semaphore(max_workers)
    pbar = tqdm_asyncio(total=len(evaluation_items), disable=not show_progress)

    async def _evaluate(query, contents_in_expected_order):
        async with semaphore:
            result = await evaluate_fn(query, contents_in_expected_order)
            pbar.update(1)
            return result

    async def run_tasks():
        tasks = [
            asyncio.create_task(_evaluate(query, contents_in_expected_order))
            for query, contents_in_expected_order in evaluation_items
        ]
        for coro in asyncio.as_completed(tasks):
            evaluation = await coro
            metrics.accumulate(evaluation.metric_vals_dict)

        pbar.close()
        return metrics

    return asyncio.run(run_tasks())

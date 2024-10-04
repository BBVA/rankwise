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
import sys

from rankwise.evaluate.calculations import build_evaluation_report
from rankwise.evaluate.io import accumulate_evaluation_metrics, build_evaluator
from rankwise.generate.data import DEFAULT_QUESTION_PROMPT
from rankwise.generate.io import generate_dataset
from rankwise.importer.io import (UndefinedEnvVarError, import_embedding_model,
                                  import_llm_model)
from rankwise.io import as_jsonlines, read_evaluate_input, read_generate_input


def exceptions_to_argument_errors(import_function):
    def _import_with_argument_errors(expression):
        try:
            return import_function(expression)
        except ImportError as exc:
            raise argparse.ArgumentTypeError(
                f"Could not import embedding model {expression!r}"
            ) from exc
        except UndefinedEnvVarError as exc:
            raise argparse.ArgumentTypeError(
                f"Undefined environment variable {exc.env_var!r}"
            ) from exc
        except Exception as exc:
            raise argparse.ArgumentTypeError(
                f"Could not instantiate embedding model {expression!r}, reason {exc}"
            )

    return _import_with_argument_errors


@as_jsonlines
def run_evaluate_subcommand(args):
    input_data = read_evaluate_input(args.input)

    for embedding_model in args.embedding_model:
        evaluate_fn = build_evaluator(
            embedding_model.instance,
            args.top_k,
            input_data.accumulated_content,
            args.metrics,
        )
        accumulated_metrics = accumulate_evaluation_metrics(
            evaluate_fn, input_data.queries_with_expected_sorted_content
        )
        evaluation_report = build_evaluation_report(
            accumulated_metrics,
            embedding_model.expression,
            args.top_k,
            input_data.accumulated_content,
            input_data.queries_with_expected_sorted_content,
        )

        yield evaluation_report


@as_jsonlines
def run_generate_subcommand(args):
    input_data = read_generate_input(args.input)

    if args.question_prompt_file is None:
        prompt = DEFAULT_QUESTION_PROMPT
    else:
        prompt = args.question_prompt_file.read()

    dataset = generate_dataset(
        args.model.instance,
        input_data,
        args.queries_count,
        prompt,
    )

    for example in dataset.examples:
        yield {"query": example.query, "contents": example.reference_contexts}


def make_parser():
    parser = argparse.ArgumentParser(description="Rankwise: A tool for evaluating embedding models")
    subparsers = parser.add_subparsers(dest="command", title="commands", required=True)

    # Evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate embedding models")
    evaluate_parser.add_argument(
        "-E",
        "--embedding-model",
        action="append",
        type=exceptions_to_argument_errors(import_embedding_model),
        help="Name of the embedding model to use",
    )
    evaluate_parser.add_argument(
        "--input",
        type=argparse.FileType("r"),
        required=False,
        default=sys.stdin,
        help="Input file",
    )
    evaluate_parser.add_argument(
        "-k",
        "--top-k",
        dest="top_k",
        type=int,
        required=False,
        default=10,
        help="Top K results to consider",
    )
    evaluate_parser.add_argument(
        "-m",
        "--metrics",
        action="append",
        type=str,
        choices=["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"],
        required=False,
        default=["hit_rate", "mrr"],
        help="Metrics to calculate",
    )
    evaluate_parser.set_defaults(func=run_evaluate_subcommand)

    # Dataset
    generate_parser = subparsers.add_parser("generate", help="Generate a dataset for evaluation")
    generate_parser.add_argument(
        "-M",
        "--model",
        type=exceptions_to_argument_errors(import_llm_model),
        required=True,
        help="Name of the model to use",
    )
    generate_parser.add_argument(
        "--input",
        type=argparse.FileType("r"),
        required=False,
        default=sys.stdin,
        help="Input file",
    )
    generate_parser.add_argument(
        "--question-prompt-file",
        type=argparse.FileType("r"),
        required=False,
        help="Question prompt file",
    )
    generate_parser.add_argument(
        "-q",
        "--queries-count",
        dest="queries_count",
        type=int,
        required=False,
        default=3,
        help="Number of queries to generate per content",
    )
    generate_parser.set_defaults(func=run_generate_subcommand)

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)

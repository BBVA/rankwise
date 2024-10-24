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
from functools import partial

import rankwise.classify.cross_entropy.io
import rankwise.classify.cosine_distance.io
import rankwise.classify.calculations

from rankwise.evaluate.calculations import build_evaluation_report
from rankwise.evaluate.io import accumulate_evaluation_metrics, build_evaluator
from rankwise.generate.data import DEFAULT_QUESTION_PROMPT
from rankwise.generate.io import generate_dataset
from rankwise.generate.calculations import format_output
from rankwise.importer.io import (
    UndefinedEnvVarError,
    import_embedding_model,
    import_llm_model,
    import_cross_encoder,
)
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

    for document in input_data:
        generation = generate_dataset(
            args.model.instance,
            [document],
            args.queries_count,
            prompt,
        )
        yield {
            "document": document,
            "questions": [example.query for example in generation.examples],
        }


@as_jsonlines
def run_classify_cross_encoder_subcommand(args):
    input_data = read_generate_input(args.input)

    cross_encoder_distance = partial(
        rankwise.classify.cosine_distance.io.calculate_distance,
        model=args.cross_encoder_model.instance,
    )

    # input_data :: {"document": "DocA", "questions": ["Q1", "Q2"]}
    all_documents = set(row["document"] for row in input_data)
    is_best_according_to_cross_encoder = partial(
        rankwise.classify.calculations.is_best,
        distance_fn=cross_encoder_distance,
        comparison_fn=rankwise.classify.calculations.strictly_greatest,
        all_documents=all_documents,
    )
    for row in input_data:
        good, bad = list(), list()
        this_document, questions = row["document"], row["questions"]

        for question in questions:
            collection = (
                good if is_best_according_to_cross_encoder(question, this_document) else bad
            )
            collection.append(question)

        yield format_output(this_document, good, bad)


@as_jsonlines
def run_classify_cosine_similarity_subcommand(args):
    input_data = read_generate_input(args.input)

    embedding_functions = [model.instance.get_text_embedding for model in args.embedding_model]
    calculate_average_cosine_distance = (
        rankwise.classify.cosine_distance.io.build_distance_function(embedding_functions)
    )

    # input_data :: {"document": "DocA", "questions": ["Q1", "Q2"]}
    all_documents = set(row["document"] for row in input_data)
    is_best_according_to_cosine_similarity = partial(
        rankwise.classify.calculations.is_best,
        distance_fn=calculate_average_cosine_distance,
        comparison_fn=rankwise.classify.calculations.strictly_greatest,
        all_documents=all_documents,
    )
    for row in input_data:
        good, bad = list(), list()
        this_document, questions = row["document"], row["questions"]

        for question in questions:
            collection = (
                good if is_best_according_to_cosine_similarity(question, this_document) else bad
            )
            collection.append(question)

        yield format_output(this_document, good, bad)


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
        help="Embedding model to use",
    )
    evaluate_parser.add_argument(
        "-i",
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
    evaluate_parser.add_argument(
        "-o",
        "--output-file",
        type=argparse.FileType("wb"),
        help="Ruta del fichero de salida en formato jsonlines donde se guardarán las preguntas generadas.",
    )
    evaluate_parser.set_defaults(func=run_evaluate_subcommand)

    generate_parser = subparsers.add_parser("generate", help="Generate a dataset for evaluation")
    generate_parser.add_argument(
        "-G",
        "--generative-model",
        type=exceptions_to_argument_errors(import_llm_model),
        required=True,
        help="Generative model to use",
    )
    generate_parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r"),
        required=False,
        default=sys.stdin,
        help="Input file",
    )
    generate_parser.add_argument(
        "-p",
        "--prompt",
        type=lambda fp: open(fp, "r").read(),
        required=False,
        help="Question prompt file",
    )
    generate_parser.add_argument(
        "-n",
        "--num-questions",
        type=int,
        default=1,
        help="Número de preguntas a generar por documento.",
    )
    generate_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=3,
        help="Límite máximo de preguntas generadas (buenas y malas) por documento.",
    )
    generate_parser.add_argument(
        "-o",
        "--output-file",
        type=argparse.FileType("wb"),
        help="Ruta del fichero de salida en formato jsonlines donde se guardarán las preguntas generadas.",
    )
    generate_parser.set_defaults(func=run_generate_subcommand)

    classify_parser = subparsers.add_parser(
        "classify", help="Categorize the generated dataset by quality"
    )
    classify_subparsers = classify_parser.add_subparsers(
        dest="command", title="commands", required=True
    )
    classify_cross_encoder_parser = classify_subparsers.add_parser(
        "cross-encoder", help="Categorize the generated dataset by cross-encoder"
    )

    classify_cross_encoder_parser.add_argument(
        "-C",
        "--cross-encoder-model",
        action="append",
        type=exceptions_to_argument_errors(import_cross_encoder),
        help="Embedding models to use",
    )
    classify_cross_encoder_parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r"),
        required=False,
        default=sys.stdin,
        help="Input file",
    )
    classify_cross_encoder_parser.add_argument(
        "-o",
        "--output-file",
        type=argparse.FileType("wb"),
        help="Ruta del fichero de salida en formato jsonlines donde se guardarán las preguntas generadas.",
    )
    classify_cross_encoder_parser.set_defaults(func=run_classify_cross_encoder_subcommand)

    classify_cosine_similarity_parser = classify_subparsers.add_parser(
        "cosine-similarity", help="Categorize the generated dataset by cosine similarity"
    )
    classify_cosine_similarity_parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r"),
        required=False,
        default=sys.stdin,
        help="Input file",
    )
    classify_cosine_similarity_parser.add_argument(
        "-E",
        "--embedding-model",
        action="append",
        type=exceptions_to_argument_errors(import_embedding_model),
        help="Embedding model to use",
    )
    classify_cosine_similarity_parser.add_argument(
        "-o",
        "--output-file",
        type=argparse.FileType("wb"),
        help="Ruta del fichero de salida en formato jsonlines donde se guardarán las preguntas generadas.",
    )

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)

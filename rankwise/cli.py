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

import rankwise.classify.calculations
import rankwise.classify.cosine_distance.io
import rankwise.classify.cross_encoder.io
from rankwise.evaluate.classification.calculations import (
    build_evaluate_classification_report,
    calculate_classification_results,
)
from rankwise.evaluate.embedding_model.calculations import build_evaluation_report
from rankwise.evaluate.embedding_model.io import accumulate_evaluation_metrics, build_evaluator
from rankwise.generate.data import DEFAULT_QUESTION_PROMPT
from rankwise.generate.io import generate_dataset
from rankwise.importer.io import (
    UndefinedEnvVarError,
    import_cross_encoder,
    import_embedding_model,
    import_llm_model,
)
from rankwise.io import (
    as_jsonlines,
    read_evaluate_classification_input,
    read_evaluate_embedding_input,
    read_json_lines_input,
)


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
def run_evaluate_embedding_subcommand(args):
    input_data = read_evaluate_embedding_input(args.input)

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
def run_evaluate_classification_subcommand(args):
    ground_truth = read_evaluate_classification_input(args.ground_truth)
    for filepath in args.classification:
        with open(filepath, "r") as file:
            classification = read_evaluate_classification_input(file)
            results = calculate_classification_results(ground_truth, classification)
            yield build_evaluate_classification_report(filepath, results)


@as_jsonlines
def run_generate_subcommand(args):
    input_data = read_json_lines_input(args.input)

    if args.question_prompt_file is None:
        prompt = DEFAULT_QUESTION_PROMPT
    else:
        prompt = args.question_prompt_file.read()

    for document in input_data:
        generation = generate_dataset(
            args.model.instance,
            [document],
            args.questions_count,
            prompt,
        )
        yield {
            "document": document,
            "questions": [example.query for example in generation.examples],
        }


@as_jsonlines
def run_classify_cross_encoder_subcommand(args):
    input_data = read_json_lines_input(args.input)

    cross_encoder_distance = partial(
        rankwise.classify.cross_encoder.io.calculate_distance,
        args.cross_encoder_model.instance,
    )

    normalize_fn = (
        rankwise.classify.calculations.normalize_min_max
        if args.threshold
        else rankwise.classify.calculations.normalize_identity
    )

    # input_data :: {"document": "DocA", "questions": ["Q1", "Q2"]}
    all_documents = set(row["document"] for row in input_data)
    is_best_according_to_cross_encoder = partial(
        rankwise.classify.calculations.is_best,
        cross_encoder_distance,
        rankwise.classify.calculations.strictly_greatest_with_threshold_fn(args.threshold),
        normalize_fn,
        all_documents,
    )
    for row in input_data:
        good, bad = list(), list()
        this_document, questions = row["document"], row["questions"]

        for question in questions:
            collection = (
                good if is_best_according_to_cross_encoder(question, this_document) else bad
            )
            collection.append(question)

        yield rankwise.classify.calculations.format_output(this_document, good, bad)


@as_jsonlines
def run_classify_cosine_similarity_subcommand(args):
    input_data = read_json_lines_input(args.input)

    embedding_functions = [model.instance.get_text_embedding for model in args.embedding_model]
    calculate_average_cosine_distance = (
        rankwise.classify.cosine_distance.io.build_distance_function(embedding_functions)
    )

    normalize_fn = (
        rankwise.classify.calculations.normalize_min_max
        if args.threshold
        else rankwise.classify.calculations.normalize_identity
    )

    # input_data :: {"document": "DocA", "questions": ["Q1", "Q2"]}
    all_documents = set(row["document"] for row in input_data)
    is_best_according_to_cosine_similarity = partial(
        rankwise.classify.calculations.is_best,
        calculate_average_cosine_distance,
        rankwise.classify.calculations.strictly_greatest_with_threshold_fn(args.threshold),
        normalize_fn,
        all_documents,
    )
    for row in input_data:
        good, bad = list(), list()
        this_document, questions = row["document"], row["questions"]

        for question in questions:
            collection = (
                good if is_best_according_to_cosine_similarity(question, this_document) else bad
            )
            collection.append(question)

        yield rankwise.classify.calculations.format_output(this_document, good, bad)


def make_parser():
    parser = argparse.ArgumentParser(description="Rankwise: A tool for evaluating embedding models")
    subparsers = parser.add_subparsers(dest="command", title="commands", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate operations")
    evaluate_subparsers = evaluate_parser.add_subparsers(
        dest="command", title="commands", required=True
    )
    evaluate_embedding_parser = evaluate_subparsers.add_parser(
        "embedding-models", help="Evaluate embedding models"
    )
    evaluate_embedding_parser.add_argument(
        "-E",
        "--embedding-model",
        action="append",
        type=exceptions_to_argument_errors(import_embedding_model),
        help="Embedding model to use",
    )
    evaluate_embedding_parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r"),
        required=False,
        default=sys.stdin,
        help="Input file",
    )
    evaluate_embedding_parser.add_argument(
        "-k",
        "--top-k",
        dest="top_k",
        type=int,
        required=False,
        default=10,
        help="Top K results to consider",
    )
    evaluate_embedding_parser.add_argument(
        "-m",
        "--metrics",
        action="append",
        type=str,
        choices=["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"],
        required=False,
        default=["hit_rate", "mrr"],
        help="Metrics to calculate",
    )
    evaluate_embedding_parser.add_argument(
        "-o",
        "--output-file",
        type=argparse.FileType("wb"),
        help="Path of the output file in jsonl format where the evaluations will be saved.",
    )
    evaluate_embedding_parser.set_defaults(func=run_evaluate_embedding_subcommand)

    evaluate_classification_parser = evaluate_subparsers.add_parser(
        "classification", help="Evaluate given classifications against ground truth"
    )
    evaluate_classification_parser.add_argument(
        "-g",
        "--ground-truth",
        type=argparse.FileType("r"),
        required=False,
        default=sys.stdin,
        help="Input file with the ground truth of the classification",
    )
    evaluate_classification_parser.add_argument(
        "-c",
        "--classification",
        action="append",
        type=str,
        required=True,
        help="Classification filepath to evaluate",
    )
    evaluate_classification_parser.add_argument(
        "-o",
        "--output-file",
        type=argparse.FileType("wb"),
        help="Path of the output file in jsonl format where the evaluations will be saved.",
    )
    evaluate_classification_parser.set_defaults(func=run_evaluate_classification_subcommand)

    generate_parser = subparsers.add_parser("generate", help="Generate a dataset for evaluation")
    generate_parser.add_argument(
        "-M",
        "--model",
        type=exceptions_to_argument_errors(import_llm_model),
        required=True,
        help="Class instance of the model to use",
    )
    generate_parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r"),
        required=False,
        default=sys.stdin,
        help="Input file with the content to generate questions from, one per row",
    )
    generate_parser.add_argument(
        "--question-prompt-file",
        type=argparse.FileType("r"),
        required=False,
        help="Question prompt file",
    )
    generate_parser.add_argument(
        "-q",
        "--questions-count",
        dest="questions_count",
        type=int,
        required=False,
        default=3,
        help="Number of questions to generate per content",
    )
    generate_parser.add_argument(
        "-o",
        "--output-file",
        type=argparse.FileType("wb"),
        help="Path of the output file in jsonl format where the generated questions will be saved.",
    )
    generate_parser.set_defaults(func=run_generate_subcommand)

    classify_parser = subparsers.add_parser(
        "classify", help="Categorize the generated questions dataset by quality"
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
        type=exceptions_to_argument_errors(import_cross_encoder),
        help="Cross-encoder model to use",
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
        help=(
            "Path of the output file in jsonl format where the classified questions will be saved."
        ),
    )
    classify_cross_encoder_parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        required=False,
        default=0,
        help="Relative threshold between the best question and the rest to consider it good.",
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
        help=(
            "Path of the output file in jsonl format where the classified questions will be saved."
        ),
    )
    classify_cosine_similarity_parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        required=False,
        default=0,
        help="Relative threshold between the best question and the rest to consider it good.",
    )
    classify_cosine_similarity_parser.set_defaults(func=run_classify_cosine_similarity_subcommand)

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)

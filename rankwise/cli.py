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

from functools import cache
import argparse
import json
import sys
from statistics import mean

from sklearn.metrics.pairwise import cosine_similarity

from rankwise.evaluate.calculations import build_evaluation_report
from rankwise.evaluate.io import accumulate_evaluation_metrics, build_evaluator
from rankwise.generate.data import DEFAULT_QUESTION_PROMPT
from rankwise.generate.io import generate_dataset
from rankwise.generate.calculations import format_output
from rankwise.importer.io import UndefinedEnvVarError, import_embedding_model, import_llm_model, import_cross_encoder
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

    def build_distance_function(*get_text_embedding_functions):
        """Builds a function that calculates the average cosine distance between two texts using multiple embedding models."""
        get_text_embedding_functions = [ cache(fn) for fn in get_text_embedding_functions ]
        def get_cosine_similarity(e1, e2):
            return cosine_similarity([e1], [e2])[0][0]

        def calculate_distance(text1, text2):
            result = mean(get_cosine_similarity(get_text_embedding(text1), get_text_embedding(text2))
                          for get_text_embedding in get_text_embedding_functions)
            print(result, text1, text2)
            return result
        return calculate_distance

	# from sentence_transformers.cross_encoder import CrossEncoder
	# model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=torch.nn.Sigmoid())
	# scores = model.predict([("How many people live in Madrid?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.")])

    input_data = read_generate_input(args.input)

    # calculate_distance = build_distance_function(*(model.instance.get_text_embedding for model in args.embedding_model))
    def calculate_distance(query, doc):
        result = args.cross_encoder_model[0].instance.predict([query, doc])
        return result

    for doc in input_data:
        good, bad = list(), list()
        considered = set()
        num_requests = 0
        while len(good) < args.num_questions and num_requests < args.limit:
            response = generate_dataset(args.generative_model.instance, [doc], args.num_questions, args.prompt)
            for examples in response.examples:
                query = examples.query
                if query in considered:
                    continue

                distance = calculate_distance(query, doc)

                is_best = all(calculate_distance(query, d) < distance for d in input_data if doc != d)
                
                if is_best:
                    good.append((distance, query))
                else:
                    bad.append((distance, query))
                num_requests += 1
                considered.add(query)

        good = [doc for (_, doc) in sorted(good, reverse=True)]
        bad = [doc for (_, doc) in sorted(bad, reverse=True)]

        yield format_output(doc, good, bad)

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

    # Dataset
    generate_parser = subparsers.add_parser("generate", help="Generate a dataset for evaluation")
    generate_parser.add_argument(
        "-G",
        "--generative-model",
        type=exceptions_to_argument_errors(import_llm_model),
        required=True,
        help="Generative model to use",
    )
    generate_parser.add_argument(
        "-C",
        "--cross-encoder-model",
        action="append",
        type=exceptions_to_argument_errors(import_cross_encoder),
        help="Name of the embedding models to use",
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

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)

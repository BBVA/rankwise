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

from operator import __ge__ as greatest_among
from operator import __gt__ as strictly_greatest
from operator import __le__ as least_among
from operator import __lt__ as strictly_least


__all__ = ["is_best", "strictly_least", "least_among", "strictly_greatest", "greatest_among"]

def strictly_greatest_with_threshold_fn(threshold: float):
    def strictly_greatest(a, b):
        return a - threshold > b

    return strictly_greatest


def normalize_min_max(distance_with_this_document, distance_with_other_documents) -> tuple[float, list[float]]:
    from sklearn.preprocessing import minmax_scale
    res = minmax_scale([[distance_with_this_document] + distance_with_other_documents], axis=1).tolist()
    return (res[0][0], res[0][1:])

def normalize_identity(distance_with_this_document, distance_with_other_documents) -> tuple[float, list[float]]:
    return distance_with_this_document, distance_with_other_documents


def is_best(distance_fn, comparison_fn, normalize_fn, all_documents, question, this_document):
    """
    Returns true if the query is best suited for the document according
    to the distance function and comparison function taking into account
    all the documents in the set.
    """

    other_documents = all_documents - set([this_document])
    distance_with_this_document = distance_fn(question, this_document)
    distance_with_other_documents = (distance_fn(question, another_document) for another_document in other_documents)

    if not normalize_fn:
        normalize_fn = normalize_identity

    distance_with_this_document_normalized, distance_with_other_documents_normalized = normalize_fn(
        distance_with_this_document, distance_with_other_documents,
    )

    return all(
        comparison_fn(distance_with_this_document_normalized, distance_with_other_document)
        for distance_with_other_document in distance_with_other_documents_normalized
    )


def format_output(document, good_questions, bad_questions):
    return {
        "document": document,
        "questions": {"good": good_questions, "bad": bad_questions},
    }

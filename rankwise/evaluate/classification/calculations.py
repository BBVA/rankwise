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

from rankwise.data import ClassificationData


def calculate_classification_results(
    ground_truth: ClassificationData, classification: ClassificationData
):
    results = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }

    for good_question, document in ground_truth.good_questions_with_document.items():
        classification_document = classification.good_questions_with_document.get(good_question)
        if classification_document:
            results[
                "true_positive" if document == classification_document else "false_positive"
            ] += 1
            continue

        if good_question in classification.bad_questions:
            results["false_negative"] += 1
        else:
            print("warning: good question not found in classification:", good_question)

    for bad_question in ground_truth.bad_questions:
        if bad_question in classification.bad_questions:
            results["true_negative"] += 1
        elif bad_question in classification.good_questions_with_document:
            results["false_positive"] += 1
        else:
            print("warning: bad question not found in classification:", bad_question)

    return results


def build_evaluate_classification_report(filename, results):
    accuracy = (results["true_positive"] + results["true_negative"]) / sum(results.values())
    precision = results["true_positive"] / (results["true_positive"] + results["false_positive"])
    recall = results["true_positive"] / (results["true_positive"] + results["false_negative"])
    f05 = 1.25 * precision * recall / (0.25 * precision + recall)
    f1 = 2 * precision * recall / (precision + recall)
    f2 = 5 * precision * recall / (4 * precision + recall)
    return {
        "filename": filename,
        "confusion_matrix": results,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
        "f1": f1,
        "f2": f2,
    }

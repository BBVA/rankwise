import argparse
import json


def open_jsonl_file(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", action="append", type=str)
    args = parser.parse_args()

    all_data = [open_jsonl_file(file) for file in args.dataset]
    merged_data = []

    for items in zip(*all_data):
        document = items[0]["document"]
        good_questions = set(items[0]["questions"]["good"])
        bad_questions = set(items[0]["questions"]["bad"])

        for item in items[1:]:
            bad_questions |= set(item["questions"]["bad"])
            bad_questions |= good_questions ^ set(item["questions"]["good"])
            good_questions &= set(item["questions"]["good"])

        merged_data.append(
            {
                "document": document,
                "questions": {"good": list(good_questions), "bad": list(bad_questions)},
            }
        )

    for item in merged_data:
        print(json.dumps(item))


if __name__ == "__main__":
    main()

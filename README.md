# rankwise

The best way to evaluate your embedding models!

> **IMPORTANT: rankwise is in an early stage of development; the interface may change without further notice.**

## Description

rankwise is a benchmark CLI tool for evaluating the quality of embedding models with your own dataset.

It also facilitates the generation of datasets using LLM models.

## Getting Started

### Pre-requisites

- [Python 3.12](https://www.python.org/downloads/release/python-3120)
- [Poetry](https://python-poetry.org/docs/#installation).

### Installation

- Install dependencies:

```bash
poetry install
```

### Executing program

- Activate the Poetry Shell:

```bash
poetry shell
```

- Run the `generate` command to create a dataset containing queries and their related documents in JSONL format from a `data.jsonl` file with one document per line.

```bash
API_KEY=xxx rankwise generate --model "azure_openai.AzureOpenAI(model='gpt-4o',deployment_name='gpt-4o',api_version='2023-07-01-preview',azure_endpoint='https://your-azure-endpoint',api_key=ENVVAR('API_KEY'))" --queries-count 3 --input data.jsonl > dataset.jsonl
```

This command uses the given LLM model to generate the specified number of queries for every document in the input file.

- Run the `evaluate` command to assess your dataset and obtain quality metrics for the specified embedding model.

```bash
API_KEY=xxx rankwise evaluate -E "azure_openai.AzureOpenAIEmbedding(model='text-embedding-3-large',deployment_name='azure-text-embedding-ada-002',api_version='2023-07-01-preview',azure_endpoint='https://your-azure-endpoint',api_key=ENVVAR('API_KEY'))" -m hit_rate -m mrr --input dataset.jsonl
```

This command uses the given embedding model to evaluate the input dataset and calculate quality metrics.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Please read the `CONTRIBUTING` file for details on our code of conduct and the process for submitting pull requests.

## Authors

- Roberto Abdelkader Martínez Pérez
- Pedro Ruiz Pareja

## Version History

- 0.1.0
  - Initial Release

## License

This project is licensed under the Apache v2.0 License - see the `LICENSE` file for details

## Acknowledgments

- Thanks to the open-source community for continuous inspiration.
- Special mentions to contributors and collaborators.

## Contact

If you have any questions or suggestions, feel free to contact the authors.
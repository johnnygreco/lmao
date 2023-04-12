# 🙊 LMAO: **L**anguage **M**odel **A**dapters and **O**rchestrators
> Leveraging the power of large LMs for downstream NLP tasks

## 🚧 **Work in progress** 🚧
This project is still in its early stages of development. The API is subject to change and the documentation is incomplete.


## Installation
The package is available on PyPI and can be installed with pip:
```bash
pip install lmao-nlp
```
> ⚠️ Don't forget the `-nlp` extension on the distribution name!

## Development

To install the development version, clone the repository and install the package in editable mode with the `dev` extra:

```bash
git clone https://github.com/johnnygreco/lmao.git
cd lmao
pip install -e ".[dev]"
```

If you want to install all the optional dependencies (e.g., to build the docs), use the `all` extra:

```bash
pip install -e ".[all]"
```

## LM Providers
The plan is to add support for all major LM providers (both for external API-based models and locally run models). At this early stage of development, [OpenAI](https://platform.openai.com/) is the only supported provider. You'll need an activate OpenAI API key to use the LMAO library.


## Quickstart

```python
from lmao import load_task_adapter

# load the sentiment analysis adapter for OpenAI's GPT-3.5-turbo model
adapter = load_task_adapter("openai", "sentiment_analysis")

# predict sentiment of the given text
response = adapter.predict("This LMAO package is awesome!")

print(response.prediction)
# output: 'positive'
```

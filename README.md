# 🙊 LMAO: **L**anguage **M**odel **A**dapters and **O**rchestrators
> Leveraging the power of large LMs for downstream NLP tasks

LMAO is an open-source library for integrating large language models (LLMs) from providers like [OpenAI](https://platform.openai.com/docs/introduction) and [Anthropic](https://console.anthropic.com/docs/api) into your NLP workflows. For example, it can be used to pre-annotate text datasets using zero- or few-shot learning with Claude or GPT-4. LMAO is in the (very) early stages of development. New features will be added with high cadence and documentation is coming soon.

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
The plan is to add support for all major LM providers (both for external API-based models and locally run models). At this early stage of development, [OpenAI](https://platform.openai.com/docs/introduction) and [Anthropic](https://console.anthropic.com/docs/api) are the only supported providers. You'll need an activate API key from OpenAI and/or Anthropic to use the LMAO library.


## Quickstart

### Perform sentiment analysis with ChatGPT
```python
from lmao import factory

# create a sentiment analysis task using an adapter for OpenAI's GPT-3.5-turbo model
model = factory.create_task("sentiment_analysis", "openai")

# predict sentiment of the given text
result = model.predict("This LMAO package is awesome!")

print(result.prediction)
# output: 'positive'
```

### Create a Claude chatbot
```python
from lmao import factory

# create a chatbot using Anthropic's Claude under the hood
chatbot = factory.create_chatbot("anthropic")

response = chatbot.chat("Hello! Tell me a joke, please!")

print(response.text)
# output: Why was six afraid of seven? Because seven eight nine!

# the chatbot object keeps track of the chat history
response = chatbot.chat("Haha! Can you please explain why this is funny?")

print(response.text)
# output: Sure! This is a pun based joke. It plays on the homophones
# 'seven' and 'ate', and 'nine'. When spoken aloud, "seven eight nine"
# sounds like the phrase "seven ate nine". The joke implies that seven ate
# (devoured) the number nine, which is why six (the previous number) would
# be afraid of seven. It's a silly pun, but that's why some people find
# that type of simple wordplay amusing.
```

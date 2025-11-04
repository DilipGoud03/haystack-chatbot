# Haystack Chatbot

This repository contains the implementation of a chatbot powered by [Haystack](https://haystack.deepset.ai/), designed for advanced natural language processing tasks such as question answering, semantic search, and conversational AI.

## Features

- **Semantic Search**: Retrieve relevant documents based on user queries using state-of-the-art models.
- **Question Answering**: Extract precise answers from documents or knowledge bases.
- **Conversational AI**: Engage in multi-turn conversations with context awareness.
- **Extensible Pipelines**: Easily add or modify components (retrievers, readers, etc.) in the Haystack pipeline.
- **Integration Ready**: Designed for easy integration with messaging platforms and web interfaces.

## Technologies

- **Haystack Framework**
- **Google-Gemini-Chat-Model And Google-Gemini-Embbeding-Model**
- **Weaviate Vector Database**

## Getting Started

### Prerequisites

- Python 3.8+
- [Haystack](https://github.com/deepset-ai/haystack)
- Other dependencies listed in `requirements.txt`

### Installation

Clone the repository and install dependencies:
```
cd
git clone https://github.com/DilipGoud03/haystack-chatbot.git
cd haystack-chatbot
```

- create vertual enviorment
```
python -m venv haystack-chatbot-venv
```

- Activate vertual enviorment
```
source haystack-chatbot-venv/bin/activate
```

### Install dependencies
```
pip install -r requirement.txt
```

### Process to run this code 
```
docker compose up -d
```
- Above comand setup a weaviate db as localy.
- Then open project directory and upload any text or doc file into documents directory
- if documents directory not available run below command.
```
mkdir documets

```
- after uploading file run
```
python services/file_upload_service.py
```
- above command upload your data into vector db (ie- weaviate db)
- Now setup is completed simply run below command to run the project
```
python chat.py
```


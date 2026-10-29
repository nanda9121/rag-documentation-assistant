# RAG-Based Technical Documentation Assistant

A Python and Streamlit assistant that retrieves relevant passages from technical documents and produces source-grounded answers with Google's Gemini API.

## Features

- Upload PDF, Markdown, and plain-text documentation.
- Split files into overlapping passages and preserve PDF page numbers.
- Create text embeddings with `gemini-embedding-2`.
- Rank passages using cosine similarity.
- Generate cited answers with `gemini-3.5-flash-lite`.
- Inspect the cited evidence under each answer.

## Requirements

- Python 3.10+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

Gemini API availability, quotas, and pricing depend on your account, region, and model. Check the current [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) and usage limits before uploading documents.

## Set up the API key

The project reads its API key from the root `.env` file. Open that file and replace the placeholder:

```env
GEMINI_API_KEY=your_google_ai_studio_api_key
```

`.env` is excluded from Git. Never commit or share the file after adding your key.

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The first indexing call sends each document passage to Google's embedding API (one request per passage). Each question and its retrieved passages are sent to Gemini to generate the answer. The index is held in the current Streamlit session; re-index documents after restarting the app.

## Run tests

```powershell
py -m unittest discover -s tests -v
```

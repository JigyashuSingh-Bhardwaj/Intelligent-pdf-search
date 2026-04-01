# Intelligent PDF Search

A local Flask application that turns uploaded PDFs into a searchable knowledge base.
It extracts text, splits it into chunks, indexes content with TF-IDF and semantic embeddings, and returns ranked search results plus short answer summaries.

## Key features

- Upload PDF files and store them in the app
- Extract and clean text from each PDF page
- Split documents into searchable chunks
- Build TF-IDF vectors and semantic embeddings
- Perform hybrid search over uploaded documents
- Generate concise answers from top search results
- Deployable on Render using `render.yaml` and `Procfile`

## Repository structure

- `app.py` — main Flask application
- `modules/` — core processing modules
- `templates/` — HTML templates
- `static/` — CSS styling
- `requirements.txt` — Python dependencies
- `Procfile` — web process for Render
- `render.yaml` — Render deployment manifest
- `.gitignore` — excluded files

## Quick local setup

1. Create and activate a Python virtual environment:

```powershell
py -m venv .\venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Download NLTK stopwords:

```powershell
python -c "import nltk; nltk.download('stopwords')"
```

4. Run the app locally:

```powershell
python app.py
```

5. Open the app in your browser:

```text
http://127.0.0.1:5000
```

## Render deployment

1. Push the repository to GitHub.
2. Create a new Render web service.
3. Connect it to your GitHub repo.
4. Render will use `render.yaml` and `Procfile` to build and run the app.

## Notes

- The app expects file-based storage for `data/` and `uploads/` during runtime.
- `.gitignore` excludes generated runtime files and virtual environments.
- `app.py` listens on `$PORT` for Render-compatible deployment.

## Important dependencies

- `Flask`
- `PyPDF2`
- `scikit-learn`
- `sentence-transformers`
- `torch`
- `waitress`
- `nltk`


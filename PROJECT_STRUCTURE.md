# Project Structure

This file describes the current repository structure after cleanup and production preparation.

```
intelligent-pdf-search/
│
├── .gitignore
├── Procfile
├── README.md
├── PROJECT_STRUCTURE.md
├── app.py
├── cleanup_phase1.py
├── config.py
├── migrate_to_sqlite.py
├── render.yaml
├── requirements.txt
├── modules/
│   ├── answer_builder_new.py
│   ├── chunking.py
│   ├── config.py
│   ├── db.py
│   ├── db_layer.py
│   ├── pdf_loader.py
│   ├── search_engine.py
│   ├── text_preprocessing.py
│   ├── vectorizer.py
│   └── utils.py
├── static/
│   └── modern-style.css
└── templates/
    ├── index_modern.html
    └── results_modern.html
```

## What each item is for

- `app.py`
  - Main Flask application and request routing.
  - Handles upload, search, and runtime startup.

- `config.py`
  - Root-level configuration loader and environment settings.
  - Keeps production parameters centralized.

- `cleanup_phase1.py`
  - Cleanup and setup helper script.
  - Useful for local maintenance before deployment.

- `migrate_to_sqlite.py`
  - Database migration utility.
  - Helps initialize or migrate the local SQLite schema.

- `requirements.txt`
  - Python package dependencies.
  - Required for Render or any Python deployment.

- `Procfile`
  - Defines the web process for Render and Heroku.
  - Starts the app through `waitress`.

- `render.yaml`
  - Render service manifest for automatic deployment.
  - Defines build and start commands.

- `modules/`
  - Application logic split into reusable modules.

  - `modules/answer_builder_new.py`
    - Builds search answers from top chunks.
    - Extracts sentences and ranks them.

  - `modules/chunking.py`
    - Splits PDF text into search-friendly chunks.

  - `modules/config.py`
    - Central configuration for vectorization, search, upload, and logging.

  - `modules/db.py`
    - SQLite database connection and schema creation.

  - `modules/db_layer.py`
    - High-level database access methods and persistence helpers.

  - `modules/pdf_loader.py`
    - PDF text extraction and cleanup.

  - `modules/search_engine.py`
    - Hybrid TF-IDF + semantic retrieval.
    - Query scoring and filtering.

  - `modules/text_preprocessing.py`
    - Text cleaning for indexing and display.

  - `modules/vectorizer.py`
    - TF-IDF vector creation and semantic embedding generation.

  - `modules/utils.py`
    - Shared helper utilities.

- `static/`
  - CSS and frontend assets.

- `templates/`
  - Flask HTML templates for the web app.

## Notes

- The repo has been cleaned to remove temporary documentation files and runtime artifacts.
- `data/`, `uploads/`, and `venv/` are not kept in source control.
- `README.md` is the primary user-facing documentation file.

## Deployment readiness

The current repository is prepared for Render deployment using:
- `render.yaml`
- `Procfile`
- `requirements.txt`
- `app.py` set to listen on `$PORT`

Make sure to keep the following in the repo before pushing:
- `app.py`
- `modules/`
- `templates/`
- `static/`
- `requirements.txt`
- `Procfile`
- `render.yaml`
- `README.md`
- `.gitignore`

Optional maintenance files that can remain if desired:
- `cleanup_phase1.py`
- `migrate_to_sqlite.py`
- `PROJECT_STRUCTURE.md`

# Web Frontend

The Web UI follows the same lightweight pattern as ICEBERG-MS: static
HTML/CSS/JavaScript in `frontends/web_app/static/`, backed by FastAPI task
endpoints in `api/server.py`. It does not use Streamlit.

It contains three tools:

- PIE / PICS fitting: either generate PIE curves from scan folders or upload
  processed PIE curve tables, then fit the selected m/z with automatic or
  manually selected PICS candidates.
- PICS query: search the maintained SQLite PICS library by m/z, species keyword,
  and ionization-energy range, then plot the selected PICS curve.
- PICS upload: upload CSV/TSV/TXT/XLSX/XLS data as either a temporary session
  library for the current user, or an administrator-restricted write to the
  maintained SQLite library.

Recommended local start command from the project root:

```bash
uvicorn api.server:app --reload
```

Open `http://127.0.0.1:8000/` after the server starts.

For public deployment, set `BL03U_ADMIN_TOKEN` before enabling server-library
uploads. Optional limits:

- `BL03U_MAX_UPLOAD_MB`: maximum upload size, default `20`.
- `BL03U_PICS_LIBRARY_TTL_HOURS`: temporary PICS library lifetime, default `24`.

# Web Frontend

The Web UI follows the same lightweight pattern as ICEBERG-MS: static
HTML/CSS/JavaScript in `src/bl03u_masstool/frontends/web_app/static/`, backed by
FastAPI task endpoints in `src/bl03u_masstool/api/server.py`. It does not use
Streamlit.

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
uvicorn bl03u_masstool.api.app:app --reload
```

Open `http://127.0.0.1:8000/` after the server starts.

If you deploy the static frontend to GitHub Pages or another static host, the
page still needs a running API backend. Pass the backend URL with `?api=` in the
page URL, for example:

```text
https://<owner>.github.io/<repo>/?api=https://api.example.com
```

The backend must allow cross-origin requests from the static site.

For public deployment, set `BL03U_ADMIN_TOKEN` before enabling server-library
uploads. Optional limits:

- `BL03U_MAX_UPLOAD_MB`: maximum upload size, default `20`.
- `BL03U_PICS_LIBRARY_TTL_HOURS`: temporary PICS library lifetime, default `24`.
- `BL03U_ALLOWED_DATA_ROOTS`: additional server-side directories that API
  folder/database path parameters may read, separated by the platform path
  separator. The project root is always allowed.
- `BL03U_PIE_JOB_TTL_HOURS`: completed or failed PIE job lifetime, default `6`.
- `BL03U_MAX_PIE_JOBS`: maximum in-memory PIE jobs retained, default `100`.

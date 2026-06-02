# Desktop Release Workflow

This project uses GitHub Actions to build the PyQt desktop app on Windows and
macOS.

## Daily Test Builds

Push to `master` to build workflow artifacts:

```bash
git push origin master
```

Open the latest `Build Desktop App` run in GitHub Actions and download:

- `BL03U-MassSpectrumTool-Windows.exe`
- `BL03U-MassSpectrumTool-macOS.dmg`

Artifacts are kept for 14 days.

## Official Releases

Create and push a version tag:

```bash
git tag -a v0.1.0 -m "BL03U MassSpectrumTool v0.1.0"
git push origin v0.1.0
```

The workflow builds both platforms and uploads the generated installers to the
GitHub Release for that tag.

The generated packages are not code-signed or notarized yet. For the current
small-group internal macOS distribution, this is acceptable: users can install
the `.dmg`, then launch the app the first time with `Right click -> Open`.
Developer ID / Authenticode signing can be added later only if the app needs
public distribution.

## Bundled Resources

The PyInstaller spec includes:

- `icons/`
- `config/`
- `database/species_database.sqlite`
- `data/examples/`

When running as a packaged app, read-only bundled resources are loaded from the
PyInstaller bundle, while writable outputs and copied runtime data are stored in
the user's application data directory.

Set `BL03U_USER_DATA_DIR` to override the user data directory during local
testing.

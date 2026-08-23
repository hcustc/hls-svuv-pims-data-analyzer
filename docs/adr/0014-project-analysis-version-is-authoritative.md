# Keep the Project Analysis Version authoritative

The Analysis Version stored in the Project is the authoritative analysis record, while Excel, CSV, images, JSON bundles, and Markdown reports are exported Derived Artifacts. Editing an export never mutates the Project implicitly; bringing edited data back requires an explicit import that creates a new Candidate or Analysis Version.

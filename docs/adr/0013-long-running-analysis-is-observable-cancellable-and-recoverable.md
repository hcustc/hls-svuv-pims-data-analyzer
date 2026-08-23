# Make long-running analysis observable, cancellable, and recoverable

Data import, large reads, peak processing, PIE and temperature analysis, and export must not block the desktop interface. Each operation exposes stage and progress, supports safe cancellation, preserves the Project and Raw Dataset on failure, and can restart from the last confirmed state; performance thresholds are set against representative Projects.

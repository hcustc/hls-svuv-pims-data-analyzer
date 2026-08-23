# Require repeatability with scientific tolerances

Given the same Raw Dataset, Analysis Baseline, parameters, and software version, an analysis must be repeatable: deterministic algorithms use fixed random state, while unavoidable nondeterminism records its state and remains within method-specific numerical tolerances. Validation compares scientific equivalence rather than requiring bitwise identity across platforms.

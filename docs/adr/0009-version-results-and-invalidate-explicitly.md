# Version results and invalidate explicitly

Every Curve Result and downstream result must be an immutable Analysis Version bound to its Raw Dataset, Analysis Baseline, complete parameters, and software version. When a dependency changes, prior versions remain available for comparison but cease to be Current Results until a Scientific Analyst explicitly selects them; recomputation creates a new version and never overwrites or silently revalidates an old one.

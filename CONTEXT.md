# SR-PIMS Data Analysis

This context covers the scientific analysis of synchrotron-radiation photoionization mass-spectrometry experiments and the evidence produced from them.

## Language

**SR-PIMS Data**:
Experimental spectra and metadata produced by synchrotron-radiation photoionization mass spectrometry. A beamline or instrument identifier describes provenance rather than the data type.
_Avoid_: BL03U data

**Project**:
A persistent boundary for one analysis effort that owns its data sources, confirmed analysis baselines, committed analysis parameters, and registered results.
_Avoid_: Workspace, session

**Exploratory Run**:
An analysis performed outside a Project. Its output is provisional and is not Project Evidence.
_Avoid_: Project run, final result

**Project Evidence**:
A reproducible proof bundle for a Project analysis that binds identifiable input data, the confirmed analysis configuration, software identity, structured intermediate and final results, quality-control findings, and Scientific Analyst confirmations.
_Avoid_: Output file, report

**Scientific Analyst**:
A researcher with sufficient SR-PIMS expertise to choose processing methods, confirm analysis baselines, review results, and form scientific interpretations.
_Avoid_: User, operator, Agent

**QC Signal**:
A non-conclusive indicator that directs a Scientific Analyst to data or results needing closer review.
_Avoid_: Pass criterion, scientific conclusion

**Scientific Interpretation**:
An explanation or claim formed by a Scientific Analyst from Project Evidence and external scientific judgment.
_Avoid_: Automated conclusion, software result

**Analysis Baseline**:
A Scientific Analyst-confirmed calibration, Peak Set, or other shared processing reference used by downstream analyses in a Project.
_Avoid_: Default setting, temporary parameter

**Candidate**:
A proposed baseline, fit, or processing choice produced by software, an Agent, or a Scientific Analyst that has no Project authority until explicitly confirmed.
_Avoid_: Active baseline, approved result

**Raw Dataset**:
An immutable, provenance-bearing collection of imported instrument data with a stable content identity.
_Avoid_: Working copy, cleaned data

**Derived Artifact**:
Data or a presentation produced by processing a Raw Dataset without replacing or altering that dataset.
_Avoid_: Raw data, source file

**Analysis Version**:
An immutable result lineage that binds a Raw Dataset, Analysis Baseline, complete parameter state, software identity, Project Evidence, and Derived Artifacts from one analysis execution.
_Avoid_: Latest file, mutable result

**Current Result**:
The Analysis Version explicitly designated by a Scientific Analyst as the Project's current result for its scope. Recency alone does not make an Analysis Version current.
_Avoid_: Newest result, last run

**PICS Record**:
A versioned photoionization cross-section curve for one species with declared units, energy coverage, source provenance, and Scientific Analyst confirmation.
_Avoid_: Anonymous cross-section values, unreviewed upload

**External Reference**:
A Project snapshot of scientific data retrieved from an external source, retaining the source record, query, retrieval time, and Scientific Analyst confirmation.
_Avoid_: Live lookup result, untracked copied value

**Legacy Project**:
A Project created by an explicitly supported earlier format and preserved unchanged while a migrated copy is produced in the current format.
_Avoid_: Current Project, arbitrary old folder

**Reference Dataset**:
A synthetic dataset with known answers or a representative real dataset with expert-approved expected results and explicit tolerances, used to judge scientific correctness.
_Avoid_: Sample data, fixture

**Representative Dataset**:
Real SR-PIMS Data selected to exercise typical workflows and Scientific Analyst review without yet supplying approved expected values and tolerances.
_Avoid_: Reference Dataset, gold standard

**PIE Curve**:
A series relating photon energy to the ion signal for one m/z channel under a declared peak range and preprocessing basis.
_Avoid_: PICS fit, species identification

**Temperature Response Curve**:
A series relating experiment temperature to the ion signal for one m/z channel at a declared photon energy and preprocessing basis.
_Avoid_: Mole-fraction curve, species identification

**Curve Result**:
An independently reviewable scientific result consisting of an approved PIE Curve or Temperature Response Curve and its Project Evidence. It remains complete without downstream species interpretation or quantification.
_Avoid_: Intermediate result, unfinished analysis

**Interpretation Result**:
An optional result that assigns scientific meaning to an approved Curve Result, such as a candidate species, a PICS fit, an isotope contribution, or a temperature-response class.
_Avoid_: Curve Result, Quantitative Result

**Quantitative Result**:
An optional result that reports a physically meaningful quantity, such as mole fraction or absolute PICS, from approved upstream evidence and declared quantitative assumptions.
_Avoid_: Curve Result, qualitative interpretation

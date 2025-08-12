Reactive‑Site Predictor – README

Quick Overview

Deterministic, structure‑only predictor that ranks putative reactive residues from a PDB/mmCIF file.
Key pillars
	1.	7‑D geometric core (atom-count, distance stats, directional variance, tetrahedral angle deviation, SASA)
	2.	Optional 20‑D residue embedding
	3.	Metal‑aware ZORA boost (distance‑weighted)
	4.	Boost / penalty blocks – coordination count, angle penalty, pocket flag

# basic
python reactive_site_predictor.py protein.pdb --top 15

# enable pocket flag  (needs mdtraj ≥1.9)
POCKET_ON=1 python reactive_site_predictor.py protein.pdb


⸻

SASA Feature

Item	Details
Algorithm	Shrake–Rupley (probe 1.4 Å) via mdtraj.shrake_rupley (residue‑level)
Vector slot	7th element of core env‑vector (after angle_dev)
Normalisation	z‑score across all residues in current structure
Scoring use	w_env += SASA_WEIGHT × max(SASA_z, 0) (ReLU – bonus only)

Buried residues (SASA_z ≤ 0) receive no penalty, highly exposed residues gain up to SASA_WEIGHT × SASA_z points.

⸻

MDTraj Requirements
	•	mdtraj ≥ 1.9 is strongly recommended.
	•	1.9  → mdtraj.geometry.pocket present (pocket flag works)
	•	<1.9 → pocket flag auto‑OFF; SASA still computed.
	•	standard_names=True is forced to standardise atom nomenclature.

Install (conda):

conda install -c conda-forge mdtraj>=1.9


⸻

Typical Parameter Ranges

Parameter	Default	Useful Range	Effect
RADIUS	3.5 Å	3.5 – 5.0 Å	Geometric resolution –↑RADIUS→hinge‑Gly recall ↑
SASA_WEIGHT	0.3	0.1 – 0.5	Strength of surface bonus


⸻

Fallback Behaviour

Module / Data	If missing	Behaviour
mdtraj absent	ImportError	SASA=0, pocket flag OFF, code still runs
geometry.pocket absent	AttributeError	SASA OK, pocket OFF


⸻

Changelog (excerpt)
	•	v0.4 – Added residue‑level SASA feature; switched env‑vector 6→7 D.
	•	v0.3 – Metal ZORA boost, deterministic seed.

"""reactive_site_predictor.py
Deterministic, full‑option reactive‑site predictor.

Features
========
✓ 6‑D core geometry + optional 20‑D residue embedding
✓ Metal‑aware ZORA distance boost
✓ Coordination count, angle penalty, pocket flag (toggle)
✓ Determinism: global random seed fixed (0) ⇒ identical output across runs
✓ Fallback: each advanced block auto‑OFF if dependency/data missing

Usage
-----
python reactive_site_predictor.py protein.pdb --top 15
# toggle pocket analysis (requires mdtraj, random‑seed deterministic)
POCKET_ON=1 python reactive_site_predictor.py protein.pdb

Dependencies
------------
numpy, biopython
mdtraj (optional, for pocket flag)
"""

# ===== SECTION 0: Imports & deterministic seed =====
import os, math, argparse, json, random, shutil
from typing import Dict
import numpy as np

# deterministic: global seed = 0
os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)

try:
    from Bio.PDB import PDBParser, MMCIFParser
except ModuleNotFoundError:
    raise SystemExit("Install biopython: pip install biopython")

# mdtraj optional
try:
    import mdtraj as mdt
    HAS_MDTRAJ = True
except ModuleNotFoundError:
    HAS_MDTRAJ = False

# ===== constants =====
RADIUS = 4.7
EPS = 1e-6
AA_LIST = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "SEC",   # Selenocysteine (U)
    "PYL",   # Pyrrolysine (O)
    "MSE",   # Selenomethionine
    "HYP",   # 4-Hydroxyproline
    "CSO",   # S-oxyCys (Cys-SOH)
    "ASX",   # Ambiguous Asn/Asp
    "GLX",   # Ambiguous Glu/Gln
    "XLE",   # Ambiguous Leu/Ile
    "XAA"    # Unknown/other
]
AA_TO_VEC = {aa: np.eye(len(AA_LIST))[i] for i, aa in enumerate(AA_LIST)}
ZORA_PARAMS: Dict[str, Dict[str, float]] = {
    "ZN": {"alpha": 1.25, "beta": 0.08},
    "MG": {"alpha": 1.15, "beta": 0.05},
    "FE": {"alpha": 1.30, "beta": 0.10},
    "CU": {"alpha": 1.35, "beta": 0.12},
    "CA": {"alpha": 1.10, "beta": 0.04},
    "MN": {"alpha": 1.18, "beta": 0.05},
    "CO": {"alpha": 1.17, "beta": 0.05},
    "NI": {"alpha": 1.16, "beta": 0.05},
    "MO": {"alpha": 1.25, "beta": 0.08},
    "W" : {"alpha": 1.26, "beta": 0.08},
    "V" : {"alpha": 1.18, "beta": 0.06},
    "NA": {"alpha": 1.05, "beta": 0.03},
    "K" : {"alpha": 0.95, "beta": 0.02},
}
def angle_bonus(angle_dev, center=0.2, width=0.15, scale=1.2):
    x = abs(angle_dev)
    return scale * (1 - abs(x - center) / width)
# ===== Core Predictor =====
class ReactivePredictor:
    def __init__(self, pdb_file: str, embed_weights=None, pocket_on: bool=False):
        self.pdb_file = pdb_file
        if pdb_file.lower().endswith((".cif", ".mmcif")):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        self.struct = parser.get_structure("prot", pdb_file)
        self.ca_atoms = [a for a in self.struct.get_atoms() if a.get_id()=="CA"]
        self.can_embed = all(a.get_parent().get_resname() in AA_TO_VEC for a in self.ca_atoms)
        self.embed_weights = np.array(embed_weights) if embed_weights is not None else np.zeros(len(AA_LIST))
        self.metal_atoms = [a for a in self.struct.get_atoms()
                            if a.element.strip().upper() in ZORA_PARAMS]
        self.pocket_on = pocket_on and HAS_MDTRAJ
# ---- MDTraj one-shot: SASA + optional pocket ----
        self._sasa = {}        # index -> SASA
        self._pocket_idx = set()

        if HAS_MDTRAJ:
            try:
        # robust loader for PDB / mmCIF
                ext = os.path.splitext(self.pdb_file.lower())[1]
                if ext in ('.pdb', '.ent'):
                    traj = mdt.load(self.pdb_file, standard_names=True)   # PDB
                else:                                                     # .cif / .mmcif
                    traj = mdt.load(self.pdb_file)                        # auto-detect

        # --- Shrake-Rupley SASA (probe 1.4 Å) ---
                sasa = mdt.shrake_rupley(traj, probe_radius=1.4,
                                        mode='residue')[0]
        # cache by zero-based residue index
                self._sasa = {idx: float(val) for idx, val in enumerate(sasa)}

        # --- pocket flag (optional: mdtraj 1.9.x only) ---
                try:
                    surf, _ = mdt.geometry.pocket.identify_pockets(traj)
                    self._pocket_idx = {int(i) for i in surf}
                except AttributeError:
                    pass  # pocket module removed in mdtraj ≥1.10

                print("DEBUG  SASA cache size:", len(self._sasa))
            except Exception as e:
                print("MDTraj failed:", e)
                self._sasa = {}
                self._pocket_idx = set()
        self.env_vecs = [self._env_vec(a) for a in self.ca_atoms]
        self.mean, self.std = self._fit_norm(np.array(self.env_vecs))
        print("SASA mean, std:", self.mean[6], self.std[6])
        print("SASA raw samples:", list(self._sasa.values())[:10])
    # ---------- helpers -----------
    @staticmethod
    def _fit_norm(mat): return mat.mean(0), mat.std(0)+EPS
    def _norm(self,v): return (v-self.mean)/self.std

    # environment vector 7 (+20)
    def _env_vec(self, ca):
        c = ca.get_coord()
        neigh = [a for a in self.struct.get_atoms() if a is not ca and np.linalg.norm(a.get_coord() - c) <= RADIUS]
        if not neigh:
            base = np.zeros(6)
        else:
            d = np.array([np.linalg.norm(a.get_coord() - c) for a in neigh])
            dirs = np.array([(a.get_coord() - c) / np.linalg.norm(a.get_coord() - c) for a in neigh])
            angles = [math.acos(np.clip(np.dot(dirs[i], dirs[j]), -1, 1))
                      for i in range(len(dirs)) for j in range(i + 1, len(dirs))]
            angle_dev = np.abs(np.array(angles) - math.radians(109.5)).mean() if angles else 0.0
        # --- SASA lookup (by zero-based residue index) ---
        res_idx = ca.get_parent().get_id()[1] - 1      # Bio.PDB is 1-based
        sasa_raw = self._sasa.get(res_idx, 0.0)

        base = np.array([
            len(neigh), d.mean(), d.max(), d.std(),
            np.var(dirs, 0).sum(), angle_dev, sasa_raw
])  # 7-D core vector (+20 residue embedding)
        resname = ca.get_parent().get_resname()
        if resname in AA_TO_VEC:
            vec = AA_TO_VEC[resname]
        else:
            vec = AA_TO_VEC["XAA"]
        return np.concatenate([base, vec])
    # scoring sub‑functions
    def _w_env(self, nv):
        entropy, dir_var, angle_dev, sasa_z = nv[3], nv[4], nv[5], nv[6]
        w = 1.2*dir_var - 0.8*entropy + angle_bonus(angle_dev)
        # --- SASA: ReLU-style bonus (positive only) ---
        relu_sasa = max(sasa_z, 0.0)    # negative values clipped to 0
        w += 0.3 * relu_sasa
        if self.can_embed:
            w += float(np.dot(self.embed_weights, nv[6:6+len(AA_LIST)]))
        #print(f"sasa_z={sasa_z:+.2f}  relu={relu_sasa:+.2f}  delta={0.3*relu_sasa:+.2f}")
        return w
        

    def _zora_factor(self, ca):
        f=1.0; c=ca.get_coord()
        for m in self.metal_atoms:
            p=ZORA_PARAMS[m.element.strip().upper()]
            d=np.linalg.norm(c-m.get_coord())
            if d<=RADIUS*2: f+=math.exp(-p["alpha"]*d)+p["beta"]
        return f

    def _coordination_boost(self, ca):
        res = ca.get_parent()
        side_atoms = [a for a in res if a.element in ("O","N","S") and a.get_id() not in ("CA","N","C","O")]
        cnt = 0
        for m in self.metal_atoms:
            for sa in side_atoms:
                if np.linalg.norm(sa.get_coord() - m.get_coord()) <= 2.6:  # 2.4~2.6 recommended
                    cnt += 1
        return cnt / 4

    def _angle_penalty(self, ca):
        if not self.metal_atoms: return 0.0
        coords=ca.get_coord(); vals=[]
        for m in self.metal_atoms:
            v1=m.get_coord()-coords
            for n in self.metal_atoms:
                if n is m: continue
                v2=n.get_coord()-coords
                vals.append(math.acos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),-1,1)))
        if not vals: return 0.0
        return -np.mean(np.abs(np.array(vals)-math.radians(109.5)))/math.pi

    def _pocket_flag(self, ca):
        """Return 1.0 if residue belongs to pocket surface (cached)."""
        if not self.pocket_on or not self._pocket_idx:
            return 0.0
        res_idx = ca.get_parent().id[1] - 1   # Bio.PDB is 1-based
        return 1.0 if res_idx in self._pocket_idx else 0.0

    # public residue score
    def residue_score(self, idx):
        v=self.env_vecs[idx]; n=self._norm(v); ca=self.ca_atoms[idx]
        base=self._zora_factor(ca)* (1/(v[1]+EPS)) * max(self._w_env(n),1e-6)
        boost=10.0*self._coordination_boost(ca)+0.1*self._angle_penalty(ca)+0.1*self._pocket_flag(ca)
        return base+boost

    def rank_residues(self, top=10):
        s=[self.residue_score(i) for i in range(len(self.ca_atoms))]
        order=np.argsort(s)[::-1][:top]
        res=[]
        for i in order:
            r=self.ca_atoms[i].get_parent()
            tag=f"{r.get_parent().get_id()}:{r.get_resname()} {r.get_id()[1]}{r.get_id()[2].strip()}"
            res.append((tag,float(s[i])))
        return res

# ===== CLI =====
def main():
    ap=argparse.ArgumentParser(description="Deterministic reactive site predictor")
    ap.add_argument("pdb")
    ap.add_argument("--top",type=int,default=10)
    ap.add_argument("--embed-weights", help="JSON list(20) for residue embedding")
    args=ap.parse_args()
    ew=json.loads(args.embed_weights) if args.embed_weights else None
    pocket_on=os.getenv("POCKET_ON", "0") == "1"
    pred = ReactivePredictor(args.pdb, ew, pocket_on)
    scores = [pred.residue_score(i) for i in range(len(pred.ca_atoms))]
    order = np.argsort(scores)[::-1][:args.top]
    print("Rank\tRes\tTotal\tw_env\tzora\tcoord\tangle\tpocket\tbase\tboost")
    for rank, i in enumerate(order, 1):
        ca = pred.ca_atoms[i]
        v = pred.env_vecs[i]
        n = pred._norm(v)
        w_env = pred._w_env(n)
        zora = pred._zora_factor(ca)
        coord = pred._coordination_boost(ca)
        angle = pred._angle_penalty(ca)
        pocket = pred._pocket_flag(ca)
        base = zora * (1/(v[1]+1e-6)) * max(w_env,1e-6)
        boost = 10.0*coord + 0.1*angle + 0.1*pocket
        total = base + boost
        r = ca.get_parent()
        tag = f"{r.get_parent().get_id()}:{r.get_resname()} {r.get_id()[1]}{r.get_id()[2].strip()}"
        print(f"{rank}\t{tag}\t{total:.4f}\t{w_env:.4f}\t{zora:.4f}\t{coord:.2f}\t{angle:.2f}\t{pocket:.2f}\t{base:.4f}\t{boost:.4f}")

if __name__=="__main__":
    main()

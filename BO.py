#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resumable Integrated SSE Optimization and Similarity Analysis Pipeline
======================================================================
Enhanced version with:
- Checkpoint saving/loading for resumable optimization
- Incremental analysis support
- Coordination-specific tracking (4 vs 6)
- Persistent state management
- Real-time monitoring and updates

NOTE
----
- This script assumes that the core optimization logic (objective, feature
  definitions, etc.) is provided by `sse_optimization_top5_by_coordination`.
- If that module is not importable, a simplified fallback objective is used
  only to test the pipeline structure (results are *not* physically meaningful).
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore')
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import optuna
import xgboost as xgb

# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """Manages saving and loading of optimization checkpoints and progress."""

    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Files inside checkpoint directory
        self.state_file = self.checkpoint_dir / 'optimization_state.json'
        self.trials_file = self.checkpoint_dir / 'trials_history.pkl'
        self.progress_file = self.checkpoint_dir / 'progress_log.csv'
        self.snapshot_dir = self.checkpoint_dir / 'snapshots'
        self.snapshot_dir.mkdir(exist_ok=True)

    # ---------- state (small JSON) ----------

    def save_state(self, state_dict: dict):
        """Save current optimization state as a JSON file."""
        with open(self.state_file, 'w') as f:
            json.dump(state_dict, f, indent=2, default=str)
        print(f"✓ State saved to {self.state_file}")

    def load_state(self):
        """Load previously saved optimization state JSON."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return None

    # ---------- trials history (large pickle) ----------

    def save_trials(self, trials_data):
        """Save Optuna trials history (pickled)."""
        with open(self.trials_file, 'wb') as f:
            pickle.dump(trials_data, f)

    def load_trials(self):
        """Load Optuna trials history from pickle."""
        if self.trials_file.exists():
            with open(self.trials_file, 'rb') as f:
                return pickle.load(f)
        return None

    # ---------- progress log (CSV) ----------

    def append_progress(self, progress_data: dict):
        """Append a row to the progress log CSV."""
        df = pd.DataFrame([progress_data])
        if self.progress_file.exists():
            df.to_csv(self.progress_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.progress_file, index=False)

    def load_progress(self) -> pd.DataFrame:
        """Load progress log CSV as DataFrame."""
        if self.progress_file.exists():
            return pd.read_csv(self.progress_file)
        return pd.DataFrame()

    # ---------- snapshot (JSON) ----------

    def save_snapshot(self, snapshot_data: dict, snapshot_name: str | None = None) -> Path:
        """Save snapshot of current results as JSON."""
        if snapshot_name is None:
            snapshot_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        snapshot_file = self.snapshot_dir / f"{snapshot_name}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2, default=str)
        return snapshot_file


# =============================================================================
# ENHANCED PROGRESS TRACKER
# =============================================================================

class EnhancedProgressTracker:
    """Track optimization progress with coordination-specific monitoring."""

    def __init__(self, checkpoint_manager: CheckpointManager, save_interval: int = 100):
        self.checkpoint_manager = checkpoint_manager
        self.save_interval = save_interval

        # attach later
        self.study = None

        # basic stats
        self.stats = {
            'start_time': None,
            'total_trials': 0,
            'trials_4coord': 0,
            'trials_6coord': 0,
            'best_sse_overall': -np.inf,
            'best_sse_4coord': -np.inf,
            'best_sse_6coord': -np.inf,
        }

    # ------------------------------------------------------------------ #
    #  basic helpers
    # ------------------------------------------------------------------ #

    def attach_study(self, study: optuna.Study):
        """Attach Optuna study object (for saving trials)."""
        self.study = study

    def _update_best_sse(self, n_nonmag_atoms: int | None, value: float):
        """Update best SSE statistics based on coordination."""
        if value > self.stats['best_sse_overall']:
            self.stats['best_sse_overall'] = value

        if n_nonmag_atoms == 4:
            self.stats['trials_4coord'] += 1
            if value > self.stats['best_sse_4coord']:
                self.stats['best_sse_4coord'] = value
        elif n_nonmag_atoms == 6:
            self.stats['trials_6coord'] += 1
            if value > self.stats['best_sse_6coord']:
                self.stats['best_sse_6coord'] = value

    # ------------------------------------------------------------------ #
    #  main entry from objective wrapper
    # ------------------------------------------------------------------ #

    def update(self, trial: optuna.Trial, value: float):
        """Update stats and progress log for each completed trial."""

        if self.stats['start_time'] is None:
            self.stats['start_time'] = datetime.now()

        self.stats['total_trials'] += 1
        n_nonmag_atoms = trial.params.get('n_nonmag_atoms', None)
        self._update_best_sse(n_nonmag_atoms, value)

        # append a progress row
        progress_row = {
            'timestamp': datetime.now().isoformat(),
            'trial_number': trial.number,
            'value': value,
            'n_nonmag_atoms': n_nonmag_atoms,
            'total_trials': self.stats['total_trials'],
            'trials_4coord': self.stats['trials_4coord'],
            'trials_6coord': self.stats['trials_6coord'],
            'best_sse_overall': self.stats['best_sse_overall'],
            'best_sse_4coord': self.stats['best_sse_4coord'],
            'best_sse_6coord': self.stats['best_sse_6coord'],
        }
        self.checkpoint_manager.append_progress(progress_row)

        # periodic checkpointing
        if self.save_interval > 0 and (trial.number + 1) % self.save_interval == 0:
            print(f"\n💾 Reached {trial.number + 1} trials – saving checkpoint.")
            self.save_checkpoint()

    # ------------------------------------------------------------------ #
    #  checkpoint & summary
    # ------------------------------------------------------------------ #

    def save_checkpoint(self):
        """Save state + trials."""
        # state
        self.checkpoint_manager.save_state(self.stats)

        # trials
        if self.study is not None:
            self.checkpoint_manager.save_trials(self.study.trials)

    def print_summary(self):
        """Print a concise summary of progress."""
        print("\n" + "=" * 80)
        print("PROGRESS SUMMARY")
        print("=" * 80)
        print(f"Total trials: {self.stats['total_trials']}")
        print(f"  - 4-coordination: {self.stats['trials_4coord']}")
        print(f"  - 6-coordination: {self.stats['trials_6coord']}")
        print("\nBest SSE values:")
        print(f"  - 4-coord best: {self.stats['best_sse_4coord']:.6f} eV")
        print(f"  - 6-coord best: {self.stats['best_sse_6coord']:.6f} eV")
        print(f"  - Overall best: {self.stats['best_sse_overall']:.6f} eV")
        if self.stats.get('start_time'):
            elapsed = (datetime.now() - self.stats['start_time']).total_seconds() / 3600
            if elapsed > 0:
                print(f"\nElapsed time: {elapsed:.2f} hours")
                print(f"Average rate: {self.stats['total_trials'] / elapsed:.1f} trials/hour")


# =============================================================================
# RESUMABLE OPTIMIZATION – STUDY MANAGEMENT
# =============================================================================

def create_or_load_study(
    study_name: str,
    storage_url: str | None = None,
    direction: str = 'maximize'
) -> optuna.Study:
    """Create a new or load an existing Optuna study."""

    if storage_url:
        # Use a database storage (e.g. sqlite:///study.db)
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_url
            )
            print(f"✓ Loaded existing study '{study_name}' with {len(study.trials)} trials")
        except Exception:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction=direction,
                sampler=optuna.samplers.TPESampler(multivariate=False),
            )
            print(f"✓ Created new study '{study_name}'")
    else:
        # In-memory study with a pickle checkpoint
        checkpoint_file = f"./checkpoints/study_{study_name}.pkl"
        os.makedirs('./checkpoints', exist_ok=True)
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                study = pickle.load(f)
            print(f"✓ Loaded study from checkpoint with {len(study.trials)} trials")
        else:
            study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
            )
            print("✓ Created new in-memory study")

    return study


def save_study_checkpoint(study: optuna.Study, study_name: str):
    """Save study object to a pickle checkpoint."""
    checkpoint_file = f"./checkpoints/study_{study_name}.pkl"
    os.makedirs('./checkpoints', exist_ok=True)
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(study, f)
    print(f"✓ Study saved to {checkpoint_file}")


# =============================================================================
# IMPORT OBJECTIVE / FEATURE DEFINITIONS
# =============================================================================

# Import optimization components from the separated module
try:
    from sse_optimization_top5_by_coordination import (
        objective, get_element_symbol,
        FEATURE_ORDER, MAGNETIC_ATOMS, NONMAGNETIC_ATOMS
    )
except ImportError:
    # Fallback if the module is not available
    print("⚠️ sse_optimization_top5_by_coordination not found. Using fallback definitions.")
    
    # ------------------------------------------------------
    # 기본 정의 (원자 리스트, feature 순서, 원자 특성 테이블)
    # ------------------------------------------------------
    MAGNETIC_ATOMS = [
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    39, 40, 41, 42, 44, 45, 46, 47, 48
    ]

    NONMAGNETIC_ATOMS = [
    5, 6, 7, 8, 9,
    13, 14, 15, 16, 17,
    31, 32, 33, 34, 35,
    49, 50, 51, 52, 53,
    81, 82, 83
    ]

    
    # 모델이 학습된 feature 순서
    FEATURE_ORDER = [
        'avg_bond_length', 'max_bond_length',
        'min_bond_length', 'std_bond_length', 'center_max_angle',
        'center_min_angle', 'center_avg_angle', 'center_std_angle',
        'nonmag_max_angle', 'nonmag_min_angle', 'nonmag_std_angle',
        'labelled_1st', 'labelled_2nd', 'labelled_3rd', 'global_1st',
        'global_2nd', 'global_3rd', 'avg_long_axis', 'avg_short_axis',
        'avg_axis_ratio', 'avg_s', 'avg_delta', 'motif0_nonmag_count',
        'magnetic_atomic_number', 'magnetic_electronegativity',
        'nonmagnetic_atomic_number', 'nonmagnetic_electronegativity',
        'hungarian_rotation_angle_deg', 'dimension', 'avg_motif_measure',
        'unit_cell_volume', 'packing_fraction',
        'p_metric', 'p_metric_std', 'd_orb_e', 'p_orb_e_non', 'd_lone_pair',
        'proxy_M_magnet', 'delta_chi', 'abs_delta_chi', 'delta_Z',
        'abs_delta_Z', 'pd_ratio', 'ax_eq_gap', 'bond_range', 'bond_cv',
        'center_angle_spread', 'nonmag_angle_spread', 'delta_chi_times_axeq',
        'd_global_local_1st', 'd_global_local_2nd', 'd_global_local_3rd'
    ]
    
    def get_element_symbol(z):
        element_map = {
        5:'B', 6:'C', 7:'N', 8:'O', 9:'F',
        13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl',
        21:'Sc', 22:'Ti', 23:'V', 24:'Cr', 25:'Mn', 26:'Fe', 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn',
        31:'Ga', 32:'Ge', 33:'As', 34:'Se', 35:'Br',
        39:'Y', 40:'Zr', 41:'Nb', 42:'Mo', 44:'Ru', 45:'Rh', 46:'Pd', 47:'Ag', 48:'Cd',
        49:'In', 50:'Sn', 51:'Sb', 52:'Te', 53:'I',
        81:'Tl', 82:'Pb', 83:'Bi'
        }

        return element_map.get(z, f'Z={z}')
    
        # 당신이 준 dict 들 (키는 문자열이므로 int로 변환해서 사용)
    d_orbit_e = {"21": 2, "22": 3, "23": 4, "24": 5 , "25": 6, "26": 7, "27": 8, "28": 9, "29": 10, "30": 10,
                "39": 2, "40": 3, "41": 4, "42": 5, "44": 7, "45": 8, "46": 9, "47": 10, "48": 10}

    non_mag_p_orbit_e = {"5": 1, "6": 2, "7": 3, "8": 4, "9": 5,
                        "13": 1, "14": 2, "15": 3, "16": 4, "17": 5,
                        "31": 1, "32": 2, "33": 3, "34": 4, "35": 5,
                        "49": 1,"50": 2, "51": 3, "52": 4, "53": 5,
                        "81": 1, "82": 2, "83": 3}

    ZVAL = {"21": 11, "22": 12, "23": 13, "24": 12 , "25": 13, "26": 8, "27": 9, "28": 10, "29": 11, "30": 12,
            "39": 11, "40": 12, "41": 13, "42": 14, "44": 14, "45": 15, "46": 10, "47": 11, "48": 12,
            "5": 3, "6": 4, "7": 5, "8": 6, "9": 7,
            "13": 3, "14": 4, "15": 5, "16": 6, "17": 7,
            "31": 13, "32": 14, "33": 5, "34": 6, "35": 7,
            "49": 13,"50": 14, "51": 5, "52": 6, "53": 7,
            "81": 13, "82": 14, "83": 15}

    d_lone_pair = {"21": 1, "22": 2, "23": 3, "24": 5, "25": 5, "26": 4, "27": 3, "28": 2, "29": 0, "30": 0,
                "39": 1, "40": 2, "41": 4, "42": 5, "44": 3, "45": 2, "46": 0, "47": 0, "48": 0}

    proxy_map = {"21":0, "22":1.73, "23":2.83, "24": 3.87, "25": 4.90, "26": 4.90, "27": 3.87, "28": 2.83,
                "29": 1.73, "30": 0,
                "39":0, "40":1.73, "41": 2.83, "42": 3.87, "44": 4.90, "45": 3.87, "46": 2.83, "47": 1.73, "48":0}

    electronegativity = {
    21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88,
    28: 1.91, 29: 1.90, 30: 1.65,
    39: 1.22, 40: 1.33, 41: 1.6, 42: 2.16, 44: 2.2, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69,
    5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96,
    49: 1.78, 50: 1.96, 51: 2.05, 52: 2.10, 53: 2.66,
    81: 1.62, 82: 1.87, 83: 2.02,
    }

    # 원자 특성 테이블 (중략된 부분은 기존 파일 그대로 사용)
    ATOM_PROPERTIES = {}

    # 1) 금속 (자성 원자 후보)
    for z_str, d_e in d_orbit_e.items():
        z = int(z_str)
        ATOM_PROPERTIES[z] = {
            'symbol': get_element_symbol(z),
            'electronegativity': electronegativity.get(z, 2.0),
            'd_orb_e': d_e,
            'proxy_M_magnet': proxy_map.get(z_str, 3),  # 기본값 3 정도
        }

    # 2) 비자성 p-블록 원소
    for z_str, p_e in non_mag_p_orbit_e.items():
        z = int(z_str)
        props = ATOM_PROPERTIES.get(z, {})
        props.update({
            'symbol': get_element_symbol(z),
            'electronegativity': electronegativity.get(z, 2.5),
            'p_orb_e_non': p_e,
            'd_lone_pair': d_lone_pair.get(z_str, 1),
            # 필요하다면 ZVAL도 같이 넣기
            'ZVAL': ZVAL.get(z_str, None),
        })
        ATOM_PROPERTIES[z] = props

    
    # ------------------------------------------------------
    # fin_data.csv 에서 Optuna 검색 범위 자동 추출
    # ------------------------------------------------------
    FIN_DATA_STATS = {}
    
    def _load_fin_data_stats(path: str = "fin_data.csv"):
        global FIN_DATA_STATS
        if not os.path.exists(path):
            return
        try:
            df = pd.read_csv(path)
            for col in [
                'unit_cell_volume',
                'p_metric', 'p_metric_std',
                'labelled_1st', 'labelled_2nd', 'labelled_3rd',
                'global_1st', 'global_2nd', 'global_3rd',
                'hungarian_rotation_angle_deg',
            ]:
                if col in df.columns:
                    s = df[col].dropna()
                    if s.empty:
                        continue
                    FIN_DATA_STATS[col] = {
                        'min': float(s.min()),
                        'max': float(s.max())
                    }
        except Exception as e:
            print(f"⚠️ Could not read fin_data.csv for stats: {e}")
    
    _load_fin_data_stats()
    
    def suggest_from_stats(trial, name, default_low, default_high, log=False):
        """fin_data.csv 값 범위를 우선 사용하고, 없으면 기본 범위를 사용"""
        stats = FIN_DATA_STATS.get(name)
        if stats:
            low = stats['min']
            high = stats['max']
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                low, high = default_low, default_high
        else:
            low, high = default_low, default_high
        
        if log:
            return trial.suggest_float(name, low, high, log=True)
        return trial.suggest_float(name, low, high)
    
    # ------------------------------------------------------
    # 기하/모티프 관련 helper 함수
    # ------------------------------------------------------
    try:
        from scipy.spatial import ConvexHull, QhullError
        _HAS_CONVEX_HULL = True
    except Exception:
        ConvexHull = None
        QhullError = Exception
        _HAS_CONVEX_HULL = False
    
    def compute_nonmag_vertex_angle_stats(coords):
        """비자성-비자성-비자성 꼭짓점 각도 통계 (BOND_df와 동일한 정의)"""
        P = np.asarray(coords, dtype=float)
        n = P.shape[0]
        if n < 3:
            return None
        
        angles = []
        for v in range(n):
            others = [i for i in range(n) if i != v]
            for i_idx in range(len(others)):
                for j_idx in range(i_idx + 1, len(others)):
                    a = P[others[i_idx]]
                    b = P[v]
                    c = P[others[j_idx]]
                    v1 = a - b
                    v2 = c - b
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 < 1e-8 or n2 < 1e-8:
                        continue
                    cosang = np.dot(v1, v2) / (n1 * n2)
                    cosang = np.clip(cosang, -1.0, 1.0)
                    ang = np.degrees(np.arccos(cosang))
                    angles.append(ang)
        
        if not angles:
            return None
        
        angles = np.array(angles)
        return {
            'max': float(np.max(angles)),
            'min': float(np.min(angles)),
            'avg': float(np.mean(angles)),
            'std': float(np.std(angles))
        }
    
    def compute_motif_s_delta(coords, eps=1e-12):
        """
        s, delta 계산 (ELONG_df 에서 사용한 정의와 동일)
        - coords: 중심(자성원자)을 원점으로 한 비자성 이웃 상대 좌표
        """
        P = np.asarray(coords, dtype=float)
        if P.shape[0] < 2:
            return 0.0, 0.0
        
        mu = P.mean(axis=0)
        r = np.linalg.norm(P - mu, axis=1)
        rbar = float(r.mean())
        if rbar < eps:
            return 0.0, 0.0
        
        s = float(r.std(ddof=0) / rbar)
        delta = float(np.linalg.norm(mu) / rbar)
        return s, delta
    
    def compute_motif_measure(coords, polyhedron):
        """
        모티프의 특성 길이 (2D 면적 혹은 3D 부피)를 ConvexHull로 계산
        - square_planar : 2D 면적 (평면 투영 convex hull)
        - 그 외 (tetrahedral, octahedral) : 3D 부피
        """
        P = np.asarray(coords, dtype=float)
        if P.shape[0] < 3 or not _HAS_CONVEX_HULL:
            return 0.0
        
        try:
            if polyhedron == 'square_planar':
                # 2D: best-fit plane에 투영해서 convex hull 면적 계산
                center = P.mean(axis=0)
                centered = P - center
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                plane_basis = Vt[:2, :]   # 첫 두 개 주성분
                projected = centered @ plane_basis.T
                hull2d = ConvexHull(projected)
                # 2D에서 hull.volume == area
                return float(hull2d.volume)
            else:
                # 3D 부피
                hull3d = ConvexHull(P)
                return float(hull3d.volume)
        except Exception:
            return 0.0
    
    # ------------------------------------------------------
    # Fallback objective: 실제 좌표 기반 feature 계산
    # ------------------------------------------------------
    def objective(trial, model):
        """Objective function with actual coordinate optimization and atom properties"""
        # 4- 혹은 6-배위 선택
        n_nonmag_atoms = trial.suggest_categorical('n_nonmag_atoms', [4, 6])
        Z_M = trial.suggest_categorical('Z_M', MAGNETIC_ATOMS)
        Z_X = trial.suggest_categorical('Z_X', NONMAGNETIC_ATOMS)
        
        # 4배위인 경우: tetrahedral / square_planar 둘 다 가능
        if n_nonmag_atoms == 4:
            polyhedron = trial.suggest_categorical('polyhedron', ['tetrahedral', 'square_planar'])
        else:
            polyhedron = 'octahedral'
        
        # --- 비자성 좌표 생성 (자성원자는 원점) ---
        coords = []
        if n_nonmag_atoms == 4 and polyhedron == 'tetrahedral':
            # Tetrahedral coordination
            for i in range(4):
                r = trial.suggest_float(f'r_{i}', 1.5, 4.0)
                if i == 0:
                    theta = trial.suggest_float(f'theta_{i}', 0, np.pi)
                    phi = trial.suggest_float(f'phi_{i}', 0, 2*np.pi)
                else:
                    # 대략적인 tetrahedral 각도 범위
                    theta = trial.suggest_float(f'theta_{i}', np.pi/3, 2*np.pi/3)
                    phi = trial.suggest_float(f'phi_{i}', 0, 2*np.pi)
                
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                coords.append([x, y, z])
        
        elif n_nonmag_atoms == 4 and polyhedron == 'square_planar':
            # Square planar: xy 평면에 가깝게 4개 점 배치
            for i in range(4):
                r = trial.suggest_float(f'r_sp_{i}', 1.5, 4.0)
                base_phi = i * (np.pi / 2)       # 0, 90, 180, 270 deg
                dphi = trial.suggest_float(f'dphi_sp_{i}', -0.2, 0.2)
                phi = base_phi + dphi
                # 거의 평면: theta ~ pi/2 (조금 위/아래 허용)
                theta = np.pi/2 + trial.suggest_float(f'dtheta_sp_{i}', -0.2, 0.2)
                
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)            # square planar면 z≈0 근처
                coords.append([x, y, z])
        
        else:
            # Octahedral coordination (6)
            for i in range(6):
                r = trial.suggest_float(f'r_{i}', 1.5, 4.0)
                if i < 4:
                    # Equatorial positions
                    theta = np.pi/2
                    phi = i * np.pi/2 + trial.suggest_float(f'phi_eq_{i}', -0.1, 0.1)
                else:
                    # Axial positions
                    theta = 0 if i == 4 else np.pi
                    phi = 0
                
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                coords.append([x, y, z])
        
        coords = np.array(coords, dtype=float)
        
        # -----------------------------
        # 좌표에서 structural features 추출
        # -----------------------------
        features = {}
        
        # Bond lengths
        bond_lengths = [np.linalg.norm(coord) for coord in coords]
        if bond_lengths:
            features['avg_bond_length'] = float(np.mean(bond_lengths))
            features['max_bond_length'] = float(np.max(bond_lengths))
            features['min_bond_length'] = float(np.min(bond_lengths))
            features['std_bond_length'] = float(np.std(bond_lengths))
            features['bond_range'] = features['max_bond_length'] - features['min_bond_length']
            features['bond_cv'] = (
                features['std_bond_length'] / features['avg_bond_length']
                if features['avg_bond_length'] > 0 else 0.0
            )
        else:
            features['avg_bond_length'] = 0.0
            features['max_bond_length'] = 0.0
            features['min_bond_length'] = 0.0
            features['std_bond_length'] = 0.0
            features['bond_range'] = 0.0
            features['bond_cv'] = 0.0
        
        # 중심(자성원자=원점) 기준 각도: 비자성-자성-비자성
        center_angles = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                v1 = coords[i]
                v2 = coords[j]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-8 or n2 < 1e-8:
                    continue
                cosang = np.dot(v1, v2) / (n1 * n2)
                cosang = np.clip(cosang, -1.0, 1.0)
                ang = np.degrees(np.arccos(cosang))
                center_angles.append(ang)
        
        if center_angles:
            center_angles = np.array(center_angles)
            features['center_max_angle'] = float(np.max(center_angles))
            features['center_min_angle'] = float(np.min(center_angles))
            features['center_avg_angle'] = float(np.mean(center_angles))
            features['center_std_angle'] = float(np.std(center_angles))
            features['center_angle_spread'] = (
                features['center_max_angle'] - features['center_min_angle']
            )
        else:
            features['center_max_angle'] = 0.0
            features['center_min_angle'] = 0.0
            features['center_avg_angle'] = 0.0
            features['center_std_angle'] = 0.0
            features['center_angle_spread'] = 0.0
        
        # 비자성 꼭짓점 각도: 비자성-비자성-비자성 (BOND_df 방식)
        nonmag_stats = compute_nonmag_vertex_angle_stats(coords)
        if nonmag_stats is not None:
            features['nonmag_max_angle'] = nonmag_stats['max']
            features['nonmag_min_angle'] = nonmag_stats['min']
            features['nonmag_std_angle'] = nonmag_stats['std']
            features['nonmag_angle_spread'] = (
                nonmag_stats['max'] - nonmag_stats['min']
            )
        else:
            features['nonmag_max_angle'] = 0.0
            features['nonmag_min_angle'] = 0.0
            features['nonmag_std_angle'] = 0.0
            features['nonmag_angle_spread'] = 0.0
        
        # 축 관련 특징 (ELONG_df의 long/short axis에 대응)
        if len(coords) >= 2:
            centered = coords - np.mean(coords, axis=0)
            try:
                cov = np.cov(centered.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = sorted(eigenvalues, reverse=True)
                if eigenvalues[0] > 1e-8:
                    features['avg_long_axis'] = float(2.0 * np.sqrt(eigenvalues[0]))
                    features['avg_short_axis'] = float(
                        2.0 * np.sqrt(max(eigenvalues[-1], 1e-8))
                    )
                    features['avg_axis_ratio'] = (
                        features['avg_short_axis'] / features['avg_long_axis']
                    )
                else:
                    # fallback: bond length 기반
                    L = features['avg_bond_length'] * 2.0
                    features['avg_long_axis'] = L
                    features['avg_short_axis'] = L
                    features['avg_axis_ratio'] = 1.0
            except Exception:
                L = features['avg_bond_length'] * 2.0
                features['avg_long_axis'] = L
                features['avg_short_axis'] = L
                features['avg_axis_ratio'] = 1.0
        else:
            L = features['avg_bond_length'] * 2.0
            features['avg_long_axis'] = L
            features['avg_short_axis'] = L
            features['avg_axis_ratio'] = 1.0
        
        # s, delta (형상 왜곡, 오프센터) – ELONG_df와 동일
        s_val, delta_val = compute_motif_s_delta(coords)
        features['avg_s'] = s_val
        features['avg_delta'] = delta_val
        
        # -----------------------------
        # 원자별 / 전자구조 feature
        # -----------------------------
        mag_props = ATOM_PROPERTIES.get(
            Z_M, {'electronegativity': 2.0, 'd_orb_e': 5, 'proxy_M_magnet': 3}
        )
        nonmag_props = ATOM_PROPERTIES.get(
            Z_X, {'electronegativity': 2.5, 'p_orb_e_non': 3, 'd_lone_pair': 1}
        )
        
        features['magnetic_atomic_number'] = float(Z_M)
        features['magnetic_electronegativity'] = float(mag_props['electronegativity'])
        features['nonmagnetic_atomic_number'] = float(Z_X)
        features['nonmagnetic_electronegativity'] = float(nonmag_props['electronegativity'])
        
        features['d_orb_e'] = float(mag_props['d_orb_e'])
        features['p_orb_e_non'] = float(nonmag_props['p_orb_e_non'])
        features['d_lone_pair'] = float(nonmag_props['d_lone_pair'])
        features['proxy_M_magnet'] = float(mag_props['proxy_M_magnet'])
        
        # Δχ, ΔZ 등
        features['delta_chi'] = (
            features['magnetic_electronegativity'] - features['nonmagnetic_electronegativity']
        )
        features['abs_delta_chi'] = abs(features['delta_chi'])
        features['delta_Z'] = (
            features['magnetic_atomic_number'] - features['nonmagnetic_atomic_number']
        )
        features['abs_delta_Z'] = abs(features['delta_Z'])
        features['pd_ratio'] = (
            features['p_orb_e_non'] / features['d_orb_e']
            if features['d_orb_e'] > 0 else 0.0
        )

        features['ax_eq_gap'] = features['max_bond_length'] - features['avg_bond_length']

        features['delta_chi_times_axeq'] = features['abs_delta_chi'] * features['ax_eq_gap']
        
        # -----------------------------
        # 모티프 부피/면적, packing fraction
        # -----------------------------
        motif_measure = compute_motif_measure(coords, polyhedron)
        features['avg_motif_measure'] = motif_measure
        
        # unit_cell_volume 은 Optuna에게 맡김 (fin_data.csv 범위 우선)
        unit_cell_volume = suggest_from_stats(
            trial,
            'unit_cell_volume',
            default_low=20.0,
            default_high=500.0
        )
        features['unit_cell_volume'] = unit_cell_volume
        
        if unit_cell_volume > 0:
            pf = motif_measure / unit_cell_volume
        else:
            pf = 0.0
        # 물리적으로는 1을 넘기기 힘들기 때문에 잘못된 경우는 1로 클램프
        if pf > 1.0:
            pf = 1.0
        if pf < 0.0:
            pf = 0.0
        features['packing_fraction'] = pf
        
        # dimension: square planar 인 경우만 2, 나머지는 3
        if (n_nonmag_atoms == 4) and (polyhedron == 'square_planar'):
            dim = 2
        else:
            dim = 3
        features['dimension'] = float(dim)
        
        # -----------------------------
        # 그 외 Optuna에 맡길 feature들
        # -----------------------------
        # p_metric, p_metric_std : fin_data 범위 기반
            # -----------------------------
    # 그 외 Optuna에 맡길 feature들
    # -----------------------------
    # p_metric, p_metric_std : fin_data 범위 기반 (기존 그대로)
        features['p_metric'] = suggest_from_stats(
            trial, 'p_metric', default_low=0.0, default_high=1.0
        )
        features['p_metric_std'] = suggest_from_stats(
            trial, 'p_metric_std', default_low=0.0, default_high=0.5
        )

        # ---- labelled/global 거리 & d_global_local_* ----
        # 1st 거리: fin_data 범위를 우선 사용
        features['labelled_1st'] = suggest_from_stats(
            trial, 'labelled_1st', default_low=0.0, default_high=0.3
        )
        features['global_1st'] = suggest_from_stats(
            trial, 'global_1st', default_low=0.0, default_high=0.3
        )

        # 2nd / 3rd 는 1st 에 Δ를 더하는 형태로 최적화
        # (범위 0.0~0.3 는 대략적인 예시이니, fin_data 값 보고 조정해도 OK)
        delta_labelled_2nd = trial.suggest_float('delta_labelled_2nd', 0.0, 0.3)
        delta_labelled_3rd = trial.suggest_float('delta_labelled_3rd', 0.0, 0.3)
        delta_global_2nd   = trial.suggest_float('delta_global_2nd',   0.0, 0.3)
        delta_global_3rd   = trial.suggest_float('delta_global_3rd',   0.0, 0.3)

        features['labelled_2nd'] = features['labelled_1st'] + delta_labelled_2nd
        features['labelled_3rd'] = features['labelled_2nd'] + delta_labelled_3rd
        features['global_2nd']   = features['global_1st']   + delta_global_2nd
        features['global_3rd']   = features['global_2nd']   + delta_global_3rd

        # d_global_local_* = global_* - labelled_*  (merge_df 정의와 동일)
        for k in ['1st', '2nd', '3rd']:
            features[f'd_global_local_{k}'] = (
                features[f'global_{k}'] - features[f'labelled_{k}']
            )

        # Hungarian 회전 각도도 fin_data 기반 범위에서 탐색
        features['hungarian_rotation_angle_deg'] = suggest_from_stats(
            trial,
            'hungarian_rotation_angle_deg',
            default_low=0.0,
            default_high=180.0,
        )

        
        # Coordination count
        features['motif0_nonmag_count'] = float(n_nonmag_atoms)
        
        # -----------------------------
        # 모델 입력/예측
        # -----------------------------
        X = np.array([features.get(f, 0.0) for f in FEATURE_ORDER]).reshape(1, -1)
        
        import xgboost as xgb
        dmat = xgb.DMatrix(X, feature_names=FEATURE_ORDER)
        y_pred = model.predict(dmat)[0]
        sse_value = float(np.expm1(y_pred))  # Inverse log1p transform
        
        # Optuna trial에 부가 정보 저장
        trial.set_user_attr('polyhedron', polyhedron)
        trial.set_user_attr('nonmag_coords', coords.tolist())
        trial.set_user_attr('features', features)
        
        return sse_value

# =============================================================================
# SAVE FINAL RESULTS
# =============================================================================

def _collect_top_trials(study: optuna.Study, n_nonmag: int | None, top_n: int = 5):
    """Collect top trials filtered by n_nonmag_atoms (if not None)."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if n_nonmag is not None:
        completed = [t for t in completed if t.params.get('n_nonmag_atoms') == n_nonmag]

    if not completed:
        return []

    # 변경: SSE 큰 순서대로 정렬
    completed = sorted(completed, key=lambda t: t.value, reverse=True)


    results = []
    for rank, trial in enumerate(completed[:top_n], start=1):
        params = trial.params
        attrs = trial.user_attrs

        Z_M = params.get('Z_M', None)
        Z_X = params.get('Z_X', None)
        Z_M_symbol = get_element_symbol(Z_M) if Z_M is not None else None
        Z_X_symbol = get_element_symbol(Z_X) if Z_X is not None else None

        result = {
            'rank': rank,
            'trial_number': trial.number,
            'sse': float(trial.value),
            'Z_M': Z_M,
            'Z_X': Z_X,
            'Z_M_symbol': Z_M_symbol,
            'Z_X_symbol': Z_X_symbol,
            'n_nonmag_atoms': params.get('n_nonmag_atoms'),
            'polyhedron': attrs.get('polyhedron'),
            'features': attrs.get('features', None),
            'nonmag_coords': attrs.get('nonmag_coords', None),
        }
        results.append(result)

    return results


def save_final_results(
    study: optuna.Study,
    progress_tracker: EnhancedProgressTracker,
    checkpoint_manager: CheckpointManager,
    results_filename: str = 'optimization_results_resumable.json',
):
    """Save all final optimization results and metadata to a JSON file."""
    print("\n💾 Saving final optimization results...")

    top5_4 = _collect_top_trials(study, n_nonmag=4, top_n=5)
    top5_6 = _collect_top_trials(study, n_nonmag=6, top_n=5)
    top5_all = _collect_top_trials(study, n_nonmag=None, top_n=5)

    metadata = {
        'total_trials': progress_tracker.stats['total_trials'],
        'trials_4coord': progress_tracker.stats['trials_4coord'],
        'trials_6coord': progress_tracker.stats['trials_6coord'],
        'best_sse_overall': progress_tracker.stats['best_sse_overall'],
        'best_sse_4coord': progress_tracker.stats['best_sse_4coord'],
        'best_sse_6coord': progress_tracker.stats['best_sse_6coord'],
    }

    results = {
        'metadata': metadata,
        'top5_4coordination': top5_4,
        'top5_6coordination': top5_6,
        'top5_overall': top5_all,
    }

    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✓ Optimization results saved to {results_filename}")
    checkpoint_manager.save_snapshot(results, snapshot_name='final_results')


# =============================================================================
# INCREMENTAL SIMILARITY ANALYZER
# =============================================================================

class IncrementalSimilarityAnalyzer:
    """
    Compare optimized structures (feature vectors) with existing dataset.

    The existing data CSV is assumed to contain:
    - A column with SSE / energy (name containing 'sse' or 'energy' or 'target')
    - A 'filename' or similar identifier column (optional)
    - A set of numeric feature columns consistent with the model features
    """

    def __init__(self, existing_data_path: str):
        if not os.path.exists(existing_data_path):
            raise FileNotFoundError(f"Existing data CSV not found: {existing_data_path}")

        self.df = pd.read_csv(existing_data_path)

        # identify filename column (if any)
        fname_candidates = [c for c in self.df.columns if 'file' in c.lower() or 'name' in c.lower()]
        self.filename_col = fname_candidates[0] if fname_candidates else None

        # identify SSE/energy column (if any)
        sse_candidates = [c for c in self.df.columns if 'sse' in c.lower() or 'energy' in c.lower() or 'target' in c.lower()]
        self.sse_col = sse_candidates[0] if sse_candidates else None

        # numeric feature columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # exclude SSE column from features
        self.feature_cols = [c for c in numeric_cols if c != self.sse_col]

        if not self.feature_cols:
            raise ValueError("No numeric feature columns found for similarity analysis.")

        self.scaler = StandardScaler()
        self.feature_matrix = self.scaler.fit_transform(self.df[self.feature_cols].values)

    # ------------------------------------------------------------------ #
    #  internal helpers
    # ------------------------------------------------------------------ #

    def _trial_vector_from_features(self, features: dict | None) -> np.ndarray | None:
        if features is None:
            return None
        vec = np.array([features.get(col, np.nan) for col in self.feature_cols], dtype=float)
        if np.isnan(vec).all():
            return None
        # fill NaNs with column means (0 after scaling) — but here before scaling
        nan_mask = np.isnan(vec)
        if nan_mask.any():
            vec[nan_mask] = np.nanmean(self.df[self.feature_cols].values, axis=0)[nan_mask]
        return vec

    # ------------------------------------------------------------------ #
    #  public API
    # ------------------------------------------------------------------ #

    def analyze_batch(self, trial_results: list[dict], top_n: int = 10) -> pd.DataFrame:
        """
        Given a list of trial summary dicts (from `save_final_results`),
        compute similarity to existing dataset and return a DataFrame.
        """
        rows = []

        for trial_info in trial_results:
            features = trial_info.get('features')
            if features is None:
                continue

            trial_vec = self._trial_vector_from_features(features)
            if trial_vec is None:
                continue

            trial_vec_scaled = self.scaler.transform(trial_vec.reshape(1, -1))[0]

            # compute cosine similarity and Euclidean distance
            sims = []
            dists = []
            for i, ref_vec in enumerate(self.feature_matrix):
                sims.append(1.0 - cosine(trial_vec_scaled, ref_vec))
                dists.append(euclidean(trial_vec_scaled, ref_vec))

            sims = np.array(sims)
            dists = np.array(dists)

            # sort by best similarity (cosine descending, then distance ascending)
            idx_sorted = np.lexsort((dists, -sims))[:top_n]

            for rank_local, idx in enumerate(idx_sorted, start=1):
                row = {
                    'trial_rank': trial_info['rank'],
                    'trial_number': trial_info['trial_number'],
                    'trial_sse': trial_info['sse'],
                    'trial_composition': f"{trial_info.get('Z_M_symbol')}-{trial_info.get('Z_X_symbol')}",
                    'similar_rank_local': rank_local,
                    'similar_index': int(idx),
                    'similar_similarity': float(sims[idx]),
                    'similar_distance': float(dists[idx]),
                }
                if self.filename_col is not None:
                    row['similar_filename'] = str(self.df.loc[idx, self.filename_col])
                if self.sse_col is not None:
                    row['similar_sse'] = float(self.df.loc[idx, self.sse_col])
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)


# =============================================================================
# MONITORING DASHBOARD
# =============================================================================

def create_monitoring_dashboard(checkpoint_dir: str, output_dir: str) -> str | None:
    """Create a multi-panel monitoring dashboard from the progress log."""
    mgr = CheckpointManager(checkpoint_dir)
    progress_df = mgr.load_progress()

    if progress_df.empty:
        print("⚠️ No progress log found – dashboard not created.")
        return None

    # parse timestamps
    try:
        progress_df['timestamp'] = pd.to_datetime(progress_df['timestamp'])
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Plot 1: total trials vs time
    ax = axes[0, 0]
    ax.plot(progress_df['timestamp'], progress_df['total_trials'], marker='o', linestyle='-')
    ax.set_xlabel('Time')
    ax.set_ylabel('Total trials')
    ax.set_title('Total Trials vs Time')
    ax.grid(True, alpha=0.3)

    # Plot 2: objective value vs trial number
    ax = axes[0, 1]
    ax.plot(progress_df['trial_number'], progress_df['value'], marker='.', linestyle='-')
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Objective (SSE)')
    ax.set_title('Objective vs Trial')
    ax.grid(True, alpha=0.3)

    # Plot 3: best SSE for 4- and 6-coordination and overall
    ax = axes[0, 2]
    ax.plot(progress_df['trial_number'], progress_df['best_sse_overall'],
            label='Best SSE overall', linewidth=2)
    ax.plot(progress_df['trial_number'], progress_df['best_sse_4coord'],
            label='Best SSE 4-coord', linestyle='--')
    ax.plot(progress_df['trial_number'], progress_df['best_sse_6coord'],
            label='Best SSE 6-coord', linestyle=':')
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Best SSE')
    ax.set_title('Best SSE by Coordination')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: counts of 4- and 6-coordination
    ax = axes[1, 0]
    ax.plot(progress_df['trial_number'], progress_df['trials_4coord'],
            label='4-coord count')
    ax.plot(progress_df['trial_number'], progress_df['trials_6coord'],
            label='6-coord count')
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Count')
    ax.set_title('Coordination Counts')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: fraction of 4- and 6-coordination
    ax = axes[1, 1]
    total = progress_df['trials_4coord'] + progress_df['trials_6coord']
    with np.errstate(divide='ignore', invalid='ignore'):
        frac4 = np.where(total > 0, progress_df['trials_4coord'] / total, 0.0)
        frac6 = np.where(total > 0, progress_df['trials_6coord'] / total, 0.0)
    ax.plot(progress_df['trial_number'], frac4, label='4-coord fraction')
    ax.plot(progress_df['trial_number'], frac6, label='6-coord fraction')
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Fraction')
    ax.set_ylim(0, 1)
    ax.set_title('Coordination Fractions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: text summary
    ax = axes[1, 2]
    ax.axis('off')
    latest = progress_df.iloc[-1]
    stats_text = "📊 Current Statistics\n" + "=" * 30 + "\n\n"
    stats_text += f"Total trials: {int(latest['total_trials'])}\n"
    stats_text += f"4-coord trials: {int(latest['trials_4coord'])}\n"
    stats_text += f"6-coord trials: {int(latest['trials_6coord'])}\n\n"
    stats_text += f"Best SSE overall: {latest['best_sse_overall']:.6f} eV\n"
    stats_text += f"Best SSE 4-coord: {latest['best_sse_4coord']:.6f} eV\n"
    stats_text += f"Best SSE 6-coord: {latest['best_sse_6coord']:.6f} eV\n\n"

    if len(progress_df) > 1:
        start = pd.to_datetime(progress_df['timestamp'].iloc[0])
        end   = pd.to_datetime(progress_df['timestamp'].iloc[-1])
        total_time = (end - start).total_seconds() / 3600
        if total_time > 0:
            avg_rate = latest['total_trials'] / total_time
            stats_text += f"Total time: {total_time:.2f} hours\n"
            stats_text += f"Average rate: {avg_rate:.1f} trials/hour\n"

    ax.text(0.05, 0.5, stats_text, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace', va='center')

    plt.suptitle('Optimization Monitoring Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    dashboard_file = os.path.join(output_dir, 'monitoring_dashboard.png')
    plt.savefig(dashboard_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Monitoring dashboard saved to {dashboard_file}")
    return dashboard_file


# =============================================================================
# RESUMABLE OPTIMIZATION (MAIN)
# =============================================================================

def run_resumable_optimization(
    model_path: str = 'final_model_all.json',
    n_trials: int = 1000,
    n_jobs: int = 8,
    study_name: str = 'sse_optimization',
    storage_url: str | None = None,
    checkpoint_interval: int = 100,
    checkpoint_dir: str = './checkpoints',
) -> optuna.Study | None:
    """
    Run optimization with checkpoint and resumability support.
    Returns the Optuna study if successful, otherwise None.
    """
    print("=" * 80)
    print("🔄 RESUMABLE SSE OPTIMIZATION")
    print("=" * 80)

    # checkpoint manager
    checkpoint_mgr = CheckpointManager(checkpoint_dir)

    # load model
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None

    model = xgb.Booster()
    model.load_model(model_path)
    print(f"✓ Loaded XGBoost model from: {model_path}")

    # create or load study
    study = create_or_load_study(study_name, storage_url, direction='maximize')

    # initialize progress tracker
    progress_tracker = EnhancedProgressTracker(checkpoint_mgr, save_interval=checkpoint_interval)
    progress_tracker.attach_study(study)
    if progress_tracker.stats['start_time'] is None:
        progress_tracker.stats['start_time'] = datetime.now()

    # compute remaining trials
    completed_trials = len(study.trials)
    remaining_trials = max(0, n_trials - completed_trials)

    if remaining_trials == 0:
        print(f"✓ Already completed {n_trials} trials")
        progress_tracker.print_summary()
        # Still save results if not already saved
        save_final_results(study, progress_tracker, checkpoint_mgr)
        return study

    print("\n📊 Optimization Status:")
    print(f"  - Completed trials: {completed_trials}")
    print(f"  - Remaining trials: {remaining_trials}")
    print(f"  - Total target: {n_trials}")
    print(f"  - Checkpoint interval: {checkpoint_interval}")
    print(f"  - Parallel jobs: {n_jobs}")

    def objective_with_tracking(trial: optuna.Trial) -> float:
        result = objective(trial, model)
        progress_tracker.update(trial, result)

        # periodic console progress
        if (trial.number + 1) % 10 == 0:
            print(f"  Trial {trial.number}: SSE={result:.6f} eV, "
                  f"n={trial.params.get('n_nonmag_atoms')}, "
                  f"Best: {progress_tracker.stats['best_sse_overall']:.6f} eV")
        return result

    print("\n🚀 Starting optimization.")
    print("-" * 60)

    try:
        study.optimize(
            objective_with_tracking,
            n_trials=remaining_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

        # save final checkpoint & study
        progress_tracker.save_checkpoint()
        if not storage_url:
            save_study_checkpoint(study, study_name)

        # save final results JSON
        save_final_results(study, progress_tracker, checkpoint_mgr)

    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user.")
        print("   Saving checkpoint and partial results...")
        progress_tracker.save_checkpoint()
        if not storage_url:
            save_study_checkpoint(study, study_name)
        save_final_results(study, progress_tracker, checkpoint_mgr)
        print("✓ Checkpoint saved. Run again to resume.")

    progress_tracker.print_summary()
    return study


# =============================================================================
# FINAL ANALYSIS PIPELINE
# =============================================================================

def generate_final_report(results: dict, similarity_df: pd.DataFrame | None, output_dir: str):
    """Generate a human-readable text report summarizing results and similarity."""
    report_file = os.path.join(output_dir, 'final_report.txt')

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RESUMABLE SSE OPTIMIZATION - FINAL REPORT\n")
        f.write("=" * 80 + "\n\n")

        # metadata
        meta = results.get('metadata', {})
        f.write("OPTIMIZATION SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total trials completed: {meta.get('total_trials')}\n")
        f.write(f"  - 4-coordination trials: {meta.get('trials_4coord')}\n")
        f.write(f"  - 6-coordination trials: {meta.get('trials_6coord')}\n")
        f.write(f"Best SSE overall: {meta.get('best_sse_overall'):.6f} eV\n")
        f.write(f"Best SSE 4-coord: {meta.get('best_sse_4coord'):.6f} eV\n")
        f.write(f"Best SSE 6-coord: {meta.get('best_sse_6coord'):.6f} eV\n")

        # top-5 blocks
        for category, title in [
            ('top5_4coordination', "TOP-5 (4-coordination)"),
            ('top5_6coordination', "TOP-5 (6-coordination)"),
            ('top5_overall', "TOP-5 (Overall)"),
        ]:
            f.write("\n\n" + title + "\n")
            f.write("-" * 80 + "\n")
            rows = results.get(category, [])
            f.write(f"{'Rank':<6} {'SSE (eV)':<12} {'M-X':<12} {'Polyhedron':<20}\n")
            f.write("-" * 80 + "\n")
            for trial in rows:
                comp = f"{trial.get('Z_M_symbol')}-{trial.get('Z_X_symbol')}"
                f.write(f"{trial['rank']:<6} "
                        f"{trial['sse']:<12.6f} "
                        f"{comp:<12} "
                        f"{str(trial.get('polyhedron')):<20}\n")

        # similarity analysis
        if similarity_df is not None and not similarity_df.empty:
            f.write("\n\nSIMILARITY ANALYSIS HIGHLIGHTS\n")
            f.write("-" * 80 + "\n")
            for trial_rank in similarity_df['trial_rank'].unique():
                sub = similarity_df[similarity_df['trial_rank'] == trial_rank].sort_values('similar_rank_local')
                best = sub.iloc[0]
                f.write(f"\nTrial rank {trial_rank} (trial #{best['trial_number']} | "
                        f"{best['trial_composition']}):\n")
                if 'similar_filename' in best:
                    f.write(f"  Best match: {best['similar_filename']}\n")
                f.write(f"  Similarity: {best['similar_similarity']:.4f}\n")
                if 'similar_sse' in best:
                    f.write(f"  Similar SSE: {best['similar_sse']:.4f} eV\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Report generated successfully.\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Final report saved to {report_file}")


def run_resumable_pipeline(
    model_path: str = 'final_model_all.json',
    existing_data_path: str = 'fin_data.csv',
    n_trials: int = 1000,
    n_jobs: int = 8,
    study_name: str = 'sse_optimization',
    storage_url: str | None = None,
    checkpoint_interval: int = 100,
    checkpoint_dir: str = './checkpoints',
    output_dir: str = './optimization_results',
    skip_optimization: bool = False,
    monitor_only: bool = False,
) -> bool:
    """
    Run the complete resumable pipeline:
    1) (Optional) optimization with checkpointing
    2) Load results
    3) Similarity analysis
    4) Monitoring dashboard and text report
    """

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("🔄 RESUMABLE SSE OPTIMIZATION AND ANALYSIS PIPELINE")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  - Model: {model_path}")
    print(f"  - Existing data: {existing_data_path}")
    print(f"  - Target trials: {n_trials}")
    print(f"  - Parallel jobs: {n_jobs}")
    print(f"  - Study name: {study_name}")
    print(f"  - Checkpoint interval: {checkpoint_interval}")
    print(f"  - Checkpoint dir: {checkpoint_dir}")
    print(f"  - Output directory: {output_dir}")

    # monitoring-only mode
    if monitor_only:
        print("\n📊 Running in monitoring mode only.")
        dashboard = create_monitoring_dashboard(checkpoint_dir, output_dir)
        if dashboard:
            print(f"✓ Monitoring complete. Dashboard: {dashboard}")
        return True

    # phase 1: optimization
    if not skip_optimization:
        print("\n" + "-" * 60)
        print("PHASE 1: OPTIMIZATION")
        print("-" * 60)
        study = run_resumable_optimization(
            model_path=model_path,
            n_trials=n_trials,
            n_jobs=n_jobs,
            study_name=study_name,
            storage_url=storage_url,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
        )
        if study is None:
            print("❌ Optimization failed.")
            return False
    else:
        print("\n⚠️ Skipping optimization phase.")

    # phase 2: load results
    print("\n" + "-" * 60)
    print("PHASE 2: ANALYSIS")
    print("-" * 60)

    results_file = 'optimization_results_resumable.json'
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return False

    with open(results_file, 'r') as f:
        results = json.load(f)
    print(f"✓ Loaded results with {results['metadata']['total_trials']} trials")

    # phase 3: similarity analysis
    print("\n📊 Running similarity analysis.")
    analyzer = IncrementalSimilarityAnalyzer(existing_data_path)

    all_similarity_results = []
    for category in ['top5_4coordination', 'top5_6coordination', 'top5_overall']:
        if category in results:
            print(f"  Analyzing {category}...")
            df_cat = analyzer.analyze_batch(results[category], top_n=10)
            if not df_cat.empty:
                df_cat['category'] = category
                all_similarity_results.append(df_cat)

    combined_similarity_df = None
    if all_similarity_results:
        combined_similarity_df = pd.concat(all_similarity_results, ignore_index=True)
        similarity_output = os.path.join(output_dir, 'similarity_analysis_by_coordination.csv')
        combined_similarity_df.to_csv(similarity_output, index=False)
        print(f"✓ Similarity analysis saved to {similarity_output}")

    # phase 4: reports & dashboard
    print("\n📈 Generating reports and dashboard.")
    dashboard = create_monitoring_dashboard(checkpoint_dir, output_dir)
    generate_final_report(results, combined_similarity_df, output_dir)

    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved in: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - optimization_results_resumable.json        - Final optimization results (in CWD)")
    print(f"  - similarity_analysis_by_coordination.csv   - Similarity analysis")
    print(f"  - monitoring_dashboard.png                  - Progress monitoring")
    print(f"  - final_report.txt                          - Text report")

    meta = results['metadata']
    print("\n📊 QUICK SUMMARY:")
    print("-" * 50)
    print(f"Total trials: {meta['total_trials']}")
    print(f"Best SSE overall: {meta['best_sse_overall']:.6f} eV")
    print(f"Best SSE 4-coord: {meta['best_sse_4coord']:.6f} eV")
    print(f"Best SSE 6-coord: {meta['best_sse_6coord']:.6f} eV")

    return True


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Resumable SSE Optimization and Analysis Pipeline'
    )

    # model & data
    parser.add_argument('--model', default='final_model_all.json',
                        help='Path to XGBoost model JSON file')
    parser.add_argument('--data', default='fin_data.csv',
                        help='Path to existing structures CSV file')

    # optimization parameters
    parser.add_argument('--trials', type=int, default=1000,
                        help='Total number of optimization trials')
    parser.add_argument('--jobs', type=int, default=8,
                        help='Number of parallel jobs')

    # study configuration
    parser.add_argument('--study-name', default='sse_optimization',
                        help='Optuna study name')
    parser.add_argument('--storage', default=None,
                        help='Optuna storage URL (e.g. sqlite:///study.db)')

    # checkpoint configuration
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Save checkpoint every N trials')
    parser.add_argument('--checkpoint-dir', default='./checkpoints',
                        help='Directory for checkpoints')

    # output configuration
    parser.add_argument('--output', default='./optimization_results',
                        help='Output directory for results')

    # control flags
    parser.add_argument('--skip-optimization', action='store_true',
                        help='Skip optimization and run analysis only')
    parser.add_argument('--monitor', action='store_true',
                        help='Run in monitoring mode only')
    parser.add_argument('--resume', action='store_true',
                        help='(Reserved) Resume from last checkpoint (default behavior)')

    args = parser.parse_args()

    success = run_resumable_pipeline(
        model_path=args.model,
        existing_data_path=args.data,
        n_trials=args.trials,
        n_jobs=args.jobs,
        study_name=args.study_name,
        storage_url=args.storage,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output,
        skip_optimization=args.skip_optimization,
        monitor_only=args.monitor,
    )

    if success:
        print("\n🎉 Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed. Check error messages above.")

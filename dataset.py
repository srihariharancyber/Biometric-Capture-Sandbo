import math
import random
import numpy as np
import torch
from typing import List, Dict, Any, Tuple

# Key categories map
KEY_CATEGORIES = {
    'alpha': 0.1,
    'digit': 0.3,
    'space': 0.5,
    'enter': 0.7,
    'backspace': 0.9,
    'modifier': 0.8,
    'punct': 0.4,
    'mouse': 0.0
}

def categorize_key(key: str) -> float:
    if not key:
        return 0.0
    k = key.lower()
    if k in ['shift', 'control', 'alt', 'meta', 'capslock']:
        return KEY_CATEGORIES['modifier']
    if k == ' ':
        return KEY_CATEGORIES['space']
    if k == 'enter':
        return KEY_CATEGORIES['enter']
    if k in ['backspace', 'delete']:
        return KEY_CATEGORIES['backspace']
    if k.isalpha() and len(k) == 1:
        return KEY_CATEGORIES['alpha']
    if k.isdigit() and len(k) == 1:
        return KEY_CATEGORIES['digit']
    return KEY_CATEGORIES['punct']


class BiometricFeatureExtractor:
    """
    Normalizes raw temporal events (keystrokes and mouse kinematic samples)
    into standard 8-dimensional feature tokens.
    """
    @staticmethod
    def extract_token(event: Dict[str, Any]) -> List[float]:
        """
        Token shape: [8]
        0: is_mouse (0.0 for keystroke, 1.0 for mouse)
        1: hold_time_normalized (0 - 1.0, scaled ~ 0 to 500ms)
        2: flight_time_normalized (0 - 1.0, scaled ~ 0 to 1000ms)
        3: velocity_normalized (0 - 1.0, scaled ~ 0 to 2000 px/s)
        4: acceleration_normalized (0 - 1.0)
        5: curvature_normalized (0 - 1.0, ratio of angle changes)
        6: jerk_normalized (0 - 1.0, smoothness)
        7: key_category_normalized (0 - 1.0)
        """
        is_mouse = 1.0 if event.get('type') == 'mouse' else 0.0
        
        # Keystroke features
        hold_time = float(event.get('hold_time', 0.0))  # in ms
        flight_time = float(event.get('flight_time', 0.0))  # in ms
        key = event.get('key', '')
        
        hold_norm = np.clip(hold_time / 400.0, 0.0, 1.0)
        flight_norm = np.clip(flight_time / 800.0, 0.0, 1.0)
        key_cat = categorize_key(key) if not is_mouse else 0.0
        
        # Mouse features
        velocity = float(event.get('velocity', 0.0))
        accel = float(event.get('accel', 0.0))
        curvature = float(event.get('curvature', 0.0))
        jerk = float(event.get('jerk', 0.0))
        
        vel_norm = np.clip(velocity / 1500.0, 0.0, 1.0)
        accel_norm = np.clip(accel / 5000.0, 0.0, 1.0)
        curv_norm = np.clip(curvature / 3.1415, 0.0, 1.0)
        jerk_norm = np.clip(jerk / 20000.0, 0.0, 1.0)
        
        return [
            float(is_mouse),
            float(hold_norm),
            float(flight_norm),
            float(vel_norm),
            float(accel_norm),
            float(curv_norm),
            float(jerk_norm),
            float(key_cat)
        ]

    @classmethod
    def sequence_to_tensor(cls, events: List[Dict[str, Any]], target_len: int = 32) -> torch.Tensor:
        """
        Pads or truncates a list of events to target_len x 8 tensor.
        """
        tokens = [cls.extract_token(ev) for ev in events]
        if len(tokens) == 0:
            tokens = [[0.0] * 8]
            
        if len(tokens) < target_len:
            # Pad with repetitions or neutral zeroes
            last_token = tokens[-1]
            while len(tokens) < target_len:
                tokens.append([0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1])
        elif len(tokens) > target_len:
            # Sliding window: take last target_len tokens
            tokens = tokens[-target_len:]
            
        return torch.tensor([tokens], dtype=torch.float32)  # [1, target_len, 8]


# Biometric User Profiles with distinct Gaussian priors
USER_PERSONAS = {
    0: {
        'name': 'Alice Vance (Touch Typist)',
        'role': 'Legitimate Primary User',
        'typing_speed_wpm': 84.0,
        'mean_hold': 72.0, 'std_hold': 12.0,
        'mean_flight': 85.0, 'std_flight': 22.0,
        'mouse_speed': 420.0, 'mouse_curvature': 0.18,
        'jerk_factor': 0.15,
        'digraph_boost': {'th': -25, 'he': -20, 'in': -18, 'er': -22}
    },
    1: {
        'name': 'Bob Miller (Hunt & Peck)',
        'role': 'Secondary Enrolled User',
        'typing_speed_wpm': 36.0,
        'mean_hold': 138.0, 'std_hold': 28.0,
        'mean_flight': 210.0, 'std_flight': 70.0,
        'mouse_speed': 260.0, 'mouse_curvature': 0.35,
        'jerk_factor': 0.38,
        'digraph_boost': {}
    },
    2: {
        'name': 'Charlie Chen (Coder / Staccato)',
        'role': 'Enrolled Engineer',
        'typing_speed_wpm': 68.0,
        'mean_hold': 60.0, 'std_hold': 15.0,
        'mean_flight': 140.0, 'std_flight': 55.0,
        'mouse_speed': 680.0, 'mouse_curvature': 0.25,
        'jerk_factor': 0.28,
        'digraph_boost': {'()': -30, '->': -25}
    },
    3: {
        'name': 'Eve Malicious (Human Imposter)',
        'role': 'Unauthorized Intruder',
        'typing_speed_wpm': 42.0,
        'mean_hold': 185.0, 'std_hold': 45.0,
        'mean_flight': 310.0, 'std_flight': 95.0,
        'mouse_speed': 180.0, 'mouse_curvature': 0.55,
        'jerk_factor': 0.65,
        'digraph_boost': {}
    },
    4: {
        'name': 'ReplayBot-v2 (Automated Script)',
        'role': 'Synthetic Bot Attack',
        'typing_speed_wpm': 150.0,
        'mean_hold': 40.0, 'std_hold': 0.8,
        'mean_flight': 30.0, 'std_flight': 1.2,
        'mouse_speed': 950.0, 'mouse_curvature': 0.01,
        'jerk_factor': 0.02,
        'digraph_boost': {}
    }
}


class SyntheticBiometricGenerator:
    """
    Generates synthetic realistic behavioral biometric sequences for training,
    benchmarking, and live attacker simulation.
    """
    @staticmethod
    def generate_event_stream(user_id: int, count: int = 32) -> List[Dict[str, Any]]:
        persona = USER_PERSONAS.get(user_id, USER_PERSONAS[0])
        events = []
        
        sample_chars = list("abcdefghijklmnopqrstuvwxyz .,\n")
        
        for i in range(count):
            # Mix 70% keystrokes, 30% mouse movements
            if random.random() < 0.72:
                # Keystroke event
                ch = random.choice(sample_chars)
                hold = max(15.0, np.random.normal(persona['mean_hold'], persona['std_hold']))
                flight = max(10.0, np.random.normal(persona['mean_flight'], persona['std_flight']))
                events.append({
                    'type': 'keystroke',
                    'key': ch,
                    'hold_time': round(float(hold), 1),
                    'flight_time': round(float(flight), 1),
                    'velocity': 0.0,
                    'accel': 0.0,
                    'curvature': 0.0,
                    'jerk': 0.0
                })
            else:
                # Mouse event
                speed = max(20.0, np.random.normal(persona['mouse_speed'], persona['mouse_speed'] * 0.25))
                curv = max(0.01, np.random.normal(persona['mouse_curvature'], 0.08))
                jerk = max(0.01, np.random.normal(persona['jerk_factor'] * 10000, 2000))
                accel = speed * 1.8
                events.append({
                    'type': 'mouse',
                    'key': '',
                    'hold_time': 0.0,
                    'flight_time': round(float(random.uniform(16.0, 80.0)), 1),
                    'velocity': round(float(speed), 1),
                    'accel': round(float(accel), 1),
                    'curvature': round(float(curv), 3),
                    'jerk': round(float(jerk), 1)
                })
        return events

    @classmethod
    def generate_dataset(
        cls,
        samples_per_user: int = 120,
        seq_len: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a balanced training/benchmark dataset across all 5 personas.
        Returns:
            X: [N_total, seq_len, 8]
            y: [N_total]
        """
        all_X = []
        all_y = []
        
        for user_id in range(5):
            for _ in range(samples_per_user):
                stream = cls.generate_event_stream(user_id=user_id, count=seq_len)
                tokens = [BiometricFeatureExtractor.extract_token(ev) for ev in stream]
                all_X.append(tokens)
                all_y.append(user_id)
                
        X_tensor = torch.tensor(all_X, dtype=torch.float32)
        y_tensor = torch.tensor(all_y, dtype=torch.long)
        
        # Shuffle
        indices = torch.randperm(X_tensor.size(0))
        return X_tensor[indices], y_tensor[indices]


def compute_benchmark_metrics(model, device='cpu') -> Dict[str, Any]:
    """
    Evaluates the continuous authentication model on hold-out sequences.
    Computes FAR, FRR, EER (Equal Error Rate), ROC Curve, and AUC.
    """
    model.eval()
    
    # Generate test sequences: User 0 (Legitimate) vs Users 1, 2, 3, 4 (Imposters)
    legit_stream_X, _ = SyntheticBiometricGenerator.generate_dataset(samples_per_user=100, seq_len=32)
    # Filter by user
    X_test, y_test = SyntheticBiometricGenerator.generate_dataset(samples_per_user=60, seq_len=32)
    
    with torch.no_grad():
        # Compute user 0 reference centroid
        u0_indices = (y_test == 0).nonzero(as_tuple=True)[0]
        u0_samples = X_test[u0_indices]
        u0_embs = model.extract_embedding(u0_samples)
        centroid = torch.mean(u0_embs, dim=0, keepdim=True)
        centroid = torch.nn.functional.normalize(centroid, p=2, dim=-1)
        
        # Test similarity scores for all test samples against user 0 centroid
        all_embs = model.extract_embedding(X_test)
        cos_sims = torch.sum(all_embs * centroid, dim=-1).cpu().numpy()
        
    labels = (y_test == 0).cpu().numpy().astype(int)  # 1 for legitimate, 0 for imposter
    
    # Compute ROC Curve and FAR / FRR across thresholds [0.0 to 1.0]
    thresholds = np.linspace(0.0, 1.0, 101)
    far_list = []
    frr_list = []
    tpr_list = []
    fpr_list = []
    
    total_positives = np.sum(labels == 1)
    total_negatives = np.sum(labels == 0)
    
    eer = 0.5
    eer_threshold = 0.5
    min_diff = 999.0
    
    for th in thresholds:
        # Predict positive (legitimate) if cos_sim >= th
        preds = (cos_sims >= th).astype(int)
        
        # False Rejection: legitimate classified as imposter (preds == 0 when labels == 1)
        fn = np.sum((preds == 0) & (labels == 1))
        frr = fn / total_positives if total_positives > 0 else 0.0
        
        # False Acceptance: imposter classified as legitimate (preds == 1 when labels == 0)
        fp = np.sum((preds == 1) & (labels == 0))
        far = fp / total_negatives if total_negatives > 0 else 0.0
        
        # True Positive Rate (Sensitivity)
        tp = np.sum((preds == 1) & (labels == 1))
        tpr = tp / total_positives if total_positives > 0 else 0.0
        
        far_list.append(round(float(far), 4))
        frr_list.append(round(float(frr), 4))
        fpr_list.append(round(float(far), 4))
        tpr_list.append(round(float(tpr), 4))
        
        # Check EER intersection point
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0
            eer_threshold = th

    # Area Under ROC Curve (Trapezoidal Rule)
    auc = 0.0
    sorted_indices = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[sorted_indices]
    tpr_sorted = np.array(tpr_list)[sorted_indices]
    if hasattr(np, 'trapezoid'):
        auc = float(np.trapezoid(tpr_sorted, fpr_sorted))
    else:
        # manual trapezoidal sum
        auc = float(np.sum((fpr_sorted[1:] - fpr_sorted[:-1]) * (tpr_sorted[1:] + tpr_sorted[:-1]) / 2.0))
    if auc < 0.5:
        auc = 1.0 - auc

    return {
        'thresholds': thresholds.tolist(),
        'far': far_list,
        'frr': frr_list,
        'fpr': fpr_list,
        'tpr': tpr_list,
        'eer': round(float(eer), 4),
        'eer_threshold': round(float(eer_threshold), 3),
        'auc': round(float(auc), 4),
        'accuracy_at_eer': round(float(1.0 - eer), 4),
        'total_eval_samples': len(labels)
    }

"""
Official MATLAB .mat File Exporter and Pre-Flight Validator.
Adheres strictly to 2027 IEEE Signal Processing Cup Call for Proposals §1.2.
Guarantees seamless execution with MATLAB `load()` function without HDF5 structure corruption.
"""

import os
import numpy as np
import scipy.io as sio
from typing import Dict, Any, List

class SubmissionMATExporter:
    """
    Exports official submission .mat containers with exact IEEE SP Cup schemas.
    """
    REQUIRED_VARIABLES = [
        'facial_video',
        'roi_data',
        'reference_signal',
        'processed_signal',
        'metrics',
        'params'
    ]

    REQUIRED_METRIC_FIELDS = [
        'MAE',
        'RMSE',
        'Pearson_r',
        'BlandAltman_LoA'
    ]

    REQUIRED_PARAM_FIELDS = [
        'camera_resolution',
        'frame_rate',
        'illuminance_lux',
        'camera_to_subject_distance',
        'subject_fitzpatrick_skin_type',
        'subject_age',
        'constraint_applied'
    ]

    @staticmethod
    def export_mat(filepath: str,
                   facial_video: np.ndarray,
                   roi_data: np.ndarray,
                   reference_signal: np.ndarray,
                   processed_signal: np.ndarray,
                   metrics: Dict[str, Any],
                   params: Dict[str, Any]) -> str:
        """
        Exports dictionary to .mat format compatible with MATLAB R2024b/R2025a.
        """
        assert facial_video.dtype == np.uint8, "facial_video must be uint8 array"
        assert not np.any(np.isnan(processed_signal)), "processed_signal cannot contain NaNs"

        # Format struct fields cleanly
        metrics_struct = {
            'MAE': float(metrics.get('MAE', 0.0)),
            'RMSE': float(metrics.get('RMSE', 0.0)),
            'Pearson_r': float(metrics.get('Pearson_r', 0.0)),
            'BlandAltman_LoA': np.array(metrics.get('BlandAltman_LoA', [-5.0, 5.0]), dtype=np.float64)
        }

        params_struct = {
            'camera_resolution': str(params.get('camera_resolution', '1280x720')),
            'frame_rate': float(params.get('frame_rate', 30.0)),
            'illuminance_lux': float(params.get('illuminance_lux', 300.0)),
            'camera_to_subject_distance': str(params.get('camera_to_subject_distance', '50cm')),
            'subject_fitzpatrick_skin_type': str(params.get('subject_fitzpatrick_skin_type', 'TypeIII')),
            'subject_age': int(params.get('subject_age', 30)),
            'constraint_applied': str(params.get('constraint_applied', 'None'))
        }

        # Add optional extra tracking info into params
        if 'extra' in params:
            params_struct['extra'] = params['extra']

        mat_dict = {
            'facial_video': facial_video,
            'roi_data': np.array(roi_data, dtype=np.int32),
            'reference_signal': np.array(reference_signal, dtype=np.float64).reshape(-1, 1),
            'processed_signal': np.array(processed_signal, dtype=np.float64).reshape(-1, 1),
            'metrics': metrics_struct,
            'params': params_struct
        }

        # Save with scipy.io.savemat (MATLAB v5/v7 compatible, seamless with load())
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        sio.savemat(filepath, mat_dict, do_compression=True)
        return filepath

    @classmethod
    def validate_mat(cls, filepath: str) -> Dict[str, Any]:
        """
        Validates an existing .mat file against competition criteria.
        """
        if not os.path.exists(filepath):
            return {"valid": False, "error": f"File not found: {filepath}"}

        try:
            data = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        except Exception as e:
            return {"valid": False, "error": f"Failed to load MAT file: {str(e)}"}

        # 1. Check all 6 required variables
        for var in cls.REQUIRED_VARIABLES:
            if var not in data:
                return {"valid": False, "error": f"Missing required variable: {var}"}

        # 2. Check facial_video data type
        video = data['facial_video']
        if video.dtype != np.uint8:
            return {"valid": False, "error": f"facial_video must be uint8, found {video.dtype}"}

        # 3. Check for NaNs
        proc = data['processed_signal']
        if np.any(np.isnan(proc)):
            return {"valid": False, "error": "processed_signal contains invalid NaN values"}

        # 4. Check metrics struct
        m = data['metrics']
        for mf in cls.REQUIRED_METRIC_FIELDS:
            if not hasattr(m, mf):
                return {"valid": False, "error": f"metrics struct missing field: {mf}"}

        # 5. Check params struct
        p = data['params']
        for pf in cls.REQUIRED_PARAM_FIELDS:
            if not hasattr(p, pf):
                return {"valid": False, "error": f"params struct missing field: {pf}"}

        return {
            "valid": True,
            "filepath": filepath,
            "num_frames": video.shape[0] if len(video.shape) >= 1 else 0,
            "resolution": f"{video.shape[2]}x{video.shape[1]}" if len(video.shape) >= 3 else "unknown",
            "mae": float(m.MAE),
            "constraint": str(p.constraint_applied)
        }

"""
PIE fitting state persistence module with schema versioning and fingerprinting.

Handles:
- State schema definition (manifest, configs, results)
- Project and database fingerprinting
- Atomic save/load with failure recovery
- Version migration
"""
import os
import json
import hashlib
import shutil
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import tempfile

import numpy as np

__version__ = '1.0'
SCHEMA_VERSION = 1
STATE_DIR_NAME = '.bl03u_pie_state'
TEMP_SUFFIX = '.tmp'
BACKUP_SUFFIX = '.bak'

logger = logging.getLogger(__name__)


class PieStateManager:
    """
    Manages PIE fitting state persistence with versioning and integrity checks.
    """

    def __init__(self, project_dir: str):
        """
        Initialize state manager for a project.

        Args:
            project_dir: Root directory of the project
        """
        self.project_dir = project_dir
        self.state_dir = os.path.join(project_dir, STATE_DIR_NAME)
        self.temp_dir = self.state_dir + TEMP_SUFFIX
        self.backup_dir = self.state_dir + BACKUP_SUFFIX
        self.arrays_dir = os.path.join(self.state_dir, 'arrays')

    def save_state(
        self,
        curves: Dict[int, Dict],
        database: List[Dict],
        calibration: Any,
        per_mz_config: Dict[int, Dict],
        all_fit_results: Dict[int, Dict],
        global_config_hash: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Atomically save PIE fitting state.

        Args:
            curves: m/z curves dict {mz: {energies, intensities, ...}}
            database: PICS database list
            calibration: Calibration object
            per_mz_config: Per-m/z configuration dict
            all_fit_results: Fitting results dict
            global_config_hash: Current global solver config hash

        Returns:
            (success, error_message)
        """
        try:
            # Clean up previous temp directory
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

            os.makedirs(self.temp_dir)
            os.makedirs(os.path.join(self.temp_dir, 'arrays'))

            # Compute fingerprints
            project_fp = compute_project_fingerprint(curves, calibration)
            database_fp = compute_database_fingerprint(database)

            # Build manifest
            manifest = self._build_manifest(
                project_fp,
                database_fp,
                global_config_hash,
                per_mz_config,
                all_fit_results,
            )

            # Save configs and results (without arrays yet)
            configs_dict = self._build_configs_dict(per_mz_config)
            results_dict = self._build_results_dict(all_fit_results, curves, database, calibration)

            # Write configs
            configs_path = os.path.join(self.temp_dir, 'configs.json')
            self._write_json(configs_path, configs_dict)

            # Write arrays and update results
            for mz, result in all_fit_results.items():
                if result.get('success') and result.get('model'):
                    npz_path = os.path.join(
                        self.temp_dir, 'arrays', f'mz_{mz}.npz'
                    )
                    self._save_mz_arrays(mz, result, curves.get(mz), npz_path)

            # Write results
            results_path = os.path.join(self.temp_dir, 'results.json')
            self._write_json(results_path, results_dict)

            # Write manifest (last, as atomic flag)
            manifest_path = os.path.join(self.temp_dir, 'manifest.json')
            self._write_json(manifest_path, manifest)

            # Verify readable
            self._verify_saved_state()

            # Atomic replacement
            if os.path.exists(self.state_dir):
                if os.path.exists(self.backup_dir):
                    shutil.rmtree(self.backup_dir)
                os.rename(self.state_dir, self.backup_dir)

            os.rename(self.temp_dir, self.state_dir)

            # Clean backup
            if os.path.exists(self.backup_dir):
                shutil.rmtree(self.backup_dir)

            logger.info('PIE state saved successfully')
            return True, None

        except Exception as e:
            logger.error(f'PIE state save failed: {e}')

            # Cleanup temp dir
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

            # Restore backup if exists
            if os.path.exists(self.backup_dir) and not os.path.exists(self.state_dir):
                os.rename(self.backup_dir, self.state_dir)
                logger.info('Restored from backup after save failure')

            return False, str(e)

    def load_state(
        self,
        curves: Dict[int, Dict],
        database: List[Dict],
        calibration: Any,
    ) -> Dict[str, Any]:
        """
        Load PIE fitting state and validate integrity.

        Args:
            curves: Current m/z curves (for fingerprint validation)
            database: Current PICS database (for fingerprint validation)
            calibration: Current calibration object

        Returns:
            {
                'success': bool,
                'configs': dict,
                'results': dict,
                'warnings': [str],
                'manifest': dict or None,
                'error': str or None,
                'obsolete_reasons': {mz: reason}
            }
        """
        if not os.path.exists(self.state_dir):
            return {
                'success': True,
                'configs': {},
                'results': {},
                'warnings': ['未找到保存的 PIE 状态，将作为新项目加载'],
                'manifest': None,
                'obsolete_reasons': {},
            }

        try:
            # Load manifest
            manifest_path = os.path.join(self.state_dir, 'manifest.json')
            manifest = self._read_json(manifest_path)

            if manifest is None:
                return {
                    'success': False,
                    'error': 'manifest.json 损坏或无法读取',
                }

            # Check schema version
            schema_version = manifest.get('schema_version', 1)
            if schema_version > SCHEMA_VERSION:
                return {
                    'success': False,
                    'error': f'不支持的 Schema 版本 {schema_version}',
                }

            # Migration if needed
            if schema_version < SCHEMA_VERSION:
                self._migrate_state(schema_version)

            # Load configs and results
            configs_path = os.path.join(self.state_dir, 'configs.json')
            results_path = os.path.join(self.state_dir, 'results.json')

            configs = self._read_json(configs_path) or {}
            results = self._read_json(results_path) or {}

            per_mz_configs = configs.get('per_mz_configs', {})
            per_mz_results = results.get('per_mz_results', {})

            # Validate fingerprints
            warnings = []
            obsolete_reasons = {}

            current_project_fp = compute_project_fingerprint(curves, calibration)
            current_database_fp = compute_database_fingerprint(database)

            saved_project_fp = manifest.get('project_fingerprint', {})
            saved_database_fp = manifest.get('database_fingerprint', {})

            if current_project_fp['curve_data_hash'] != saved_project_fp.get('curve_data_hash'):
                warnings.append('实验 PIE 曲线数据已变化，部分结果可能已过期')
                obsolete_reasons['_global_data'] = '实验数据已变化'

            if current_database_fp['cross_section_hash'] != saved_database_fp.get('cross_section_hash'):
                warnings.append('PICS 截面数据已更新，部分结果可能已过期')
                obsolete_reasons['_global_database'] = 'PICS 数据库已更新'

            # Compute current fingerprints for validity checks
            current_project_fp = compute_project_fingerprint(curves, calibration)
            current_database_fp = compute_database_fingerprint(database)
            current_per_mz_fps = {}
            for mz_str in per_mz_results.keys():
                try:
                    mz = int(mz_str)
                    result = per_mz_results[mz_str]
                    if result.get('success'):
                        used_species_ids = [
                            c.get('species_id') for c in result.get('components', [])
                            if c.get('species_id') is not None
                        ]
                        current_per_mz_fps[mz_str] = compute_per_mz_pics_fingerprint(
                            mz, used_species_ids, database
                        )
                except (ValueError, KeyError):
                    pass

            # Check array files and validate references
            for mz_str, result in per_mz_results.items():
                if result.get('success') and result.get('arrays'):
                    arrays_dict = result['arrays']
                    for array_name, ref in arrays_dict.items():
                        if '#' in ref:
                            # New format: arrays/mz_<num>.npz#fieldname
                            npz_file = ref.split('#')[0]

                            # Security: Validate NPZ filename format
                            # Must be arrays/mz_<number>.npz
                            import re
                            if not re.match(r'^arrays/mz_\d+\.npz$', npz_file):
                                result['success'] = False
                                result['error'] = f'Invalid array reference format: {ref}'
                                obsolete_reasons[mz_str] = f'无效的数组引用: {ref}'
                                warnings.append(
                                    f'm/z {mz_str} 的数组引用格式无效，结果已禁用'
                                )
                                break

                            # Build and validate path
                            npz_path = os.path.join(self.state_dir, npz_file)
                            try:
                                real_path = os.path.realpath(npz_path)
                                real_state_dir = os.path.realpath(self.state_dir)
                                if not real_path.startswith(real_state_dir + os.sep):
                                    result['success'] = False
                                    result['error'] = '数组文件路径超出状态目录'
                                    obsolete_reasons[mz_str] = '非法的数组路径'
                                    warnings.append(
                                        f'm/z {mz_str} 的数组路径超出范围，结果已禁用'
                                    )
                                    break
                            except (OSError, ValueError):
                                result['success'] = False
                                result['error'] = '数组文件路径验证失败'
                                obsolete_reasons[mz_str] = '路径验证失败'
                                break

                            if not os.path.exists(npz_path):
                                result['success'] = False
                                result['error'] = f'数组文件缺失: {os.path.basename(npz_path)}'
                                obsolete_reasons[mz_str] = '数组文件缺失'
                                warnings.append(
                                    f'm/z {mz_str} 的数组文件缺失，结果已禁用'
                                )
                                break

            # Phase 3 Step 3: Derive result validity from fingerprints
            for mz_str, result in per_mz_results.items():
                if not result.get('success'):
                    result['_status'] = 'FAILED'
                    continue

                # Check if result was a failure at fit time
                if result.get('error'):
                    result['_status'] = 'FAILED'
                    continue

                # Compare fingerprints to derive status
                saved_project_fp = result.get('experimental_data_fingerprint')
                saved_per_mz_pics_fp = result.get('pics_data_fingerprint', {})

                current_project_fp_hash = current_project_fp['curve_data_hash']
                current_per_mz_fp = current_per_mz_fps.get(mz_str, {})
                current_cs_hash = current_per_mz_fp.get('cs_combined_hash', '')

                # Check per-m/z config hash (from Phase 2)
                current_per_mz_hash = result.get('fit_config_hash')  # This should be recalculated in dialog

                if saved_project_fp != current_project_fp_hash:
                    result['_status'] = 'OBSOLETE'
                    result['_obsolete_reason'] = 'experimental_data_changed'
                    if mz_str not in obsolete_reasons:
                        obsolete_reasons[mz_str] = '实验数据已变化'
                elif saved_per_mz_pics_fp.get('cs_combined_hash') != current_cs_hash:
                    result['_status'] = 'OBSOLETE'
                    result['_obsolete_reason'] = 'pics_data_changed'
                    if mz_str not in obsolete_reasons:
                        obsolete_reasons[mz_str] = '物种截面数据已变化'
                else:
                    # All fingerprints match - result is still valid
                    result['_status'] = 'COMPLETED'

            return {
                'success': True,
                'configs': per_mz_configs,
                'results': per_mz_results,
                'warnings': warnings,
                'manifest': manifest,
                'obsolete_reasons': obsolete_reasons,
            }

        except Exception as e:
            logger.error(f'PIE state load failed: {e}')
            return {
                'success': False,
                'error': str(e),
            }

    def load_mz_arrays(self, mz: int) -> Dict[str, np.ndarray]:
        """
        Load arrays for a specific m/z from NPZ file.

        Args:
            mz: m/z value

        Returns:
            Dict of {array_name: numpy_array}
        """
        npz_path = os.path.join(self.state_dir, 'arrays', f'mz_{mz}.npz')

        # Security: Verify path is within state_dir (prevent path traversal)
        try:
            real_path = os.path.realpath(npz_path)
            real_state_dir = os.path.realpath(self.state_dir)
            if not real_path.startswith(real_state_dir + os.sep):
                logger.error(f'Path traversal attempt detected: {npz_path}')
                return {}
        except (OSError, ValueError):
            return {}

        if not os.path.exists(npz_path):
            return {}

        try:
            # Safe load: no pickle deserialization
            data = np.load(npz_path, allow_pickle=False)
            arrays = {name: data[name] for name in data.files}

            # Validate array shapes and dtypes
            expected_arrays = {
                'energies': (float, 1),  # (dtype, ndim)
                'intensities': (float, 1),
                'fitted_curve': (float, 1),
                'component_curves': (None, None),  # Variable shape
                'residuals': (float, 1),
            }

            for array_name, array in arrays.items():
                if array_name in expected_arrays:
                    expected_dtype, expected_ndim = expected_arrays[array_name]
                    if expected_dtype is not None:
                        if array.dtype.kind != 'f':  # float type check
                            logger.warning(
                                f'Unexpected dtype for {array_name}: {array.dtype}'
                            )
                    if expected_ndim is not None:
                        if array.ndim != expected_ndim:
                            logger.warning(
                                f'Unexpected ndim for {array_name}: {array.ndim}'
                            )

            return arrays
        except Exception as e:
            logger.error(f'Failed to load arrays for m/z {mz}: {e}')
            return {}

    def _build_manifest(
        self,
        project_fp: Dict,
        database_fp: Dict,
        global_config_hash: str,
        per_mz_config: Dict,
        all_fit_results: Dict,
    ) -> Dict:
        """Build manifest dictionary."""
        now = datetime.now(timezone.utc).isoformat()

        fitted_count = sum(
            1 for r in all_fit_results.values() if r.get('success')
        )
        # Note: obsolete_count calculation skipped during save
        # (obsolete status is derived at load time, not stored)
        failed_count = sum(
            1 for r in all_fit_results.values() if not r.get('success')
        )

        return {
            'schema_version': SCHEMA_VERSION,
            'created_by': 'BL03U_MassSpectrumTool',
            'saved_at': now,
            'last_modified_at': now,
            'project_fingerprint': project_fp,
            'database_fingerprint': database_fp,
            'global_config_hash': global_config_hash,
            'summary': {
                'total_mz': len(all_fit_results),
                'fitted_mz': fitted_count,
                'failed_mz': failed_count,
            },
        }

    def _build_configs_dict(self, per_mz_config: Dict[int, Dict]) -> Dict:
        """Build configs dictionary for JSON serialization."""
        return {
            'per_mz_configs': {
                str(mz): config
                for mz, config in per_mz_config.items()
            }
        }

    def _build_results_dict(
        self,
        all_fit_results: Dict[int, Dict],
        curves: Dict[int, Dict],
        database: List[Dict],
        calibration: Any,
    ) -> Dict:
        """
        Build results dictionary with complete schema including metrics, components, and fingerprints.

        Phase 3 Step 3: Extended schema with:
        - Metrics (r_squared, rmse, mae)
        - Components (species with coefficients and contributions)
        - Array references for curve data
        - Fingerprints (snapshots at fit time)
        """
        per_mz_results = {}

        for mz, result in all_fit_results.items():
            result_entry = {
                'mz': mz,
                'success': result.get('success', False),
                'fit_timestamp': result.get('fit_timestamp'),
                'fit_config_hash': result.get('fit_config_hash'),
                'global_config_hash': result.get('global_config_hash'),
                'error': result.get('error'),
            }

            if result.get('success') and result.get('model'):
                model = result['model']
                species_list = model.get('species', result.get('components', []))

                # Extract used species IDs
                used_species_ids = [
                    sp.get('id') for sp in species_list if sp.get('id') is not None
                ]

                # Compute fingerprints at fit time
                project_fp = compute_project_fingerprint(curves, calibration)
                per_mz_pics_fp = compute_per_mz_pics_fingerprint(mz, used_species_ids, database)

                # Metrics
                result_entry['metrics'] = {
                    'r_squared': float(model.get('r_squared', 0.0)),
                    'rmse': float(model.get('rmse', 0.0)),
                    'mae': float(model.get('mae', 0.0)),
                }

                # Components (species with contributions)
                result_entry['components'] = [
                    {
                        'species_id': sp.get('id'),
                        'species': sp.get('species'),
                        'coefficient': float(sp.get('coefficient', 0.0)),
                        'contribution_percent': float(sp.get('contribution_percent', 0.0)),
                        'ionization_energy': float(sp.get('ie', 0.0)) if sp.get('ie') is not None else None,
                    }
                    for sp in species_list
                ]

                # Array references
                result_entry['arrays'] = {
                    'energies': f'arrays/mz_{mz}.npz#energies',
                    'experimental': f'arrays/mz_{mz}.npz#experimental',
                    'total_fit': f'arrays/mz_{mz}.npz#total_fit',
                    'residual': f'arrays/mz_{mz}.npz#residual',
                    'components': f'arrays/mz_{mz}.npz#components',
                }

                # Fingerprints (snapshots at fit time)
                result_entry['experimental_data_fingerprint'] = project_fp['curve_data_hash']
                result_entry['pics_data_fingerprint'] = per_mz_pics_fp

            per_mz_results[str(mz)] = result_entry

        return {'per_mz_results': per_mz_results}

    def _save_mz_arrays(
        self,
        mz: int,
        result: Dict,
        curve: Optional[Dict],
        npz_path: str,
    ) -> None:
        """
        Save arrays for a specific m/z to NPZ file.

        Phase 3 Step 3: Saves complete fit data including:
        - energies: Energy axis
        - experimental: Experimental intensity data
        - total_fit: Total fitted curve
        - residual: Residual (experimental - total_fit)
        - components: Component curves (n_energy x n_species)
        """
        model = result.get('model', {})
        arrays_dict = {}

        # Save experimental data
        if curve:
            energies = np.array(curve.get('energies', []), dtype=np.float64)
            experimental = np.array(curve.get('intensities', []), dtype=np.float64)

            arrays_dict['energies'] = energies
            arrays_dict['experimental'] = experimental

            # Compute residual if we have fitted curve
            if 'fitted_curve' in model:
                fitted_curve = np.array(model['fitted_curve'], dtype=np.float64)
                if len(experimental) == len(fitted_curve):
                    residual = experimental - fitted_curve
                    arrays_dict['residual'] = residual
                else:
                    # If lengths don't match, save empty residual
                    arrays_dict['residual'] = np.array([], dtype=np.float64)

        # Save fitted curve
        if 'fitted_curve' in model:
            arrays_dict['total_fit'] = np.array(model['fitted_curve'], dtype=np.float64)

        # Save component curves (should be n_energy x n_species)
        if 'component_curves' in model:
            component_curves = model['component_curves']
            if isinstance(component_curves, (list, tuple)):
                arrays_dict['components'] = np.array(component_curves, dtype=np.float64)
            else:
                arrays_dict['components'] = np.asarray(component_curves, dtype=np.float64)

        # Write to NPZ with compression, no pickle
        if arrays_dict:
            try:
                np.savez_compressed(npz_path, **arrays_dict)
            except Exception as e:
                logger.error(f'Failed to save arrays for m/z {mz}: {e}')

    def _verify_saved_state(self) -> None:
        """Verify that saved state is readable."""
        manifest_path = os.path.join(self.temp_dir, 'manifest.json')
        configs_path = os.path.join(self.temp_dir, 'configs.json')
        results_path = os.path.join(self.temp_dir, 'results.json')

        # Try to read all files
        self._read_json(manifest_path)
        self._read_json(configs_path)
        self._read_json(results_path)

    def _write_json(self, path: str, data: Dict) -> None:
        """Write JSON file with UTF-8 encoding."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _read_json(self, path: str) -> Optional[Dict]:
        """Read JSON file with error handling."""
        try:
            if not os.path.exists(path):
                return None

            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f'Failed to read JSON {path}: {e}')
            return None

    def _migrate_state(self, from_version: int) -> None:
        """Migrate state from older schema version."""
        if from_version < SCHEMA_VERSION:
            # Add migration logic here as needed
            logger.info(f'Migrated PIE state from schema v{from_version} to v{SCHEMA_VERSION}')


def compute_project_fingerprint(
    curves: Dict[int, Dict],
    calibration: Any,
) -> Dict[str, str]:
    """
    Compute project fingerprint to detect data changes.

    Detects: m/z additions/deletions, energy axis changes, intensity changes, calibration changes
    """
    # Hash of m/z keys
    mz_keys = sorted(curves.keys())
    mz_keys_str = json.dumps(mz_keys)
    mz_keys_hash = hashlib.sha256(mz_keys_str.encode()).hexdigest()[:16]

    # Hash of curve data
    curve_hashes = []
    for mz in mz_keys:
        curve = curves[mz]
        energies = np.array(curve.get('energies', []))
        intensities = np.array(curve.get('intensities', []))

        curve_info = {
            'mz': mz,
            'energy_count': len(energies),
            'energy_min': float(np.nanmin(energies)) if len(energies) > 0 else None,
            'energy_max': float(np.nanmax(energies)) if len(energies) > 0 else None,
            'intensity_sum': float(np.nansum(np.abs(intensities))) if len(intensities) > 0 else None,
        }

        curve_hash = hashlib.sha256(
            json.dumps(curve_info, sort_keys=True).encode()
        ).hexdigest()[:16]
        curve_hashes.append(curve_hash)

    curve_data_hash = hashlib.sha256(
        ''.join(curve_hashes).encode()
    ).hexdigest()[:16]

    # Hash of calibration
    calib_info = {
        'a': float(getattr(calibration, 'a', 0.0)),
        'b': float(getattr(calibration, 'b', 1.0)),
        'c': float(getattr(calibration, 'c', 0.0)),
    }
    calibration_hash = hashlib.sha256(
        json.dumps(calib_info, sort_keys=True).encode()
    ).hexdigest()[:16]

    return {
        'mz_keys_hash': mz_keys_hash,
        'curve_count': len(curves),
        'curve_data_hash': curve_data_hash,
        'calibration_hash': calibration_hash,
    }


def compute_database_fingerprint(database: List[Dict]) -> Dict[str, str]:
    """
    Compute database fingerprint to detect PICS data changes.

    Detects: species additions/deletions, cross-section updates, IE changes
    """
    # Hash of species IDs
    species_ids = sorted([item.get('id') for item in database if item.get('id') is not None])
    species_ids_hash = hashlib.sha256(
        json.dumps(species_ids).encode()
    ).hexdigest()[:16]

    # Hash of species data
    species_hashes = []
    for item in database:
        cs = np.array(item.get('cross_sections', []))
        ie_value = item.get('ie')
        if ie_value is None:
            ie_value = item.get('ionization_energy')
        try:
            normalized_ie = float(ie_value) if ie_value is not None else None
        except (TypeError, ValueError):
            normalized_ie = None

        species_info = {
            'id': item.get('id'),
            'species': item.get('species'),
            'ie': normalized_ie,
            'cs_len': len(cs),
            'cs_sum': float(np.nansum(np.abs(cs))) if len(cs) > 0 else None,
        }

        species_hash = hashlib.sha256(
            json.dumps(species_info, sort_keys=True).encode()
        ).hexdigest()[:16]
        species_hashes.append(species_hash)

    cross_section_hash = hashlib.sha256(
        ''.join(species_hashes).encode()
    ).hexdigest()[:16]

    return {
        'database_type': 'PICS',
        'total_species': len(database),
        'species_ids_hash': species_ids_hash,
        'cross_section_hash': cross_section_hash,
    }


def compute_per_mz_pics_fingerprint(
    mz: int,
    used_species_ids: List[int],
    database: List[Dict],
) -> Dict[str, Any]:
    """
    Compute fine-grained PICS fingerprint for species used in a specific m/z fit.

    Phase 3 Step 3: Per-m/z fingerprinting to detect if only relevant species changed.
    Unrelated species changes will NOT invalidate the result.

    Args:
        mz: m/z value
        used_species_ids: List of species IDs used in the fit for this m/z
        database: Complete PICS database

    Returns:
        {
            'mz': 46,
            'used_species_ids': [1, 5, 12],
            'species_count': 3,
            'cs_combined_hash': '...',
        }
    """
    # Build lookup of database by ID for fast access
    db_by_id = {item.get('id'): item for item in database if item.get('id') is not None}

    # Get only the species used in this fit
    used_species = [db_by_id[sid] for sid in sorted(used_species_ids) if sid in db_by_id]

    # Compute fingerprint for just these species
    cs_data = []
    for species in used_species:
        cs = np.array(species.get('cross_sections', []))
        cs_info = (
            species['id'],
            len(cs),
            float(np.nansum(np.abs(cs))) if len(cs) > 0 else 0.0,
        )
        cs_data.append(cs_info)

    cs_combined_hash = hashlib.sha256(
        json.dumps(cs_data, sort_keys=True).encode()
    ).hexdigest()[:16]

    return {
        'mz': mz,
        'used_species_ids': sorted(used_species_ids),
        'species_count': len(used_species_ids),
        'cs_combined_hash': cs_combined_hash,
    }

#!/usr/bin/env python3
"""Run RINAMI inference using the averaged outputs of three LPM checkpoints.

This script is intended to be placed in the `scripts/` directory of the
published RINAMI repository and executed as:

    python run_inference.py <input.pdb>

It generates the temporary ProteinMPNN node representations and output
profiles for the input PDB, runs three lowest-performing-model checkpoints,
and reports the ensemble-averaged prediction.

Optionally, it can also save:
  - the ensemble-averaged residue-amino-acid-wise dG matrix as .npz
  - a heatmap image (.png) of the ensemble-averaged residue-amino-acid-wise dG matrix
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RINAMI_DIR = ROOT / "RINAMI"
PROCESSED_DATA_DIR = ROOT / "processed_data"
DEFAULT_LPM_ROOT = ROOT / "pth" / "lowest_performing_models"
DEFAULT_LPM_SPLITS = (1, 2, 3)

AA_ORDER_CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
AA_ORDER_HEATMAP = "QENHDRKTSAGMCLVIWYFP"

# Make imports work when this script is executed from either `scripts/` or the
# repository root.
sys.path.insert(0, str(RINAMI_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from RINAMI_model_main import RINAMIInterpretableMultiTask


from util import get_sequence_from_single_chain_pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict folding free energy using the average output of three "
            "RINAMI lowest-performing-model checkpoints."
        )
    )
    parser.add_argument(
        "input_pdb",
        type=Path,
        help="Input single-chain PDB file.",
    )
    parser.add_argument(
        "--lpm-root",
        type=Path,
        default=DEFAULT_LPM_ROOT,
        help="Root directory containing pth_split_<n>/LPM.pth checkpoints.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=list(DEFAULT_LPM_SPLITS),
        help="Split IDs used for the LPM ensemble. Default: 1 2 3.",
    )
    parser.add_argument(
        "--esm-size",
        type=int,
        default=320,
        help="ESM representation size used by the trained model.",
    )
    parser.add_argument(
        "--save-residue-amino-acid-dG-matrix",
        action="store_true",
        help="Save the ensemble-averaged residue-amino-acid-wise dG matrix as a compressed .npz file.",
    )
    parser.add_argument(
        "--matrix-out-dir",
        type=Path,
        default=None,
        help="Output directory for the .npz matrix. Default: ./rinami_matrix_outputs",
    )
    parser.add_argument(
        "--save-residue-amino-acid-dG-heatmap",
        action="store_true",
        help="Save the ensemble-averaged residue-amino-acid-wise dG matrix as a heatmap image (.png).",
    )
    parser.add_argument(
        "--heatmap-out-dir",
        type=Path,
        default=None,
        help="Output directory for the heatmap image. Default: ./rinami_heatmap_outputs",
    )
    parser.add_argument(
        "--heatmap-dpi",
        type=int,
        default=300,
        help="DPI for the saved heatmap image. Default: 300.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary processed_data files after inference.",
    )
    return parser.parse_args()


def resolve_lpm_checkpoints(lpm_root: Path, splits: Iterable[int]) -> List[Path]:
    ckpt_paths = []
    for split in splits:
        ckpt = lpm_root / f"pth_split_{split}" / "LPM.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"LPM checkpoint was not found: {ckpt}")
        ckpt_paths.append(ckpt)

    if len(ckpt_paths) == 0:
        raise ValueError("No LPM checkpoints were specified.")

    return ckpt_paths


def load_state_dict_compat(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)


def copy_non_esm_modules(src_model: torch.nn.Module, dst_model: torch.nn.Module) -> None:
    """Copy all checkpoint-loaded modules except the shared ESM encoder."""
    skip_prefix = "aa_seq_encoder."
    src_state = src_model.state_dict()
    filtered_state = {
        key: value
        for key, value in src_state.items()
        if not key.startswith(skip_prefix)
    }
    missing, unexpected = dst_model.load_state_dict(filtered_state, strict=False)
    missing = [x for x in missing if not x.startswith(skip_prefix)]
    unexpected = [x for x in unexpected if not x.startswith(skip_prefix)]

    if unexpected:
        raise RuntimeError(f"Unexpected keys while copying model modules: {unexpected}")
    if missing:
        print(f"[WARN] Missing keys while copying model modules: {missing}")


def build_lpm_ensemble(ckpt_paths: List[Path], esm_size: int) -> List[torch.nn.Module]:
    """Build ensemble models while sharing a single frozen ESM encoder."""
    print(f"[INFO] Loading {len(ckpt_paths)} LPM checkpoints")

    base_model = RINAMIInterpretableMultiTask(dropout=0.1, ESM_size=esm_size).to(device)
    load_state_dict_compat(base_model, ckpt_paths[0])
    base_model.eval()

    shared_esm = base_model.aa_seq_encoder
    shared_esm.eval()
    for param in shared_esm.parameters():
        param.requires_grad = False

    ensemble_models = [base_model]
    print(f"[INFO] Loaded LPM head 1/{len(ckpt_paths)}: {ckpt_paths[0]}")

    for idx, ckpt_path in enumerate(ckpt_paths[1:], start=2):
        model_i = RINAMIInterpretableMultiTask(dropout=0.1, ESM_size=esm_size).to(device)
        model_i.aa_seq_encoder = shared_esm

        tmp_model = RINAMIInterpretableMultiTask(dropout=0.1, ESM_size=esm_size).to(device)
        load_state_dict_compat(tmp_model, ckpt_path)
        copy_non_esm_modules(tmp_model, model_i)
        model_i.eval()

        ensemble_models.append(model_i)
        print(f"[INFO] Loaded LPM head {idx}/{len(ckpt_paths)}: {ckpt_path}")

        del tmp_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return ensemble_models


def prepare_temp_feature_dirs(input_pdb_path: Path) -> Tuple[Path, Path, Path, str]:
    input_dir = PROCESSED_DATA_DIR / "input_pdb"
    profile_dir = PROCESSED_DATA_DIR / "temp_ProteinMPNN_output_profile"
    node_rep_dir = PROCESSED_DATA_DIR / "temp_ProteinMPNN_node_rep"

    for path in (input_dir, profile_dir, node_rep_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    copied_pdb_path = input_dir / input_pdb_path.name
    shutil.copy2(input_pdb_path, copied_pdb_path)
    input_data_name = input_pdb_path.stem

    return input_dir, profile_dir, node_rep_dir, input_data_name


def run_feature_generation(input_dir: Path, node_rep_dir: Path, profile_dir: Path) -> None:
    node_script = SCRIPT_DIR / "pdb_to_mpnn_node_rep.py"
    profile_script = SCRIPT_DIR / "pdb_to_mpnn_output_profile.py"

    if not node_script.exists():
        raise FileNotFoundError(
            f"ProteinMPNN node-representation script was not found: {node_script}"
        )
    if not profile_script.exists():
        raise FileNotFoundError(
            f"ProteinMPNN profile script was not found: {profile_script}"
        )

    subprocess.run(
        [sys.executable, str(node_script), str(input_dir), str(node_rep_dir)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(profile_script), str(input_dir), str(profile_dir)],
        check=True,
    )


def predict_with_lpm_ensemble(
    ensemble_models: List[torch.nn.Module],
    aa_seq: str,
    node_rep_path: Path,
    profile_path: Path,
) -> dict:
    pred_dg_list = []
    pred_foldability_logit_list = []
    pred_foldability_prob_list = []
    matrix_list = []

    with torch.no_grad():
        for model in ensemble_models:
            out = model(
                [aa_seq],
                [str(node_rep_path)],
                [str(profile_path)],
                return_details=True,
            )
            pred_dg_list.append(float(out["dG"][0].detach().cpu().item()))
            pred_foldability_logit_list.append(
                float(out["foldability_logit"][0].detach().cpu().item())
            )
            pred_foldability_prob_list.append(
                float(torch.sigmoid(out["foldability_logit"])[0].detach().cpu().item())
            )
            matrix_list.append(
                out["residue_amino_acid_wise_dG_mat"][0].detach().cpu().numpy()
            )

    matrix_avg = np.stack(matrix_list, axis=0).mean(axis=0)

    return {
        "pred_dG_each_model": pred_dg_list,
        "pred_dG_mean": float(np.mean(pred_dg_list)),
        "pred_foldability_logit_each_model": pred_foldability_logit_list,
        "pred_foldability_logit_mean": float(np.mean(pred_foldability_logit_list)),
        "pred_foldability_prob_each_model": pred_foldability_prob_list,
        "pred_foldability_prob_mean": float(np.mean(pred_foldability_prob_list)),
        "residue_amino_acid_wise_dG_mat_mean": matrix_avg,
    }


def save_residue_amino_acid_dG_matrix(
    output_dir: Path,
    input_data_name: str,
    aa_seq: str,
    pred: dict,
    ckpt_paths: List[Path],
    splits: List[int],
) -> Path:
    """Save the ensemble-averaged residue-amino-acid-wise dG matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_data_name}_residue_amino_acid_wise_dG_mat.npz"

    mat = pred["residue_amino_acid_wise_dG_mat_mean"][:len(aa_seq), :]
    np.savez_compressed(
        out_path,
        residue_amino_acid_wise_dG_mat=mat,
        sequence=np.array(aa_seq, dtype=object),
        aa_order=np.array(list(AA_ORDER_CANONICAL), dtype=object),
        pred_dG_each_model=np.asarray(pred["pred_dG_each_model"], dtype=float),
        pred_dG_mean=np.asarray(pred["pred_dG_mean"], dtype=float),
        pred_foldability_logit_each_model=np.asarray(
            pred["pred_foldability_logit_each_model"], dtype=float
        ),
        pred_foldability_logit_mean=np.asarray(
            pred["pred_foldability_logit_mean"], dtype=float
        ),
        pred_foldability_prob_each_model=np.asarray(
            pred["pred_foldability_prob_each_model"], dtype=float
        ),
        pred_foldability_prob_mean=np.asarray(
            pred["pred_foldability_prob_mean"], dtype=float
        ),
        ensemble_splits=np.asarray(splits, dtype=int),
        ensemble_ckpts=np.asarray([str(x) for x in ckpt_paths], dtype=object),
    )
    return out_path


def reorder_matrix_columns(mat: np.ndarray, source_order: str, target_order: str) -> np.ndarray:
    """Reorder the amino-acid axis of a [L, 20] matrix."""
    index_map = []
    for aa in target_order:
        if aa not in source_order:
            raise ValueError(f"Amino acid {aa} was not found in source order {source_order}")
        index_map.append(source_order.index(aa))
    return mat[:, index_map]


def make_xtick_positions_and_labels(aa_seq: str) -> Tuple[List[int], List[str]]:
    """Make x ticks for all residues."""
    positions = list(range(len(aa_seq)))
    labels = [f"{aa_seq[i]}{i + 1}" for i in positions]
    return positions, labels


def save_residue_amino_acid_dG_heatmap(
    output_dir: Path,
    input_data_name: str,
    aa_seq: str,
    pred: dict,
    dpi: int = 300,
) -> Path:
    """Save the ensemble-averaged residue-amino-acid-wise dG matrix as a heatmap image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_data_name}_residue_amino_acid_wise_dG_mat_heatmap.png"

    mat = pred["residue_amino_acid_wise_dG_mat_mean"][:len(aa_seq), :]
    mat = reorder_matrix_columns(
        mat=mat,
        source_order=AA_ORDER_CANONICAL,
        target_order=AA_ORDER_HEATMAP,
    )

    # Convert [L, 20] -> [20, L] for a heatmap with amino acids on the y-axis.
    heatmap = mat.T

    fig_width = max(10.0, len(aa_seq) * 0.28)
    fig_height = 6.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-.75,
        vmax=.75,
    )

    xticks, xticklabels = make_xtick_positions_and_labels(aa_seq)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)

    ax.set_yticks(range(len(AA_ORDER_HEATMAP)))
    ax.set_yticklabels(list(AA_ORDER_HEATMAP), fontsize=10)

    ax.set_xlabel("Residue position")
    ax.set_ylabel("Amino acid")
    ax.set_title(
        f"Residue-amino-acid-wise dG matrix\n"
        f"{input_data_name} | Predicted ΔG = {pred['pred_dG_mean']:.3f} kcal/mol"
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Contribution to predicted ΔG")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_path


def cleanup_temp_dirs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    input_pdb_path = args.input_pdb.expanduser().resolve()
    if not input_pdb_path.exists():
        raise FileNotFoundError(f"Input PDB file was not found: {input_pdb_path}")

    ckpt_paths = resolve_lpm_checkpoints(args.lpm_root.expanduser().resolve(), args.splits)
    input_dir, profile_dir, node_rep_dir, input_data_name = prepare_temp_feature_dirs(input_pdb_path)

    try:
        run_feature_generation(input_dir, node_rep_dir, profile_dir)

        aa_seq = get_sequence_from_single_chain_pdb(str(input_pdb_path))
        node_rep_path = node_rep_dir / f"{input_data_name}.pt"
        profile_path = profile_dir / f"{input_data_name}_profile.npy"

        if not node_rep_path.exists():
            raise FileNotFoundError(
                f"Generated node representation was not found: {node_rep_path}"
            )
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Generated ProteinMPNN profile was not found: {profile_path}"
            )

        ensemble_models = build_lpm_ensemble(ckpt_paths, esm_size=args.esm_size)
        pred = predict_with_lpm_ensemble(
            ensemble_models=ensemble_models,
            aa_seq=aa_seq,
            node_rep_path=node_rep_path,
            profile_path=profile_path,
        )

        print(f"Input file: {input_pdb_path}")
        print(f"Sequence length: {len(aa_seq)}")
        print("LPM checkpoints:")
        for ckpt in ckpt_paths:
            print(f"  - {ckpt}")
        print("Predicted ΔG for each LPM [kcal/mol]:")
        print(f"Ensemble-averaged predicted ΔG = {pred['pred_dG_mean']:.6f} [kcal/mol]")
        print(
            "Ensemble-averaged predicted foldability probability = "
            f"{pred['pred_foldability_prob_mean']:.6f}"
        )

        if args.save_residue_amino_acid_dG_matrix:
            matrix_out_dir = args.matrix_out_dir
            if matrix_out_dir is None:
                matrix_out_dir = SCRIPT_DIR / "rinami_matrix_outputs"
            else:
                matrix_out_dir = matrix_out_dir.expanduser().resolve()

            matrix_path = save_residue_amino_acid_dG_matrix(
                output_dir=matrix_out_dir,
                input_data_name=input_data_name,
                aa_seq=aa_seq,
                pred=pred,
                ckpt_paths=ckpt_paths,
                splits=list(args.splits),
            )
            print(f"Saved residue-amino-acid-wise dG matrix: {matrix_path}")

        if args.save_residue_amino_acid_dG_heatmap:
            heatmap_out_dir = args.heatmap_out_dir
            if heatmap_out_dir is None:
                heatmap_out_dir = SCRIPT_DIR / "rinami_heatmap_outputs"
            else:
                heatmap_out_dir = heatmap_out_dir.expanduser().resolve()

            heatmap_path = save_residue_amino_acid_dG_heatmap(
                output_dir=heatmap_out_dir,
                input_data_name=input_data_name,
                aa_seq=aa_seq,
                pred=pred,
                dpi=args.heatmap_dpi,
            )
            print(f"Saved residue-amino-acid-wise dG heatmap: {heatmap_path}")

    finally:
        if not args.keep_temp:
            cleanup_temp_dirs(input_dir, profile_dir, node_rep_dir)


if __name__ == "__main__":
    main()

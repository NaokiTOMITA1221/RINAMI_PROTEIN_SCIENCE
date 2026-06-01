from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RINAMI_DIR = ROOT / "RINAMI"
PROCESSED_DATA_DIR = ROOT / "processed_data"

DEFAULT_LPM_ROOT = ROOT / "pth" / "lowest_performing_models"
DEFAULT_LPM_SPLITS = (1, 2, 3)

AA_ORDER_CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
AA_ORDER_PLOT = "QENHDRKTSAGMCLVIWYFP"
AA_REORDER_INDEX = [AA_ORDER_CANONICAL.index(aa) for aa in AA_ORDER_PLOT]

# Make imports work when this script is executed from either scripts/ or repository root.
sys.path.insert(0, str(RINAMI_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from RINAMI_model_main import RINAMI

from util import get_sequence_from_single_chain_pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict folding free energy for all PDB files in a directory using "
            "the ensemble-averaged outputs of RINAMI LPM checkpoints."
        )
    )

    parser.add_argument(
        "pdb_dir",
        type=Path,
        help="Directory containing input *.pdb files.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "batch_rinami_predictions.csv",
        help="Output CSV path. Default: ../batch_rinami_predictions.csv",
    )
    parser.add_argument(
        "--lpm-root",
        type=Path,
        default=DEFAULT_LPM_ROOT,
        help="Root directory containing pth_split_*/LPM.pth checkpoints.",
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
        help="ESM representation size used by the trained model. Default: 320.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary processed_data files after inference.",
    )
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
        help=(
            "Remove and regenerate temporary feature directories even if they exist. "
            "This is normally recommended for clean batch inference."
        ),
    )

    parser.add_argument(
        "--save-residue-amino-acid-dG-heatmap",
        action="store_true",
        help=(
            "Save residue-amino-acid-wise dG heatmap images for each successfully "
            "predicted PDB."
        ),
    )
    parser.add_argument(
        "--heatmap-out-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save heatmap images. Required when "
            "--save-residue-amino-acid-dG-heatmap is specified."
        ),
    )
    parser.add_argument(
        "--heatmap-dpi",
        type=int,
        default=300,
        help="DPI for saved heatmap PNG files. Default: 300.",
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
    """
    Copy all checkpoint-loaded modules except the shared ESM encoder.
    """
    skip_prefix = "aa_seq_encoder."

    src_state = src_model.state_dict()
    filtered_state = {
        key: value for key, value in src_state.items() if not key.startswith(skip_prefix)
    }

    missing, unexpected = dst_model.load_state_dict(filtered_state, strict=False)

    missing = [x for x in missing if not x.startswith(skip_prefix)]
    unexpected = [x for x in unexpected if not x.startswith(skip_prefix)]

    if unexpected:
        raise RuntimeError(f"Unexpected keys while copying model modules: {unexpected}")
    if missing:
        print(f"[WARN] Missing keys while copying model modules: {missing}")


def build_lpm_ensemble(ckpt_paths: List[Path], esm_size: int) -> List[torch.nn.Module]:
    """
    Build ensemble models while sharing a single frozen ESM encoder.
    """
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading {len(ckpt_paths)} LPM checkpoints")

    base_model = RINAMI(dropout=0.1, ESM_size=esm_size).to(device)
    load_state_dict_compat(base_model, ckpt_paths[0])
    base_model.eval()

    shared_esm = base_model.aa_seq_encoder
    shared_esm.eval()
    for param in shared_esm.parameters():
        param.requires_grad = False

    ensemble_models = [base_model]

    for idx, ckpt_path in enumerate(ckpt_paths[1:], start=2):
        model_i = RINAMI(dropout=0.1, ESM_size=esm_size).to(device)
        model_i.aa_seq_encoder = shared_esm

        tmp_model = RINAMI(dropout=0.1, ESM_size=esm_size).to(device)
        load_state_dict_compat(tmp_model, ckpt_path)
        copy_non_esm_modules(tmp_model, model_i)

        model_i.eval()
        ensemble_models.append(model_i)

        print(f"[INFO] Loaded LPM head {idx}/{len(ckpt_paths)}: {ckpt_path}")

        del tmp_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[INFO] Loaded LPM head 1/{len(ckpt_paths)}: {ckpt_paths[0]}")
    return ensemble_models


def get_pdb_paths(pdb_dir: Path) -> List[Path]:
    pdb_paths = sorted(pdb_dir.glob("*.pdb"))
    if len(pdb_paths) == 0:
        raise FileNotFoundError(f"No *.pdb files were found in: {pdb_dir}")
    return pdb_paths


def prepare_temp_feature_dirs(
    pdb_paths: List[Path],
    overwrite_features: bool = True,
) -> tuple[Path, Path, Path]:
    """
    Copy all input PDBs into a temporary input directory and prepare feature directories.
    """
    input_dir = PROCESSED_DATA_DIR / "batch_input_pdb"
    profile_dir = PROCESSED_DATA_DIR / "batch_ProteinMPNN_output_profile"
    node_rep_dir = PROCESSED_DATA_DIR / "batch_ProteinMPNN_node_rep"

    if overwrite_features:
        for path in (input_dir, profile_dir, node_rep_dir):
            if path.exists():
                shutil.rmtree(path)

    for path in (input_dir, profile_dir, node_rep_dir):
        path.mkdir(parents=True, exist_ok=True)

    seen_names = set()
    for pdb_path in pdb_paths:
        if pdb_path.name in seen_names:
            raise ValueError(f"Duplicated PDB filename detected: {pdb_path.name}")
        seen_names.add(pdb_path.name)

        dst = input_dir / pdb_path.name
        shutil.copy2(pdb_path, dst)

    return input_dir, profile_dir, node_rep_dir


def run_feature_generation(input_dir: Path, node_rep_dir: Path, profile_dir: Path) -> None:
    node_script = SCRIPT_DIR / "pdb_to_mpnn_node_rep.py"
    profile_script = SCRIPT_DIR / "pdb_to_mpnn_output_profile.py"

    if not node_script.exists():
        raise FileNotFoundError(f"ProteinMPNN node-representation script was not found: {node_script}")
    if not profile_script.exists():
        raise FileNotFoundError(f"ProteinMPNN profile script was not found: {profile_script}")

    print("[INFO] Generating ProteinMPNN node representations...")
    subprocess.run(
        [sys.executable, str(node_script), str(input_dir), str(node_rep_dir)],
        check=True,
    )

    print("[INFO] Generating ProteinMPNN output profiles...")
    subprocess.run(
        [
            sys.executable,
            str(profile_script),
            str(input_dir),
            str(profile_dir),
            "--overwrite",
        ],
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

            foldability_logit = out["foldability_logit"][0].detach().cpu()
            pred_foldability_logit_list.append(float(foldability_logit.item()))
            pred_foldability_prob_list.append(float(torch.sigmoid(foldability_logit).item()))

            matrix = (
                out["residue_amino_acid_wise_dG_mat"][0]
                .detach()
                .cpu()
                .numpy()
            )
            matrix_list.append(matrix)

    matrix_avg = np.stack(matrix_list, axis=0).mean(axis=0)

    return {
        "pred_dG_each_model": pred_dg_list,
        "pred_dG_mean": float(np.mean(pred_dg_list)),
        "pred_dG_std": float(np.std(pred_dg_list, ddof=0)),
        "pred_foldability_logit_each_model": pred_foldability_logit_list,
        "pred_foldability_logit_mean": float(np.mean(pred_foldability_logit_list)),
        "pred_foldability_prob_each_model": pred_foldability_prob_list,
        "pred_foldability_prob_mean": float(np.mean(pred_foldability_prob_list)),
        "residue_amino_acid_wise_dG_mat_mean": matrix_avg,
    }


def save_residue_amino_acid_dg_heatmap(
    matrix: np.ndarray,
    aa_seq: str,
    out_path: Path,
    title: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Save residue-amino-acid-wise dG matrix as a heatmap.

    Expected matrix shape: (20, L)

    This function normalizes it to:
        y-axis: amino acids
        x-axis: residue positions
    """
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError(f"Heatmap matrix must be 2D, but got shape: {matrix.shape}")

    seq_len = len(aa_seq)

    if matrix.shape == (seq_len, 20):
        plot_mat = matrix.T
    elif matrix.shape == (20, seq_len):
        plot_mat = matrix
    else:
        raise ValueError(
            "Unexpected residue-amino-acid-wise dG matrix shape. "
            f"matrix.shape={matrix.shape}, sequence_length={seq_len}. "
            "Expected (20, L)."
        )

    plot_mat = plot_mat[AA_REORDER_INDEX, :]
    y_labels = list(AA_ORDER_PLOT)
    x_labels = [f"{i + 1}{aa}" for i, aa in enumerate(aa_seq)]

    width = max(8.0, min(0.22 * seq_len, 40.0))
    height = 6.5

    fig, ax = plt.subplots(figsize=(width, height))

    im = ax.imshow(plot_mat, vmin=-1.0, vmax=1.0, aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Residue-amino-acid-wise predicted ΔG contribution")

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Amino acid")

    if seq_len <= 80:
        ax.set_xticks(np.arange(seq_len))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    else:
        step = max(1, seq_len // 40)
        tick_positions = np.arange(0, seq_len, step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [str(i + 1) for i in tick_positions],
            rotation=90,
            fontsize=8,
        )

    ax.set_xlabel("Residue position")
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def cleanup_temp_dirs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            shutil.rmtree(path)


def write_csv(rows: List[dict], out_csv: Path, splits: List[int]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pdb_file",
        "status",
        "sequence_length",
        "pred_dG_mean",
        "pred_dG_std",
        "pred_foldability_prob_mean",
        "pred_foldability_logit_mean",
    ]

    for split in splits:
        fieldnames.append(f"pred_dG_split_{split}")

    fieldnames.extend(
        [
            "heatmap_path",
            "error",
        ]
    )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            safe_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(safe_row)


def main() -> None:
    args = parse_args()

    pdb_dir = args.pdb_dir.expanduser().resolve()
    out_csv = args.out_csv.expanduser().resolve()
    lpm_root = args.lpm_root.expanduser().resolve()

    if not pdb_dir.is_dir():
        raise NotADirectoryError(f"Input PDB directory was not found: {pdb_dir}")

    if args.save_residue_amino_acid_dG_heatmap and args.heatmap_out_dir is None:
        raise ValueError(
            "--heatmap-out-dir is required when "
            "--save-residue-amino-acid-dG-heatmap is specified."
        )

    heatmap_out_dir = None
    if args.heatmap_out_dir is not None:
        heatmap_out_dir = args.heatmap_out_dir.expanduser().resolve()
        heatmap_out_dir.mkdir(parents=True, exist_ok=True)

    pdb_paths = get_pdb_paths(pdb_dir)
    ckpt_paths = resolve_lpm_checkpoints(lpm_root, args.splits)

    print(f"[INFO] Found {len(pdb_paths)} PDB files in: {pdb_dir}")
    print(f"[INFO] Output CSV: {out_csv}")
    print("[INFO] LPM checkpoints:")
    for ckpt in ckpt_paths:
        print(f"  - {ckpt}")

    input_dir, profile_dir, node_rep_dir = prepare_temp_feature_dirs(
        pdb_paths=pdb_paths,
        overwrite_features=True if args.overwrite_features else True,
    )

    rows: List[dict] = []

    try:
        run_feature_generation(input_dir, node_rep_dir, profile_dir)
        ensemble_models = build_lpm_ensemble(ckpt_paths, esm_size=args.esm_size)

        for idx, pdb_path in enumerate(pdb_paths, start=1):
            print(f"[INFO] Predicting {idx}/{len(pdb_paths)}: {pdb_path.name}")

            row = {
                "pdb_file": str(pdb_path),
                "status": "failed",
                "sequence_length": "",
                "pred_dG_mean": "",
                "pred_dG_std": "",
                "pred_foldability_prob_mean": "",
                "pred_foldability_logit_mean": "",
                "heatmap_path": "",
                "error": "",
            }

            try:
                input_data_name = pdb_path.stem
                node_rep_path = node_rep_dir / f"{input_data_name}.pt"
                profile_path = profile_dir / f"{input_data_name}_profile.npy"

                if not node_rep_path.exists():
                    raise FileNotFoundError(f"Generated node representation was not found: {node_rep_path}")
                if not profile_path.exists():
                    raise FileNotFoundError(f"Generated ProteinMPNN profile was not found: {profile_path}")

                aa_seq = get_sequence_from_single_chain_pdb(str(pdb_path))

                pred = predict_with_lpm_ensemble(
                    ensemble_models=ensemble_models,
                    aa_seq=aa_seq,
                    node_rep_path=node_rep_path,
                    profile_path=profile_path,
                )

                row["status"] = "success"
                row["sequence_length"] = len(aa_seq)
                row["pred_dG_mean"] = f"{pred['pred_dG_mean']:.8f}"
                row["pred_dG_std"] = f"{pred['pred_dG_std']:.8f}"
                row["pred_foldability_prob_mean"] = f"{pred['pred_foldability_prob_mean']:.8f}"
                row["pred_foldability_logit_mean"] = f"{pred['pred_foldability_logit_mean']:.8f}"

                for split, value in zip(args.splits, pred["pred_dG_each_model"]):
                    row[f"pred_dG_split_{split}"] = f"{value:.8f}"

                if args.save_residue_amino_acid_dG_heatmap:
                    assert heatmap_out_dir is not None
                    heatmap_path = heatmap_out_dir / f"{pdb_path.stem}_residue_amino_acid_dG_heatmap.png"

                    save_residue_amino_acid_dg_heatmap(
                        matrix=pred["residue_amino_acid_wise_dG_mat_mean"],
                        aa_seq=aa_seq,
                        out_path=heatmap_path,
                        title=f"{pdb_path.stem}: residue-amino-acid-wise predicted ΔG",
                        dpi=args.heatmap_dpi,
                    )
                    row["heatmap_path"] = str(heatmap_path)

                print(
                    "  "
                    f"pred_dG_mean={row['pred_dG_mean']} kcal/mol, "
                    f"foldability_prob_mean={row['pred_foldability_prob_mean']}"
                )

            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"[ERROR] Failed to process {pdb_path.name}: {row['error']}", file=sys.stderr)
                traceback.print_exc()

            rows.append(row)
            write_csv(rows, out_csv, splits=args.splits)

        print(f"[INFO] Done. Results were saved to: {out_csv}")

    finally:
        if not args.keep_temp:
            cleanup_temp_dirs(input_dir, profile_dir, node_rep_dir)


if __name__ == "__main__":
    main()

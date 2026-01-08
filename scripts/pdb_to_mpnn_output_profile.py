#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import copy
import glob
import os
import re
import sys
import warnings

import numpy as np
import torch
import tqdm


def _add_local_paths(script_dir: Path) -> None:
    # scripts/ProteinMPNN_to_get_emb を最優先で import できるようにする
    sys.path.insert(0, str(script_dir / "ProteinMPNN_to_get_emb"))
    
    '''
    # ProteinMPNN 本体（scripts/ProteinMPNN が無ければ clone）
    if not (script_dir / "ProteinMPNN").is_dir():
        os.system("git clone -q https://github.com/dauparas/ProteinMPNN.git " + str(script_dir / "ProteinMPNN"))
    sys.path.append(str(script_dir / "ProteinMPNN"))
    '''

def _load_model(
    script_dir: Path,
    device: torch.device,
    model_name: str,
    backbone_noise: float,
    hidden_dim: int,
    num_layers: int,
) -> torch.nn.Module:
    from protein_mpnn_utils_to_get_emb import ProteinMPNN  # type: ignore

    weights_dir = script_dir / "ProteinMPNN_to_get_emb" / "vanilla_model_weights"
    checkpoint_path = weights_dir / f"{model_name}.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Number of edges:", checkpoint["num_edges"])
    print(f"Training noise level: {checkpoint['noise_level']}A")

    model = ProteinMPNN(
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=backbone_noise,
        k_neighbors=checkpoint["num_edges"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded")
    return model


def make_tied_positions_for_homomers(pdb_dict_list):
    my_dict = {}
    for result in pdb_dict_list:
        all_chain_list = sorted([item[-1:] for item in list(result) if item[:9] == "seq_chain"])
        tied_positions_list = []
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
        for i in range(1, chain_length + 1):
            temp_dict = {}
            for chain in all_chain_list:
                temp_dict[chain] = [i]
            tied_positions_list.append(temp_dict)
        my_dict[result["name"]] = tied_positions_list
    return my_dict


@torch.no_grad()
def compute_profile_for_pdb(
    pdb_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    homomer: bool = True,
    designed_chain: str = "A",
    fixed_chain: str = "",
    max_length: int = 20000,
    sampling_temp: str = "0.1",
    batch_size: int = 1,
) -> np.ndarray:
    """
    Returns:
        profile: np.ndarray shape (20, L)  (元コードと同じ: exp(log_probs)[0,:,:20].T)
    """
    from protein_mpnn_utils_to_get_emb import parse_PDB, StructureDatasetPDB, tied_featurize  # type: ignore

    if designed_chain == "":
        designed_chain_list = []
    else:
        designed_chain_list = re.sub("[^A-Za-z]+", ",", designed_chain).split(",")

    if fixed_chain == "":
        fixed_chain_list = []
    else:
        fixed_chain_list = re.sub("[^A-Za-z]+", ",", fixed_chain).split(",")

    chain_list = list(set(designed_chain_list + fixed_chain_list))

    # ここは元コードと同じ流れ
    pdb_dict_list = parse_PDB(str(pdb_path), input_chain_list=chain_list)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)

    chain_id_dict = {pdb_dict_list[0]["name"]: (designed_chain_list, fixed_chain_list)}
    fixed_positions_dict = None
    pssm_dict = None
    omit_AA_dict = None
    bias_by_res_dict = None

    tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list) if homomer else None

    # dataset_valid は通常 1 要素（その pdb）なので 1 回だけ回る想定
    for protein in dataset_valid:
        batch_clones = [copy.deepcopy(protein) for _ in range(batch_size)]

        (
            X,
            S,
            mask,
            lengths,
            chain_M,
            chain_encoding_all,
            chain_list_list,
            visible_list_list,
            masked_list_list,
            masked_chain_length_list_list,
            chain_M_pos,
            omit_AA_mask,
            residue_idx,
            dihedral_mask,
            tied_pos_list_of_lists_list,
            pssm_coef,
            pssm_bias,
            pssm_log_odds_all,
            bias_by_res_all,
            tied_beta,
        ) = tied_featurize(
            batch_clones,
            device,
            chain_id_dict,
            fixed_positions_dict,
            omit_AA_dict,
            tied_positions_dict,
            pssm_dict,
            bias_by_res_dict,
        )

        # 元コードでは model(...) の最後の引数 randn を torch.randn で渡している（重要）
        randn_1 = torch.randn(chain_M.shape, device=X.device)

        log_probs = model(
            X,
            S,
            mask,
            chain_M * chain_M_pos,
            residue_idx,
            chain_encoding_all,
            randn_1,
        )

        prof = np.exp(log_probs.to("cpu").detach().numpy().copy())[0, :, :20].T  # (20, L)
        return prof

    raise RuntimeError(f"Failed to parse/featurize PDB: {pdb_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ProteinMPNN output profiles for all PDBs in a folder and save as *_profile.npy."
    )
    parser.add_argument("pdb_dir", type=str, help="Folder containing *.pdb files")
    parser.add_argument("out_dir", type=str, help="Output folder to save *_profile.npy")
    parser.add_argument("--model-name", type=str, default="v_48_020", help="ProteinMPNN vanilla weights name")
    parser.add_argument("--backbone-noise", type=float, default=0.0, help="Backbone noise (augment_eps)")
    parser.add_argument("--max-length", type=int, default=20000, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for MPNN featurize")
    parser.add_argument("--homomer", action="store_true", help="Tie positions across chains (homomer mode)")
    parser.add_argument("--designed-chain", type=str, default="A", help="Designed chain letters, e.g. 'A' or 'A,B'")
    parser.add_argument("--fixed-chain", type=str, default="", help="Fixed chain letters, e.g. 'B'")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing *_profile.npy files")

    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdb_dir.is_dir():
        raise SystemExit(f"Not a directory: {pdb_dir}")

    script_dir = Path(__file__).resolve().parent
    _add_local_paths(script_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(
        script_dir=script_dir,
        device=device,
        model_name=args.model_name,
        backbone_noise=args.backbone_noise,
        hidden_dim=128,
        num_layers=3,
    )

    pdb_paths = sorted(pdb_dir.glob("*.pdb"))
    if len(pdb_paths) == 0:
        raise SystemExit(f"No PDB files found in: {pdb_dir}")

    for pdb_path in tqdm.tqdm(pdb_paths, total=len(pdb_paths)):
        out_path = out_dir / f"{pdb_path.stem}_profile.npy"
        if out_path.exists() and (not args.overwrite):
            continue

        try:
            prof = compute_profile_for_pdb(
                pdb_path=pdb_path,
                model=model,
                device=device,
                homomer=args.homomer,
                designed_chain=args.designed_chain,
                fixed_chain=args.fixed_chain,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
            np.save(out_path, prof)
        except Exception as e:
            print(f"Error processing {pdb_path.name}: {e}", file=sys.stderr)

    print(f"Done. Output dir: {out_dir}")


if __name__ == "__main__":
    # PyTorch の FutureWarning を抑えたい場合は適宜調整（挙動は変えない）
    warnings.filterwarnings("default")
    main()

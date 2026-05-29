"""ProteinMPNN utilities for node-embedding extraction.

Re-exports the canonical ProteinMPNN implementation from
ProteinMPNN_to_get_emb/protein_mpnn_utils.py, then replaces ProteinMPNN
with a subclass whose forward() returns (all_hidden, h_ES_list, log_probs)
using deterministic left-to-right decoding, and adds alt_parse_PDB* helpers
that also return the raw PDB residue-number list.
"""
from __future__ import print_function

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Canonical ProteinMPNN package lives one level down.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "ProteinMPNN_to_get_emb"))

# Re-export everything callers expect under the protein_mpnn_utils namespace.
from protein_mpnn_utils import (  # type: ignore  # noqa: E402, F401
    _scores,
    _S_to_seq,
    parse_PDB_biounits,
    parse_PDB,
    tied_featurize,
    loss_nll,
    loss_smoothed,
    StructureDataset,
    StructureDatasetPDB,
    StructureLoader,
    gather_edges,
    gather_nodes,
    gather_nodes_t,
    cat_neighbors_nodes,
    EncLayer,
    DecLayer,
    PositionWiseFeedForward,
    PositionalEncodings,
    CA_ProteinFeatures,
    ProteinFeatures,
    ProteinMPNN as _ProteinMPNN,
)


class ProteinMPNN(_ProteinMPNN):
    """ProteinMPNN variant for node-embedding extraction.

    Differences from the canonical class:
      - forward() returns (all_hidden, h_ES_list, log_probs) instead of log_probs.
      - Decoding order is deterministic left-to-right (all context always visible),
        giving stable embeddings without needing a random seed.
    """

    def forward(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        randn,
        use_input_decoding_order=False,
        decoding_order=None,
    ):
        device = X.device

        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        # Keep intermediate embeddings on CPU to save GPU memory.
        h_ES_list = {
            "h_S": h_S.to("cpu"),
            "h_E": h_E.to("cpu"),
            "E_idx": E_idx.to("cpu"),
        }

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask

        if not use_input_decoding_order:
            # Deterministic left-to-right order: all residues are fully visible.
            decoding_order = torch.tensor([list(range(X.size(1)))], device=device)

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        # Set all residues fully visible (override backward/forward masks).
        order_mask_backward = torch.ones_like(order_mask_backward)

        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        all_hidden = []
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
            all_hidden.append(h_V)

        h_ES_list["h_V"] = h_V

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return list(reversed(all_hidden)), h_ES_list, log_probs


# ---------------------------------------------------------------------------
# Additional helpers not present in the canonical code
# ---------------------------------------------------------------------------

def alt_parse_PDB_biounits(x, atoms=None, chain=None):
    """Like parse_PDB_biounits, but also returns the raw PDB residue-number list.

    Returns:
        xyz       : np.ndarray (L, len(atoms), 3)
        seq       : list[str]
        resn_list : list of raw PDB residue-number strings (preserves insertion codes)
    On failure returns ('no_chain', 'no_chain', 'no_chain').
    """
    if atoms is None:
        atoms = ["N", "CA", "C"]

    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "GAP",
    ]
    aa_1_N = {a: n for n, a in enumerate(alpha_1)}
    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = {n: a for n, a in enumerate(alpha_1)}

    def N_to_AA(x):
        x = np.array(x)
        if x.ndim == 1:
            x = x[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

    xyz, seq = {}, {}
    min_resn, max_resn = 1e6, -1e6
    resn_list = []

    with open(x, "rb") as f:
        for line in f:
            line = line.decode("utf-8", "ignore").rstrip()
            if line[:6] == "HETATM" and line[17:20] == "MSE":
                line = line.replace("HETATM", "ATOM  ").replace("MSE", "MET")
            if line[:4] != "ATOM":
                continue
            ch = line[21:22]
            if ch != chain and chain is not None:
                continue
            atom = line[12:16].strip()
            resi = line[17:20]
            resn_raw = line[22:27].strip()
            if resn_raw not in resn_list:
                resn_list.append(resn_raw)
            coords = [float(line[i : i + 8]) for i in [30, 38, 46]]
            if resn_raw[-1].isalpha():
                resa, resn_int = resn_raw[-1], int(resn_raw[:-1]) - 1
            else:
                resa, resn_int = "", int(resn_raw) - 1
            min_resn = min(min_resn, resn_int)
            max_resn = max(max_resn, resn_int)
            xyz.setdefault(resn_int, {}).setdefault(resa, {})
            seq.setdefault(resn_int, {}).setdefault(resa, resi)
            if atom not in xyz[resn_int][resa]:
                xyz[resn_int][resa][atom] = np.array(coords)

    seq_, xyz_ = [], []
    try:
        for resn_int in range(int(min_resn), int(max_resn) + 1):
            if resn_int in seq:
                for k in sorted(seq[resn_int]):
                    seq_.append(aa_3_N.get(seq[resn_int][k], 20))
            else:
                seq_.append(20)
            if resn_int in xyz:
                for k in sorted(xyz[resn_int]):
                    for atom in atoms:
                        xyz_.append(
                            xyz[resn_int][k].get(atom, np.full(3, np.nan))
                        )
            else:
                for _ in atoms:
                    xyz_.append(np.full(3, np.nan))
        return (
            np.array(xyz_).reshape(-1, len(atoms), 3),
            N_to_AA(np.array(seq_)),
            list(dict.fromkeys(resn_list)),
        )
    except TypeError:
        return "no_chain", "no_chain", "no_chain"


def alt_parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False, side_chains=False):
    """Like parse_PDB, but each dict in the returned list contains 'resn_list'.

    'resn_list' holds the raw PDB residue-number strings from the last parsed
    chain, preserving insertion codes (e.g. '100A').
    """
    init_alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    extra_alphabet = [str(i) for i in range(300)]
    chain_alphabet = input_chain_list if input_chain_list else init_alphabet + extra_alphabet

    pdb_dict_list = []
    for biounit in [path_to_pdb]:
        my_dict = {"resn_list": []}
        s = 0
        concat_seq = ""
        resn_list = []

        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ["CA"]
            elif side_chains:
                sidechain_atoms = [
                    "N", "CA", "C", "O", "CB",
                    "CG", "CG1", "OG1", "OG2", "CG2", "OG", "SG",
                    "CD", "SD", "CD1", "ND1", "CD2", "OD1", "OD2", "ND2",
                    "CE", "CE1", "NE1", "OE1", "NE2", "OE2", "NE", "CE2", "CE3",
                    "NZ", "CZ", "CZ2", "CZ3", "CH2", "OH", "NH1", "NH2",
                ]
            else:
                sidechain_atoms = ["N", "CA", "C", "O"]

            xyz, seq, resn_list = alt_parse_PDB_biounits(
                biounit, atoms=sidechain_atoms, chain=letter
            )
            if isinstance(xyz, str):
                continue

            concat_seq += seq[0]
            my_dict[f"seq_chain_{letter}"] = seq[0]
            coords = {}
            if ca_only:
                coords[f"CA_chain_{letter}"] = xyz.tolist()
            else:
                coords[f"N_chain_{letter}"] = xyz[:, 0, :].tolist()
                coords[f"CA_chain_{letter}"] = xyz[:, 1, :].tolist()
                coords[f"C_chain_{letter}"] = xyz[:, 2, :].tolist()
                coords[f"O_chain_{letter}"] = xyz[:, 3, :].tolist()
            my_dict[f"coords_chain_{letter}"] = coords
            s += 1

        fi = biounit.rfind("/")
        my_dict["name"] = biounit[fi + 1 : -4]
        my_dict["num_of_chains"] = s
        my_dict["seq"] = concat_seq
        my_dict["resn_list"] = list(resn_list)

        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)

    return pdb_dict_list

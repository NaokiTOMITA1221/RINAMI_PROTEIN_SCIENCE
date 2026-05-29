#!/usr/bin/env python3
"""
ProteinMPNN utilities for node-embedding extraction.

This wrapper loads the canonical ProteinMPNN implementation from:

    scripts/ProteinMPNN_to_get_emb/protein_mpnn_utils.py

using importlib to avoid circular imports caused by this wrapper having the
same module name, protein_mpnn_utils.py.

It then re-exports the canonical utilities and replaces ProteinMPNN with a
subclass whose forward() returns:

    (all_hidden, h_ES_list, log_probs)

This behavior is used by pdb_to_mpnn_node_rep.py to extract node embeddings.
"""

from __future__ import print_function

from pathlib import Path
import importlib.util
import sys

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Load the canonical ProteinMPNN utility module without name collision
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_REAL_UTILS_PATH = _HERE / "ProteinMPNN_to_get_emb" / "protein_mpnn_utils.py"

if not _REAL_UTILS_PATH.exists():
    raise FileNotFoundError(
        f"Could not find canonical ProteinMPNN utility file: {_REAL_UTILS_PATH}"
    )

_spec = importlib.util.spec_from_file_location(
    "_rinami_canonical_protein_mpnn_utils",
    str(_REAL_UTILS_PATH),
)

if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module spec from {_REAL_UTILS_PATH}")

_real = importlib.util.module_from_spec(_spec)
sys.modules["_rinami_canonical_protein_mpnn_utils"] = _real
_spec.loader.exec_module(_real)


# ---------------------------------------------------------------------------
# Re-export canonical names expected by existing scripts
# ---------------------------------------------------------------------------

_scores = _real._scores
_S_to_seq = _real._S_to_seq

parse_PDB_biounits = _real.parse_PDB_biounits
parse_PDB = _real.parse_PDB
tied_featurize = _real.tied_featurize

loss_nll = _real.loss_nll
loss_smoothed = _real.loss_smoothed

StructureDataset = _real.StructureDataset
StructureDatasetPDB = _real.StructureDatasetPDB
StructureLoader = _real.StructureLoader

gather_edges = _real.gather_edges
gather_nodes = _real.gather_nodes
gather_nodes_t = _real.gather_nodes_t
cat_neighbors_nodes = _real.cat_neighbors_nodes

EncLayer = _real.EncLayer
DecLayer = _real.DecLayer
PositionWiseFeedForward = _real.PositionWiseFeedForward
PositionalEncodings = _real.PositionalEncodings
CA_ProteinFeatures = _real.CA_ProteinFeatures
ProteinFeatures = _real.ProteinFeatures

_ProteinMPNN = _real.ProteinMPNN


# ---------------------------------------------------------------------------
# ProteinMPNN subclass for node-embedding extraction
# ---------------------------------------------------------------------------

class ProteinMPNN(_ProteinMPNN):
    """
    ProteinMPNN variant for node-embedding extraction.

    Differences from the canonical class:
    - forward() returns (all_hidden, h_ES_list, log_probs) instead of only log_probs.
    - Decoding order is deterministic left-to-right by default.
    - Intermediate embeddings needed by RINAMI are stored in h_ES_list.
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

        h_V = torch.zeros(
            (E.shape[0], E.shape[1], E.shape[-1]),
            device=E.device,
        )
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        h_ES_list = {
            "h_S": h_S.to("cpu"),
            "h_E": h_E.to("cpu"),
            "E_idx": E_idx.to("cpu"),
        }

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask

        if not use_input_decoding_order:
            # Deterministic left-to-right decoding order.
            batch_size = X.size(0)
            seq_len = X.size(1)
            decoding_order = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)

        mask_size = E_idx.shape[1]

        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order,
            num_classes=mask_size,
        ).float()

        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (
                1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            ),
        )

        # For embedding extraction, make all residues visible.
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

        h_ES_list["h_V"] = h_V.to("cpu")

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)

        return list(reversed(all_hidden)), h_ES_list, log_probs


__all__ = [
    "_scores",
    "_S_to_seq",
    "parse_PDB_biounits",
    "parse_PDB",
    "tied_featurize",
    "loss_nll",
    "loss_smoothed",
    "StructureDataset",
    "StructureDatasetPDB",
    "StructureLoader",
    "gather_edges",
    "gather_nodes",
    "gather_nodes_t",
    "cat_neighbors_nodes",
    "EncLayer",
    "DecLayer",
    "PositionWiseFeedForward",
    "PositionalEncodings",
    "CA_ProteinFeatures",
    "ProteinFeatures",
    "ProteinMPNN",
]

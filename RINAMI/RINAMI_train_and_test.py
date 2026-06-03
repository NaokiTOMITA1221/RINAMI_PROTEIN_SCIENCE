import copy
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import get_cosine_schedule_with_warmup

from RINAMI_model_main import RINAMI
from util import (
    AA_ORDER_PLOT,
    DEFAULT_ENSEMBLE_ROOT,
    DEFAULT_ENSEMBLE_ROOT_TRAINED_BY_USER,
    DEFAULT_ENSEMBLE_ROOT_BASELINE,
    DEFAULT_ENSEMBLE_SPLITS,
    ESM_SIZE,
    TRAIN_SAMPLE_FRACTION,
    _bootstrap_regression_metrics,
    _avg_ensemble_matrix_outputs,
    align_residue_info_to_sequence,
    build_maxwell_dataset,
    build_baseline_ensemble_models,
    build_shared_ensemble_models,
    composite_score,
    device,
    evaluate_foldability_ensemble_from_avg_matrix,
    evaluate_maxwell_ensemble_from_avg_matrix,
    evaluate_regression,
    extract_residue_info_and_rsa,
    iterate_batches,
    load_garcia_benchmark_dataset,
    load_mega_test,
    load_mega_test_and_val_wt_only,
    load_train_val,
    make_optimizer,
    resolve_ensemble_ckpts,
    save_interpretability_heatmap,
    train_one_epoch_classification,
    train_one_epoch_regression,
)

def train_model(
    model_save_path,
    trained_model_param=None,
    num_sets=20,
    batch_size=256,
    dropout=0.1,
    ESM_size=ESM_SIZE,
    split_num=1,
    patience=8,
):
    reg_train, reg_val, cls_train = load_train_val(
        split_num=split_num,
        add_extremes_to_cls=True,
    )

    model = RINAMI(dropout=dropout, ESM_size=ESM_size).to(device)
    if trained_model_param is not None:
        state = torch.load(trained_model_param, map_location=device)
        model.load_state_dict(state, strict=False)

    optimizer = make_optimizer(model)

    effective_cls_size = max(1, int(np.floor(len(cls_train["seqs"]) * TRAIN_SAMPLE_FRACTION)))
    effective_reg_size = max(1, int(np.floor(len(reg_train["seqs"]) * TRAIN_SAMPLE_FRACTION)))

    steps_per_cls_epoch = math.ceil(effective_cls_size / batch_size)
    steps_per_reg_epoch = math.ceil(effective_reg_size / batch_size)
    total_steps = num_sets * (steps_per_cls_epoch + 3 * steps_per_reg_epoch)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.01 * total_steps)),
        num_training_steps=max(2, total_steps),
    )

    print(
        f"[INFO] TRAIN_SAMPLE_FRACTION={TRAIN_SAMPLE_FRACTION:.3f} "
        f"| cls_train_used_per_epoch≈{effective_cls_size}/{len(cls_train['seqs'])} "
        f"| reg_train_used_per_epoch≈{effective_reg_size}/{len(reg_train['seqs'])}"
    )

    best_state = None
    best_metrics = None
    best_score = -1e18
    best_tag = None
    stale = 0

    def evaluate_and_maybe_save(tag, train_loss, update_best_and_stale=False):
        nonlocal best_state, best_metrics, best_score, best_tag, stale

        metrics = evaluate_regression(model, reg_val, batch_size=max(32, batch_size // 2))
        score = composite_score(metrics)

        print(
            f"[{tag}] "
            f"TrainLoss: {train_loss:.6f} "
            f"ValLoss: {metrics['val_loss']:.6f} "
            f"Pearson: {metrics['pearson']:.4f} "
            f"Spearman: {metrics['spearman']:.4f} "
            f"RMSE: {metrics['rmse']:.4f} "
            f"Composite: {score:.4f}"
        )

        epoch_ckpt_path = model_save_path.replace(".pth", f"_{tag}.pth")
        torch.save(model.state_dict(), epoch_ckpt_path)

        if update_best_and_stale:
            if score > best_score:
                best_score = score
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())
                best_tag = tag
                stale = 0
                torch.save(best_state, model_save_path)
                print(f"[BEST UPDATED] {tag}")
            else:
                stale += 1
                print(f"[NO IMPROVEMENT] stale={stale}/{patience}")
        else:
            print("[INFO] classification epoch: best/stale unchanged")

        return metrics, score

    stop_training = False

    for set_idx in range(num_sets):
        cls_loss = train_one_epoch_classification(
            model,
            cls_train,
            optimizer,
            scheduler,
            batch_size=batch_size,
            epoch_label=f"set {set_idx+1} cls(1/1)",
            sample_fraction=TRAIN_SAMPLE_FRACTION,
        )
        evaluate_and_maybe_save(
            tag=f"set{set_idx+1}_cls",
            train_loss=cls_loss,
            update_best_and_stale=False,
        )

        for reg_epoch_idx in range(3):
            reg_loss = train_one_epoch_regression(
                model,
                reg_train,
                optimizer,
                scheduler,
                batch_size=batch_size,
                epoch_label=f"set {set_idx+1} reg({reg_epoch_idx+1}/3)",
                sample_fraction=TRAIN_SAMPLE_FRACTION,
            )
            evaluate_and_maybe_save(
                tag=f"set{set_idx+1}_reg{reg_epoch_idx+1}",
                train_loss=reg_loss,
                update_best_and_stale=True,
            )
            if stale >= patience:
                print(f"Early stopping after set {set_idx+1} reg epoch {reg_epoch_idx+1}.")
                stop_training = True
                break

        if stop_training:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Best validation metrics:")
    print(json.dumps(
        {
            "best_tag": best_tag,
            **{k: v for k, v in best_metrics.items() if k not in ["pred_dg", "true_dg"]},
        },
        indent=2
    ))


def test_model(trained_model_param, ESM_size=ESM_SIZE, batch_size=256, split_num=1, use_baseline=False):
    test_data = load_mega_test(split_num=split_num)

    if use_baseline:
        from Baseline_RINAMI_model_main import BaselineRINAMI
        model = BaselineRINAMI(dropout=0.1, ESM_size=ESM_size).to(device)
    else:
        model = RINAMI(dropout=0.1, ESM_size=ESM_size).to(device)
    model.load_state_dict(torch.load(trained_model_param, map_location=device), strict=False)

    metrics = evaluate_regression(model, test_data, batch_size=batch_size)
    print("Mega test metrics:")
    print(json.dumps(
        {
            "pearson": metrics["pearson"],
            "spearman": metrics["spearman"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "val_loss": metrics["val_loss"],
        },
        indent=2,
    ))



def export_residue_contributions(trained_model_param, output_dir, ESM_size=ESM_SIZE, split_num=1, batch_size=1):
    os.makedirs(output_dir, exist_ok=True)
    test_data = load_mega_test_and_val_wt_only(split_num=split_num)

    model = RINAMI(dropout=0.1, ESM_size=ESM_size).to(device)
    model.load_state_dict(torch.load(trained_model_param, map_location=device), strict=False)
    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(
            iterate_batches(test_data, batch_size=batch_size, shuffle=False, sample_fraction=1.0),
            desc="export_wt_only"
        ):
            out = model(batch["seqs"], batch["structs"], batch["profiles"], return_details=True)

            batch_indices = []
            for name in batch["names"]:
                idx = test_data["names"].index(name)
                batch_indices.append(idx)

            batch_pdbs = [test_data["pdbs"][i] for i in batch_indices]

            for i, (name, seq, pdb_path) in enumerate(zip(batch["names"], batch["seqs"], batch_pdbs)):
                residue_amino_acid_wise_dG_mat = np.square(
                    out["residue_amino_acid_wise_dG_mat"][i].detach().cpu().numpy()
                )
                valid_len = len(seq)

                residue_infos_raw = extract_residue_info_and_rsa(pdb_path)
                residue_infos = align_residue_info_to_sequence(seq, residue_infos_raw, pdb_path)

                png_path = os.path.join(output_dir, f"{name}_interpretability_heatmap.png")
                buried_mask, rsa_array, residue_numbers = save_interpretability_heatmap(
                    residue_amino_acid_wise_dG_mat=residue_amino_acid_wise_dG_mat,
                    valid_len=valid_len,
                    seq=seq,
                    residue_infos=residue_infos,
                    png_path=png_path,
                    vmin=-.75,
                    vmax=.75,
                )
                xticklabels = np.array(
                    [f"{residue_numbers[j]}{seq[j]}" for j in range(valid_len)],
                    dtype=object,
                )
                aa_labels_plot_order = np.array(list(AA_ORDER_PLOT), dtype=object)

                record = {
                    "residue_amino_acid_wise_dG_mat": residue_amino_acid_wise_dG_mat,
                    "pred_dG": float(out["dG"][i].item()),
                    "pred_foldability_prob": float(torch.sigmoid(out["foldability_logit"][i]).item()),
                    "rsa": rsa_array,
                    "buried_mask": buried_mask,
                    "residue_numbers": residue_numbers,
                    "xticklabels": xticklabels,
                    "aa_order_for_plot": aa_labels_plot_order,
                    "pdb_path": np.array(pdb_path, dtype=object),
                    "sequence": np.array(seq, dtype=object),
                }

                np.savez_compressed(
                    os.path.join(output_dir, f"{name}_interpretability.npz"),
                    **record
                )

###############################################
# Garcia benchmark and Maxwell ensemble tests #
###############################################
def test_model_with_maxwell_ensemble(
    ensemble_root=DEFAULT_ENSEMBLE_ROOT,
    ESM_size=ESM_SIZE,
    batch_size=256,
    use_baseline=False,
):
    dataset = build_maxwell_dataset()
    ckpt_paths = resolve_ensemble_ckpts(ensemble_root=ensemble_root, splits=DEFAULT_ENSEMBLE_SPLITS)
    if use_baseline:
        ensemble_models = build_baseline_ensemble_models(ckpt_paths, ESM_size=ESM_size, dropout=0.1)
    else:
        ensemble_models = build_shared_ensemble_models(ckpt_paths, ESM_size=ESM_size, dropout=0.1)

    print("[INFO] Maxwell ensemble evaluation uses averaged residue_amino_acid_wise_dG_mat over 3 heads")
    print("[INFO] foldability_logit is recomputed from the averaged matrix-derived dG")

    metrics = evaluate_maxwell_ensemble_from_avg_matrix(
        ensemble_models,
        dataset,
        batch_size=batch_size,
        print_predictions=True,
        show_progress=True,
        progress_desc="Maxwell ensemble",
    )

    bootstrap_summary = _bootstrap_regression_metrics(
        pred_dg=metrics["pred_dg"],
        true_dg=metrics["true_dg"],
        B=10000,
        seed=1234,
        alpha=0.05,
    )

    print("\nMaxwell ensemble metrics:")

    print(json.dumps(
        {
            "ensemble_ckpts": ckpt_paths,
            "pearson": metrics["pearson"],
            "spearman": metrics["spearman"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "val_loss": metrics["val_loss"],
            "mean_pred_foldability_logit": float(np.mean(metrics["pred_foldability_logit"])),
            "mean_pred_foldability_prob": float(np.mean(metrics["pred_foldability_prob"])),
            "bootstrap": bootstrap_summary,
        },
        indent=2,
    ))
    print(f"\nCorrelation of Maxwell ensemble test: {metrics['pearson']}")

def test_model_with_garcia_benchmark_set_ensemble(
    ensemble_root=DEFAULT_ENSEMBLE_ROOT,
    ESM_size=ESM_SIZE,
    batch_size=1,
    threshold=0.5,
    seq_len_threshold=300,
    print_predictions=False,
    use_baseline=False,
):
    dataset = load_garcia_benchmark_dataset(seq_len_threshold=seq_len_threshold)
    ckpt_paths = resolve_ensemble_ckpts(
        ensemble_root=ensemble_root,
        splits=DEFAULT_ENSEMBLE_SPLITS,
    )
    if use_baseline:
        ensemble_models = build_baseline_ensemble_models(
            ckpt_paths,
            ESM_size=ESM_size,
            dropout=0.1,
        )
    else:
        ensemble_models = build_shared_ensemble_models(
            ckpt_paths,
            ESM_size=ESM_size,
            dropout=0.1,
        )


    metrics = evaluate_foldability_ensemble_from_avg_matrix(
        ensemble_models,
        dataset,
        batch_size=batch_size,
        threshold=threshold,
        print_predictions=print_predictions,
        show_progress=True,
        progress_desc=f"Garcia ensemble (len<={seq_len_threshold})",
    )

    af_probs = np.array(dataset["af_plddt"], dtype=float)
    esmfold_probs = np.array(dataset["esmfold_plddt"], dtype=float)
    ref_labels = np.array(dataset["af_label"], dtype=int)

    auc_roc_AF_pLDDT_3rec = float(roc_auc_score(ref_labels, af_probs))
    auc_roc_ESMFold_pLDDT = float(roc_auc_score(ref_labels, esmfold_probs))

    auc_pr_AF_pLDDT_3rec = float(average_precision_score(ref_labels, af_probs))
    auc_pr_ESMFold_pLDDT = float(average_precision_score(ref_labels, esmfold_probs))

    print(f"True Foldable: {metrics['true_pos_count']}, True Not Foldable: {metrics['true_neg_count']}")
    print(
        f"ROC-AUC(foldable=1): {metrics['auc_roc']:.4f} | "
        f"PR-AUC(foldable=1): {metrics['auc_pr']:.4f} | "
        f"Accuracy: {metrics['acc']:.4f} | "
        f"AF2 pLDDT ROC-AUC: {auc_roc_AF_pLDDT_3rec:.4f} | "
        f"AF2 pLDDT PR-AUC: {auc_pr_AF_pLDDT_3rec:.4f} | "
        f"ESMFold pLDDT ROC-AUC: {auc_roc_ESMFold_pLDDT:.4f} | "
        f"ESMFold pLDDT PR-AUC: {auc_pr_ESMFold_pLDDT:.4f}"
    )

    return (
        metrics["p_f_probs"],
        dataset["af_plddt"],
        dataset["esmfold_plddt"],
        dataset["af_label"],
        metrics["auc_roc"],
        metrics["auc_pr"],
        auc_roc_AF_pLDDT_3rec,
        auc_pr_AF_pLDDT_3rec,
        auc_roc_ESMFold_pLDDT,
        auc_pr_ESMFold_pLDDT,
        metrics["true_pos_count"],
        metrics["true_neg_count"],
    )


def run_garcia_benchmark_ensemble(
    ensemble_root=DEFAULT_ENSEMBLE_ROOT,
    ESM_size=ESM_SIZE,
    use_baseline=False,
):
    print("Test mode: Garcia_benchmark_ensemble_test")

    seq_len_threshold_list = [80, 130, 180, 230]

    ROC_AUC_list = []
    ROC_AUC_AF2_list = []
    ROC_AUC_ESMFold_list = []

    PR_AUC_list = []
    PR_AUC_AF2_list = []
    PR_AUC_ESMFold_list = []

    true_pos_num_list = []
    true_neg_num_list = []

    for seq_len_threshold in seq_len_threshold_list:
        print("#" * 80)
        print(f"seq_len_threshold = {seq_len_threshold}")

        (
            foldability_prob_RINAMI,
            AF2_plddt,
            ESMFold_plddt,
            exp_foldability,
            roc_auc,
            pr_auc,
            auc_roc_AF_pLDDT_3rec,
            auc_pr_AF_pLDDT_3rec,
            auc_roc_ESMFold_pLDDT,
            auc_pr_ESMFold_pLDDT,
            tpc,
            tnc,
        ) = test_model_with_garcia_benchmark_set_ensemble(
            ensemble_root=ensemble_root,
            ESM_size=ESM_size,
            seq_len_threshold=seq_len_threshold,
            batch_size=1,
            print_predictions=False,
            use_baseline=use_baseline,
        )
        

#######
# CLI #
#######
if __name__ == "__main__":
    args = sys.argv

    # Train from scratch
    if len(args) == 3 and args[1] not in ["Maxwell_test", "Garcia_test"]:
        model_save_path = args[1]
        split_num = int(args[2])
        print("Training mode: from scratch")
        train_model(model_save_path, trained_model_param=None, split_num=split_num)


    # Finetune        
    elif len(args) == 4 and args[2] not in [
        "Mega_test",
        "Maxwell_test",
        "Garcia_test",
        "export_interpretability",
    ]:
        model_save_path = args[1]
        trained_model_param = args[2]
        split_num = int(args[3])
        print("Training mode: finetune")
        train_model(model_save_path, trained_model_param=trained_model_param, split_num=split_num)

    elif len(args) == 5 and args[2] == "export_interpretability":
        export_residue_contributions(args[1], args[4], split_num=args[3])
    
    elif len(args) == 5 and args[2] == "Mega_test" and args[4] == "BASELINE":
        print("Mode: Mega_test (baseline)")
        test_model(args[1], split_num=int(args[3]), use_baseline=True)

    elif len(args) == 4 and args[2] == "Mega_test":
        print("Mode: Mega_test")
        test_model(args[1], split_num=int(args[3]))

    elif len(args) == 2 and args[1] == "Maxwell_test":
        print("Mode: Maxwell_test")
        test_model_with_maxwell_ensemble()

    elif len(args) == 2 and args[1] == "Garcia_test":
        print("Mode: Garcia_test")
        run_garcia_benchmark_ensemble()

    elif len(args) == 3 and args[1] == "Maxwell_test" and args[2] == "USER_TRAINED":
        print("Mode: Maxwell_test (user trained)")
        test_model_with_maxwell_ensemble(ensemble_root=DEFAULT_ENSEMBLE_ROOT_TRAINED_BY_USER)

    elif len(args) == 3 and args[1] == "Garcia_test" and args[2] == "USER_TRAINED":
        print("Mode: Garcia_test (user trained)")
        run_garcia_benchmark_ensemble(ensemble_root=DEFAULT_ENSEMBLE_ROOT_TRAINED_BY_USER)

    elif len(args) == 3 and args[1] == "Maxwell_test" and args[2] == "BASELINE":
        print("Mode: Maxwell_test (baseline models)")
        test_model_with_maxwell_ensemble(ensemble_root=DEFAULT_ENSEMBLE_ROOT_BASELINE, use_baseline=True)

    elif len(args) == 3 and args[1] == "Garcia_test" and args[2] == "BASELINE":
        print("Mode: Garcia_test (baseline models)")
        run_garcia_benchmark_ensemble(ensemble_root=DEFAULT_ENSEMBLE_ROOT_BASELINE, use_baseline=True)




    else:
        print(
        
            "Usage:\n"
            "  train from scratch:\n"
            "    python3 RINAMI_train_and_test.py <save_path> <split_num>\n"
            "\n"
            "  finetune:\n"
            "    python3 RINAMI_train_and_test.py <save_path> <init_ckpt> <split_num>\n"
            "\n"
            "  Mega test:\n"
            "    python3 RINAMI_train_and_test.py <ckpt> Mega_test <split_num>\n"
            "\n"
            "  export:\n"
            "    python3 RINAMI_train_and_test.py <ckpt> export_interpretability <split_num> <out_dir>\n"
            "\n"
            "  Maxwell test with the manuscript-reported models:\n"
            "    python3 RINAMI_train_and_test.py Maxwell_test\n"
            "\n"
            "  Garcia test with the manuscript-reported models:\n"
            "    python3 RINAMI_train_and_test.py Garcia_test\n"
            "\n"
            "  Maxwell test with user-trained models:\n"
            "    python3 RINAMI_train_and_test.py Maxwell_test USER_TRAINED\n"
            "\n"
            "  Garcia test with user-trained models:\n"
            "    python3 RINAMI_train_and_test.py Garcia_test USER_TRAINED\n"
            "\n"
            "  Maxwell test with the baseline models:\n"
            "    python3 RINAMI_train_and_test.py Maxwell_test BASELINE\n"
            "\n"
            "  Garcia test with the baseline models:\n"
            "    python3 RINAMI_train_and_test.py Garcia_test BASELINE\n"
        
            )





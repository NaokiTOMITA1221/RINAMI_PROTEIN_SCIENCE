import re
import torch

# RINAMI を import（あなたの環境に合わせてパス調整）
from RINAMI_regression_train_and_test import RINAMI

def remap_keys(old_sd: dict) -> dict:
    rules = [
        # projection / refine
        (r"^ESM_rep_projection\.", "ESM_rep_refine."),
        (r"^ProteinMPNN_profile_projection\.", "ProteinMPNN_rep_refine."),

        # BN
        (r"^bn_proj_aa\.", "bn_refine_aa."),
        (r"^bn_proj_node\.", "bn_refine_node."),

        # 以降は「同名」なので基本そのまま：
        # layer_norm_*, MLP_pe_*, CA_aa_seq_rep_and_node_rep, interaction_converter*
    ]

    new_sd = {}
    for k, v in old_sd.items():
        nk = k
        for pat, repl in rules:
            nk = re.sub(pat, repl, nk)
        new_sd[nk] = v
    return new_sd

def load_ckpt(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    # 形式が {"state_dict": ...} の場合にも対応
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt

def main(old_ckpt_path: str, out_path: str, ESM_size: int = 320):
    old_sd = load_ckpt(old_ckpt_path)

    # キー変換
    remapped = remap_keys(old_sd)

    # RINAMI 側にロードして、入るところだけ入れる（まずは strict=False 推奨）
    model = RINAMI(dropout=0.0, ESM_size=ESM_size).cpu()
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    print("\n=== load_state_dict report (strict=False) ===")
    print("missing keys   :", len(missing))
    print("unexpected keys:", len(unexpected))

    # 重要：ここで missing/unexpected を見て、追加ルールが必要か判断
    # とりあえず「入った重み」を新命名で保存
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved remapped RINAMI state_dict -> {out_path}")

if __name__ == "__main__":
    import sys
    # 使い方:
    # python remap_dGpredictor_to_RINAMI.py old.pth rinami_from_old.pth 320
    old_pth = sys.argv[1]
    out_pth = sys.argv[2]
    esm = int(sys.argv[3]) if len(sys.argv) >= 4 else 320
    main(old_pth, out_pth, ESM_size=esm)


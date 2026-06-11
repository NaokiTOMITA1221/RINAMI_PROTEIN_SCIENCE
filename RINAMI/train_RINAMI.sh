#!/usr/bin/env bash
set -euo pipefail

############# user settings ############
SCRIPT_DIR="${SCRIPT_DIR:-./}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-RINAMI_train_and_test.py}"

OUT_ROOT="${OUT_ROOT:-../pth/pth_RINAMI_trained}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"

SPLITS="${SPLITS:-1 2 3}"

# To resume or initialize from an existing checkpoint:
#   INIT_CKPT_MODE=none  -> train each split from scratch
#   INIT_CKPT_MODE=path  -> use INIT_CKPT_PATH as the initial checkpoint for all splits
INIT_CKPT_MODE="${INIT_CKPT_MODE:-none}"
INIT_CKPT_PATH="${INIT_CKPT_PATH:-}"

# Run Mega-scale test after training
RUN_MEGA_TEST="${RUN_MEGA_TEST:-1}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

cd "${SCRIPT_DIR}"

echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "SPLITS=${SPLITS}"
echo "INIT_CKPT_MODE=${INIT_CKPT_MODE}"
echo "RUN_MEGA_TEST=${RUN_MEGA_TEST}"
echo

for split in ${SPLITS}; do
    echo "############################################################"
    echo "Starting training for split ${split}"
    echo "############################################################"

    SPLIT_DIR="${OUT_ROOT}/pth_split_${split}"
    mkdir -p "${SPLIT_DIR}"

    BEST_CKPT="${SPLIT_DIR}/best.pth"
    TRAIN_LOG="${LOG_ROOT}/train_split_${split}.log"
    MEGA_LOG="${LOG_ROOT}/mega_test_split_${split}.log"

    if [[ "${INIT_CKPT_MODE}" == "path" ]]; then
        if [[ -z "${INIT_CKPT_PATH}" ]]; then
            echo "[ERROR] INIT_CKPT_MODE=path but INIT_CKPT_PATH is empty." >&2
            exit 1
        fi

        echo "[INFO] finetune mode with init ckpt: ${INIT_CKPT_PATH}"
        "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${BEST_CKPT}" "${INIT_CKPT_PATH}" "${split}" 2>&1 | tee "${TRAIN_LOG}"
    else
        echo "[INFO] train from scratch"
        "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${BEST_CKPT}" "${split}" 2>&1 | tee "${TRAIN_LOG}"
    fi

    if [[ "${RUN_MEGA_TEST}" == "1" ]]; then
        echo "[INFO] Mega_test for split ${split}"
        "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${BEST_CKPT}" "Mega_test" "${split}" 2>&1 | tee "${MEGA_LOG}"
    fi

    echo "[INFO] finished split ${split}"
    echo
done

echo "All requested splits finished."

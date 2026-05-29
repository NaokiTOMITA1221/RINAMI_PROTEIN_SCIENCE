#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-./}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-RINAMI_train_and_test.py}"

BASELINE_ROOT="${BASELINE_ROOT:-../pth/baseline_models}"
LOG_ROOT="${LOG_ROOT:-${BASELINE_ROOT}/logs}"
SPLITS="${SPLITS:-1 2 3}"

RUN_MEGA_TEST="${RUN_MEGA_TEST:-1}"
RUN_MAXWELL_TEST="${RUN_MAXWELL_TEST:-1}"
RUN_GARCIA_TEST="${RUN_GARCIA_TEST:-1}"

mkdir -p "${LOG_ROOT}"
cd "${SCRIPT_DIR}"

echo "BASELINE_ROOT=${BASELINE_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "SPLITS=${SPLITS}"
echo "RUN_MEGA_TEST=${RUN_MEGA_TEST}"
echo "RUN_MAXWELL_TEST=${RUN_MAXWELL_TEST}"
echo "RUN_GARCIA_TEST=${RUN_GARCIA_TEST}"
echo

# --- Mega test (split-wise) ---
if [[ "${RUN_MEGA_TEST}" == "1" ]]; then
    for split in ${SPLITS}; do
        CKPT="${BASELINE_ROOT}/pth_split_${split}/baseline.pth"
        LOG="${LOG_ROOT}/mega_test_split_${split}.log"

        echo "###########################################################"
        echo "# Mega test: split ${split}                               "
        echo "# ckpt : ${CKPT}                                         "
        echo "# log  : ${LOG}                                          "
        echo "###########################################################"

        "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${CKPT}" Mega_test "${split}" BASELINE 2>&1 | tee "${LOG}"
        echo
    done
fi

# --- Maxwell ensemble test ---
if [[ "${RUN_MAXWELL_TEST}" == "1" ]]; then
    LOG="${LOG_ROOT}/maxwell_test.log"
    echo "###############################################################"
    echo "# Maxwell ensemble test (baseline models)                     "
    echo "# log  : ${LOG}                                              "
    echo "###############################################################"
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" Maxwell_test BASELINE 2>&1 | tee "${LOG}"
    echo
fi


echo "All baseline tests finished."
~
~
~

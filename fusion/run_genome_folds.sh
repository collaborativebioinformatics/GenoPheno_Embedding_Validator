#!/usr/bin/env bash
# Genome side of the fusion inputs, fold by fold, for one SynthTAN cohort:
#   step 1 (graph from the fold's training patients) -> edge eligibility filter on the training patients ->
#   step 3 (node2vec) -> per-patient node2vec means; with EVO2=1 also step 4a (Evo 2 fine-tuned on the node2vec
#   embeddings, objective "next", no labels) -> per-patient Evo 2 embeddings.
# Writes fold<k>/genome_embeddings_node2vec.csv (and genome_embeddings_evo2.csv) into the cohort's fusion folder.
#
# Usage:
#   FUSION_DIR=fusion_inputs/cohort_n200 HAPLO_INPUT=haplograph_input/cohort_n200 WORK=work/cohort_n200 \
#     bash fusion/run_genome_folds.sh
#   EVO2=1 MOCK_EVO2=0 PY=$HOME/evo2-env/bin/python FUSION_DIR=... HAPLO_INPUT=... WORK=... \
#     bash fusion/run_genome_folds.sh
#
# Variables
#   FUSION_DIR   cohort folder from fusion/make_fusion_inputs.py
#   HAPLO_INPUT  the same cohort's folder from fusion/synthtan_to_haplograph.py (checked: same builder files)
#   WORK         working folder for THIS conversion only (fold graphs, step-1 cache); new, or made by this
#                script from the same HAPLO_INPUT
#   PY           python with numpy, pandas, scipy, torch (and evo2 for EVO2=1)      (default: python)
#   DEVICE       cpu or cuda for step 3 (step 4a uses the GPU that evo2 picks)    (default: cpu)
#   FOLDS        space-separated folds                                             (default: all)
#   EVO2         1 = also run step 4a and export Evo 2 embeddings                   (default: 0)
#   EVO2_ARGS    extra options for 4a_finetune_evo2_embeddings.py, e.g. "--epochs 1 --train-blocks 4"
#                (the driver sets --method, --objective, --graph-dir and --train-individuals itself)
#   MOCK_EVO2    1 = use the fake evo2 in fusion/tests/mock_evo2 (plumbing test; outputs go to *_MOCK files)
#
# Every product records what it was made from and its own checksum (<product>.prov). A rerun skips a product whose
# record matches both, and stops with an error otherwise (remove the product and its .prov to recompute).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
DEVICE="${DEVICE:-cpu}"
EVO2="${EVO2:-0}"
EVO2_ARGS="${EVO2_ARGS:-}"
MOCK_EVO2="${MOCK_EVO2:-0}"
: "${FUSION_DIR:?set FUSION_DIR}" "${HAPLO_INPUT:?set HAPLO_INPUT}" "${WORK:?set WORK}"
FUSION_DIR="$(cd "$FUSION_DIR" && pwd)"
HAPLO_INPUT="$(cd "$HAPLO_INPUT" && pwd)"
mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"

die() { echo "error: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
json() { "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]])" "$1" "$2"; }

read -r -a EVO2_ARGV <<< "$EVO2_ARGS"
for a in "${EVO2_ARGV[@]}"; do
  case "$a" in
    --method*|--objective*|--graph-dir*|--train-individuals*|--sample-map*)
      die "EVO2_ARGS must not set $a (the driver sets it)" ;;
  esac
done
if [ "$MOCK_EVO2" = "1" ]; then
  export PYTHONPATH="$REPO/fusion/tests/mock_evo2${PYTHONPATH:+:$PYTHONPATH}"
  TAG="_MOCK"; MOCK_FLAG=(--mock)
else
  TAG=""; MOCK_FLAG=()
fi

RUN=$(json "$FUSION_DIR/manifest.json" run_name)
NF=$(json "$FUSION_DIR/manifest.json" n_folds)
[ "$(json "$HAPLO_INPUT/conversion.json" sha256_builder_nodes)" = "$(json "$FUSION_DIR/manifest.json" sha256_builder_nodes)" ] ||
  die "$HAPLO_INPUT was converted from other builder files than $FUSION_DIR was made from (other cohort?)"
CONVERSION_SHA=$(sha "$HAPLO_INPUT/conversion.json")      # builder node + edge checksums, settings and counts
FOLDS_SHA=$(sha "$FUSION_DIR/folds.csv")
EXPORTER_SHA=$(sha "$REPO/fusion/genome_patient_embeddings.py")
CHROMS=$(cat "$HAPLO_INPUT/chroms.txt")
FOLDS="${FOLDS:-$(seq 0 $((NF - 1)))}"

# a WORK folder belongs to one conversion: stamp it on first use; refuse unstamped non-empty folders
STAMP="$WORK/conversion.sha256"
if [ -f "$STAMP" ]; then
  [ "$(cat "$STAMP")" = "$CONVERSION_SHA" ] || die "$WORK was made from another genome input; use a separate WORK"
elif [ -n "$(ls -A "$WORK")" ]; then
  die "$WORK is not empty and was not made by this script"
else
  echo "$CONVERSION_SHA" > "$STAMP"
fi

if [ "$EVO2" = "1" ]; then
  # same import context as step 4a (a script in $REPO): run from $REPO
  EVO2_INFO=$(cd "$REPO" && "$PY" -c "
import evo2, importlib.util
from importlib import metadata
try: v = metadata.version('evo2')
except metadata.PackageNotFoundError: v = 'unknown'
# the real package has an evo2.models module and no IS_MOCK; anything else counts as a stand-in
real = importlib.util.find_spec('evo2.models') is not None and not getattr(evo2, 'IS_MOCK', False)
print(int(not real), v, evo2.__file__)") || die "cannot import evo2 with $PY"
  read -r EVO2_IS_MOCK EVO2_VERSION EVO2_FILE <<< "$EVO2_INFO"
  [ "$EVO2_IS_MOCK" = "$MOCK_EVO2" ] ||
    die "evo2 resolves to $EVO2_FILE (mock=$EVO2_IS_MOCK) but MOCK_EVO2=$MOCK_EVO2"
  echo "evo2: $EVO2_FILE (version $EVO2_VERSION, mock=$EVO2_IS_MOCK)"
fi

# make_step <prov> <product> [companions...] -- <command...>
#   run the command unless the product exists with the same recipe and unchanged checksums of product+companions
make_step() {
  local prov="$1"; shift
  local files=()
  while [ "$1" != "--" ]; do files+=("$1"); shift; done
  shift
  local product="${files[0]}"
  local rec="$product.prov"
  if [ -e "$product" ]; then
    [ -f "$rec" ] || die "$product exists without a record ($rec); remove it to recompute"
    local want; want="$prov"$'\n'"$(for f in "${files[@]}"; do echo "$(basename "$f")=$(sha "$f" 2>/dev/null || echo missing)"; done)"
    [ "$(cat "$rec")" = "$want" ] && return 0
    die "$product was made from other inputs or settings, or was changed; remove it (and $rec) to recompute"
  fi
  rm -f "$rec"
  "$@"
  for f in "${files[@]}"; do [ -e "$f" ] || die "$f was not written"; done
  { printf '%s\n' "$prov"; for f in "${files[@]}"; do echo "$(basename "$f")=$(sha "$f")"; done; } > "$rec.tmp"
  head -c -1 "$rec.tmp" > "$rec" && rm "$rec.tmp"      # same text as $want (no trailing newline)
}

step1() { (cd "$REPO" && "$PY" 1_build_graph.py --local-dir "$HAPLO_INPUT" --chroms "$CHROMS" --edges all \
            --train-individuals "$TR" --workdir "$WORK" --out "$G" &&
          "$PY" fusion/filter_graph_support.py --graph-dir "$G"); }
step3() { (cd "$REPO" && "$PY" 3_node_embeddings.py --method node2vec --graph-dir "$G" --train-individuals "$TR" \
            --device "$DEVICE"); }
step4a() { (cd "$REPO" && "$PY" 4a_finetune_evo2_embeddings.py --method node2vec --objective next --graph-dir "$G" \
             --train-individuals "$TR" "${EVO2_ARGV[@]}"); }
export_node2vec() { "$PY" "$REPO/fusion/genome_patient_embeddings.py" --graph-dir "$G" --method node2vec \
                      --expect-train "$TR" --expect-patients "$FUSION_DIR/folds.csv" --out "$OUT"; }
export_evo2() { "$PY" "$REPO/fusion/genome_patient_embeddings.py" --graph-dir "$G" --method evo2 --evo2-pt "$PT" \
                  --expect-train "$TR" --expect-patients "$FUSION_DIR/folds.csv" "${MOCK_FLAG[@]}" --out "$OUT"; }

for k in $FOLDS; do
  TR="$FUSION_DIR/fold$k/${RUN}_fold${k}_train.txt"
  G="$WORK/graph_fold$k"
  [ -f "$TR" ] || die "missing $TR"
  TR_SHA=$(sha "$TR")
  echo "== $RUN fold $k"

  make_step "conversion=$CONVERSION_SHA chroms=$CHROMS train=$TR_SHA step1=--edges,all filter=min_support3" \
    "$G/graph_filter.json" "$G/graph.npz" "$G/nodes.csv.gz" "$G/carriers.npz" "$G/individuals.txt" \
    "$G/train_individuals.txt" -- step1
  diff -q <(sort "$G/train_individuals.txt") <(sort "$TR") > /dev/null ||
    die "$G was built from other training patients than $TR"

  GRAPH_SHA=$(sha "$G/graph.npz")
  make_step "graph=$GRAPH_SHA train=$TR_SHA step3=node2vec,defaults,seed0" \
    "$G/node_emb_node2vec.npy" "$G/node_emb_node2vec.json" -- step3
  EMB_SHA=$(sha "$G/node_emb_node2vec.npy")

  OUT="$FUSION_DIR/fold$k/genome_embeddings_node2vec.csv"
  make_step "node_emb=$EMB_SHA graph=$GRAPH_SHA carriers=$(sha "$G/carriers.npz") train=$TR_SHA folds=$FOLDS_SHA exporter=$EXPORTER_SHA" \
    "$OUT" "${OUT%.csv}.json" -- export_node2vec

  if [ "$EVO2" = "1" ]; then
    PT="$G/evo2_next_node2vec.pt"
    make_step "node_emb=$EMB_SHA train=$TR_SHA evo2=$EVO2_FILE,$EVO2_VERSION,mock$EVO2_IS_MOCK args=$EVO2_ARGS" \
      "$PT" -- step4a
    OUT="$FUSION_DIR/fold$k/genome_embeddings_evo2$TAG.csv"
    make_step "model=$(sha "$PT") node_emb=$EMB_SHA train=$TR_SHA folds=$FOLDS_SHA evo2=$EVO2_FILE,mock$EVO2_IS_MOCK exporter=$EXPORTER_SHA" \
      "$OUT" "${OUT%.csv}.json" -- export_evo2
  fi
done
echo "done: $FUSION_DIR"

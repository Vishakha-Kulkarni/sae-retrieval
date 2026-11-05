#!/bin/bash
# One-time Pyserini environment for scripts/nscc/bm25.pbs. Run on a login node (it downloads), from the repo:
#   bash scripts/nscc/bm25_setup.sh
# Builds a venv on the PyTorch module, so torch comes from the module and pip skips its slow download;
# takes Java from the NSCC module (Pyserini asks for 21; newer JVMs run its bytecode and BM25 scores do not
# depend on the JVM), or with JAVA_MODULE= unpacks a JDK 21 into $JDK; downloads the prebuilt
# msmarco-v1-passage index into $PYSERINI_CACHE and searches one query. Rerunning reuses whatever is
# already there. bm25.pbs uses the same defaults.
set -euo pipefail
PYTORCH_MODULE=${PYTORCH_MODULE:-pytorch/2.10.0-py3-cu12.6}
JAVA_MODULE=${JAVA_MODULE-java/26.0.1-jdk}
VENV=${VENV:-$HOME/pyserini-env}
JDK=${JDK:-$HOME/scratch/sae/jdk-21}
export PYSERINI_CACHE=${PYSERINI_CACHE:-$HOME/scratch/sae/pyserini}

module purge
module load "$PYTORCH_MODULE"

if [ -n "$JAVA_MODULE" ]; then
    module load "$JAVA_MODULE"  # sets JAVA_HOME, which pyjnius uses to find libjvm
else
    if ! "$JDK/bin/java" -version 2>&1 | grep -q 'version "21'; then
        mkdir -p "$JDK"
        curl -fSL --retry 3 https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse \
            | tar -xz -C "$JDK" --strip-components=1
    fi
    export JAVA_HOME=$JDK PATH=$JDK/bin:$PATH
fi
java -version 2>&1 | head -n 1

[ -x "$VENV/bin/python" ] || python -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -c 'import sys; assert sys.version_info >= (3, 12), f"Pyserini 2.x needs Python >= 3.12, got {sys.version}"'
"$VENV/bin/pip" install pyserini

mkdir -p "$PYSERINI_CACHE"
"$VENV/bin/python" - <<'EOF'
import torch
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
assert searcher.num_docs == 8841823, f'expected 8,841,823 MS MARCO passages, index has {searcher.num_docs}'
hits = searcher.search('what is a sparse autoencoder', k=3)
print('torch', torch.__version__, '| index', searcher.num_docs, 'passages | top hits',
      [(h.docid, round(h.score, 2)) for h in hits])
print('OK: Pyserini ready; next: qsub -P $PROJECT_ID scripts/nscc/bm25.pbs')
EOF

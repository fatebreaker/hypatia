#!/bin/bash
# Phase F: fetch the PMC full text cited by Open Targets evidence.
#
# Runs pmc_build_keylist.py to produce keys.txt, then pulls each .txt with a
# pool of parallel curls. ~1.87M small objects, so:
#   - shard output by PMCID prefix; 1.87M files in one dir would cripple the FS
#   - skip anything already on disk, so re-running resumes for free
#   - -f so 4xx doesn't leave truncated files behind
#
# Plain text only. PDFs and figure images are ~200x the bytes for no gain to a
# retrieval index.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"   # sets BASE, LOG, PY (never hardcode scratch)
PMC=$BASE/pmc
S3=https://pmc-oa-opendata.s3.amazonaws.com
JOBS=${JOBS:-48}

mkdir -p $LOG $PMC/txt
exec > >(tee -a $LOG/fetch_pmc.log) 2>&1

echo "=================================================="
echo "Phase F (PMC full text) start: $(date)"
echo "=================================================="

if [ ! -s "$PMC/keys.txt" ]; then
  echo "--- building key list"
  # $PROJECT_ROOT, not $BASE: env.sh sets BASE to the DATA root, which has no
  # scripts/ dir. This exited 1 before fetching anything.
  $PY "$PROJECT_ROOT/scripts/pmc_build_keylist.py" || exit 1
fi

TOTAL=$(wc -l < $PMC/keys.txt)
echo "--- $TOTAL keys to fetch, $JOBS parallel workers"

# One article: shard by the first 6 chars of the PMCID (~1000 dirs).
cat > $PMC/get_one.sh <<'EOF'
#!/bin/bash
key="$1"; PMC="$2"; S3="$3"
base=$(basename "$key")                 # PMC10000000.1.txt
id=${base%%.*}                          # PMC10000000
shard=${id:0:6}                         # PMC100
dest="$PMC/txt/$shard/$base"
[ -s "$dest" ] && exit 0
mkdir -p "$PMC/txt/$shard"
curl -fsS --max-time 120 --retry 3 --retry-delay 2 \
     -o "$dest.part" "$S3/$key" && mv "$dest.part" "$dest" || { rm -f "$dest.part"; exit 1; }
EOF
chmod +x $PMC/get_one.sh

echo "--- starting fetch $(date)"
xargs -a $PMC/keys.txt -P $JOBS -I{} $PMC/get_one.sh {} "$PMC" "$S3" 2>>$LOG/pmc_errors.log

GOT=$(find $PMC/txt -name "*.txt" -type f | wc -l)
echo "=================================================="
echo "Phase F complete: $(date)"
echo "fetched $GOT / $TOTAL"
echo "size: $(du -sh $PMC/txt 2>/dev/null | cut -f1)"
echo "errors: $(wc -l < $LOG/pmc_errors.log 2>/dev/null || echo 0)"
echo "=================================================="

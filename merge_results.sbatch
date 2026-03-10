#!/bin/bash
#SBATCH --job-name=gate_merge
#SBATCH --partition=128x24
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Merge results from GATE CZT simulation array jobs + run diagnostics

# SLURM copies scripts to spool dir, so resolve project dir from env or home
GATE_SIM_DIR="${GATE_SIM_DIR:-$HOME/gate_sim}"

echo "============================================================"
echo "GATE CZT Merge Results + Diagnostics"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Project dir: ${GATE_SIM_DIR}"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gate_env

echo "Python: $(which python)"

cd "${GATE_SIM_DIR}"
echo "Working dir: $(pwd)"

# ── Step 1: Merge results ──
echo ""
echo "Step 1: Merging simulation results..."
python "${GATE_SIM_DIR}/merge_results.py" --output-dir "${GATE_SIM_DIR}/output"
MERGE_EXIT=$?
echo "Merge finished with exit code: ${MERGE_EXIT}"

# ── Step 2: Run diagnostic 3D plots ──
echo ""
echo "Step 2: Generating diagnostic plots..."
if [[ -f "${GATE_SIM_DIR}/diagnostic_3d_plot.py" ]]; then
    python "${GATE_SIM_DIR}/diagnostic_3d_plot.py"
    DIAG_EXIT=$?
    echo "Diagnostics finished with exit code: ${DIAG_EXIT}"
else
    echo "Warning: diagnostic_3d_plot.py not found, skipping"
    DIAG_EXIT=0
fi

# ── Step 3: Run standard analysis ──
echo ""
echo "Step 3: Running standard analysis..."
MERGED_PHSP="${GATE_SIM_DIR}/output/merged/phsp_detector_merged.root"
if [[ -f "${GATE_SIM_DIR}/analyze_phsp.py" ]] && [[ -f "${MERGED_PHSP}" ]]; then
    python "${GATE_SIM_DIR}/analyze_phsp.py" --input "${MERGED_PHSP}" --output-dir "${GATE_SIM_DIR}/output/analysis"
    ANALYSIS_EXIT=$?
    echo "Analysis finished with exit code: ${ANALYSIS_EXIT}"
else
    echo "Warning: analyze_phsp.py or merged phsp file not found, skipping"
    ANALYSIS_EXIT=0
fi

# ── Step 4: Post results summary to Slack ──
WEBHOOK="${AUTOPILOT_SLACK_WEBHOOK:-}"
if [[ -n "${WEBHOOK}" ]]; then
    echo ""
    echo "Step 4: Posting results to Slack..."

    # Collect diagnostic stats
    STATS_FILE="${GATE_SIM_DIR}/output/merged/merged_statistics.txt"
    STATS_SUMMARY=""
    if [[ -f "${STATS_FILE}" ]]; then
        STATS_SUMMARY="$(cat "${STATS_FILE}")"
    fi

    # Count output files
    N_PLOTS=$(ls "${GATE_SIM_DIR}/output/diagnostics/"*.png 2>/dev/null | wc -l)
    N_ANALYSIS=$(ls "${GATE_SIM_DIR}/output/analysis/"*.png 2>/dev/null | wc -l)

    # Build Slack message
    python3 -c "
import json, urllib.request, sys, os, socket

webhook = '${WEBHOOK}'
stats = '''${STATS_SUMMARY}'''
n_plots = ${N_PLOTS}
n_analysis = ${N_ANALYSIS}
merge_ok = ${MERGE_EXIT} == 0
diag_ok = ${DIAG_EXIT} == 0
analysis_ok = ${ANALYSIS_EXIT} == 0
host = socket.gethostname()

all_ok = merge_ok and diag_ok and analysis_ok
emoji = ':tada:' if all_ok else ':warning:'
status_line = 'Everything looks good!' if all_ok else 'Some steps had issues — check logs.'

body = f'''{emoji} *Post-processing is done!* {status_line}

:file_folder: *Results Merge:* {'Passed — all job outputs combined' if merge_ok else 'FAILED — check merge logs'}
:bar_chart: *3D Diagnostics:* {'Generated {n_plots} plots' if diag_ok else 'FAILED'}
:chart_with_upwards_trend: *Analysis:* {'Generated {n_analysis} plots (energy spectrum, hit maps, FWHM)' if analysis_ok else 'Skipped or failed'}

:clipboard: *Run Statistics:*
\`\`\`
{stats[:1500]}
\`\`\`

:framed_picture: *Diagnostic Visualizations Created:*
1. *3D Energy Heatmap* — photon interactions colored by energy deposit
2. *3D Particle Map* — gammas (gold) vs secondary electrons (blue)
3. *Full Geometry View* — source, collimator, and detector in 3D
4. *6-Panel Summary* — energy spectrum, hit maps, depth profile, particle breakdown

:open_file_folder: *Where to find everything:*
\`~/projects/gate_sim/output/diagnostics/\` — 3D plots
\`~/projects/gate_sim/output/analysis/\` — spectral analysis
\`~/projects/gate_sim/output/merged/\` — merged ROOT files for further analysis
'''

payload = json.dumps({
    'blocks': [
        {'type': 'header', 'text': {'type': 'plain_text', 'text': 'CZT Simulation — Results Ready', 'emoji': True}},
        {'type': 'section', 'text': {'type': 'mrkdwn', 'text': body}},
        {'type': 'context', 'elements': [{'type': 'mrkdwn', 'text': f':microscope: autopilot post-processing | {host} | {os.popen(\"date +%H:%M:%S\").read().strip()}'}]},
    ]
}).encode('utf-8')

req = urllib.request.Request(webhook, data=payload, headers={'Content-Type': 'application/json'})
try:
    urllib.request.urlopen(req, timeout=10)
    print('Slack notification sent')
except Exception as e:
    print(f'Slack post failed: {e}', file=sys.stderr)
"

    # ── Step 4b: Upload diagnostic images to Slack ──
    SLACK_TOKEN="${AUTOPILOT_SLACK_TOKEN:-}"
    SLACK_CHANNEL="${AUTOPILOT_SLACK_CHANNEL:-}"

    if [[ -n "${SLACK_TOKEN}" && -n "${SLACK_CHANNEL}" ]]; then
        echo "Step 4b: Uploading diagnostic images to Slack..."
        for img in "${GATE_SIM_DIR}/output/diagnostics/"*.png "${GATE_SIM_DIR}/output/analysis/"*.png; do
            [[ -f "$img" ]] || continue
            FNAME="$(basename "$img")"
            TITLE="$(basename "$img" .png | tr '_' ' ')"
            FSIZE="$(stat -c%s "$img")"
            echo "  Uploading: ${FNAME} (${FSIZE} bytes)"

            # Step A: Get upload URL
            UPLOAD_URL_RESP=$(curl -s -X POST \
                -H "Authorization: Bearer ${SLACK_TOKEN}" \
                -F "filename=${FNAME}" \
                -F "length=${FSIZE}" \
                "https://slack.com/api/files.getUploadURLExternal" 2>&1)
            UPLOAD_URL=$(echo "$UPLOAD_URL_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('upload_url',''))" 2>/dev/null)
            FILE_ID=$(echo "$UPLOAD_URL_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('file_id',''))" 2>/dev/null)

            if [[ -z "$UPLOAD_URL" || -z "$FILE_ID" ]]; then
                echo "    Failed to get upload URL — skipping remaining uploads"
                break
            fi

            # Step B: Upload file content
            curl -s -X POST -F "file=@${img}" "$UPLOAD_URL" >/dev/null 2>&1

            # Step C: Complete upload and share to channel
            COMPLETE_RESP=$(curl -s -X POST \
                -H "Authorization: Bearer ${SLACK_TOKEN}" \
                -H "Content-Type: application/json" \
                -d "{\"files\":[{\"id\":\"${FILE_ID}\",\"title\":\"${TITLE}\"}],\"channel_id\":\"${SLACK_CHANNEL}\",\"initial_comment\":\":bar_chart: ${TITLE}\"}" \
                "https://slack.com/api/files.completeUploadExternal" 2>&1)
            if echo "$COMPLETE_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('ok') else 1)" 2>/dev/null; then
                echo "    Uploaded successfully"
            else
                echo "    Upload failed: ${COMPLETE_RESP}"
                break
            fi
        done
    else
        echo "Step 4b: Skipping image uploads (AUTOPILOT_SLACK_TOKEN or AUTOPILOT_SLACK_CHANNEL not set)"
    fi
fi

# ── Step 5: Publish results to GitHub Pages ──
echo ""
echo "Step 5: Publishing results to GitHub Pages..."
DOCS_RESULTS="${GATE_SIM_DIR}/docs/results"
mkdir -p "${DOCS_RESULTS}"

# Copy analysis and diagnostic PNGs
COPIED=0
for img in "${GATE_SIM_DIR}/output/analysis/"*.png "${GATE_SIM_DIR}/output/diagnostics/"*.png; do
    [[ -f "$img" ]] || continue
    cp "$img" "${DOCS_RESULTS}/"
    COPIED=$((COPIED + 1))
done
echo "Copied ${COPIED} images to docs/results/"

# Copy merged statistics if available
if [[ -f "${GATE_SIM_DIR}/output/merged/merged_statistics.txt" ]]; then
    cp "${GATE_SIM_DIR}/output/merged/merged_statistics.txt" "${DOCS_RESULTS}/"
fi

# Commit and push to GitHub
if [[ ${COPIED} -gt 0 ]]; then
    cd "${GATE_SIM_DIR}"
    git add docs/results/
    git commit -m "Update simulation results ($(date +%Y-%m-%d), ${COPIED} plots)"
    git push origin main
    PAGES_EXIT=$?
    if [[ ${PAGES_EXIT} -eq 0 ]]; then
        echo "Results published — site will update at https://maxteicheiraucsc.github.io/gate_sim/"
    else
        echo "Warning: git push failed (exit ${PAGES_EXIT}). Results saved locally in docs/results/"
    fi
else
    echo "No images to publish"
fi

# Exit with worst code
FINAL_EXIT=0
[[ ${MERGE_EXIT} -ne 0 ]] && FINAL_EXIT=${MERGE_EXIT}
[[ ${DIAG_EXIT} -ne 0 ]] && FINAL_EXIT=${DIAG_EXIT}
echo ""
echo "All steps complete. Final exit code: ${FINAL_EXIT}"
exit ${FINAL_EXIT}

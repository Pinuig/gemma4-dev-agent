#!/bin/bash
# Helper to create a clean zip for the Paper Track attachments
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Creating patch_examples.zip ..."
mkdir -p /tmp/patch_examples
cp examples/sample_trajectory.json /tmp/patch_examples/
# Add a few more real before/after examples if you have them
(cd /tmp && zip -r "$ROOT/patch_examples.zip" patch_examples)
echo "Done: patch_examples.zip"

echo "Paper PDF suggestion:"
echo "  Convert paper/paper.md to PDF (pandoc or Overleaf) and name it gemma4_devagent_paper.pdf"
echo ""
echo "Then on Kaggle Paper Track:"
echo "  1. New Writeup"
echo "  2. Paste content of paper/paper.md (or upload the PDF)"
echo "  3. Attach gemma4_devagent_paper.pdf and patch_examples.zip"
echo "  4. Link this GitHub repo + public notebook + YouTube demo"
echo "  5. Publish before 12 Nov 2026 23:59 UTC"

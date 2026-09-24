#!/bin/sh
# Copies the pages in docs/wiki/ into the project's GitHub wiki and pushes them.
# The wiki is a separate git repository, so this needs your own GitHub login.
#
# Usage (from the repository root): sh docs/wiki/publish.sh
set -e
WIKI_URL=${WIKI_URL:-https://github.com/thomasmd321/AI_Programming_with_Python_Project_1.wiki.git}
SRC=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

git clone --quiet "$WIKI_URL" "$TMP/wiki"
# Every page except this folder's README (which describes the folder itself).
for page in "$SRC"/*.md; do
    [ "$(basename "$page")" = README.md ] || cp "$page" "$TMP/wiki/"
done
cd "$TMP/wiki"
git add -A
if git diff --cached --quiet; then
    echo "The wiki is already up to date."
else
    git commit --quiet -m "Update wiki from docs/wiki"
    git push --quiet
    echo "Published: https://github.com/thomasmd321/AI_Programming_with_Python_Project_1/wiki"
fi

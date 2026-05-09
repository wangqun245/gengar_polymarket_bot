#!/bin/bash
# PolyBot Startup Script
# Handles: tmux, caffeinate
# Usage: ./start.sh

set -e

PROJ_DIR="$HOME/gengar_bot/gengar_polybot"
SESSION_NAME="polybot"

echo "PolyBot Startup"
echo "================"

# 1. Kill existing tmux session if running
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session '$SESSION_NAME'..."
    tmux kill-session -t "$SESSION_NAME"
    echo "Old session killed"
else
    echo "No existing tmux session"
fi

# 2. Launch tmux with caffeinate + bot
echo "Starting tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" -c "$PROJ_DIR" \
    "caffeinate -i python bot.py; echo 'Bot exited. Press enter to close.'; read"

echo ""
echo "================"
echo "PolyBot running in tmux"
echo ""
echo "   Attach:  tmux attach -t $SESSION_NAME"
echo "   Detach:  Ctrl+B then D"
echo "   Stop:    tmux attach, then Ctrl+C"
echo "   Status:  tmux ls"
echo "================"

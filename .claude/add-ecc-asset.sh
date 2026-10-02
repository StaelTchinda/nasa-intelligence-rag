#!/usr/bin/env bash

set -e

REPO="affaan-m/ECC"
BRANCH="main"

show_help() {
    echo "Usage: $0 <type> <name1,name2,...>"
    echo ""
    echo "Arguments:"
    echo "  type   The type of asset (skill, agent, command, rule)"
    echo "  name   A comma-separated list of asset names"
    echo ""
    echo "Options:"
    echo "  --help, -h   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 skill continuous-learning-v2,django-tdd"
    echo "  $0 agent code-reviewer,planner"
    echo "  $0 command verify,plan"
    exit 0
}

# Check for help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
fi

# Check if both arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments."
    echo "Run '$0 --help' for usage instructions."
    exit 1
fi

ASSET_TYPE="$1"
ASSET_NAMES="$2"

# Map asset type to remote paths and determine if it's a directory or a file
case "$ASSET_TYPE" in
    skill|skills)
        TARGET_BASE=".claude/skills"
        REMOTE_BASE="skills"
        IS_DIR=true
        ;;
    agent|agents)
        TARGET_BASE=".claude/agents"
        REMOTE_BASE="agents"
        IS_DIR=false
        ;;
    command|commands)
        TARGET_BASE=".claude/commands"
        REMOTE_BASE="commands"
        IS_DIR=false
        ;;
    rule|rules)
        TARGET_BASE=".claude/rules"
        REMOTE_BASE="rules"
        IS_DIR=false
        ;;
    *)
        echo "Error: Unknown asset type '$ASSET_TYPE'."
        echo "Supported types: skill, agent, command, rule"
        exit 1
        ;;
esac

# Create temporary files and directories, cleanup automatically on exit
TEMP_TAR=$(mktemp)
EXTRACT_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_TAR" "$EXTRACT_DIR"' EXIT

echo "Fetching ECC repository..."
curl -sL "https://github.com/$REPO/tarball/$BRANCH" -o "$TEMP_TAR"

echo "Extracting repository..."
tar xz -f "$TEMP_TAR" -C "$EXTRACT_DIR"

# Locate the root folder created inside the temp extraction directory
REPO_ROOT=$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)

# Split comma-separated names into an array
IFS=',' read -ra ASSET_ARRAY <<< "$ASSET_NAMES"

for ASSET_NAME in "${ASSET_ARRAY[@]}"; do
    # Trim leading/trailing whitespace
    ASSET_NAME=$(echo "$ASSET_NAME" | xargs)
    if [ -z "$ASSET_NAME" ]; then
        continue
    fi

    if [ "$IS_DIR" = true ]; then
        # Directory-based asset (Skills)
        SRC_PATH="$REPO_ROOT/$REMOTE_BASE/$ASSET_NAME"
        TARGET_DIR="$TARGET_BASE/$ASSET_NAME"

        if [ -d "$SRC_PATH" ]; then
            echo "Installing skill '$ASSET_NAME'..."
            mkdir -p "$TARGET_DIR"
            cp -R "$SRC_PATH/." "$TARGET_DIR/"
            echo "Successfully installed to $TARGET_DIR"
        else
            echo "Warning: Skill '$ASSET_NAME' not found in repository."
        fi
    else
        # File-based asset (Agents, Commands, Rules) - strip .md if user included it
        ASSET_NAME="${ASSET_NAME%.md}"
        SRC_PATH="$REPO_ROOT/$REMOTE_BASE/$ASSET_NAME.md"
        TARGET_DIR="$TARGET_BASE"

        if [ -f "$SRC_PATH" ]; then
            echo "Installing $ASSET_TYPE file '$ASSET_NAME.md'..."
            mkdir -p "$TARGET_DIR"
            cp "$SRC_PATH" "$TARGET_DIR/"
            echo "Successfully installed to $TARGET_DIR/$ASSET_NAME.md"
        else
            echo "Warning: $ASSET_TYPE '$ASSET_NAME.md' not found in repository."
        fi
    fi
done

echo "All requested assets processed successfully!"
#!/bin/bash

# This file provide some git related utilities, especially shortcut regarding github

# Note: this script uses return 1 to exit the script if an error occurs because if it is sourced,
#       it will not exit the shell

echo " -- enabling git utils --"

# Check that gh is avail and is logged in
if ! command -v gh &>/dev/null; then
    echo "gh is not installed" >&2
    return 1
fi

if ! gh auth status &>/dev/null; then
    echo "gh is not logged in, do 'gh auth login' to login" >&2
    return 1
fi

function current_gh_username {
    gh api user -q .login
}

gh_username=$(current_gh_username)
repo_name="$gh_username/Shamrock"
upstream_repo="Shamrock-code/Shamrock"
main_branch="main"

echo " -- gh username: $gh_username"

function sync_fork {
    echo " -- syncing fork $repo_name ($main_branch -> $main_branch) with upstream"
    gh repo sync $repo_name -b $main_branch
    echo " -- fetched latest changes from upstream"
    git fetch --all
}

function current_branch {
    echo $(git rev-parse --abbrev-ref HEAD)
}

# Update the fork + create branch up to date with main and push it to the fork
function new_branch {
    if [[ -z "$1" ]]; then
        echo "Usage: new_branch <branch_name>" >&2
        return 1
    fi
    sync_fork
    echo " -- checking out $main_branch"
    git checkout "$main_branch"
    echo " -- pulling $main_branch"
    git pull
    echo " -- creating new branch $1"
    git switch --create "$1"
    echo " -- pushing new branch $1 to origin"
    git push --set-upstream origin "$1"
}

# Open a PR from the fork (web interface)
function open_pr_web {
    echo " -- opening PR ($upstream_repo:$main_branch <= $gh_username:$(current_branch)) in web browser"
    gh pr create --repo $upstream_repo --base $main_branch --head $gh_username:$(current_branch) --web
}

# Open a PR from the fork (CLI)
function open_pr_cli {
    echo " -- opening PR ($upstream_repo:$main_branch <= $gh_username:$(current_branch)) in CLI"
    gh pr create --repo $upstream_repo --base $main_branch --head $gh_username:$(current_branch)
}

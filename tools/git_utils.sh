#!/bin/bash

# This file provide some git related utilities, especially shortcut regarding github

# Note: this script uses return 1 to exit the script if an error occurs because if it is sourced,
#       it will not exit the shell
#
# In principle the workflow with this script to do a PR would be just:
#   new_branch <branch name>
#   <do your changes>
#   <stage your changes, can be with git add -a but be carefull of what you commit>
#   git commit -m "[<section of the code>] <commit message>"
#   git push
#   open_pr_web

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
    gh api user -q .login || return
}

gh_username=$(current_gh_username)
repo_name="$gh_username/Shamrock"
upstream_repo="Shamrock-code/Shamrock"
main_branch="main"

echo " -- gh username: $gh_username"

function sync_fork {
    echo " -- syncing fork $repo_name ($main_branch -> $main_branch) with upstream"
    gh repo sync $repo_name -b $main_branch || return
    echo " -- fetched latest changes from upstream"
    git fetch --all || return
}

function current_branch {
    git rev-parse --abbrev-ref HEAD || return
}

# Update the fork + create branch up to date with main and push it to the fork
function new_branch {
    if [[ -z "$1" ]]; then
        echo "Usage: new_branch <branch_name>" >&2
        return 1
    fi
    sync_fork || return
    echo " -- checking out $main_branch"
    git checkout "$main_branch" || return
    echo " -- pulling $main_branch"
    git pull || return
    echo " -- creating new branch $1"
    git switch --create "$1" || return
    echo " -- pushing new branch $1 to origin"
    git push --set-upstream origin "$1" || return
}

# Fast github checkout from the github user:branch format
function gco {
    if [[ -z "$1" ]]; then
        echo "Usage: gco <github user:branch>, or gco <branch name>" >&2
        return 1
    fi
    echo " -- fetching origin"
    git fetch origin || return
    echo " -- checking out $1 => git checkout ${1#$gh_username:}"
    git checkout "${1#$gh_username:}" || return
    echo " -- pulling changes ..."
    git pull || return
    echo " -- done !"
}

# Open a PR from the fork (web interface)
function open_pr_web {
    echo " -- opening PR ($upstream_repo:$main_branch <= $gh_username:$(current_branch)) in web browser"
    gh pr create --repo $upstream_repo --base $main_branch --head $gh_username:$(current_branch) --web || return
}

# Open a PR from the fork (CLI)
function open_pr_cli {
    echo " -- opening PR ($upstream_repo:$main_branch <= $gh_username:$(current_branch)) in CLI"
    gh pr create --repo $upstream_repo --base $main_branch --head $gh_username:$(current_branch) || return
}

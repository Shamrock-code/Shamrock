import os
import subprocess

# start allow utf-8
title_wide = """
  █████████  █████   █████   █████████   ██████   ██████ ███████████      ███████      █████████  █████   ████
 ███░░░░░███░░███   ░░███   ███░░░░░███ ░░██████ ██████ ░░███░░░░░███   ███░░░░░███   ███░░░░░███░░███   ███░
░███    ░░░  ░███    ░███  ░███    ░███  ░███░█████░███  ░███    ░███  ███     ░░███ ███     ░░░  ░███  ███
░░█████████  ░███████████  ░███████████  ░███░░███ ░███  ░██████████  ░███      ░███░███          ░███████
 ░░░░░░░░███ ░███░░░░░███  ░███░░░░░███  ░███ ░░░  ░███  ░███░░░░░███ ░███      ░███░███          ░███░░███
 ███    ░███ ░███    ░███  ░███    ░███  ░███      ░███  ░███    ░███ ░░███     ███ ░░███     ███ ░███ ░░███
░░█████████  █████   █████ █████   █████ █████     █████ █████   █████ ░░░███████░   ░░█████████  █████ ░░████
 ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░ ░░░░░     ░░░░░ ░░░░░   ░░░░░    ░░░░░░░      ░░░░░░░░░  ░░░░░   ░░░░
"""

title_normal = """
███████╗██╗  ██╗ █████╗ ███╗   ███╗██████╗  ██████╗  ██████╗██╗  ██╗
██╔════╝██║  ██║██╔══██╗████╗ ████║██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝
███████╗███████║███████║██╔████╔██║██████╔╝██║   ██║██║     █████╔╝
╚════██║██╔══██║██╔══██║██║╚██╔╝██║██╔══██╗██║   ██║██║     ██╔═██╗
███████║██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝╚██████╗██║  ██╗
╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝
"""

title_small = """
╔═╗╦ ╦╔═╗╔╦╗╦═╗╔═╗╔═╗╦╔═
╚═╗╠═╣╠═╣║║║╠╦╝║ ║║  ╠╩╗
╚═╝╩ ╩╩ ╩╩ ╩╩╚═╚═╝╚═╝╩ ╩
"""
# end allow utf-8

abs_proj_dir = os.path.abspath(os.path.join(__file__, "../../.."))
abs_src_dir = os.path.join(abs_proj_dir, "src")


def is_a_precommit_call():
    return os.environ.get("PRE_COMMIT") == "1"


def print_tool_info(utility_name):
    if is_a_precommit_call():
        return

    col_cnt = 100

    try:
        col_cnt = os.get_terminal_size().columns
    except:
        print("Warn : couldn't get terminal size")

    if col_cnt > 112:
        print(title_wide)
    elif col_cnt > 69:  # nice
        print(title_normal)
    else:
        print(title_small)

    print("\033[1;90m" + "-" * col_cnt + "\033[0;0m\n")

    print("\033[1;34mCurrent tool      \033[0;0m: ", utility_name)
    print("\033[1;34mProject directory \033[0;0m: ", abs_proj_dir)
    print("\033[1;34mSource  directory \033[0;0m: ", abs_src_dir)

    print()

    try:
        r_log = subprocess.run(
            ["git", "log", "-n", "1", "--decorate=full"],
            capture_output=True,
            text=True,
        )
        if r_log.returncode != 0:
            raise RuntimeError("git log failed")

        str_git = r_log.stdout
        git_hash = str_git.split()[1]
        git_head = str_git[str_git.find("HEAD -> ") + 8 : str_git.find(")")]

        git_head = git_head.split(",")

        if len(git_head) == 1:
            git_head = "\033[1;92m" + git_head[0] + "\033[0;0m"
        else:
            git_head = (
                "\033[1;92m" + git_head[0] + "\033[0;0m , \033[1;91m" + git_head[0] + "\033[0;0m"
            )

        print("\033[1;34mGit status \033[0;0m: ")
        print("     \033[1;93mcommit \033[0;0m: ", git_hash)
        print("     \033[1;36mHEAD   \033[0;0m: ", git_head)
        print("     \033[1;31mmodified files\033[0;0m (since last commit):")
        print(os.popen('git diff-index --name-only HEAD -- | sed "s/^/        /g"').read())
        print("\033[1;90m" + "-" * col_cnt + "\033[0;0m\n")
    except Exception:  # noqa: BLE001
        print("Warn : couldn't get git status")

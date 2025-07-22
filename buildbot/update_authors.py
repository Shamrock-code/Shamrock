import glob
import subprocess
import sys

from lib.buildbot import *

print_buildbot_info("Authors check tool")

if len(sys.argv) > 1:
    print("Updating authors for files: ", sys.argv[1:])
    file_list = sys.argv[1:]
else:
    file_list = glob.glob(str(abs_src_dir) + "/**", recursive=True)

file_list.sort()

missing_doxygenfilehead = []

authorlist = []


def apply_mailmap(authors):
    ret = []
    for a in authors:

        try:
            cmd = f'git check-mailmap "{a['author']} <{a['email']}>"'
            output = subprocess.check_output(cmd, shell=True).decode()

            match = re.search(r"(.*) <(.*)>", output)
            if match is not None:
                app = {"author": match.group(1), "email": match.group(2)}
                if not app in ret:
                    ret.append(app)

        except subprocess.CalledProcessError as err:
            print(err)

    return ret


def get_author_list_from_blame(path):
    authors = []
    coauthors = []
    try:
        output = subprocess.check_output(R"git log " + path, shell=True).decode()
        for l in output.split("\n"):
            # if we get an answer like
            # Author: Timothée David--Cléris <tim.shamrock@proton.me>
            # extract the author name and email
            match = re.search(r"Author: (.*) <(.*)>", l)
            if match is not None:
                app = {"author": match.group(1), "email": match.group(2)}
                if not app in authors:
                    authors.append(app)

            match = re.search(r"Co-authored-by: (.*) <(.*)>", l)
            if match is not None:
                app = {"author": match.group(1), "email": match.group(2)}
                if not app in coauthors:
                    coauthors.append(app)

        # print(authors, coauthors)

    except subprocess.CalledProcessError as err:
        print(err)

    authors = apply_mailmap(authors)
    coauthors = apply_mailmap(coauthors)

    for a in coauthors:
        if not a in authors:
            authors.append(a)

    for a in authors:
        if not a in authorlist:
            authorlist.append(a)

    return authors


def extract_current_authors_from_header(splt, l_start, l_end):
    current_authors = []
    for l in splt[l_start : l_end + 1]:
        match = re.search(r"^\s*\*?\s*@author\s+(.+?)\s+\((.+?)\)", l)
        # print(match)
        if match:
            current_authors.append({"author": match.group(1), "email": match.group(2)})
    # Apply mailmap correction before returning
    current_authors = apply_mailmap(current_authors)

    for a in current_authors:
        if not a in authorlist:
            authorlist.append(a)

    return current_authors


def merge_author_lists(list_blame, list_other):
    """
    Merge two lists of authors (dicts with 'author' and 'email').
    Deduplicate by email. If duplicate, the entry from list_blame takes precedence.
    Returns a list of dicts with 'author', 'email', and 'from_blame' (True if from blame, False otherwise).
    """
    # Start with blame list
    merged = {a["email"]: {**a, "from_blame": True} for a in list_blame}
    # Add others only if not already present
    for a in list_other:
        if a["email"] not in merged:
            merged[a["email"]] = {**a, "from_blame": False}
    # Sort by author name
    merged_list = sorted(merged.values(), key=lambda x: x["author"])
    return merged_list


def get_doxstring(path, filename, current_authors=None):
    tmp = " * @file " + filename
    try:
        lst = merge_author_lists(get_author_list_from_blame(path), current_authors)
        for a in lst:
            suffix = " --no git blame--" if not a.get("from_blame", False) else ""
            tmp += f"\n * @author {a['author']} ({a['email']}){suffix}"
        # tmp+= (subprocess.check_output(R'git log --pretty=format:" * @author %aN (%aE)" '+path+' |sort |uniq',shell=True).decode())[:-1]
    except subprocess.CalledProcessError as err:
        print(err)

    return tmp


import difflib
import re


def print_diff(before, after, beforename, aftername):
    sys.stdout.writelines(
        difflib.context_diff(
            before.split("\n"), after.split("\n"), fromfile=beforename, tofile=aftername
        )
    )


def autocorect(source, filename, path):

    l_start = 0
    l_end = 0
    i = 0

    splt = source.split("\n")
    for l in splt:
        if l_start > 0:
            if not ("@author" in l):
                break
        if "@file" in l:
            l_start = i
        if "@author" in l:
            l_end = i
        i += 1

    if l_end == 0:
        l_end = l_start

    # Extract current authors from header
    current_authors = extract_current_authors_from_header(splt, l_start, l_end)
    # print(current_authors)

    new_splt = splt[:l_start]
    new_splt.append(get_doxstring(path, filename, current_authors))
    new_splt += splt[l_end + 1 :]

    new_src = ""
    for l in new_splt:
        new_src += l + "\n"
    new_src = new_src[:-1]

    do_replace = not (new_src == source)

    if do_replace:
        print("autocorect : ", filename)
        print_diff(source, new_src, filename, filename + " (corec)")

    return do_replace, new_src


def run_autocorect():

    errors = []

    for fname in file_list:

        if (not fname.endswith(".cpp")) and (not fname.endswith(".hpp")):
            continue

        if fname.endswith("version.cpp"):
            continue

        if "cmake/feature_test" in fname:
            continue
        if "src/tests/" in fname:
            continue
        if "exemple.cpp" in fname:
            continue
        if "godbolt.cpp" in fname:
            continue

        f = open(fname, "r")
        source = f.read()
        f.close()

        change, source = autocorect(source, os.path.basename(fname), fname)

        if change:
            print("autocorect : ", fname.split(abs_proj_dir)[-1])
            f = open(fname, "w")
            f.write(source)
            f.close()
            errors.append(fname.split(abs_proj_dir)[-1])

    return errors


missing_doxygenfilehead = run_autocorect()

print("--------------------------------")
print("Current author list:")
for a in authorlist:
    print(f"    {a['author']} <{a['email']}>")


if missing_doxygenfilehead:
    print("--------------------------------")

    # Write markdown report
    report = "## ❌ Authorship update required\n\n"
    report += (
        "The following files had their author headers updated by the author update script.\n\n"
    )
    report += "Please run the script again (python3 buildbot/update_authors.py) and commit these changes.\n\n"
    report += "**Note:** The list below is only partial. Only the first 10 files are shown.\n\n"
    for fname in missing_doxygenfilehead[:10]:
        report += f"- `{fname}`\n"
    with open("log_precommit_check-Authorship-update", "w") as f:
        f.write(report)

    sys.exit("authors were not up to date -> exiting")

import argparse

# usage : python clang_tidy_diff_report.py -i <input file> <output file>
parser = argparse.ArgumentParser(description="Clang tidy diff report generator")

parser.add_argument("-i", "--input", help="input file", required=True)
parser.add_argument("-f", "--fixes", help="input file", required=True)
parser.add_argument("-o", "--output", help="output file", required=True)

args = parser.parse_args()


def filter_double_newline(lines):
    ret = []

    is_last_line_only_return = False
    for l in lines:
        if l == "\n":
            if is_last_line_only_return:
                continue
            is_last_line_only_return = True
        else:
            is_last_line_only_return = False

        ret.append(l)

    return ret


def filter_non_warnings(lines):
    ret = []
    len_lines = len(lines)

    enable_skip = -1

    for i, l in enumerate(lines):
        if (" warnings generated." in l) or (" warning generated." in l):
            warn_cnt = int(l.split()[0])

            if lines[i + 1].startswith("Suppressed"):
                suppr_warn = int(lines[i + 1].split()[1])

                if suppr_warn == warn_cnt:
                    if lines[i + 2].startswith("Use -header-filter=.* to "):
                        enable_skip = 3
                    else:
                        enable_skip = 2
        if enable_skip <= 0:
            ret.append(l)
        else:
            print("skipped :", l)
            enable_skip -= 1

    return ret


f_in = open(args.input, "r")
no_double_newline = filter_double_newline(f_in.readlines())
lines_buf = filter_double_newline(filter_non_warnings(no_double_newline))

f_fixes = open(args.fixes, "r")
fixes_lines = f_fixes.readlines()

print(lines_buf)
print(fixes_lines)

no_relevant = not (lines_buf == ["No relevant changes found.\n"])
print_warn = (not (lines_buf == ["\n"])) and no_relevant
print_fixes = (not (fixes_lines == [])) and no_relevant


buf = "# Clang-tidy diff report\n"

if print_warn:
    buf += "```\n"

    for l in lines_buf:
        buf += l

    buf += "```\n"

if print_fixes:
    buf += "## Suggested changes\n"
    buf += "<details>\n"
    buf += "<summary>\n"
    buf += "Detailed changes :\n"
    buf += "</summary>\n"
    buf += " \n"
    buf += "```diff\n"
    for l in fixes_lines:
        buf += l
    buf += "```\n"
    buf += "\n"
    buf += "</details>\n"

if not print_warn and not print_fixes:
    buf += "\n"
    buf += "No relevant changes found.\n"
    buf += "**Well done!**  \n\n"
    buf += "You should now go back to your normal life and enjoy a hopefully sunny day "
    buf += "while waiting for the review.\n"

f_out = open(args.output, "w")
f_out.write(buf)
f_out.close()

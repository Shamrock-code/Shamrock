import argparse

# usage : python clang_tidy_diff_report.py -i <input file> <output file>
parser = argparse.ArgumentParser(description='Clang tidy diff report generator')

parser.add_argument('-i', '--input', help='input file', required=True)
parser.add_argument('-f', '--fixes', help='input file', required=True)
parser.add_argument('-o', '--output', help='output file', required=True)

args = parser.parse_args()



buf = "# Cland tidy diff report\n"
buf += "```\n"

f_in = open(args.input, 'r')
is_last_line_only_return = False
for l in f_in.readlines():

    if l == "\n":

        if is_last_line_only_return:
            continue

        is_last_line_only_return = True
    else:
        is_last_line_only_return = False

    buf += l
buf += "```\n"


f_fixes = open(args.fixes, 'r')
buf += "## Suggested changes\n"
buf += "<details>\n"
buf += "<summary>\n"
buf += "Detailed changes :\n"
buf += "</summary>\n"
buf += " \n"
buf += "```diff\n"
for l in f_fixes.readlines():
    buf += l
buf += "```\n"
buf += "\n"
buf += "</details>\n"

f_out = open(args.output, 'w')
f_out.write(buf)
f_out.close()

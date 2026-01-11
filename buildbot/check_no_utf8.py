import sys
from pathlib import Path

RED = "\033[31m"
RESET = "\033[0m"


def highlight_line(line: bytes):
    rendered = []
    carets = []

    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        # Fallback: show replacement characters
        text = line.decode("utf-8", errors="replace")

    i = 0
    for ch in text:
        b = ch.encode("utf-8")
        if len(b) > 1:
            rendered.append(f"{RED}{ch}{RESET}")
            carets.append("^" * len(ch))
        else:
            rendered.append(ch)
            carets.append(" ")

    return "".join(rendered), "".join(carets)


def main(files):
    failed = False

    for file in files:
        path = Path(file)
        try:
            data = path.read_bytes()
        except (OSError, UnicodeDecodeError) as e:
            print(f"{file}: error reading file ({e})")
            failed = True
            continue

        allow_utf8 = False
        for lineno, line in enumerate(data.splitlines(), 1):

            if "@author" in line.decode("utf-8"):
                continue
            if "Copyright" in line.decode("utf-8"):
                continue
            if "# start allow utf-8" in line.decode("utf-8"):
                allow_utf8 = True
            if "# end allow utf-8" in line.decode("utf-8"):
                allow_utf8 = False
            if "// start allow utf-8" in line.decode("utf-8"):
                allow_utf8 = True
            if "// end allow utf-8" in line.decode("utf-8"):
                allow_utf8 = False

            if allow_utf8:
                continue

            if any(b > 0x7F for b in line):
                rendered, carets = highlight_line(line)
                print(f"{file}:{lineno}: non-ASCII character(s) found")
                print(rendered)
                print(carets)
                failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

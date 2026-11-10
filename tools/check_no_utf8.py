import os
import re
import sys
from pathlib import Path

import shamrock_tool_banner

abs_proj_dir = os.path.join(os.path.dirname(__file__), "..")
abs_src_dir = os.path.join(abs_proj_dir, "src")


RED = "\033[31m"
RESET = "\033[0m"


def load_mailmap_names():
    """Names of the people listed in the .mailmap (canonical and commit names)"""
    mailmap = Path(__file__).resolve().parent.parent / ".mailmap"
    if not mailmap.exists():
        return []

    names = set()
    for line in mailmap.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0]
        # a name is the text before an email "<...>"
        for name in re.findall(r"([^<>]+?)\s*<[^<>]*>", line):
            name = name.strip()
            if name:
                names.add(name)

    # longest first, so that a name containing another one takes precedence
    return sorted(names, key=len, reverse=True)


def allowed_mask(text: str, names):
    """True for the characters of text that are part of a name of the .mailmap"""
    allowed = [False] * len(text)
    for name in names:
        start = text.find(name)
        while start != -1:
            for i in range(start, start + len(name)):
                allowed[i] = True
            start = text.find(name, start + len(name))
    return allowed


def highlight_line(text: str, allowed):
    rendered = []
    carets = []

    for ch, ok in zip(text, allowed):
        b = ch.encode("utf-8")
        if len(b) > 1 and not ok:
            rendered.append(f"{RED}{ch}{RESET}")
            carets.append("^" * len(ch))
        else:
            rendered.append(ch)
            carets.append(" ")

    return "".join(rendered), "".join(carets)


def main(files, names=None):
    failed = False

    if names is None:
        names = load_mailmap_names()

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
                text = line.decode("utf-8", errors="replace")
                allowed = allowed_mask(text, names)
                if all(ok or ord(ch) <= 0x7F for ch, ok in zip(text, allowed)):
                    # only names of people listed in the .mailmap
                    continue
                rendered, carets = highlight_line(text, allowed)
                print(f"{file}:{lineno}: non-ASCII character(s) found")
                print(rendered)
                print(carets)
                failed = True

    return 1 if failed else 0


# Unit tests, run with: pytest tools/check_no_utf8.py
# (non-ASCII written as escapes so that this file passes its own check)
NAMES = ["L\u00e9odasce Sewanou", "Timoth\u00e9e David--Cl\u00e9ris"]


def _check(tmp_path, content):
    f = tmp_path / "sample.cpp"
    f.write_text(content, encoding="utf-8")
    return main([str(f)], names=NAMES)


def test_ascii_only(tmp_path):
    assert _check(tmp_path, "// plain ascii\n") == 0


def test_mailmap_name_allowed(tmp_path):
    assert _check(tmp_path, "// With some sweet AMR from L\u00e9odasce Sewanou.\n") == 0


def test_stray_accent_rejected(tmp_path):
    assert _check(tmp_path, "// a stray accent: caf\u00e9\n") == 1


def test_mixed_line_highlights_only_non_name(tmp_path, capsys):
    line = "// mixed: Timoth\u00e9e David--Cl\u00e9ris is taking way too much caf\u00e9\n"
    assert _check(tmp_path, line) == 1
    out = capsys.readouterr().out
    # only the last \u00e9 (in caf\u00e9) is highlighted, not the ones of the name
    assert out.count(f"{RED}\u00e9{RESET}") == 1
    assert f"caf{RED}\u00e9{RESET}" in out


def test_allow_markers(tmp_path):
    content = "// start allow utf-8\n// caf\u00e9\n// end allow utf-8\n"
    assert _check(tmp_path, content) == 0
    assert _check(tmp_path, content + "// caf\u00e9\n") == 1


def test_allowed_mask():
    mask = allowed_mask("x No\u00e9 y", ["No\u00e9"])
    assert mask == [False, False, True, True, True, False, False]


if __name__ == "__main__":
    shamrock_tool_banner.print_tool_info("No UTF-8 in files (except for people names)")
    sys.exit(main(sys.argv[1:]))

import subprocess, sys, os, re

def test_no_old_prefixes():
    # crude but effective
    roots = ["experiments", "analysis", "tests", "run_phase1.py"]
    bad = re.compile(r"wbit_project/(results|analysis|experiments|tests)")
    for r in roots:
        if os.path.isdir(r):
            for dirpath, _, files in os.walk(r):
                for f in files:
                    if f.endswith(".py") or f.endswith(".md"):
                        p = os.path.join(dirpath, f)
                        with open(p, "r", encoding="utf-8") as fh:
                            txt = fh.read()
                        assert not bad.search(txt), f"Old prefix found in {p}"
        elif os.path.isfile(r):
            with open(r, "r", encoding="utf-8") as fh:
                txt = fh.read()
            assert not bad.search(txt), f"Old prefix found in {r}"

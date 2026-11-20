"""environments/Dockerfile inlines the pyproject pins (the Anyscale image builder forbids COPY)."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_has_no_copy_and_pins_match_pyproject():
    text = (ROOT / "environments/Dockerfile").read_text()
    assert not re.search(r"^\s*(COPY|ADD)\s", text, re.M)
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    want = set(proj["dependencies"] + proj["optional-dependencies"]["test"])
    got = set(re.findall(r"[A-Za-z0-9_.\-]+==[0-9][A-Za-z0-9_.+\-]*", text))
    assert want <= got, f"missing in Dockerfile: {sorted(want - got)}"

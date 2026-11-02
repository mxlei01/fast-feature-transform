"""The hashed lists release.yaml installs from agree with pyproject.toml and uv.lock.

make release-requirements writes them. These tests fail when a version changes in pyproject.toml, uv.lock or a .in
file while the lists still hold the old one, and when the wheel tests' list trusts a file uv.lock does not.
"""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / ".github" / "requirements"
COMPILED = ["build-constraints", "cibuildwheel", "cibuildwheel-tools", "twine"]
LISTS = [*COMPILED, "wheel-tests"]
STALE = "run make release-requirements"


def _name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(text: str) -> dict[str, set[str]]:
    """Map the name of each name==version line to its versions."""
    pins: dict[str, set[str]] = {}
    for match in re.finditer(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;\\]+)", text, re.MULTILINE):
        pins.setdefault(_name(match[1]), set()).add(match[2])
    return pins


def test_build_constraints_match_pyproject() -> None:
    pins = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["build-constraint-dependencies"]
    wanted = _pins("\n".join(pins))
    assert len(wanted) == len(pins), f"not every pin is name==version: {pins}"
    assert (REQUIREMENTS / "build-constraints.in").read_text().splitlines() == pins, STALE
    hashed = _pins((REQUIREMENTS / "build-constraints.txt").read_text())
    assert {name: hashed.get(name) for name in wanted} == wanted, STALE


def _hashes(text: str) -> dict[str, set[str]]:
    """Map the name of each requirement in a hashed list to its hashes."""
    hashes: dict[str, set[str]] = {}
    for requirement in text.replace("\\\n", " ").splitlines():
        if match := re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*==", requirement):
            hashes.setdefault(_name(match[1]), set()).update(re.findall(r"--hash=(sha256:[0-9a-f]+)", requirement))
    return hashes


def test_wheel_tests_match_uv_lock() -> None:
    versions: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    for package in tomllib.loads((ROOT / "uv.lock").read_text())["package"]:
        name = _name(package["name"])
        versions.setdefault(name, set()).add(package.get("version", ""))
        files = [package.get("sdist", {}), *package.get("wheels", [])]
        hashes.setdefault(name, set()).update(file["hash"] for file in files if "hash" in file)
    text = (REQUIREMENTS / "wheel-tests.txt").read_text()
    assert {name: pins for name, pins in _pins(text).items() if pins != versions.get(name)} == {}, STALE
    assert {
        name: listed - hashes.get(name, set())
        for name, listed in _hashes(text).items()
        if not listed <= hashes.get(name, set())
    } == {}, STALE


@pytest.mark.parametrize("name", COMPILED)
def test_hashed_list_holds_its_input(name: str) -> None:
    hashed = _pins((REQUIREMENTS / f"{name}.txt").read_text())
    for line in (REQUIREMENTS / f"{name}.in").read_text().splitlines():
        # A name, its extras, then the version of an == pin; a comment or blank line has no name.
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^]]*\])?\s*(?:==\s*([^\s;#]+))?", line)
        if match:
            versions = hashed.get(_name(match[1]), set())
            assert versions, f"{line!r}: {STALE}"
            assert match[2] in (None, *versions), f"{line!r}: {STALE}"


@pytest.mark.parametrize("name", LISTS)
def test_every_requirement_has_a_hash(name: str) -> None:
    text = (REQUIREMENTS / f"{name}.txt").read_text().replace("\\\n", " ")
    requirements = [line for line in text.splitlines() if re.match(r"[A-Za-z0-9]", line)]
    assert requirements
    assert [line for line in requirements if "--hash=sha256:" not in line] == []

"""
Pipeline layout: step scripts are numbered 1-6 (4a/4b), and every module a script imports by name exists.
Run:  python -m pytest part1_haplograph/tests -q
"""
import re

from conftest import PART1

STEPS = ["1_download_haplograph", "2_build_graph", "3_node_embeddings", "4a_finetune_evo2_embeddings",
         "4b_finetune_evo2_hash", "5_download_gwas", "6_validate_evo2"]


def test_step_scripts_are_numbered():
    assert sorted(p.stem for p in PART1.glob("[0-9]*.py")) == STEPS


def test_every_imported_pipeline_module_exists():
    """import_module("...") / script("...") / plain imports must name an existing module (a rename
    left `import download_haplograph` in a self-test that pytest never ran)."""
    local = {p.stem for p in PART1.glob("*.py")}
    for f in list(PART1.glob("*.py")) + [t for t in (PART1 / "tests").glob("*.py") if t.name != "test_layout.py"]:
        s = f.read_text()
        names = re.findall(r'(?:import_module|script)\("([^"]+)"\)', s)
        names += [n for n in re.findall(r"^\s*(?:import|from) (\w+)", s, re.M)
                  if re.search(r"haplo|evo2_model|download|build_graph|validate", n)]
        missing = [n for n in names if n not in local and n != "conftest"]
        assert not missing, (f.name, missing)

"""核验发布文件哈希、报告链接、源码快照和研究参数。"""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET

from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "PUBLICATION-MANIFEST.json").read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        data = (root / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"], relative
        assert len(data) == record["bytes"], relative
        if Path(relative).suffix in {".md", ".json", ".xml", ".toml", ".py"}:
            assert re.search(rb"/(?:Users|home)/[^/\s\"']+/", data) is None, relative
    summary = json.loads((root / "reports/classic-factors/summary.json").read_text(encoding="utf-8"))
    assert summary["study_fingerprint"] == manifest["study_fingerprint"]
    documents = [root / "README.md", root / "THIRD_PARTY_NOTICES.md", root / "reports/classic-factors/report.md"]
    documents.extend(root / "runs" / item["run_id"] / "report.md" for item in summary["ranking"])
    links = 0
    for path in documents:
        content = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if target.startswith(("https:", "http:", "#")):
                continue
            target = target.split("#", 1)[0]
            if target:
                assert (path.parent / target).resolve().is_file(), (path, target)
                links += 1
    images = sorted(root.glob("reports/classic-factors/*.png")) + sorted(root.glob("runs/classic-*/*.png"))
    assert len(images) == 15
    for path in images:
        with Image.open(path) as picture:
            picture.verify()
    for item in summary["ranking"]:
        directory = root / "runs" / item["run_id"]
        run_manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in run_manifest["source_sha256"].items():
            assert hashlib.sha256((directory / "source" / name).read_bytes()).hexdigest() == expected, (directory, name)
        for name, expected in run_manifest["publication"]["files"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, (directory, name)
    suites = list(ET.parse(root / "validation/classic-factors-tests.xml").getroot().iter("testsuite"))
    test_count = sum(int(suite.get("tests", "0")) for suite in suites)
    checks = json.loads((root / "reports/classic-factors/artifact_validation.json").read_text(encoding="utf-8"))
    assert test_count >= 73 and test_count == checks["python_tests"]
    assert sum(int(suite.get(key, "0")) for suite in suites for key in ("failures", "errors", "skipped")) == 0
    print(json.dumps({"status": "passed", "files": len(manifest["files"]) + 1,
                      "report_documents": len(documents), "links": links, "images": len(images),
                      "tests": test_count, "source_snapshots": 10}, ensure_ascii=False))


if __name__ == "__main__":
    main()

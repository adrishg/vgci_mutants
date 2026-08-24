from pathlib import Path
import zipfile

from shared.docx_integrity import compare_packages, package_snapshot
from shared.manuscript_consistency import audit_text


def test_consistency_checker_finds_deliberately_stale_values_and_wording():
    findings = audit_text(
        "The risk ratio was 2.345%. Subset CI covered complete effect. Old=17.389.",
        stale_values=["17.389"],
    )
    checks = {finding["check"] for finding in findings}
    assert {"risk_ratio_terminology", "subset_ci_coverage_wording", "three_decimal_percentage", "stale_registered_value"} <= checks


def _docx(path: Path, document: str, media: bytes = b"image"):
    rels = """<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"r1\" Type=\"x\" Target=\"word/document.xml\"/></Relationships>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/media/image1.png", media)
        archive.writestr("_rels/.rels", rels)


def test_docx_integrity_detects_media_or_tracked_change_damage(tmp_path):
    original, same, damaged = (tmp_path / name for name in ("original.docx", "same.docx", "damaged.docx"))
    document = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:ins><w:r><w:t>x</w:t></w:r></w:ins></w:document>'
    _docx(original, document)
    _docx(same, document)
    _docx(damaged, document.replace("w:ins", "w:r"), media=b"different")
    assert package_snapshot(original)["unresolved_relationships"] == ()
    assert compare_packages(original, same) == []
    issues = compare_packages(original, damaged)
    assert "media_sha256 changed" in issues
    assert "insertions changed" in issues


def test_docx_integrity_allows_declared_figure_replacement(tmp_path):
    original, revised = (tmp_path / name for name in ("original.docx", "revised.docx"))
    document = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'
    _docx(original, document, media=b"draft")
    _docx(revised, document, media=b"final")
    assert compare_packages(
        original,
        revised,
        allowed_changed_parts={"word/media/image1.png"},
    ) == []

"""Render a simple Markdown-like text file into a minimal .docx document.

Supported syntax:
- # Title
- ## Heading 1
- ### Heading 2
- blank line to separate paragraphs
- lines starting with "- " become bullet-like paragraphs

This renderer intentionally stays minimal and uses only the Python standard
library so it can run in restricted environments.
"""

from __future__ import annotations

import argparse
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a simple report to DOCX.")
    parser.add_argument("--input", required=True, help="Path to the source markdown-like file.")
    parser.add_argument("--output", required=True, help="Path to the output DOCX file.")
    parser.add_argument(
        "--title",
        default="Bao cao de tai HGT Drug Protein Disease",
        help="Document title metadata.",
    )
    parser.add_argument(
        "--author",
        default="OpenAI Codex",
        help="Document author metadata.",
    )
    return parser.parse_args()


def parse_source(text: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            merged = " ".join(line.strip() for line in paragraph_buffer if line.strip())
            if merged:
                items.append(("normal", merged))
            paragraph_buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            items.append(("title", stripped[2:].strip()))
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            items.append(("heading1", stripped[3:].strip()))
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            items.append(("heading2", stripped[4:].strip()))
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            items.append(("bullet", "• " + stripped[2:].strip()))
            continue

        paragraph_buffer.append(stripped)

    flush_paragraph()
    return items


def build_document_xml(items: list[tuple[str, str]]) -> str:
    paragraphs = "\n".join(make_paragraph_xml(kind, text) for kind, text in items)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 w15 wp14">\n'
        "  <w:body>\n"
        f"{indent_xml(paragraphs, 4)}\n"
        "    <w:sectPr>\n"
        '      <w:pgSz w:w="11906" w:h="16838"/>\n'
        '      <w:pgMar w:top="1440" w:right="1134" w:bottom="1440" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/>\n'
        '      <w:cols w:space="708"/>\n'
        '      <w:docGrid w:linePitch="360"/>\n'
        "    </w:sectPr>\n"
        "  </w:body>\n"
        "</w:document>\n"
    )


def make_paragraph_xml(kind: str, text: str) -> str:
    escaped_text = escape(text)
    if kind == "title":
        return (
            "  <w:p>\n"
            "    <w:pPr>\n"
            '      <w:jc w:val="center"/>\n'
            '      <w:spacing w:before="120" w:after="180"/>\n'
            "    </w:pPr>\n"
            "    <w:r>\n"
            f"{indent_xml(run_properties(size=32, bold=True), 6)}\n"
            f"      <w:t>{escaped_text}</w:t>\n"
            "    </w:r>\n"
            "  </w:p>"
        )
    if kind == "heading1":
        return (
            "  <w:p>\n"
            "    <w:pPr>\n"
            '      <w:spacing w:before="220" w:after="100"/>\n'
            "    </w:pPr>\n"
            "    <w:r>\n"
            f"{indent_xml(run_properties(size=28, bold=True), 6)}\n"
            f"      <w:t>{escaped_text}</w:t>\n"
            "    </w:r>\n"
            "  </w:p>"
        )
    if kind == "heading2":
        return (
            "  <w:p>\n"
            "    <w:pPr>\n"
            '      <w:spacing w:before="140" w:after="80"/>\n'
            "    </w:pPr>\n"
            "    <w:r>\n"
            f"{indent_xml(run_properties(size=26, bold=True), 6)}\n"
            f"      <w:t>{escaped_text}</w:t>\n"
            "    </w:r>\n"
            "  </w:p>"
        )
    if kind == "bullet":
        return (
            "  <w:p>\n"
            "    <w:pPr>\n"
            '      <w:ind w:left="360"/>\n'
            '      <w:spacing w:after="60"/>\n'
            "    </w:pPr>\n"
            "    <w:r>\n"
            f"{indent_xml(run_properties(size=24), 6)}\n"
            f"      <w:t>{escaped_text}</w:t>\n"
            "    </w:r>\n"
            "  </w:p>"
        )
    return (
        "  <w:p>\n"
        "    <w:pPr>\n"
        '      <w:jc w:val="both"/>\n'
        '      <w:spacing w:after="100"/>\n'
        "    </w:pPr>\n"
        "    <w:r>\n"
        f"{indent_xml(run_properties(size=24), 6)}\n"
        f"      <w:t xml:space=\"preserve\">{escaped_text}</w:t>\n"
        "    </w:r>\n"
        "  </w:p>"
    )


def run_properties(*, size: int, bold: bool = False) -> str:
    bold_xml = "      <w:b/>\n" if bold else ""
    return (
        "      <w:rPr>\n"
        '        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman"/>\n'
        f"{bold_xml}"
        f'        <w:sz w:val="{size * 2}"/>\n'
        f'        <w:szCs w:val="{size * 2}"/>\n'
        '        <w:lang w:val="vi-VN"/>\n'
        "      </w:rPr>"
    )


def build_content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="xml" ContentType="application/xml"/>\n'
        '  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>\n'
        '  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>\n'
        '  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>\n'
        "</Types>\n"
    )


def build_root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>\n'
        '  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>\n'
        '  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>\n'
        "</Relationships>\n"
    )


def build_document_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>\n'
    )


def build_core_xml(title: str, author: str) -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        f"  <dc:title>{escape(title)}</dc:title>\n"
        f"  <dc:creator>{escape(author)}</dc:creator>\n"
        f"  <cp:lastModifiedBy>{escape(author)}</cp:lastModifiedBy>\n"
        f'  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>\n'
        f'  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>\n'
        "</cp:coreProperties>\n"
    )


def build_app_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">\n'
        "  <Application>Microsoft Office Word</Application>\n"
        "  <DocSecurity>0</DocSecurity>\n"
        "  <ScaleCrop>false</ScaleCrop>\n"
        "  <Company></Company>\n"
        "  <LinksUpToDate>false</LinksUpToDate>\n"
        "  <SharedDoc>false</SharedDoc>\n"
        "  <HyperlinksChanged>false</HyperlinksChanged>\n"
        "  <AppVersion>16.0000</AppVersion>\n"
        "</Properties>\n"
    )


def indent_xml(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def write_docx(output_path: Path, document_xml: str, title: str, author: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", build_content_types_xml())
        archive.writestr("_rels/.rels", build_root_rels_xml())
        archive.writestr("docProps/core.xml", build_core_xml(title=title, author=author))
        archive.writestr("docProps/app.xml", build_app_xml())
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/_rels/document.xml.rels", build_document_rels_xml())


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    source_text = input_path.read_text(encoding="utf-8")
    items = parse_source(source_text)
    document_xml = build_document_xml(items)
    write_docx(output_path=output_path, document_xml=document_xml, title=args.title, author=args.author)

    print("Rendered DOCX successfully.")


if __name__ == "__main__":
    main()

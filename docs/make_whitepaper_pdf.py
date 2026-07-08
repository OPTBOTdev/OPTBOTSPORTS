"""Render README.md (controlled markdown subset) into the OptBot CIN whitepaper PDF."""
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

NAVY = colors.HexColor("#16294A")
RED = colors.HexColor("#D7263D")
GRAY = colors.HexColor("#5B6B82")
ICE = colors.HexColor("#EDF3FB")

SANITIZE = {"→": "->", "≥": ">=", "≤": "<=", "─": "-", "┐": "+",
            "┼": "+", "┘": "+", "┌": "+", "└": "+", "│": "|",
            "↓": "v", "↑": "^", "✓": "[x]",
            "✅": "[DONE]", "\U0001f504": "[RUNNING]", "⏳": "[QUEUED]",
            "·": "·"}


def sanitize(s: str) -> str:
    for k, v in SANITIZE.items():
        s = s.replace(k, v)
    return s


def inline(s: str) -> str:
    s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r'<font face="Courier" size="9">\1</font>', s)
    return s


def build(md_path: str, out_path: str):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=10, leading=14.5, alignment=TA_JUSTIFY,
                          textColor=colors.HexColor("#222222"), spaceAfter=7)
    h1 = ParagraphStyle("h1x", parent=body, fontName="Helvetica-Bold", fontSize=21,
                        leading=25, textColor=NAVY, spaceBefore=6, spaceAfter=10,
                        alignment=0)
    h2 = ParagraphStyle("h2x", parent=body, fontName="Helvetica-Bold", fontSize=14.5,
                        leading=18, textColor=NAVY, spaceBefore=16, spaceAfter=6,
                        alignment=0)
    h3 = ParagraphStyle("h3x", parent=body, fontName="Helvetica-Bold", fontSize=11.5,
                        leading=15, textColor=RED, spaceBefore=11, spaceAfter=4,
                        alignment=0)
    mono = ParagraphStyle("monox", parent=body, fontName="Courier", fontSize=8.2,
                          leading=11, backColor=ICE, borderPadding=6, alignment=0)
    cell = ParagraphStyle("cellx", parent=body, fontSize=8.6, leading=11.5,
                          alignment=0, spaceAfter=0)
    cell_h = ParagraphStyle("cellhx", parent=cell, fontName="Helvetica-Bold",
                            textColor=colors.white)

    lines = sanitize(Path(md_path).read_text(encoding="utf-8")).splitlines()
    story, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("    ") and ln.strip():                       # code block
            block = []
            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                if not lines[i].strip() and not (i + 1 < len(lines) and lines[i + 1].startswith("    ")):
                    break
                block.append(lines[i][4:])
                i += 1
            story.append(Preformatted("\n".join(block).rstrip(), mono))
            story.append(Spacer(1, 7))
            continue
        if ln.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            rows.pop(1)                                                 # separator row
            data = [[Paragraph(inline(c), cell_h if r == 0 else cell) for c in row]
                    for r, row in enumerate(rows)]
            ncol = len(rows[0])
            w = (letter[0] - 1.5 * inch) / ncol
            widths = [w * 0.55] + [w * (0.45 / (ncol - 1) + 1)] * (ncol - 1) if ncol > 2 else None
            t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ICE]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D6E8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
            story.append(Spacer(1, 9))
            continue
        if ln.startswith("### "):
            story.append(Paragraph(inline(ln[4:]), h3))
        elif ln.startswith("## "):
            story.append(Paragraph(inline(ln[3:]), h2))
        elif ln.startswith("# "):
            story.append(Paragraph(inline(ln[2:]), h1))
        elif ln.strip() == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#C9D6E8")))
            story.append(Spacer(1, 4))
        elif re.match(r"^\d+\. ", ln.strip()):
            story.append(Paragraph(inline(ln.strip()), body))
        elif ln.strip().startswith("- "):
            story.append(Paragraph("• " + inline(ln.strip()[2:]), body))
        elif ln.strip():
            para = [ln.strip()]
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and not re.match(r"^(#|\||    |---|\d+\. |- )", lines[i + 1])):
                i += 1
                para.append(lines[i].strip())
            story.append(Paragraph(inline(" ".join(para)), body))
        i += 1

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(0.75 * inch, 0.5 * inch, "OptBot · Portability CIN · Confidential")
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(out_path, pagesize=letter,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                            title="OptBot CIN — Technical Whitepaper", author="OptBot")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out_path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    build(str(root / "README.md"), str(root / "artifacts" / "OptBot_CIN_Whitepaper.pdf"))

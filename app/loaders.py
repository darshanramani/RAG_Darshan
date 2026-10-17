"""Turn raw resume files (PDF, DOCX, LaTeX, Markdown/TXT) into plain text."""
from __future__ import annotations

import re
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".tex", ".md", ".txt"}


def load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages, links = [], []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
        # Hyperlinks (e.g. the word "LinkedIn") only exist as annotations.
        for annot in page.get("/Annots") or []:
            obj = annot.get_object()
            uri = (obj.get("/A") or {}).get("/URI")
            if uri:
                links.append(str(uri))
    text = "\n".join(pages)
    if links:
        text += "\n\nLinks:\n" + "\n".join(dict.fromkeys(links))
    return text


def load_docx(path: Path) -> str:
    import docx

    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs)


def load_tex(path: Path) -> str:
    """Rough LaTeX -> text. Good enough for resume templates (Jake's resume etc.)."""
    src = path.read_text(encoding="utf-8", errors="ignore")
    src = re.sub(r"(?<!\\)%.*", "", src)                                   # comments
    body = src.split(r"\begin{document}", 1)[-1].split(r"\end{document}", 1)[0]
    body = re.sub(r"\\href\{([^}]*)\}\{([^}]*)\}", r"\2 (\1)", body)
    body = re.sub(r"\\(section|subsection)\*?\{([^}]*)\}", r"\n\2\n", body)
    body = re.sub(r"\\item\s*", "\n• ", body)
    body = re.sub(r"\\(textbf|textit|emph|underline|small|large|Large|scshape)\{([^}]*)\}", r"\2", body)
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", body)              # other commands
    body = body.replace("{", " ").replace("}", " ").replace("\\\\", "\n").replace("$|$", "|")
    body = body.replace("--", "–").replace("\\&", "&").replace("\\%", "%")
    return re.sub(r"[ \t]+", " ", body)


def load(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext == ".tex":
        return load_tex(path)
    if ext in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {path}")

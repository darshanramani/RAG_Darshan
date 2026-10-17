"""Structure-aware resume parsing and chunking.

Resumes are short but dense, so naive fixed-size chunking mixes roles together and
retrieval gets fuzzy. Instead we parse the resume into sections/roles and emit:

  * one chunk per experience bullet, prefixed with its role header
    (contextual chunk headers: "Data Scientist at Deloitte (Jan 2023 – Jun 2023): ...")
  * one overview chunk per role (title, company, location, dates, all tools mentioned)
  * one chunk per skills category, the summary, each education entry, contact info
  * a synthesized career-timeline chunk, so "walk me through his career" works
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

SECTION_NAMES = [
    "summary", "profile", "objective", "skills", "technical skills", "experience",
    "work experience", "professional experience", "projects", "education",
    "certifications", "publications", "awards", "achievements", "volunteering", "links",
]
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_DATE = rf"(?:{_MONTH}\s+)?\d{{4}}"
DATE_RANGE = re.compile(rf"({_DATE})\s*[–—-]\s*({_DATE}|Present|Current|Now)\s*$", re.I)
BULLET = re.compile(r"^\s*[•●▪◦\-\*]\s*")


@dataclass
class Role:
    header: str
    dates: str
    bullets: list[str] = field(default_factory=list)

    @property
    def title_company(self) -> str:
        return self.header


@dataclass
class Chunk:
    id: str
    text: str
    section: str
    source: str
    title: str  # short human-readable label used in citations

    def to_dict(self) -> dict:
        return asdict(self)


def _cid(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode()).hexdigest()[:12]


def split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        key = line.lower().rstrip(":")
        if key in SECTION_NAMES:
            current = key
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def join_bullets(lines: list[str]) -> list[str]:
    """Merge PDF line-wraps back into whole bullets."""
    items: list[str] = []
    for line in lines:
        if BULLET.match(line) or not items:
            items.append(BULLET.sub("", line))
        else:
            items[-1] += " " + line
    return [re.sub(r"\s+", " ", i).strip() for i in items if i.strip()]


def parse_roles(lines: list[str]) -> list[Role]:
    roles: list[Role] = []
    buf: list[str] = []
    for line in lines:
        m = DATE_RANGE.search(line)
        if m and not BULLET.match(line):
            if roles:
                roles[-1].bullets = join_bullets(buf)
            buf = []
            header = line[: m.start()].strip().rstrip(",|")
            roles.append(Role(header=header, dates=f"{m.group(1)} – {m.group(2)}"))
        else:
            buf.append(line)
    if roles:
        roles[-1].bullets = join_bullets(buf)
    return roles


def parse_education(lines: list[str]) -> list[str]:
    """Pair 'Degree  dates' lines with the following 'School  location' line."""
    entries, cur = [], None
    for line in lines:
        m = DATE_RANGE.search(line)
        if m:
            if cur:
                entries.append(cur)
            cur = f"{line[: m.start()].strip()} ({m.group(1)} – {m.group(2)})"
        elif cur:
            cur += f", {line}"
        else:
            entries.append(line)
    if cur:
        entries.append(cur)
    return entries


def chunk_resume(text: str, source: str, owner: str, extra_links: list[str] | None = None) -> list[Chunk]:
    s = split_sections(text)
    chunks: list[Chunk] = []

    def add(section: str, title: str, body: str):
        chunks.append(Chunk(_cid(source, section, body), body.strip(), section, source, title))

    # contact / header: keep professional links only (PII already scrubbed upstream)
    header = " ".join(s.get("header", [])[1:])
    contact_bits = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text) + re.findall(
        r"https?://(?:www\.)?linkedin\.com/in/\S+", text) + [l for l in (extra_links or []) if l]
    if header or contact_bits:
        loc = header.split("|")[0].strip() if header else ""
        add("contact", "Contact",
            f"{owner} — contact details. Location: {loc or 'not listed'}. "
            f"Contact: {', '.join(dict.fromkeys(contact_bits)) or 'see LinkedIn'}.")

    for key in ("summary", "profile", "objective"):
        if s.get(key):
            add("summary", "Professional summary", f"{owner} — professional summary: " + " ".join(s[key]))

    for key in ("skills", "technical skills"):
        for item in join_bullets(s.get(key, [])):
            cat, _, vals = item.partition(":")
            if vals:
                add("skills", f"Skills: {cat.strip()}", f"{owner}'s skills — {cat.strip()}: {vals.strip()}")
            else:
                add("skills", "Skills", f"{owner}'s skills: {item}")

    timeline = []
    for key in ("experience", "work experience", "professional experience"):
        for role in parse_roles(s.get(key, [])):
            ctx = f"{owner} worked as {role.header} ({role.dates})"
            timeline.append(f"{role.header} ({role.dates})")
            add("experience", f"{role.header} ({role.dates})",
                f"{ctx}. Role overview: {len(role.bullets)} key contributions. "
                f"Highlights: " + " ".join(b.split(",")[0] + "." for b in role.bullets))
            for b in role.bullets:
                add("experience", f"{role.header} ({role.dates})", f"{ctx}: {b}")

    for item in join_bullets(s.get("projects", [])):
        add("projects", "Projects", f"{owner}'s project: {item}")

    for entry in parse_education(s.get("education", [])):
        add("education", "Education", f"{owner}'s education: {entry}")

    for key in ("certifications", "publications", "awards", "achievements", "volunteering"):
        for item in join_bullets(s.get(key, [])):
            add(key, key.title(), f"{owner} — {key}: {item}")

    if timeline:
        add("timeline", "Career timeline",
            f"{owner}'s career timeline (work history / employment history), most recent first: " + "; ".join(timeline) + ".")
    return chunks

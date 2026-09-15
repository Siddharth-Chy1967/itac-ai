"""Pre-assessment checklist retrieval (Feature 2).

A student preparing for an assessment gives a SIC code and/or a short facility
description; this finds relevant past reports two ways, merges them, and distills
the most common AR themes into a "what to look for" checklist.

Two passes (spec Feature 2, steps 4-9):

- PASS A, SIC waterfall: exhaustive metadata filter on sic4, widening to sic3 then
  sic2 when fewer than `min_reports` distinct reports match. No semantic ranking.
- PASS B, process similarity: embed the facility description and semantic-search the
  facility_profile chunks (no SIC filter), threshold-filtered. This is what lets two
  facilities with different SIC codes but shared materials/processes find each other.

MERGE unions the two report sets. Then a recency preference (era policy): prefer
2017+ reports; only fall back to including <=2006 reports when fewer than
`min_reports` modern reports matched (the same threshold the SIC waterfall uses).
AR-theme grouping draws ONLY from modern reports -- old-era reports lack reliable
per-AR headings, so they contribute to facility-match coverage but never to themes.

AR COLLECTION is exhaustive (every AR of every grouped report, metadata-filtered,
no relevance cutoff), deduped by (report_id, ar_number) to drop the bare "AR-N"
summary-table labels the extractor also tags. GROUPING greedily merges
near-duplicate AR titles by cosine similarity and takes the largest groups.

Everything is driven through the `_Store` / `_Embedder` protocols so it unit-tests
against fakes without ChromaDB or torch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

# Reports from this year onward are "modern" and eligible for AR-theme grouping.
MODERN_MIN_YEAR = 2007


# --- data carriers ----------------------------------------------------------

@dataclass
class ReportRef:
    """A report matched by a pass, with its era and (for Pass B) similarity."""

    report_id: str
    era: str
    score: Optional[float] = None


@dataclass
class ARSection:
    """One AR heading from a report (before dedup/cleaning)."""

    report_id: str
    ar_number: str
    title: str  # the raw section heading
    era: str = ""


@dataclass
class ThemeGroup:
    """A cluster of near-duplicate AR titles across reports."""

    representative: str
    examples: list[str]
    report_ids: list[str]
    # The subset of `report_ids` from 2017+ reports. Frequency alone does not say
    # whether a theme is still current advice: "Install High Efficiency T-8 Lamps"
    # is one of the most-written ARs in the corpus and has not been recommended
    # since the LED era. Carrying the modern subset lets a ranking show both.
    modern_report_ids: list[str] = field(default_factory=list)

    @property
    def report_count(self) -> int:
        return len(self.report_ids)

    @property
    def modern_count(self) -> int:
        return len(self.modern_report_ids)

    @property
    def dated(self) -> bool:
        """True when every report writing this theme predates 2017."""
        return bool(self.report_ids) and not self.modern_report_ids


@dataclass
class MatchedReport:
    """A report in the merged set, with which pass(es) surfaced it."""

    report_id: str
    era: str
    from_industry: bool = False
    from_process: bool = False


@dataclass
class ChecklistResult:
    """The structured checklist, ready to render into reference material."""

    themes: list[ThemeGroup]
    total_reports: int
    n_industry: int
    n_process: int
    sic_level: Optional[str]  # "sic4" | "sic3" | "sic2" | None
    sic_pattern: Optional[str]  # e.g. "259x" when widened to sic3
    thin: bool
    matched: list[MatchedReport] = field(default_factory=list)
    fallback_used: bool = False


# --- store / embedder protocols (real impls wired by the orchestrator) ------

class _Store(Protocol):
    def reports_for_sic(self, level: str, value: str) -> list[ReportRef]: ...
    def profile_search(self, query_embedding: list[float], k: int) -> list[ReportRef]: ...
    def ar_sections(self, report_ids: set[str]) -> list[ARSection]: ...


class _BPStore(Protocol):
    """The Best Practice collection (Section B), same (doc, meta, sim) query shape."""

    def query(self, query_embedding: list[float], k: int) -> list[tuple[str, dict, float]]: ...


class _Embedder(Protocol):
    def encode_query(self, text: str) -> list[float]: ...
    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...


# --- era helpers ------------------------------------------------------------

def _era_year(era: str) -> Optional[int]:
    m = re.search(r"(\d{4})", era or "")
    return int(m.group(1)) if m else None


def is_modern(era: str) -> bool:
    """True for a 2017+ era; unknown/old eras are treated as not modern."""
    year = _era_year(era)
    return year is not None and year >= MODERN_MIN_YEAR


# --- Pass A: SIC waterfall ---------------------------------------------------

def sic_waterfall(store: _Store, sic: str, *, min_reports: int) -> tuple[list[ReportRef], Optional[str]]:
    """Filter reports by SIC, widening sic4 -> sic3 -> sic2 until >= min_reports.

    Returns the matched reports and the level used ("sic4"/"sic3"/"sic2"), or
    ([], None) when no SIC is given. Falls back to the widest (sic2) result if even
    that yields fewer than min_reports.
    """
    digits = re.sub(r"\D", "", sic or "")
    if not digits:
        return [], None

    last: list[ReportRef] = []
    last_level: Optional[str] = None
    for level, width in (("sic4", 4), ("sic3", 3), ("sic2", 2)):
        value = digits[:width]
        if len(value) < width:
            continue
        reports = _distinct(store.reports_for_sic(level, value))
        last, last_level = reports, level
        if len(reports) >= min_reports:
            return reports, level
    return last, last_level


def _distinct(reports: list[ReportRef]) -> list[ReportRef]:
    """One ReportRef per report_id (first occurrence wins)."""
    seen: dict[str, ReportRef] = {}
    for r in reports:
        seen.setdefault(r.report_id, r)
    return list(seen.values())


# --- Pass B: process similarity ---------------------------------------------

def process_similarity(
    store: _Store, embedder: _Embedder, description: str, *, k: int, threshold: float
) -> list[ReportRef]:
    """Semantic-search facility_profile chunks; keep distinct reports over threshold.

    One ReportRef per report_id, keeping its best-scoring profile chunk.
    """
    vec = embedder.encode_query(description)
    best: dict[str, ReportRef] = {}
    for hit in store.profile_search(vec, k):
        if hit.score is None or hit.score < threshold:
            continue
        cur = best.get(hit.report_id)
        if cur is None or (hit.score > (cur.score or 0.0)):
            best[hit.report_id] = hit
    return list(best.values())


# --- merge + recency preference ---------------------------------------------

def merge_passes(pass_a: list[ReportRef], pass_b: list[ReportRef]) -> list[MatchedReport]:
    """Union Pass A and Pass B by report_id, tracking which pass(es) matched each."""
    merged: dict[str, MatchedReport] = {}
    for r in pass_a:
        m = merged.setdefault(r.report_id, MatchedReport(r.report_id, r.era))
        m.from_industry = True
    for r in pass_b:
        m = merged.setdefault(r.report_id, MatchedReport(r.report_id, r.era))
        m.from_process = True
    return list(merged.values())


def apply_recency_preference(
    merged: list[MatchedReport], *, min_reports: int
) -> tuple[list[MatchedReport], bool]:
    """Prefer modern reports; keep <=2006 ones only as fallback when modern is thin.

    Returns the kept reports and whether the old-era fallback was used.
    """
    modern = [m for m in merged if is_modern(m.era)]
    if len(modern) >= min_reports:
        return modern, False
    # Too few modern matches: pad with the old-era reports for coverage.
    return merged, any(not is_modern(m.era) for m in merged)


# --- AR collection + cleaning ------------------------------------------------

_AR_LEAD = re.compile(r"^\s*AR[\s#\-.:]*(?:No\.?\s*)?\d+\s*[:.\-)]?\s*", re.IGNORECASE)
_ARC_CITE = re.compile(r"\(?\s*ARC\s*#?\s*\d\.\d{3,4}(?:\.\d+)?\s*\)?", re.IGNORECASE)
# A table-of-contents page number trailing the heading ("...Insulate the Thermoformers\t43").
# The TAB is what identifies it: 445 headings in the corpus end that way, and every
# heading ending in a SPACE plus a number is instead a real part of the title
# ("...in Plant 2", "...Add Automatic Economizers, Building 3"). Anchoring on the tab
# strips the page numbers without truncating those.
_TOC_PAGE = re.compile(r"\t\s*\d{1,3}\s*$")


def clean_ar_title(heading: str) -> str:
    """Strip the 'AR-N' lead, an '(ARC X.YYYY)' citation, and a trailing TOC page number."""
    title = _TOC_PAGE.sub("", heading or "")
    title = _ARC_CITE.sub("", title)
    title = _AR_LEAD.sub("", title)
    return re.sub(r"\s+", " ", title).strip(" .:-(),")


# Words that say what KIND of action an AR is, not what it is about. Two titles sharing
# only these ("Install ... High Efficiency ...") are not the same recommendation.
_TITLE_FILLER = frozenset(
    """
    a an the of on in at for to with and or from by into as per via
    install installing installation replace replacing repair repairing reduce reducing
    use using add adding implement improve upgrade adjust increase decrease change switch
    convert retrofit utilize apply maintain turn shut off when not new high higher
    efficiency efficient energy system systems unit units equipment existing current
    recommended recommendation ar arc recycle recycling sell selling purchase
    """.split()
)


def _stem(word: str) -> str:
    """A crude, deterministic stem: lowercase, drop an apostrophe-s and one suffix, cap at 6."""
    w = re.sub(r"['’]s$", "", word.lower())
    for suffix in ("ing", "ers", "er", "es", "ed", "s"):
        if len(w) > 3 and w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)]
            break
    return w[:6]


def title_words(text: str) -> set[str]:
    """The distinctive word stems of an AR title or topic (action/filler words removed)."""
    return {
        _stem(w)
        for w in re.findall(r"[A-Za-z]{3,}(?:['’]s)?", text or "")
        if re.sub(r"['’]s$", "", w.lower()) not in _TITLE_FILLER
    }


def _word_overlap(a: set[str], b: set[str]) -> float:
    """Share of the smaller word set found in the other; 1.0 when either is empty."""
    if not a or not b:
        return 1.0
    return len(a & b) / min(len(a), len(b))


def collect_ar_titles(sections: list[ARSection]) -> list[ARSection]:
    """Dedupe raw AR sections by (report_id, ar_number), keeping the richest title.

    The extractor also tags the bare "AR-1"/"AR-2" labels in summary tables, which
    clean to an empty title; those are dropped in favour of the real heading. Titles
    are returned cleaned (no "AR-N" lead, no ARC citation).
    """
    best: dict[tuple[str, str], ARSection] = {}
    for s in sections:
        cleaned = clean_ar_title(s.title)
        if not cleaned:
            continue
        key = (s.report_id, s.ar_number)
        cur = best.get(key)
        if cur is None or len(cleaned) > len(cur.title):
            best[key] = ARSection(s.report_id, s.ar_number, cleaned, s.era)
    return list(best.values())


# --- grouping ----------------------------------------------------------------

# A title joins a group only when at least this share of its distinctive words (or the
# group seed's, whichever set is smaller) are shared. Cosine alone merged titles that
# differ only in their object: "Install High Efficiency Motors" absorbed "... Air
# Conditioners" and "... Lighting", and "Reduce Compressor Air Pressure" absorbed
# "Reduce Inlet Compressor Temperature" -- different ARC codes counted as one theme.
MIN_TITLE_WORD_OVERLAP = 0.5


def group_ar_titles(
    sections: list[ARSection], embedder: _Embedder, *, threshold: float, top_n: int
) -> list[ThemeGroup]:
    """Greedily cluster AR titles by cosine similarity; return the largest groups.

    A title joins the most similar existing group that clears `threshold` AND shares
    its distinctive words with that group's seed (`MIN_TITLE_WORD_OVERLAP`). Titles are
    visited most-common phrasing first, so each group is seeded by its standard wording
    and the result does not depend on store order.

    Group size is measured in DISTINCT reports (a report with two similar ARs counts
    once), so a theme's count is "how many past reports wrote this". The label is the
    group's most common title, not whichever odd phrasing sat nearest the centroid.
    Ties in size go to the theme more reports since 2017 wrote, then alphabetically.
    """
    if not sections:
        return []

    import numpy as np

    reports_by_title: dict[str, set[str]] = {}
    for s in sections:
        reports_by_title.setdefault(s.title, set()).add(s.report_id)
    order = sorted(
        range(len(sections)),
        key=lambda i: (
            -len(reports_by_title[sections[i].title]),
            sections[i].title.lower(),
            sections[i].report_id,
            sections[i].ar_number,
        ),
    )

    vecs = np.asarray(embedder.encode_documents([s.title for s in sections]), dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    words = [title_words(s.title) for s in sections]

    clusters: list[dict] = []
    for i in order:
        v = vecs[i]
        best_c, best_sim = None, -1.0
        for c in clusters:
            if _word_overlap(words[i], c["seed_words"]) < MIN_TITLE_WORD_OVERLAP:
                continue
            sim = float(v @ c["centroid"])
            if sim > best_sim:
                best_c, best_sim = c, sim
        if best_c is not None and best_sim >= threshold:
            best_c["members"].append(i)
            best_c["sum"] = best_c["sum"] + v
            best_c["centroid"] = best_c["sum"] / (np.linalg.norm(best_c["sum"]) or 1.0)
        else:
            clusters.append(
                {"members": [i], "sum": v.copy(), "centroid": v.copy(), "seed_words": words[i]}
            )

    groups: list[ThemeGroup] = []
    for c in clusters:
        members = [sections[j] for j in c["members"]]
        centroid = c["centroid"]
        title_reports: dict[str, set[str]] = {}
        for m in members:
            title_reports.setdefault(m.title, set()).add(m.report_id)
        rep_idx = max(
            c["members"],
            key=lambda j: (len(title_reports[sections[j].title]), float(vecs[j] @ centroid)),
        )
        report_ids = sorted({m.report_id for m in members})
        modern_ids = sorted({m.report_id for m in members if is_modern(m.era)})
        examples = list(dict.fromkeys(m.title for m in members))  # distinct, order-stable
        groups.append(
            ThemeGroup(
                representative=sections[rep_idx].title,
                examples=examples,
                report_ids=report_ids,
                modern_report_ids=modern_ids,
            )
        )

    groups.sort(key=lambda g: (-g.report_count, -g.modern_count, g.representative.lower()))
    return groups[:top_n]


# --- top-level orchestration -------------------------------------------------

def _sic_pattern(digits: str, level: Optional[str]) -> Optional[str]:
    if not level:
        return None
    width = {"sic4": 4, "sic3": 3, "sic2": 2}[level]
    stem = re.sub(r"\D", "", digits)[:width]
    return stem + "x" * (4 - width)


def build_checklist(
    *,
    sic: Optional[str],
    description: Optional[str],
    store: _Store,
    embedder: _Embedder,
    min_reports: int,
    process_k: int,
    process_threshold: float,
    group_threshold: float,
    top_themes: int,
) -> ChecklistResult:
    """Run the two passes, merge, collect ARs, group, and assemble the checklist.

    SIC only -> Pass A alone. Description only -> Pass B alone. Both -> both passes.
    """
    pass_a, sic_level = ([], None)
    if sic:
        pass_a, sic_level = sic_waterfall(store, sic, min_reports=min_reports)
    pass_b: list[ReportRef] = []
    if description:
        pass_b = process_similarity(
            store, embedder, description, k=process_k, threshold=process_threshold
        )

    merged, fallback_used = apply_recency_preference(
        merge_passes(pass_a, pass_b), min_reports=min_reports
    )
    n_industry = sum(1 for m in merged if m.from_industry)
    n_process = sum(1 for m in merged if m.from_process)

    # AR grouping prefers modern reports (era policy). But when the recency preference
    # already had to widen -- fewer than `min_reports` modern matches -- restricting the
    # themes to that handful is what made thin-SIC checklists come back near-empty
    # (SIC 25xx matches 8 past reports, exactly 1 of them modern). In that case group
    # over everything `apply_recency_preference` kept, so the checklist has a basis; the
    # older themes it surfaces are flagged `dated` for the caller.
    theme_ids = {
        m.report_id for m in merged if fallback_used or is_modern(m.era)
    }
    raw_ars = store.ar_sections(theme_ids) if theme_ids else []
    ars = collect_ar_titles(raw_ars)
    themes = group_ar_titles(ars, embedder, threshold=group_threshold, top_n=top_themes)

    return ChecklistResult(
        themes=themes,
        total_reports=len(merged),
        n_industry=n_industry,
        n_process=n_process,
        sic_level=sic_level,
        sic_pattern=_sic_pattern(sic or "", sic_level),
        thin=len(merged) < min_reports,
        matched=merged,
        fallback_used=fallback_used,
    )


# --- "Most common ARs for SIC X" ranking ------------------------------------
# The pre-assessment checklist above answers "what should I look for"; this answers
# the narrower factual question "what does this industry actually get recommended,
# and how often". Same SIC waterfall, but three deliberate differences:
#
#   NO PROCESS PASS. The question names an industry, not a facility, so there is no
#     description to embed and nothing to widen the match on beyond the SIC.
#   ALL ERAS. This is a frequency count over the corpus, and 302 of the 474 ingested
#     reports predate 2017. Excluding them (as the checklist does by default) would
#     not make the count more current, just wrong -- and their AR headings parse fine.
#     Each theme carries its modern subset so a dated one can be labeled, not hidden.
#   RANKED, NOT SAMPLED. Output is the top N by report count, descending, with the
#     counts intact, because the count IS the answer.


@dataclass
class SicARRanking:
    """The ranked AR frequency table for one SIC query."""

    themes: list[ThemeGroup]
    total_reports: int  # comparable past reports the waterfall matched
    modern_reports: int  # how many of those are 2017+
    total_ars: int  # distinct ARs written across them
    sic_level: Optional[str]  # "sic4" | "sic3" | "sic2" | None
    sic_pattern: Optional[str]  # e.g. "302x" when the exact code was not in the corpus
    requested_pattern: str  # what the user asked for, normalized ("3028")
    widened: bool  # True when the exact code had too little and we fell back
    thin: bool
    # Whether "no modern report wrote this" is evidence of anything. With only one or
    # two modern reports in the matched set, EVERY theme has zero modern occurrences,
    # which says the sample is small, not that the advice is superseded. Below
    # `min_reports` modern reports the caller must not draw the currency inference.
    era_confident: bool = False


def rank_ars_for_sic(
    *,
    sic: str,
    store: _Store,
    embedder: _Embedder,
    min_reports: int,
    group_threshold: float,
    top_n: int,
) -> SicARRanking:
    """Rank the AR themes most often written for a SIC, most frequent first.

    Runs the same sic4 -> sic3 -> sic2 waterfall, so a code with no reports of its own
    (e.g. 3028) answers from its family (302x, then 30xx) instead of returning nothing.
    """
    reports, sic_level = sic_waterfall(store, sic, min_reports=min_reports)
    report_ids = {r.report_id for r in reports}
    modern_reports = sum(1 for r in reports if is_modern(r.era))
    ars = collect_ar_titles(store.ar_sections(report_ids)) if report_ids else []
    themes = group_ar_titles(ars, embedder, threshold=group_threshold, top_n=top_n)

    requested = re.sub(r"[^0-9xX*]", "", sic or "").lower().replace("*", "x")
    pattern = _sic_pattern(sic or "", sic_level)
    # "Widened" means the answer is about a broader family than the code asked for --
    # worth saying out loud, since the user's exact industry may not be in the corpus.
    widened = bool(pattern and requested and pattern != requested.ljust(4, "x"))
    return SicARRanking(
        themes=themes,
        total_reports=len(report_ids),
        modern_reports=modern_reports,
        total_ars=len(ars),
        sic_level=sic_level,
        sic_pattern=pattern,
        requested_pattern=requested,
        widened=widened,
        thin=len(report_ids) < min_reports,
        era_confident=modern_reports >= min_reports,
    )


# --- Best Practice retrieval (Section B, B3) --------------------------------
# The pre-assessment above answers "what did similar facilities get recommended".
# B3 answers "which best practices should this facility be checked against", by
# surfacing candidate best-practice items two ways, mirroring the hybrid pattern
# above and reusing its SIC waterfall and AR-collection machinery:
#
#   PRECEDENT: the SIC waterfall (sic4->sic3->sic2) finds comparable past
#     facilities; each AR title from the modern ones (era policy: old-era reports
#     have no reliable AR headings) is mapped onto its nearest bp_reference item.
#     What facilities like this one were told to do is a strong relevance signal.
#   SEMANTIC FALLBACK: when precedent is thin (fewer than min_reports modern
#     reports, the same threshold the SIC waterfall widens on), embed the facility
#     description directly against the 115-item bp_reference, top-K per category.
#
# Output is unfiltered candidates (item, category, template, fill-in record
# prompts, source tag). The already/partial/not synthesis against the student's
# own description happens downstream (B4), not here.


@dataclass
class BPCandidate:
    """One candidate best practice, before the B4 already/partial/not synthesis."""

    name: str
    category: str
    template: str  # the practice description (the already/partial/not check signal)
    fill_ins: list[str]  # "(Insert ...)" record prompts: what to note on-site
    source: str  # "precedent" | "semantic"
    score: float
    report_ids: list[str] = field(default_factory=list)  # precedent provenance


@dataclass
class BPResult:
    """The B3 candidate set with its basis, ready to render for B4 synthesis."""

    candidates: list[BPCandidate]
    precedent_reports: int  # comparable modern facilities the SIC waterfall found
    precedent_report_ids: list[str]  # those that actually surfaced a practice
    sic_level: Optional[str]
    sic_pattern: Optional[str]
    used_semantic_fallback: bool
    thin: bool


def _bp_candidate(meta: dict, source: str, score: float) -> BPCandidate:
    """Build a BPCandidate from a bp_reference metadata row (see ingest_bp.metadata)."""
    raw = meta.get("fill_ins") or ""
    fill_ins = [f for f in (p.strip() for p in raw.split(" | ")) if f]
    return BPCandidate(
        name=meta.get("name", ""),
        category=meta.get("category", ""),
        template=meta.get("template", ""),
        fill_ins=fill_ins,
        source=source,
        score=score,
    )


def bp_precedent_candidates(
    store: _Store,
    bp_store: _BPStore,
    embedder: _Embedder,
    reports: list[ReportRef],
    *,
    per_ar_k: int,
    threshold: float,
) -> list[BPCandidate]:
    """Map each AR title of the modern comparable reports onto its nearest BP item.

    Distinct by BP item: an item surfaced by several ARs/reports is merged, keeping
    the best score and unioning the reports that pointed to it (its provenance).
    """
    modern_ids = {r.report_id for r in reports if is_modern(r.era)}
    if not modern_ids:
        return []
    ars = collect_ar_titles(store.ar_sections(modern_ids))
    if not ars:
        return []

    vecs = embedder.encode_documents([a.title for a in ars])
    by_name: dict[str, BPCandidate] = {}
    for ar, vec in zip(ars, vecs):
        for _doc, meta, sim in bp_store.query(vec, per_ar_k):
            if sim < threshold:
                continue
            cand = by_name.get(meta.get("name", ""))
            if cand is None:
                cand = _bp_candidate(meta, "precedent", sim)
                by_name[cand.name] = cand
            cand.score = max(cand.score, sim)
            if ar.report_id not in cand.report_ids:
                cand.report_ids.append(ar.report_id)
    return list(by_name.values())


def bp_semantic_candidates(
    bp_store: _BPStore,
    embedder: _Embedder,
    description: str,
    *,
    k: int,
    threshold: float,
    per_category: int,
) -> list[BPCandidate]:
    """Embed the facility description against bp_reference; keep top per category.

    Hits come back best-first, so taking the first `per_category` per category keeps
    the strongest few of each rather than flooding the checklist from one category.
    """
    vec = embedder.encode_query(description)
    kept_per_cat: dict[str, int] = {}
    out: list[BPCandidate] = []
    for _doc, meta, sim in bp_store.query(vec, k):
        if sim < threshold:
            continue
        cand = _bp_candidate(meta, "semantic", sim)
        if kept_per_cat.get(cand.category, 0) >= per_category:
            continue
        kept_per_cat[cand.category] = kept_per_cat.get(cand.category, 0) + 1
        out.append(cand)
    return out


def build_bp_candidates(
    *,
    sic: Optional[str],
    description: Optional[str],
    store: _Store,
    bp_store: _BPStore,
    embedder: _Embedder,
    min_reports: int,
    precedent_k: int,
    precedent_threshold: float,
    semantic_k: int,
    semantic_threshold: float,
    per_category: int,
) -> BPResult:
    """Assemble the B3 candidate set: precedent first, semantic fallback when thin.

    Precedent draws on modern comparable reports (era policy). The semantic fallback
    runs only when those are fewer than `min_reports` AND a description was given,
    and its items are unioned in (deduped by BP name, precedent winning on overlap).
    """
    reports, sic_level = ([], None)
    if sic:
        reports, sic_level = sic_waterfall(store, sic, min_reports=min_reports)
    modern = [r for r in reports if is_modern(r.era)]

    precedent = bp_precedent_candidates(
        store, bp_store, embedder, reports, per_ar_k=precedent_k, threshold=precedent_threshold
    )
    thin = len(modern) < min_reports

    candidates = list(precedent)
    seen = {c.name for c in candidates}
    used_semantic = False
    if thin and description:
        used_semantic = True
        for cand in bp_semantic_candidates(
            bp_store, embedder, description,
            k=semantic_k, threshold=semantic_threshold, per_category=per_category,
        ):
            if cand.name not in seen:
                seen.add(cand.name)
                candidates.append(cand)

    return BPResult(
        candidates=candidates,
        precedent_reports=len(modern),
        precedent_report_ids=sorted({rid for c in precedent for rid in c.report_ids}),
        sic_level=sic_level,
        sic_pattern=_sic_pattern(sic or "", sic_level),
        used_semantic_fallback=used_semantic,
        thin=thin,
    )


def lookup_bp_item(
    bp_store: _BPStore, embedder: _Embedder, topic: str, *, threshold: float
) -> Optional[BPCandidate]:
    """Return the single best-matching bp_reference item for a write-up topic, or None.

    Backs the drafting path ("write me the LED lighting best practice"): embed the
    topic, take the top bp_reference hit, and return it (tagged "writeup", fill-in
    prompts intact) only when it clears `threshold`. A topic that matches nothing on
    the list confidently returns None so the caller can fall through to a normal
    answer rather than dressing a weak match up as the official template.
    """
    topic = (topic or "").strip()
    if not topic:
        return None
    hits = bp_store.query(embedder.encode_query(topic), 1)
    if not hits:
        return None
    _doc, meta, sim = hits[0]
    if sim < threshold:
        return None
    return _bp_candidate(meta, "writeup", sim)

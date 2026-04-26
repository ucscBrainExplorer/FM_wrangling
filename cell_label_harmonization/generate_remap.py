#!/usr/bin/env python3
"""
generate_remap.py — map author cell type labels to Cell Ontology (CL) term IDs.

Reads an annotated single-cell TSV (or .tsv.gz), auto-detects the most granular
annotation column, and produces a remap TSV consumed by cellLabelMap_obs.py.

Usage
-----
  python3 generate_remap.py \
      --input  obs.tsv.gz \
      --obo    cl-basic.obo \
      --output remap.tsv \
      [--column author_cell_type] \
      [--pdf   paper.pdf] \
      [--doi   10.1234/example] \
      [--context "IPC = Intermediate Progenitor Cell"] \
      [--no-llm] \
      [--cellxgene-fallback EN EN-fetal-early]

Output columns
--------------
  original_label                         author label string
  mapped_cell_label                      plain-English description
  mapped_cell_label_method               cl_term_name | glossary | llm | original
  mapped_cell_label_ontology_term_id     resolved CL ID (empty if unresolved)
  mapped_cell_label_ontology_term_name   canonical OBO name for the resolved CL term
  mapped_cell_label_ontology_term_match_method  resolution method (see cascade below)
  cellxgene_cell_type                    cellxgene display label(s) co-occurring with
                                         this author label (pipe-separated if multiple)
  cellxgene_cell_type_ontology_term_id   corresponding cellxgene CL IDs

-------------------------------------------------------------------------------
STEP 1 — Column selection
-------------------------------------------------------------------------------
If --column is not given, the annotation column hierarchy is detected
automatically.  Candidate columns are filtered by removing:
  - Numeric columns (>80 % of first 200 values parse as float)
  - High-cardinality columns (>1000 unique values — likely barcodes/IDs)
  - Low-cardinality columns (<3 unique values — likely flags)
  - CL-ID columns (>90 % values match CL:NNNNNNN pattern)
  - CellxGene display columns (paired with a *_ontology_term_id column)

Among remaining candidates, parent–child relationships are inferred by
functional dependency: C is a child of P if every value in C maps to exactly
one value in P and P has fewer unique values.  The leaf column (no children)
is selected; ties broken by preferring the latest-appearing column.

-------------------------------------------------------------------------------
STEP 2 — Resolution cascade
-------------------------------------------------------------------------------
Each unique label is attempted in order, stopping at first success:

  cellxgene_fallback  label listed in --cellxgene-fallback → use cellxgene CL ID
  already_cl          label already matches CL:NNNNNNN — passed through
  exact               case-insensitive match to canonical OBO name
  fuzzy               fuzzy_normalize(label) vs fuzzy_normalize(OBO names)
  synonym*            case-insensitive match to OBO synonym string
  fuzzy_synonym*      fuzzy_normalize(label) vs fuzzy_normalize(OBO synonyms)

  * synonym/fuzzy_synonym results are validated against the paper glossary:
    if the glossary defines this label AND the CL term name contradicts the
    glossary definition (no token/substring overlap between glossary value and
    CL name, AND the label itself is not similar to the CL name), the match is
    discarded and the label is sent to the LLM stage instead.

Uninformative labels (unknown, unassigned, doublet, ambiguous, etc.) skip all
resolution and are written as no_cl_term_found immediately.

Obsolete CL terms (is_obsolete: true in OBO) are excluded from all lookups.

-------------------------------------------------------------------------------
STEP 3 — Pre-LLM preparation
-------------------------------------------------------------------------------
3a. PDF / DOI / context
    If --pdf is given, the first 20 pages (up to 6,000 words) are extracted and
    sent to an LLM to produce a compact glossary: "LABEL: plain English name".
    If --doi is given, title + abstract are fetched from CrossRef.
    If --context is given, the string is used directly.
    Glossary keys are normalised (lowercased, em/en-dash → hyphen) before lookup.

    The glossary is parsed before the cascade and used in two ways:
      - Contradiction check: override wrong synonym matches (Step 2 above).
      - Cell label hints: provide plain-English names for unresolved labels
        (Step 3c below).

3b. Parent subtree precomputation
    Parent columns detected via functional dependency; each parent label is
    resolved to a CL ID using the exact/synonym/fuzzy cascade (no LLM).
    The ontology subtree of each resolved parent is precomputed for candidate
    scoring in Step 4.

3c. Cell label generation (one per unresolved label)
    Plain-English description produced in priority order:
      1. PDF glossary lookup (exact, ±plural 's', dash-normalised)  → [glossary]
      2. Claude LLM: "describe this cell type in plain English"      → [llm]
      3. Original label string as-is                                 → [original]
    Used as the candidate-filter query in Step 4 and as the cell_label output.

-------------------------------------------------------------------------------
STEP 4 — Candidate filtering  (_apply_marker_filter + _filter_candidate_terms)
-------------------------------------------------------------------------------
Narrows ~2,500 CL terms to ≤100 candidates for the LLM prompt.

4a. Marker gene pre-filter  (_apply_marker_filter)
    If the label or hint contains a recognised marker gene token (VIP, SST,
    PV/PVALB, SNCG, LAMP5, CCK, NPY), the pool is restricted to CL terms
    whose canonical name contains the corresponding marker string.

4b. Keyword scoring  (_filter_candidate_terms)
    - Keywords extracted from "label + hint" (stop words removed, ≥2 chars).
    - Each CL term scored by keyword overlap; zero-overlap terms excluded.
    - Anchor set: terms matching original label keywords guaranteed inclusion.
    - Synonym expansion: maps author-vocabulary words to OBO vocabulary
      (e.g. excitatory → glutamatergic) via OBO synonym lines.
    - Parent subtree boost: in-lineage terms receive a 2× score multiplier.
    - Top 100 candidates returned.

-------------------------------------------------------------------------------
STEP 5 — LLM resolution  (_resolve_with_llm)   [skipped with --no-llm]
-------------------------------------------------------------------------------
Two LLMs are queried per label:
  Primary:   Claude (claude-sonnet-4-6)  via ANTHROPIC_API_KEY
  Secondary: Gemini 2.5 Flash            via GOOGLE_API_KEY
Each call retried up to 2× (5 s / 10 s backoff) on 529/429 overload.

Main pass: pick the most specific matching CL term.
  Both agree     → llm_consensus
  Disagree       → more specific (ontology child) wins; Claude on tie
  One NONE       → other model's answer accepted
  Both NONE      → proceeds to ancestor fallback

Ancestor fallback: relaxed prompt allows dropping maturity/region qualifiers.
  Result tagged llm_ancestor.

Post-resolution: species-specific terms (e.g. "(Mus musculus)") replaced by
nearest species-agnostic is_a ancestor.

-------------------------------------------------------------------------------
STEP 6 — Glossary hint cascade fallback
-------------------------------------------------------------------------------
For labels still unresolved after LLM, the plain-English glossary hint is used
as input to the exact/synonym/fuzzy cascade.  Progressively shorter sub-phrases
are tried by dropping leading words (e.g. "Postnatal excitatory neuron" →
"excitatory neuron") until a match is found.  Requires at least 2 words.
Result tagged glossary_<method> (e.g. glossary_fuzzy).

-------------------------------------------------------------------------------
STEP 7 — CellxGene fallback
-------------------------------------------------------------------------------
For labels still unresolved after Step 6, the cellxgene CL term ID associated
with that author label is used — provided it is unambiguous (no pipe-separated
multiple values).  Result tagged cellxgene_fallback.

-------------------------------------------------------------------------------
STEP 8 — Output TSV + provenance header
-------------------------------------------------------------------------------
The output file begins with comment lines:
  # generated: <ISO timestamp>
  # input: <path>  column: <col>  obo: <path>
Followed by a header row and one data row per unique label.

-------------------------------------------------------------------------------
STEP 9 — Hierarchy consistency check  (stdout only)
-------------------------------------------------------------------------------
For each group of labels sharing the same immediate parent value, the Most
Informative Common Ancestor (MICA) of all resolved CL IDs is computed:
  IC > 0.1                 → consistent
  IC ≤ 0.1                 → weak
  MICA = CL:0000000 (cell) → WARNING
  MICA not within parent's subtree → WARNING
Individual terms outside the parent's subtree are flagged [!].
"""

import argparse
import os
import re
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="networkx")

import pandas as pd
import requests

from obo_parser import parse_obo_names
from annotate_cl_terms import fuzzy_normalize, parse_obo_synonyms, query_llm_mapping, query_llm_ancestor, query_llm_label, FatalAPIError, _load_prompt
from column_hierarchy import _is_cl_id, _is_metadata_column, _get_cellxgene_cols, detect_leaf_column, _is_functional


def build_label_mapping(obo_path, cl_names=None):
    """Build lookup tables for resolving cell type names to CL term IDs."""
    if cl_names is None:
        cl_names = parse_obo_names(obo_path)
    name_to_id = {name.lower(): cl_id for cl_id, name in cl_names.items()}
    synonym_to_id = parse_obo_synonyms(obo_path)
    fuzzy_name_to_id = {fuzzy_normalize(name): cl_id for cl_id, name in cl_names.items()}
    fuzzy_synonym_to_id = {fuzzy_normalize(syn): cl_id for syn, cl_id in synonym_to_id.items()}
    return cl_names, name_to_id, synonym_to_id, fuzzy_name_to_id, fuzzy_synonym_to_id


def resolve_name(name, name_to_id, synonym_to_id, fuzzy_name_to_id,
                 fuzzy_synonym_to_id, cl_names):
    """Resolve a single cell type name to a CL term ID (exact → synonym → fuzzy)."""
    name_lower = name.lower().strip()
    if name_lower in name_to_id:
        cl_id = name_to_id[name_lower]
        return cl_id, cl_names.get(cl_id, ''), 'exact'
    if name_lower in synonym_to_id:
        cl_id = synonym_to_id[name_lower]
        return cl_id, cl_names.get(cl_id, ''), 'synonym'
    fuzzy = fuzzy_normalize(name)
    if fuzzy in fuzzy_name_to_id:
        cl_id = fuzzy_name_to_id[fuzzy]
        return cl_id, cl_names.get(cl_id, ''), 'fuzzy'
    if fuzzy in fuzzy_synonym_to_id:
        cl_id = fuzzy_synonym_to_id[fuzzy]
        return cl_id, cl_names.get(cl_id, ''), 'fuzzy_synonym'
    return None, '', None


def fetch_doi_context(doi):
    """Fetch title and abstract for a DOI via the CrossRef API.

    Returns:
        A context string combining title and abstract, or None on failure.
    """
    url = f"https://api.crossref.org/works/{doi}"
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "generate_remap/1.0 (cell-type-annotation)"})
        resp.raise_for_status()
        msg = resp.json().get("message", {})
        title = ' '.join(msg.get("title", []))
        abstract = msg.get("abstract", "")
        # Strip JATS XML tags that CrossRef sometimes includes
        abstract = re.sub(r'<[^>]+>', ' ', abstract).strip()
        parts = [p for p in [title, abstract] if p]
        return ' '.join(parts) if parts else None
    except Exception as e:
        print(f"  Warning: could not fetch DOI context: {e}")
        return None


def extract_pdf_text(pdf_path, max_pages=20, start_page=0):
    """Extract plain text from a PDF, capped at max_pages to avoid supplements.

    Returns:
        Extracted text string, or None on failure.
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        pages = reader.pages[start_page:start_page + max_pages]
        text = '\n'.join(page.extract_text() or '' for page in pages)
        return text.strip() or None
    except Exception as e:
        print(f"  Warning: could not read PDF: {e}")
        return None


def extract_cell_type_glossary(pdf_text, api='claude'):
    """Ask an LLM to extract a compact cell type glossary from paper text.

    Makes a single LLM call and returns a concise glossary string for use
    as context in subsequent label queries.

    Returns:
        Glossary string, or None on failure.
    """
    # Truncate to ~6000 words — abbreviations/glossaries appear early in papers
    words = pdf_text.split()
    if len(words) > 6000:
        pdf_text = ' '.join(words[:6000]) + ' [truncated]'

    prompt = _load_prompt("glossary_extraction.txt", pdf_text=pdf_text)
    def _call():
        if api == 'claude':
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        elif api == 'gemini':
            from google import genai as google_genai
            client = google_genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            return ''.join(p.text for p in response.candidates[0].content.parts if hasattr(p, 'text')).strip()
        return None

    for attempt in range(3):
        try:
            return _call()
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ('credit', 'billing', 'unauthorized', 'invalid_api_key', 'permission')):
                print(f"  Warning: glossary extraction failed (fatal): {e}")
                return None
            if attempt < 2 and any(k in err_str for k in ('529', 'overloaded', '429', 'rate_limit', 'timeout')):
                wait = 10 * (attempt + 1)
                print(f"  Warning: {api} overloaded, retrying in {wait}s...")
                import time
                time.sleep(wait)
                continue
            print(f"  Warning: glossary extraction failed: {e}")
            return None
    return None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Auto-detect annotation columns and generate a remap TSV"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Ground truth TSV file with annotation columns")
    parser.add_argument("--obo", type=str, required=True,
                        help="Cell Ontology OBO file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output remap TSV path (default: <input_stem>_remap.tsv alongside input)")
    parser.add_argument("--column", type=str, default=None,
                        help="Annotation column to use (auto-detect if omitted)")
    parser.add_argument("--min-score", type=float, default=0.10,
                        help="Minimum resolvability fraction to consider a column (default: 0.10)")
    parser.add_argument("--doi", type=str, default=None,
                        help="DOI of the source paper (e.g. 10.1038/s41586-024-12345-6). "
                             "Abstract and title are fetched from CrossRef and used as "
                             "context when generating cell labels.")
    parser.add_argument("--pdf", type=str, default=None,
                        help="Path to the source paper PDF. The LLM extracts a cell type "
                             "glossary in one upfront call, which is then used as context "
                             "for all label queries. Overrides --doi if both are given.")
    parser.add_argument("--context", type=str, default=None,
                        help="Free-text biological context passed directly to the LLM "
                             "(e.g. 'brain organoid scRNA-seq; IPC = Intermediate Progenitor "
                             "Cell'). Useful when no paper is available. Overridden by --pdf.")
    parser.add_argument("--no-llm", action="store_true", default=False,
                        help="Skip LLM resolution entirely. Unresolved labels fall back to "
                             "glossary hint cascade then no_cl_term_found.")
    parser.add_argument("--cellxgene-fallback", nargs="+", metavar="LABEL", default=[],
                        help="Labels to resolve directly from cellxgene CL term, skipping "
                             "all other resolution steps (e.g. --cellxgene-fallback EN EN-fetal-early).")
    return parser


def score_column(values, name_to_id, synonym_to_id, fuzzy_name_to_id,
                 fuzzy_synonym_to_id, cl_names):
    """Score a column by the fraction of unique labels that resolve to CL terms.

    Returns:
        Tuple of (score, n_unique, n_resolved).
    """
    unique_labels = set(str(v).strip() for v in values
                        if pd.notna(v) and str(v).strip())
    if not unique_labels:
        return 0.0, 0, 0

    n_resolved = 0
    for label in unique_labels:
        if _is_cl_id(label):
            n_resolved += 1
            continue
        cl_id, _, method = resolve_name(
            label, name_to_id, synonym_to_id,
            fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)
        if cl_id is not None:
            n_resolved += 1

    score = n_resolved / len(unique_labels)
    return score, len(unique_labels), n_resolved


# ---------------------------------------------------------------------------
# Species qualifier patterns
# ---------------------------------------------------------------------------
# Matches CL term names that end with a parenthesised species name, e.g.
# "(Mus musculus)", "(Homo sapiens)", "(Mmus)", "(Hsap)".
_SPECIES_QUALIFIER = re.compile(
    r'\(\s*(?:Mus musculus|Homo sapiens|Rattus norvegicus|Mmus|Hsap|[A-Z][a-z]+ [a-z]+)\s*\)\s*$'
)


def _despecify_cl_term(cl_id, cl_names, G):
    """Return the nearest species-agnostic is_a ancestor of a species-specific CL term.

    If cl_id's canonical name ends with a species qualifier (e.g. '(Mus musculus)'),
    walks up the is_a hierarchy (edges point child→parent) level by level.  At each
    level, all species-agnostic candidates are scored by keyword overlap with the
    original term name (species qualifier stripped) and the best-scoring one is
    returned.  This prevents picking a broad morphology parent (e.g. 'bipolar neuron')
    over a specific functional parent (e.g. 'VIP GABAergic interneuron') when both
    are direct is_a parents.
    Returns (cl_id, cl_name) unchanged when already species-agnostic or no agnostic
    ancestor exists.
    """
    name = cl_names.get(cl_id, '')
    if not _SPECIES_QUALIFIER.search(name):
        return cl_id, name

    # Strip species qualifier from original name for keyword scoring
    base_name = _SPECIES_QUALIFIER.sub('', name).strip().lower()
    base_words = set(re.split(r'[\s\-_/]+', base_name)) - {'', 'cell', 'the', 'of', 'a'}

    def _score(parent_name):
        pwords = set(re.split(r'[\s\-_/]+', parent_name.lower())) - {'', 'cell', 'the', 'of', 'a'}
        return len(base_words & pwords)

    visited = {cl_id}
    # BFS level by level so we prefer closer ancestors
    current_level = [p for p in G.successors(cl_id) if p.startswith('CL:')]
    while current_level:
        next_level = []
        agnostic = []
        for parent_id in current_level:
            if parent_id in visited:
                continue
            visited.add(parent_id)
            parent_name = cl_names.get(parent_id, '')
            if not parent_name:
                next_level.extend(p for p in G.successors(parent_id) if p.startswith('CL:'))
                continue
            if _SPECIES_QUALIFIER.search(parent_name):
                next_level.extend(p for p in G.successors(parent_id) if p.startswith('CL:'))
            else:
                agnostic.append((parent_id, parent_name))

        if agnostic:
            # Pick best keyword-overlap match among agnostic parents at this level
            best_id, best_name = max(agnostic, key=lambda x: _score(x[1]))
            return best_id, best_name

        current_level = next_level

    return cl_id, name  # no agnostic ancestor found — keep original


# ---------------------------------------------------------------------------
# Marker gene pre-filter
# ---------------------------------------------------------------------------
# Maps token strings (as they appear in author labels/hints after splitting on
# whitespace and [-_/] separators) to the canonical string used in CL term
# names.  When a label/hint contains one of these tokens, candidate CL terms
# are restricted to those whose name contains the canonical marker string —
# before the LLM call.  This makes marker gene resolution deterministic and
# independent of the LLM prompt.
#
# Extend this dict when working with a new dataset that introduces new marker
# abbreviations.  Two-letter tokens (cr, cb) are intentionally omitted to
# avoid false positives.
MARKER_GENES = {
    'vip':        'vip',
    'sst':        'sst',
    'pv':         'pvalb',
    'pvalb':      'pvalb',
    'parvalbumin':'pvalb',
    'sncg':       'sncg',
    'lamp5':      'lamp5',
    'cck':        'cck',
    'npy':        'npy',
}

_UNINFORMATIVE_LABELS = re.compile(
    r'^(unknown|unassigned|unannotated|other|na|n/a|none|ambiguous|doublet|'
    r'low quality|low_quality|noise|artifact|mixed|undefined)s?$',
    re.IGNORECASE
)


def _apply_marker_filter(candidates, label, hint):
    """Restrict candidates to marker-specific CL terms if label/hint contains a marker gene.

    Tokenises ``label + " " + hint`` on whitespace and [-_/] separators and
    checks each token against MARKER_GENES.  If a match is found, the
    candidate dict is filtered to only those CL terms whose name contains the
    canonical marker string (case-insensitive).  The filter is skipped if
    fewer than 3 marker-specific candidates would remain, so the LLM is never
    left with a trivially small or empty set.

    Returns the (possibly filtered) candidates dict.
    """
    tokens = set(w.lower() for w in re.split(r'[\s\-_/]+', f"{label} {hint}") if len(w) >= 2)
    for token in tokens:
        canonical = MARKER_GENES.get(token)
        if canonical:
            filtered = {cl_id: name for cl_id, name in candidates.items()
                        if canonical in name.lower()}
            if len(filtered) >= 1:
                return filtered
    return candidates


def build_keyword_expansion(cl_names, synonym_to_id):
    """Build a word-level keyword expansion map from OBO synonyms.

    For each synonym phrase, maps each word in that phrase to the canonical
    words of the corresponding CL term name.  This allows _filter_candidate_terms
    to expand author-label keywords (e.g. "excitatory") into ontology vocabulary
    (e.g. "glutamatergic") so the right candidate terms surface for the LLM.

    Returns:
        dict {word: set_of_canonical_words} — only entries where the synonym
        word differs from the canonical term's vocabulary are included.
    """
    expansion = {}
    for syn_phrase, cl_id in synonym_to_id.items():
        canonical = cl_names.get(cl_id, '')
        if not canonical:
            continue
        syn_words = set(w.lower() for w in re.split(r'[\s\-_/]+', syn_phrase) if len(w) >= 2)
        can_words = set(w.lower() for w in re.split(r'[\s\-_/]+', canonical) if len(w) >= 2)
        for w in syn_words - can_words:
            expansion.setdefault(w, set()).update(can_words)
    return expansion


def _filter_candidate_terms(label, cl_names, max_candidates=100, expansion=None, parent_subtree=None):
    """Select a focused subset of CL terms relevant to a label.

    Extracts keywords from the label, optionally expands them via the OBO
    synonym expansion map, scores CL terms by keyword overlap, and returns
    the top max_candidates by score. Falls back to a capped sample of the
    full dict if no keywords match.
    """
    # Extract keywords (2+ chars, lowercased, skip common stop words)
    stop = {'of', 'the', 'and', 'or', 'in', 'to', 'a', 'an', 'by', 'is',
            'at', 'on', 'for', 'from', 'with', 'derived', 'type', 'cell'}
    words = set(w.lower() for w in re.split(r'[\s\-_/]+', label)
                if len(w) >= 2 and w.lower() not in stop)

    if not words:
        return dict(list(cl_names.items())[:max_candidates])

    # Score each CL term by how many original keywords it shares with the label.
    # Original-keyword matches are always included (anchor set) so that expansion
    # cannot crowd out direct matches (e.g. "astrocyte" for "Astrocyte-Immature").
    original_words = words
    scored = {}
    for cl_id, name in cl_names.items():
        name_words = set(w.lower() for w in re.split(r'[\s\-_/]+', name))
        overlap = len(original_words & name_words)
        if overlap > 0:
            scored[cl_id] = (name, overlap)

    if len(scored) < 5:
        # Too few matches — broaden: include partial substring matches
        for cl_id, name in cl_names.items():
            if cl_id in scored:
                continue
            name_lower = name.lower()
            if any(w in name_lower for w in original_words):
                scored[cl_id] = (name, 0.5)

    # Anchor set: all terms matching original keywords (guaranteed inclusion)
    anchor_ids = set(scored.keys())

    # Expand keywords using OBO synonym map (e.g. "excitatory" → "glutamatergic")
    # and score additional terms, but only fill slots not taken by anchors.
    if expansion:
        extra_words = set()
        for w in original_words:
            if w in expansion:
                extra_words.update(expansion[w] - stop)
        if extra_words:
            expanded_words = original_words | extra_words
            for cl_id, name in cl_names.items():
                if cl_id in scored:
                    continue
                name_words = set(w.lower() for w in re.split(r'[\s\-_/]+', name))
                overlap = len(expanded_words & name_words)
                if overlap > 0:
                    scored[cl_id] = (name, overlap * 0.9)  # slightly lower priority than anchors

    # Apply parent subtree boost: terms within the parent's ontology subtree get
    # a 2x score multiplier so they rank above same-scoring out-of-lineage terms.
    if parent_subtree:
        scored = {
            cl_id: (name, score * 2.0 if cl_id in parent_subtree else score)
            for cl_id, (name, score) in scored.items()
        }

    # Return top max_candidates: anchors first, then expansion terms by score
    anchor_top = sorted(
        ((cl_id, data) for cl_id, data in scored.items() if cl_id in anchor_ids),
        key=lambda x: x[1][1], reverse=True
    )[:max_candidates]
    remaining_slots = max_candidates - len(anchor_top)
    expansion_top = sorted(
        ((cl_id, data) for cl_id, data in scored.items() if cl_id not in anchor_ids),
        key=lambda x: x[1][1], reverse=True
    )[:remaining_slots]

    combined = {cl_id: name for cl_id, (name, _) in anchor_top + expansion_top}
    return combined if combined else dict(list(cl_names.items())[:max_candidates])


def _apply_voting(label, results, cl_names, resolved, method_prefix, G=None):
    """Apply consensus/preference voting and store result in resolved dict.

    When both LLMs return different CL terms, checks if one is an ontology
    ancestor of the other — if so, the more specific (child) term wins
    regardless of which LLM provided it.  Falls back to Claude on disagreement
    when neither answer is a descendant of the other.

    Prints the outcome. method_prefix is 'llm' or 'llm_ancestor'.
    """
    if len(results) == 2 and results.get('claude') == results.get('gemini'):
        cl_id = results['claude']
        cl_name = cl_names.get(cl_id, '')
        resolved[label] = (cl_id, cl_name, f'{method_prefix}_consensus' if method_prefix == 'llm' else method_prefix)
        print(f" {cl_id} ({cl_name}) [consensus]")
    elif len(results) == 2:
        claude_id = results['claude']
        gemini_id = results['gemini']
        winner, winner_api = claude_id, 'claude'

        # If one answer is an ontology ancestor of the other, pick the child (more specific).
        if G is not None:
            try:
                import networkx as nx
                # In this graph edges go parent→child, so nx.ancestors(G, node)
                # returns subtypes (children) of node, not supertypes.
                # claude_id in nx.ancestors(G, gemini_id) means Claude is a subtype
                # of Gemini → Claude is more specific → keep Claude.
                # gemini_id in nx.ancestors(G, claude_id) means Gemini is a subtype
                # of Claude → Gemini is more specific → use Gemini.
                claude_subtypes = set(nx.ancestors(G, claude_id))
                gemini_subtypes = set(nx.ancestors(G, gemini_id))
                if gemini_id in claude_subtypes:
                    # Gemini is a subtype of Claude → Gemini is more specific
                    winner, winner_api = gemini_id, 'gemini'
                elif claude_id in gemini_subtypes:
                    # Claude is a subtype of Gemini → Claude is more specific
                    pass  # keep claude
            except Exception:
                pass  # fall back to Claude on graph errors

        cl_name = cl_names.get(winner, '')
        loser_id = gemini_id if winner_api == 'claude' else claude_id
        loser_name = cl_names.get(loser_id, '')
        loser_api = 'gemini' if winner_api == 'claude' else 'claude'
        method = f'{method_prefix}_{winner_api}' if method_prefix == 'llm' else method_prefix
        resolved[label] = (winner, cl_name, method)
        print(f" {winner} ({cl_name}) [{winner_api}] vs {loser_id} ({loser_name}) [{loser_api}] — using {winner_api}")
    elif results:
        api_used = 'claude' if 'claude' in results else 'gemini'
        cl_id = results[api_used]
        cl_name = cl_names.get(cl_id, '')
        resolved[label] = (cl_id, cl_name, f'{method_prefix}_{api_used}' if method_prefix == 'llm' else method_prefix)
        print(f" {cl_id} ({cl_name}) [{api_used}]")
    else:
        print(" NONE")


def _resolve_with_llm(unresolved_labels, cl_names, paper_context=None, label_hints=None, expansion=None, label_to_parent_subtree=None, G=None, n_workers=10):
    """Attempt LLM resolution for a list of unresolved labels.

    Fires all label×API combinations in parallel (ThreadPoolExecutor), then
    applies voting sequentially in original label order.  n_workers controls
    concurrency — tune down if hitting API rate limits.
    Returns dict {label: (cl_id, cl_name, method)} for successfully resolved labels.
    """
    has_anthropic = bool(os.environ.get('ANTHROPIC_API_KEY'))
    has_gemini = bool(os.environ.get('GOOGLE_API_KEY'))

    if not has_anthropic and not has_gemini:
        print("\n  Error: ANTHROPIC_API_KEY or GOOGLE_API_KEY is required")
        print("  Set one of these environment variables:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("    export GOOGLE_API_KEY=sk-...")
        return {}

    apis = []
    if has_anthropic:
        apis.append('claude')
    if has_gemini:
        apis.append('gemini')
    print(f"\n  LLM CL term resolution using: {', '.join(apis)} (n_workers={n_workers})")

    # Pre-compute candidates for all labels (CPU-bound, fast) before firing threads.
    # Apply marker filter to full cl_names first so marker-specific terms are
    # never crowded out of the top-100 candidate cut by unrelated high-scoring terms.
    label_data = {}
    for label in unresolved_labels:
        hint = label_hints.get(label, label) if label_hints else label
        query = f"{label} {hint}" if hint != label else label
        parent_subtree = label_to_parent_subtree.get(label) if label_to_parent_subtree else None
        pool = _apply_marker_filter(cl_names, label, hint)
        candidates = _filter_candidate_terms(query, pool, expansion=expansion, parent_subtree=parent_subtree)
        label_data[label] = (hint, candidates)

    disabled_apis = set()
    _lock = threading.Lock()

    def _submit_calls(labels, query_fn):
        """Fire all label×api combinations in parallel; return {label: {api: cl_id}}."""
        raw = {lbl: {} for lbl in labels}
        active = [a for a in apis if a not in disabled_apis]
        tasks = [(lbl, a) for lbl in labels for a in active]
        if not tasks:
            return raw

        def _call(lbl, api):
            with _lock:
                if api in disabled_apis:
                    return lbl, api, None, None
            _, candidates = label_data[lbl]
            try:
                cl_id = query_fn(lbl, candidates, api=api, paper_context=paper_context)
                return lbl, api, cl_id, None
            except FatalAPIError as e:
                return lbl, api, None, e

        with ThreadPoolExecutor(max_workers=min(len(tasks), n_workers)) as ex:
            futures = {ex.submit(_call, lbl, api): (lbl, api) for lbl, api in tasks}
            for fut in as_completed(futures):
                lbl, api, cl_id, err = fut.result()
                if err:
                    print(f"\n  {err} — disabling {api} for remaining labels")
                    with _lock:
                        disabled_apis.add(api)
                elif cl_id is not None:
                    raw[lbl][api] = cl_id
        return raw

    # Main mapping pass — fire all in parallel, vote sequentially in label order
    raw = _submit_calls(unresolved_labels, query_llm_mapping)
    resolved = {}
    for label in unresolved_labels:
        hint, candidates = label_data[label]
        print(f"    \"{label}\" ({len(candidates)} candidate terms)...", end="", flush=True)
        results = {}
        for api, cl_id in raw[label].items():
            if cl_id == 'NONE':
                print(f" [{api}: no match]", end="", flush=True)
            elif cl_id and cl_id != 'CL:0000000':
                results[api] = cl_id
        _apply_voting(label, results, cl_names, resolved, method_prefix='llm', G=G)

    # Ancestor fallback: for labels still unresolved, try closest parent CL term
    still_unresolved = [l for l in unresolved_labels if l not in resolved]
    if still_unresolved:
        print(f"\n  Ancestor fallback for {len(still_unresolved)} labels...")
        raw_anc = _submit_calls(still_unresolved, query_llm_ancestor)
        for label in still_unresolved:
            hint, candidates = label_data[label]
            print(f"    \"{label}\" → \"{hint[:50]}\" ({len(candidates)} candidate terms)...", end="", flush=True)
            results = {}
            claude_vetoed = raw_anc[label].get('claude') == 'NONE'
            for api, cl_id in raw_anc[label].items():
                if cl_id == 'NONE':
                    if api == 'claude':
                        print(f" [claude: no ancestor]", end="", flush=True)
                    else:
                        print(f" [gemini: no ancestor]", end="", flush=True)
                elif cl_id:
                    if api == 'gemini' and claude_vetoed:
                        print(f" [claude vetoed gemini ancestor {cl_id} ({cl_names.get(cl_id, '?')})]", end="", flush=True)
                    else:
                        results[api] = cl_id
            _apply_voting(label, results, cl_names, resolved, method_prefix='llm_ancestor', G=G)

    return resolved


def _parse_glossary(paper_context):
    """Parse a glossary string into a {normalized_key: value} lookup dict."""
    lookup = {}
    if not paper_context:
        return lookup
    for line in paper_context.splitlines():
        sep = ':' if ':' in line else ('=' if '=' in line else None)
        if sep:
            key, _, val = line.partition(sep)
            key = key.strip().lower()
            key = re.sub(r'[–—‒]', '-', key)
            val = val.strip()
            if key and val:
                lookup[key] = val
    return lookup


def _glossary_covers(label, lookup):
    """Return True if the label or any of its tokens appears as a glossary key."""
    lbl_lower = label.lower().strip()
    lbl_norm = re.sub(r'[–—‒]', '-', lbl_lower)
    if (lookup.get(lbl_lower) or lookup.get(lbl_norm) or
            lookup.get(lbl_lower + 's') or lookup.get(lbl_lower.rstrip('s'))):
        return True
    tokens = [t for t in re.split(r'[\s\-_/]+', lbl_norm) if len(t) >= 2]
    return any(lookup.get(t) or lookup.get(t.rstrip('s')) or lookup.get(t + 's')
               for t in tokens)


def generate_remap(df, column, obo_path, paper_context=None, use_llm=True, cellxgene_fallback_labels=None):
    """Generate a remap table from a column of labels.

    Args:
        df: DataFrame with the annotation column.
        column: Column name to remap.
        obo_path: Path to OBO file.
        paper_context: Optional paper title/abstract for LLM context.

    Returns:
        Tuple of (remap_df, summary_report).
    """
    cl_names, name_to_id, synonym_to_id, fuzzy_name_to_id, fuzzy_synonym_to_id = \
        build_label_mapping(obo_path)
    keyword_expansion = build_keyword_expansion(cl_names, synonym_to_id)

    unique_labels = sorted(set(
        str(v).strip() for v in df[column]
        if pd.notna(v) and str(v).strip()
    ))

    # Build cellxgene lookup: author label -> (cell_type values, cell_type_ontology_term_id values)
    has_cellxgene = 'cell_type' in df.columns and 'cell_type_ontology_term_id' in df.columns
    cellxgene_lookup = {}
    if has_cellxgene:
        for label, grp in df.groupby(column):
            label = str(label).strip()
            types = sorted(grp['cell_type'].dropna().astype(str).unique())
            ids = sorted(grp['cell_type_ontology_term_id'].dropna().astype(str).unique())
            cellxgene_lookup[label] = (' | '.join(types), ' | '.join(ids))

    # Parse glossary before the cascade so it can override synonym/fuzzy matches
    # for labels that appear in the paper context (e.g. IPC = Intermediate Progenitor Cell,
    # not plasmacytoid dendritic cell).
    glossary_lookup = {}
    if paper_context:
        for line in paper_context.splitlines():
            # Accept both "KEY: value" and "KEY = value" separators
            sep = ':' if ':' in line else ('=' if '=' in line else None)
            if sep:
                key, _, val = line.partition(sep)
                key = key.strip().lower()
                key = re.sub(r'[\u2013\u2014\u2012]', '-', key)  # normalize dashes
                val = val.strip()
                if key and val:
                    glossary_lookup[key] = val

    def _in_glossary(label):
        lbl_lower = label.lower()
        lbl_norm = re.sub(r'[\u2013\u2014\u2012]', '-', lbl_lower)
        return (glossary_lookup.get(lbl_lower)
                or glossary_lookup.get(lbl_norm)
                or glossary_lookup.get(lbl_lower + 's')
                or glossary_lookup.get(lbl_lower.rstrip('s')))

    _STOPWORDS = {'cell', 'cells', 'type', 'types', 'and', 'or', 'of', 'the', 'a', 'an'}

    def _glossary_contradicts(glossary_hit, canon_name):
        """Return True if glossary definition and CL name share no meaningful tokens.

        Checks both exact token overlap and prefix overlap (e.g. 'microglia' vs
        'microglial') to avoid false contradictions from morphological variants.
        """
        gloss_tokens = set(re.findall(r'\w+', glossary_hit.lower())) - _STOPWORDS
        cl_tokens = set(re.findall(r'\w+', canon_name.lower())) - _STOPWORDS
        if not gloss_tokens or not cl_tokens:
            return False
        if gloss_tokens & cl_tokens:
            return False
        # prefix/substring match: 'microglia' vs 'microglial', 'glial' vs 'microglial'
        for g in gloss_tokens:
            for c in cl_tokens:
                if len(g) >= 4 and len(c) >= 4 and (
                    g.startswith(c) or c.startswith(g) or g in c or c in g
                ):
                    return False
        return True

    print(f"\nRunning exact/synonym/fuzzy cascade on {len(unique_labels)} unique labels...")
    rows = []
    unresolved_labels = []
    method_counts = {'exact': 0, 'synonym': 0, 'fuzzy': 0, 'fuzzy_synonym': 0,
                     'already_cl': 0, 'llm_consensus': 0, 'llm_claude': 0,
                     'llm_gemini': 0, 'llm_ancestor': 0, 'no_cl_term_found': 0}

    cellxgene_fallback_set = set(cellxgene_fallback_labels or [])

    for label in unique_labels:
        cxg_type, cxg_id = cellxgene_lookup.get(label, ('', ''))

        if label in cellxgene_fallback_set:
            cxg_cl_id = cxg_id if cxg_id and '|' not in cxg_id else ''
            cxg_cl_name = cl_names.get(cxg_cl_id, '') if cxg_cl_id else ''
            if cxg_cl_id and cxg_cl_name:
                print(f"  [cellxgene_fallback] \"{label}\" → {cxg_cl_id} ({cxg_cl_name})")
                rows.append({
                    'original_label': label,
                    'cl_term_id': cxg_cl_id,
                    'cl_term_name': cxg_cl_name,
                    'cl_term_match_method': 'cellxgene_fallback',
                    'cellxgene_cell_type': cxg_type,
                    'cellxgene_cell_type_ontology_term_id': cxg_id,
                })
                method_counts['cellxgene_fallback'] = method_counts.get('cellxgene_fallback', 0) + 1
            else:
                print(f"  [cellxgene_fallback] \"{label}\" — no unambiguous cellxgene CL term, leaving unresolved")
                rows.append({
                    'original_label': label,
                    'cl_term_id': '', 'cl_term_name': '',
                    'cl_term_match_method': 'no_cl_term_found',
                    'cellxgene_cell_type': cxg_type,
                    'cellxgene_cell_type_ontology_term_id': cxg_id,
                })
                method_counts['no_cl_term_found'] += 1
            continue

        if _UNINFORMATIVE_LABELS.match(label):
            rows.append({
                'original_label': label,
                'cl_term_id': '',
                'cl_term_name': '',
                'cl_term_match_method': 'no_cl_term_found',
                'cellxgene_cell_type': cxg_type,
                'cellxgene_cell_type_ontology_term_id': cxg_id,
                'cell_label': label,
                'cell_label_method': 'original',
            })
            method_counts['no_cl_term_found'] += 1
            continue
        if _is_cl_id(label):
            rows.append({
                'original_label': label,
                'cl_term_id': label,
                'cl_term_name': cl_names.get(label, ''),
                'cl_term_match_method': 'already_cl',

                'cellxgene_cell_type': cxg_type,
                'cellxgene_cell_type_ontology_term_id': cxg_id,
            })
            method_counts['already_cl'] += 1
            continue

        cl_id, canon_name, method = resolve_name(
            label, name_to_id, synonym_to_id,
            fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)

        # If cascade found a synonym/fuzzy match but the glossary defines this label
        # with a contradicting meaning, defer to LLM — the glossary context is more
        # reliable (e.g. IPC → plasmacytoid dendritic cell contradicts "Intermediate
        # progenitor cells"). Exact matches are always trusted regardless of glossary.
        if cl_id is not None and method in ('synonym', 'fuzzy_synonym'):
            glossary_hit = _in_glossary(label)
            # Only override if the glossary VALUE contradicts the CL name AND the
            # glossary KEY itself is not similar to the CL name. If the key is
            # similar (e.g. "microglia" ~ "microglial cell"), the synonym match is
            # trustworthy even if the description uses different vocabulary.
            if glossary_hit and _glossary_contradicts(glossary_hit, canon_name):
                label_norm = re.sub(r'[\u2013\u2014\u2012]', '-', label.lower())
                key_tokens = set(re.findall(r'\w+', label_norm)) - _STOPWORDS
                cl_tokens = set(re.findall(r'\w+', canon_name.lower())) - _STOPWORDS
                key_similar = any(
                    len(k) >= 4 and len(c) >= 4 and (k == c or k in c or c in k)
                    for k in key_tokens for c in cl_tokens
                )
                if not key_similar:
                    print(f"  [glossary override] \"{label}\" — [{method}] {cl_id} ({canon_name}) contradicts glossary \"{glossary_hit}\"")
                    unresolved_labels.append(label)
                    continue

        if cl_id is not None:
            rows.append({
                'original_label': label,
                'cl_term_id': cl_id,
                'cl_term_name': canon_name,
                'cl_term_match_method': method,

                'cellxgene_cell_type': cxg_type,
                'cellxgene_cell_type_ontology_term_id': cxg_id,
            })
            method_counts[method] += 1
            print(f"  [{method}] \"{label}\" → {cl_id} ({canon_name})")
        else:
            unresolved_labels.append(label)

    label_hints = {}       # {label: plain_english_name}
    label_hint_method = {} # {label: 'glossary'|'llm'|'original'}
    if unresolved_labels:
        label_api = 'claude' if os.environ.get('ANTHROPIC_API_KEY') else \
                    'gemini' if os.environ.get('GOOGLE_API_KEY') else None
        print(f"\n  Generating cell labels for {len(unresolved_labels)} unresolved labels (via {label_api})...")
        for lbl in unresolved_labels:
            print(f"    \"{lbl}\"...", end="", flush=True)
            # Check glossary first before calling LLM.
            # Normalize dashes (em-dash, en-dash → hyphen) to handle LLM formatting
            # variation in extracted glossary keys, then also try plural forms.
            lbl_lower = lbl.lower()
            lbl_norm = re.sub(r'[\u2013\u2014\u2012]', '-', lbl_lower)
            # Full-label match first (handles paper-extracted glossaries)
            glossary_hit = (glossary_lookup.get(lbl_lower)
                            or glossary_lookup.get(lbl_norm)
                            or glossary_lookup.get(lbl_lower + 's')
                            or glossary_lookup.get(lbl_lower.rstrip('s')))
            # Token-level match: expand abbreviation tokens within the label
            # (e.g. "ENs - Activated" → token "en" → "excitatory neuron")
            if not glossary_hit and glossary_lookup:
                tokens = re.split(r'[\s\-_/]+', lbl_norm)
                expanded_parts = []
                matched = False
                for tok in tokens:
                    hit = (glossary_lookup.get(tok)
                           or glossary_lookup.get(tok.rstrip('s'))
                           or glossary_lookup.get(tok + 's'))
                    if hit:
                        expanded_parts.append(hit)
                        matched = True
                    else:
                        expanded_parts.append(tok)
                if matched:
                    glossary_hit = ' '.join(expanded_parts)
            if glossary_hit:
                label_hints[lbl] = glossary_hit
                label_hint_method[lbl] = 'glossary'
                print(f" {glossary_hit[:60]} [glossary]")
                continue
            try:
                desc = query_llm_label(lbl, api=label_api, paper_context=paper_context) if label_api else None
            except FatalAPIError:
                desc = None
            if desc and desc != lbl:
                label_hints[lbl] = desc
                label_hint_method[lbl] = 'llm'
            else:
                label_hints[lbl] = lbl
                label_hint_method[lbl] = 'original'
            print(f" {label_hints[lbl][:60]}")

    # Build parent subtree map for boosting candidate scoring.
    # For each unresolved child label, resolve its parent label (exact/synonym/fuzzy)
    # and precompute the parent's ontology subtree for use in _filter_candidate_terms.
    label_to_parent_subtree = {}
    G = None
    cellxgene_cols = _get_cellxgene_cols(df)
    parent_cols = detect_parent_columns(df, column, exclude_cols=cellxgene_cols)
    if parent_cols and unresolved_labels:
        from ontology_utils import load_ontology
        import networkx as nx
        G = load_ontology(obo_path)
        # Build per-column lookup: child_label -> parent_val for each parent column
        col_lookups = []
        for parent_col, _ in parent_cols:
            lookup = (
                df[[column, parent_col]].drop_duplicates()
                .set_index(column)[parent_col].to_dict()
            )
            col_lookups.append(lookup)

        parent_subtree_cache = {}
        for label in unresolved_labels:
            # Try each parent level in order (most specific first) until one resolves.
            # This handles cases where the immediate parent label is the same as the
            # child (e.g. IPC-EN -> Subclass "IPC-EN" -> unresolvable) by falling
            # through to the next level (e.g. Class "Progenitor").
            subtree = None
            for lookup in col_lookups:
                parent_val = str(lookup.get(label, '')).strip()
                if not parent_val or parent_val == label:
                    continue  # same label or missing — try next level
                if parent_val not in parent_subtree_cache:
                    parent_cl_id, _, _ = resolve_name(
                        parent_val, name_to_id, synonym_to_id,
                        fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)
                    if parent_cl_id:
                        try:
                            parent_subtree_cache[parent_val] = (
                                set(nx.ancestors(G, parent_cl_id)) | {parent_cl_id}
                            )
                        except nx.NodeNotFound:
                            parent_subtree_cache[parent_val] = None
                    else:
                        parent_subtree_cache[parent_val] = None
                subtree = parent_subtree_cache.get(parent_val)
                if subtree:
                    break  # found a usable parent subtree
            if subtree:
                label_to_parent_subtree[label] = subtree

    # LLM pass for unresolved labels — uses label_hints for ancestor candidate filtering
    llm_resolved = {}
    if unresolved_labels and use_llm:
        print(f"\n  {len(unresolved_labels)} labels unresolved — trying LLM...")
        llm_resolved = _resolve_with_llm(unresolved_labels, cl_names,
                                         paper_context=paper_context,
                                         label_hints=label_hints,
                                         expansion=keyword_expansion,
                                         label_to_parent_subtree=label_to_parent_subtree or None,
                                         G=G)
    elif unresolved_labels and not use_llm:
        print(f"\n  {len(unresolved_labels)} labels unresolved — LLM skipped (--no-llm).")

    # De-specify any species-specific LLM results (e.g. "(Mus musculus)" terms).
    # If the resolved term has a species qualifier, replace it with the nearest
    # species-agnostic is_a ancestor.
    if llm_resolved:
        despecify_G = G
        for label, (cl_id, cl_name, method) in list(llm_resolved.items()):
            if _SPECIES_QUALIFIER.search(cl_name):
                if despecify_G is None:
                    from ontology_utils import load_ontology
                    despecify_G = load_ontology(obo_path)
                new_id, new_name = _despecify_cl_term(cl_id, cl_names, despecify_G)
                if new_id != cl_id:
                    print(f"    De-specified \"{label}\": {cl_id} ({cl_name}) → {new_id} ({new_name})")
                    llm_resolved[label] = (new_id, new_name, method)

    for label in unresolved_labels:
        cxg_type, cxg_id = cellxgene_lookup.get(label, ('', ''))
        if label in llm_resolved:
            cl_id, cl_name, method = llm_resolved[label]
            rows.append({
                'original_label': label,
                'cl_term_id': cl_id,
                'cl_term_name': cl_name,
                'cl_term_match_method': method,
                'cellxgene_cell_type': cxg_type,
                'cellxgene_cell_type_ontology_term_id': cxg_id,
            })
            method_counts[method] += 1
        else:
            # Glossary hint fallback: try cascade on the plain-English glossary
            # description (e.g. "postnatal excitatory neurons" for "EN") before
            # giving up. This catches broad labels the LLM won't pin to a specific
            # term but whose expanded name matches a CL ancestor.
            hint = label_hints.get(label, '')
            hint_cl_id, hint_cl_name, hint_method, matched_phrase = (None, '', None, '')
            if hint and hint.lower() != label.lower():
                # Try full hint first, then progressively drop leading words to find
                # a CL ancestor (e.g. "Postnatal excitatory neuron" → "excitatory neuron").
                # Require at least 2 words to avoid overly generic single-word matches.
                words = hint.split()
                for start in range(len(words) - 1):
                    phrase = ' '.join(words[start:])
                    cl_id_, cl_name_, method_ = resolve_name(
                        phrase, name_to_id, synonym_to_id,
                        fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)
                    if cl_id_:
                        hint_cl_id, hint_cl_name, hint_method, matched_phrase = \
                            cl_id_, cl_name_, f'glossary_{method_}', phrase
                        print(f"  [glossary_hint] \"{label}\" → {hint_cl_id} ({hint_cl_name}) via \"{matched_phrase}\"")
                        break

            if hint_cl_id:
                rows.append({
                    'original_label': label,
                    'cl_term_id': hint_cl_id,
                    'cl_term_name': hint_cl_name,
                    'cl_term_match_method': hint_method,
                    'cellxgene_cell_type': cxg_type,
                    'cellxgene_cell_type_ontology_term_id': cxg_id,
                })
                method_counts[hint_method] = method_counts.get(hint_method, 0) + 1
            else:
                # CellxGene fallback: use the cellxgene CL ID if there's exactly
                # one associated CL term (pipe-separated means multiple — too ambiguous).
                cxg_cl_id = cxg_id if cxg_id and '|' not in cxg_id else ''
                cxg_cl_name = cl_names.get(cxg_cl_id, '') if cxg_cl_id else ''
                if cxg_cl_id and cxg_cl_name:
                    print(f"  [cellxgene_fallback] \"{label}\" → {cxg_cl_id} ({cxg_cl_name})")
                    rows.append({
                        'original_label': label,
                        'cl_term_id': cxg_cl_id,
                        'cl_term_name': cxg_cl_name,
                        'cl_term_match_method': 'cellxgene_fallback',
                        'cellxgene_cell_type': cxg_type,
                        'cellxgene_cell_type_ontology_term_id': cxg_id,
                    })
                    method_counts['cellxgene_fallback'] = method_counts.get('cellxgene_fallback', 0) + 1
                else:
                    rows.append({
                        'original_label': label,
                        'cl_term_id': '',
                        'cl_term_name': '',
                        'cl_term_match_method': 'no_cl_term_found',
                        'cellxgene_cell_type': cxg_type,
                        'cellxgene_cell_type_ontology_term_id': cxg_id,
                    })
                    method_counts['no_cl_term_found'] += 1

    # Populate cell_label for every row.
    # Priority: (1) CL term name for exact matches, (2) pre-generated LLM label
    # for llm_ancestor and no_cl_term_found (preserves specificity), (3) original label.
    _EXACT_METHODS = {'already_cl', 'exact', 'synonym', 'fuzzy', 'fuzzy_synonym',
                      'llm_consensus', 'llm_claude', 'llm_gemini'}
    for row in rows:
        if row['cl_term_match_method'] in _EXACT_METHODS:
            row['cell_label'] = row['cl_term_name']
            row['cell_label_method'] = 'cl_term_name'
        else:
            lbl = row['original_label']
            row['cell_label'] = label_hints.get(lbl) or lbl
            row['cell_label_method'] = label_hint_method.get(lbl, 'original')

    remap_df = pd.DataFrame(rows, columns=[
        'original_label', 'cell_label', 'cell_label_method', 'cl_term_id', 'cl_term_name',
        'cl_term_match_method', 'cellxgene_cell_type', 'cellxgene_cell_type_ontology_term_id',
    ]).rename(columns={
        'cell_label':          'mapped_cell_label',
        'cell_label_method':   'mapped_cell_label_method',
        'cl_term_id':          'mapped_cell_label_ontology_term_id',
        'cl_term_name':        'mapped_cell_label_ontology_term_name',
        'cl_term_match_method':'mapped_cell_label_ontology_term_match_method',
    })

    total = len(unique_labels)
    resolved = total - method_counts['no_cl_term_found']
    lines = [
        f"Remap summary for column '{column}':",
        f"  Total unique labels: {total}",
        f"  Resolved:            {resolved} ({resolved/max(total,1):.0%})",
    ]
    for m in ('exact', 'synonym', 'fuzzy', 'fuzzy_synonym', 'already_cl',
              'llm_consensus', 'llm_claude', 'llm_gemini', 'llm_ancestor'):
        if method_counts[m]:
            lines.append(f"    {m}: {method_counts[m]}")
    if method_counts['no_cl_term_found']:
        lines.append(f"  Unresolved:          {method_counts['no_cl_term_found']}")
        still_unresolved = [r['original_label'] for r in rows if r['cl_term_match_method'] == 'no_cl_term_found']
        for name in still_unresolved:
            lines.append(f"    - \"{name}\"")

    return remap_df, '\n'.join(lines)


def detect_parent_columns(df, child_col, exclude_cols=None):
    """Find columns that are valid hierarchical parents of child_col.

    A column P is a parent of C if every unique value in C maps to exactly
    one value in P (functional dependency C -> P), and P has fewer unique
    values than C (coarser granularity).

    Returns:
        List of (parent_col, n_unique) sorted from coarsest to finest.
    """
    child_nunique = df[child_col].nunique()
    exclude_cols = exclude_cols or set()

    parents = []
    for col in df.columns:
        if col == child_col or col in exclude_cols:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        if _is_metadata_column(col, vals.tolist()):
            continue
        col_nunique = vals.nunique()
        if col_nunique >= child_nunique or col_nunique < 2:
            continue
        pairs = df[[child_col, col]].dropna()
        if _is_functional(pairs, child_col, col):
            parents.append((col, col_nunique))

    parents.sort(key=lambda x: x[1])
    return parents


def detect_child_columns(df, parent_col):
    """Find columns that are valid hierarchical children of parent_col.

    A column C is a child of P if every unique value in C maps to exactly
    one value in P (functional dependency C -> P), and C has more unique
    values than P (finer granularity).

    Returns:
        List of (child_col, n_unique) sorted from finest to coarsest.
    """
    parent_nunique = df[parent_col].nunique()

    children = []
    for col in df.columns:
        if col == parent_col:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        if _is_metadata_column(col, vals.tolist()):
            continue
        col_nunique = vals.nunique()
        if col_nunique <= parent_nunique or col_nunique < 2:
            continue
        pairs = df[[col, parent_col]].dropna()
        if _is_functional(pairs, col, parent_col):
            children.append((col, col_nunique))

    children.sort(key=lambda x: x[1])
    return children


def build_hierarchy_string(df, parent_cols, selected_col, child_cols):
    """Build a full hierarchy display string, e.g. 'Class (6) -> Subclass (12) -> Type/Type_updated (34)'.

    Columns with the same number of unique values are grouped with '/'.
    """
    parts = []
    for col, n in parent_cols:
        parts.append(f"{col} ({n})")
    parts.append(f"{selected_col} ({df[selected_col].nunique()})")

    # Group children by n_unique
    groups = {}
    for col, n in child_cols:
        groups.setdefault(n, []).append(col)
    for n in sorted(groups):
        parts.append(f"{'/'.join(groups[n])} ({n})")

    return ' -> '.join(parts)


def check_hierarchy_consistency(df, child_col, remap_dict, parent_cols, obo_path):
    """Check that CL term mappings are consistent within parent groups.

    For each parent group, finds the Most Informative Common Ancestor (MICA)
    of all CL terms in that group. Reports whether the MICA is biologically
    sensible.

    Returns:
        Consistency report string.
    """
    from ontology_utils import load_ontology, precompute_ic
    import networkx as nx

    cl_names, name_to_id, synonym_to_id, fuzzy_name_to_id, fuzzy_synonym_to_id = \
        build_label_mapping(obo_path)
    G = load_ontology(obo_path)
    ic_values = precompute_ic(G, k=0.5)

    lines = ["\n=== Hierarchy consistency check ==="]

    for parent_col, _ in parent_cols:
        lines.append(f"\nParent column: '{parent_col}'")

        # Build: parent_value -> [child_labels] -> [CL terms]
        pairs = df[[child_col, parent_col]].drop_duplicates()
        parent_groups = pairs.groupby(parent_col)[child_col].apply(list).to_dict()

        for parent_val in sorted(parent_groups.keys()):
            child_labels = sorted(set(str(v).strip() for v in parent_groups[parent_val]))
            cl_ids = []
            label_to_cl = {}
            for label in child_labels:
                cl_id = remap_dict.get(label)
                if cl_id and _is_cl_id(cl_id):
                    cl_ids.append(cl_id)
                    label_to_cl[label] = cl_id

            if len(cl_ids) < 2:
                if len(cl_ids) == 1:
                    cl_name = cl_names.get(cl_ids[0], '?')
                    lines.append(f"  {parent_val}: 1 CL term — {cl_ids[0]} ({cl_name})")
                else:
                    lines.append(f"  {parent_val}: no resolved CL terms")
                continue

            # Find common ancestors of all CL terms in this group
            unique_cl = list(set(cl_ids))
            # Start with ancestors of the first term, intersect with the rest
            try:
                common_anc = set(nx.descendants(G, unique_cl[0])) | {unique_cl[0]}
                for cl_id in unique_cl[1:]:
                    anc_i = set(nx.descendants(G, cl_id)) | {cl_id}
                    common_anc &= anc_i
            except nx.NodeNotFound:
                lines.append(f"  {parent_val}: ontology lookup failed")
                continue

            if not common_anc:
                lines.append(f"  {parent_val} ({len(unique_cl)} CL terms): "
                             f"WARNING — no common ancestor found!")
                for label, cl_id in sorted(label_to_cl.items()):
                    lines.append(f"    {label} -> {cl_id} ({cl_names.get(cl_id, '?')})")
                continue

            # Find MICA (highest IC among common ancestors)
            mica_id = max(common_anc, key=lambda a: ic_values.get(a, 0))
            mica_name = cl_names.get(mica_id, '?')
            mica_ic = ic_values.get(mica_id, 0)

            # Assess: is the MICA reasonable?
            if mica_ic > 0.1:
                status = "consistent"
            elif mica_id == 'CL:0000000':
                status = "WARNING — only common ancestor is root 'cell'"
            else:
                status = "weak — common ancestor is very generic"

            # Validate MICA against the resolved parent label; also build subtree
            # for per-term violation flagging below.
            parent_subtree = set()
            parent_cl_id, _, _ = resolve_name(
                parent_val, name_to_id, synonym_to_id,
                fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)
            if parent_cl_id:
                try:
                    parent_subtree = set(nx.ancestors(G, parent_cl_id)) | {parent_cl_id}
                    mica_ancestors = set(nx.descendants(G, mica_id)) | {mica_id}
                    if parent_cl_id not in mica_ancestors:
                        parent_cl_name = cl_names.get(parent_cl_id, '?')
                        status += (f"; WARNING — MICA is not under parent label "
                                   f"{parent_cl_id} ({parent_cl_name})")
                except nx.NodeNotFound:
                    pass

            # Determine per-term violations before printing
            term_violations = {
                cl_id: (parent_subtree and cl_id not in parent_subtree)
                for cl_id in label_to_cl.values()
            }
            any_violations = any(term_violations.values())

            # Suppress "MICA not under parent" warning when all terms are actually
            # under the parent (false positive from multiple inheritance paths).
            if not any_violations and 'MICA is not under parent label' in status:
                status = status.replace(
                    status[status.index('; WARNING — MICA is not under parent label'):], ''
                )

            n_cl = len(unique_cl)
            if 'WARNING' in status:
                lines.append(f"  {parent_val} ({n_cl} CL term{'s' if n_cl != 1 else ''}): "
                             f"MICA = {mica_id} ({mica_name}, IC={mica_ic:.3f}) — {status}")
                for label, cl_id in sorted(label_to_cl.items()):
                    marker = " [!]" if term_violations.get(cl_id) else ""
                    lines.append(f"    {label} -> {cl_id} ({cl_names.get(cl_id, '?')}){marker}")

    return '\n'.join(lines)


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Validate LLM API keys early before any processing
    has_anthropic = bool(os.environ.get('ANTHROPIC_API_KEY'))
    has_gemini = bool(os.environ.get('GOOGLE_API_KEY'))
    if not has_anthropic and not has_gemini:
        print("Error: ANTHROPIC_API_KEY or GOOGLE_API_KEY is required.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  export GOOGLE_API_KEY=sk-...")
        sys.exit(1)

    # Fetch paper context — PDF takes priority over DOI
    paper_context = None
    if args.pdf:
        glossary_cache = re.sub(r'\.pdf$', '', args.pdf, flags=re.IGNORECASE) + '_glossary.txt'
        if os.path.exists(glossary_cache):
            with open(glossary_cache) as f:
                paper_context = f.read().strip()
            lines = paper_context.splitlines()
            print(f"\nLoaded cached glossary from {glossary_cache} ({len(lines)} entries):")
            for entry in lines:
                print(f"    {entry}")
        else:
            print(f"\nExtracting cell type glossary from PDF: {args.pdf}...")
            pdf_text = extract_pdf_text(args.pdf)
            if pdf_text:
                # Prefer Gemini for glossary extraction — large input, saves Claude TPM quota
                llm_api = 'gemini' if os.environ.get('GOOGLE_API_KEY') else \
                          'claude' if os.environ.get('ANTHROPIC_API_KEY') else None
                if llm_api:
                    paper_context = extract_cell_type_glossary(pdf_text, api=llm_api)
                    if paper_context:
                        with open(glossary_cache, 'w') as f:
                            f.write(paper_context)
                        lines = paper_context.splitlines()
                        print(f"  Glossary extracted ({len(lines)} entries) — cached to {glossary_cache}:")
                        for entry in lines:
                            print(f"    {entry}")
                    else:
                        print("  Glossary extraction failed — proceeding without it.")
                else:
                    print("  Skipping glossary extraction (no LLM API key).")
            else:
                print("  Could not read PDF — proceeding without it.")
    elif args.doi:
        print(f"\nFetching paper context for DOI: {args.doi}...")
        paper_context = fetch_doi_context(args.doi)
        if paper_context:
            print(f"  {paper_context[:120]}...")
        else:
            print("  Could not retrieve context — proceeding without it.")
    elif args.context:
        paper_context = args.context
        print(f"\nUsing provided context: {paper_context[:120]}")

    # Load input
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input, sep='\t')
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # Detect or validate column
    if args.column:
        if args.column not in df.columns:
            print(f"Error: column '{args.column}' not found.")
            print(f"  Available columns: {list(df.columns)}")
            sys.exit(1)
        selected_col = args.column
        print(f"  Using column '{selected_col}' (user-specified)")
    else:
        print(f"\nAuto-detecting annotation column hierarchy...")
        selected_col, detect_report = detect_leaf_column(df)
        print(detect_report)
        if selected_col is None:
            print("\nError: No suitable annotation column found.")
            print(f"  Available columns: {list(df.columns)}")
            print("  Use --column to specify one explicitly.")
            sys.exit(1)

    # Show full hierarchy with selected column highlighted
    cellxgene_cols = _get_cellxgene_cols(df)
    parent_cols = detect_parent_columns(df, selected_col, exclude_cols=cellxgene_cols)
    child_cols = detect_child_columns(df, selected_col)
    hierarchy_str = build_hierarchy_string(df, parent_cols, selected_col, child_cols)
    print(f"\nDetected hierarchy: {hierarchy_str}")

    # Glossary coverage check: if --pdf was given, read additional pages for any
    # labels not covered by the initial extraction.
    if args.pdf and paper_context:
        _checkable = sorted({
            str(v).strip() for v in df[selected_col]
            if pd.notna(v) and str(v).strip()
            and not _UNINFORMATIVE_LABELS.match(str(v).strip())
            and not _is_cl_id(str(v).strip())
        })
        _lookup = _parse_glossary(paper_context)
        uncovered = [l for l in _checkable if not _glossary_covers(l, _lookup)]

        print(f"\nGlossary coverage: {len(_checkable) - len(uncovered)}/{len(_checkable)} "
              f"labels from '{selected_col}' found.")
        if uncovered:
            print(f"  {len(uncovered)} uncovered label(s):")
            for l in uncovered[:10]:
                print(f"    - {l}")
            if len(uncovered) > 10:
                print(f"    ... and {len(uncovered) - 10} more")

            try:
                import pypdf
                total_pages = len(pypdf.PdfReader(args.pdf).pages)
            except Exception:
                total_pages = 0

            sup_api = 'gemini' if has_gemini else 'claude'
            BATCH = 20
            page = BATCH  # pages 0–BATCH-1 already processed
            while uncovered and page < total_pages:
                end = min(page + BATCH, total_pages)
                print(f"\n  Reading PDF pages {page + 1}–{end} of {total_pages} "
                      f"({len(uncovered)} label(s) still uncovered)...")
                extra_text = extract_pdf_text(args.pdf, max_pages=BATCH, start_page=page)
                if extra_text:
                    supplemental = extract_cell_type_glossary(extra_text, api=sup_api)
                    if supplemental:
                        paper_context += '\n' + supplemental
                        _lookup = _parse_glossary(paper_context)
                        new_uncovered = [l for l in uncovered
                                         if not _glossary_covers(l, _lookup)]
                        found = len(uncovered) - len(new_uncovered)
                        if found:
                            print(f"    Resolved {found} label(s).")
                        uncovered = new_uncovered
                        with open(glossary_cache, 'w') as f:
                            f.write(paper_context)
                page += BATCH

            if uncovered:
                print(f"  {len(uncovered)} label(s) remain uncovered after full PDF "
                      f"(will fall back to LLM).")
            else:
                print(f"  All labels now covered.")

    # Generate remap
    print(f"\nResolving labels from '{selected_col}' to CL term IDs...")
    remap_df, summary = generate_remap(df, selected_col, args.obo,
                                       paper_context=paper_context,
                                       use_llm=not args.no_llm,
                                       cellxgene_fallback_labels=args.cellxgene_fallback)
    print(summary)

    # Derive output path if not specified
    if args.output is None:
        from pathlib import Path
        stem = re.sub(r'\.tsv(\.gz)?$', '', Path(args.input).name)
        args.output = str(Path(args.input).parent / f"{stem}_remap.tsv")
        print(f"\nOutput: {args.output}")

    # Write output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    comment_lines = [
        f"# Generated by generate_remap.py",
        f"# Input file: {args.input}",
        f"# Source column: {selected_col}",
        f"# OBO file: {args.obo}",
    ]
    with open(args.output, 'w') as f:
        f.write('\n'.join(comment_lines) + '\n')
        remap_df.to_csv(f, sep='\t', index=False)

    print(f"\nSaved remap to {args.output}")
    print(f"  Use with: evaluate.py --remap-file {args.output}")

    # Hierarchy consistency check — immediate parent only
    if parent_cols:
        immediate_parent = [parent_cols[-1]]
        remap_dict = dict(zip(remap_df['original_label'], remap_df['mapped_cell_label_ontology_term_id']))
        report = check_hierarchy_consistency(
            df, selected_col, remap_dict, immediate_parent, args.obo)
        print(report)
    else:
        print("No parent columns detected — skipping consistency check.")


if __name__ == "__main__":
    main()

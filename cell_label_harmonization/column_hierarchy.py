"""
column_hierarchy.py — shared logic for detecting cell type annotation column
hierarchy in single-cell TSV files.

Imported by both generate_remap.py and detect_hierarchy.py so that any
improvement to the detection logic benefits both tools automatically.

Public API
----------
    _is_cl_id(value)                   — True if string looks like CL:NNNNNNN
    _is_metadata_column(col, values)   — True if column is metadata, not annotations
    _get_cellxgene_cols(df)            — set of cellxgene display-name columns
    detect_leaf_column(df)             — (selected_col, report_str)  ← used by generate_remap.py
    detect_full_hierarchy(df)          — (candidates, parent_of, children_of,
                                          roots, leaves, nunique, cellxgene_cols)
                                          ← used by detect_hierarchy.py
"""

import re

import pandas as pd


# ---------------------------------------------------------------------------
# CL ID check
# ---------------------------------------------------------------------------

def _is_cl_id(value):
    """Check if a string looks like a CL term ID (e.g. CL:0000540)."""
    return bool(re.match(r'^CL:\d+$', str(value).strip()))


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

_METADATA_NAME_TOKENS = {
    'id', 'ids', 'barcode', 'barcodes', 'index', 'indices',
    'sample', 'samples', 'donor', 'donors', 'batch', 'batches',
    'uuid', 'accession', 'key',
}

# Standard cellxgene schema metadata columns that are never cell type annotations
_CELLXGENE_METADATA_COLS = {
    'assay', 'development_stage', 'disease', 'donor_id', 'self_reported_ethnicity',
    'sex', 'suspension_type', 'tissue', 'tissue_type',
}


def _is_metadata_column(col_name, values):
    """Heuristic: return True if the column looks like metadata, not annotations."""
    # Ontology term ID columns are always IDs, never annotation labels
    if col_name.endswith('_ontology_term_id'):
        return True

    # Known cellxgene schema metadata columns
    if col_name in _CELLXGENE_METADATA_COLS:
        return True

    # Column name tokens (split on _, -, space, and CamelCase boundaries)
    expanded = re.sub(r'([a-z])([A-Z])', r'\1_\2', col_name)
    tokens = set(re.split(r'[_\-\s]+', expanded.lower()))
    if tokens & _METADATA_NAME_TOKENS:
        return True

    non_null = [v for v in values if pd.notna(v)]
    if not non_null:
        return True

    # Numeric columns (ages, coordinates, counts)
    numeric_count = 0
    for v in non_null[:200]:
        try:
            float(v)
            numeric_count += 1
        except (ValueError, TypeError):
            pass
    if numeric_count / max(len(non_null[:200]), 1) > 0.8:
        return True

    # Count unique values across the full column (cheap set operation)
    n_unique = len(set(str(v) for v in non_null))

    # High-cardinality columns (barcodes, IDs)
    if n_unique > 1000:
        return True

    # Very low cardinality (≤ 3 unique values) — likely a constant/flag field
    if n_unique <= 3:
        return True

    # Columns that are already CL term IDs (we want to IMPROVE upon those)
    cl_count = sum(1 for v in non_null[:200] if _is_cl_id(str(v).strip()))
    if cl_count / max(len(non_null[:200]), 1) > 0.9:
        return True

    return False


def _get_cellxgene_cols(df):
    """Return the set of cellxgene display-name columns to skip.

    A column is a cellxgene display column if a paired "{col}_ontology_term_id"
    column exists and contains >90% CL IDs.
    """
    cellxgene_cols = set()
    for col in df.columns:
        paired = col + '_ontology_term_id'
        if paired in df.columns:
            sample = [v for v in df[paired] if pd.notna(v) and str(v).strip()][:200]
            cl_frac = sum(1 for v in sample if _is_cl_id(str(v).strip())) / max(len(sample), 1)
            if cl_frac > 0.9:
                cellxgene_cols.add(col)
    return cellxgene_cols


# ---------------------------------------------------------------------------
# Cell-type vocabulary scorer
# ---------------------------------------------------------------------------

# Whole-word tokens that are strong indicators of cell type annotation values
_CELL_TYPE_TOKENS = {
    # Generic
    'cell', 'cells', 'type', 'types', 'subtype', 'subtypes', 'cluster', 'clusters',
    # Neural / brain
    'neuron', 'neurons', 'neuronal', 'neural',
    'glia', 'glial', 'astrocyte', 'astrocytes', 'astro',
    'oligodendrocyte', 'oligodendrocytes', 'oligo', 'opc',
    'microglia', 'microglial',
    'radial', 'progenitor', 'progenitors', 'npc',
    'interneuron', 'interneurons', 'excitatory', 'inhibitory',
    'ependymal', 'choroid', 'cajal', 'retzius',
    'ipc', 'rg', 'org', 'trg', 'cr',
    # Developmental state
    'stem', 'precursor', 'precursors',
    'cycling', 'dividing', 'proliferating', 'quiescent',
    'early', 'late', 'mature', 'immature', 'newborn', 'activated', 'stressed',
    # Blood / immune
    'macrophage', 'macrophages', 'monocyte', 'monocytes', 'dendritic',
    'lymphocyte', 'lymphocytes', 'erythrocyte', 'erythrocytes',
    'neutrophil', 'neutrophils', 'eosinophil', 'basophil',
    'platelet', 'platelets',
    # Other tissue
    'fibroblast', 'fibroblasts', 'epithelial', 'endothelial',
    'hepatocyte', 'hepatocytes', 'cardiomyocyte', 'cardiomyocytes',
    'pericyte', 'pericytes', 'schwann',
    # Common prefix / pan terms
    'pan', 'panglia', 'panneuron',
}

# Suffixes that — when found at word-end — strongly suggest a cell type
_CELL_TYPE_SUFFIXES = ('cyte', 'blast', 'glia', 'neuron', 'phage', 'phil')

_VALUE_TOKEN_RE = re.compile(r'[A-Za-z]+')


def _cell_type_score(values):
    """Return fraction of values that look like cell type labels (0.0–1.0).

    Each value is tokenised on non-alpha boundaries; a value scores a hit if
    any token matches the cell-type vocabulary or ends with a cell-type suffix.
    """
    if not values:
        return 0.0
    hits = 0
    for v in values:
        tokens = [t.lower() for t in _VALUE_TOKEN_RE.findall(str(v))]
        if any(t in _CELL_TYPE_TOKENS or t.endswith(_CELL_TYPE_SUFFIXES)
               for t in tokens):
            hits += 1
    return hits / len(values)


# ---------------------------------------------------------------------------
# Functional dependency check (strict or majority-vote relaxed)
# ---------------------------------------------------------------------------

# Fraction of rows that must agree on the parent label for a given child value.
# 1.0 = strict (original behaviour); lower values allow "pan-" / "stressed-"
# type labels that don't map perfectly to one parent.
_MV_THRESHOLD = 0.7


def _is_functional(pairs, C, P, threshold=_MV_THRESHOLD):
    """Return True if C → P is a functional dependency at the given threshold.

    For each unique value of C, the most-common value of P must account for
    at least `threshold` of that group's rows.  At threshold=1.0 this is
    identical to the original strict test.
    """
    for _, group in pairs.groupby(C)[P]:
        top_frac = group.value_counts(normalize=True).iloc[0]
        if top_frac < threshold:
            return False
    return True


# ---------------------------------------------------------------------------
# Leaf selection (used by generate_remap.py)
# ---------------------------------------------------------------------------

_REVISED_PATTERN = re.compile(
    r'\b(updated|revised|final|corrected|curated|clean|v\d)\b', re.IGNORECASE
)


def detect_leaf_column(df):
    """Detect the most granular annotation column via hierarchy inference.

    Filters candidate annotation columns, builds parent-child relationships
    among them using functional dependency, and returns the leaf (the column
    with no finer-grained children).  If multiple leaves exist, the one
    appearing latest in the file is preferred (updated columns come after
    their predecessors).

    Returns:
        Tuple of (column_name_or_None, report_string).
    """
    cellxgene_cols = _get_cellxgene_cols(df)

    # Collect candidate annotation columns in file order, excluding cellxgene cols
    candidates = []
    for col in df.columns:
        if col in cellxgene_cols:
            continue
        if not _is_metadata_column(col, df[col].tolist()):
            candidates.append(col)

    # If filtering cellxgene cols leaves nothing, fall back to including them —
    # they may be the only annotation columns available
    if not candidates and cellxgene_cols:
        print(f"  No author annotation columns found — falling back to cellxgene label columns: {sorted(cellxgene_cols)}")
        for col in df.columns:
            if col in cellxgene_cols and not _is_metadata_column(col, df[col].tolist()):
                candidates.append(col)
    elif cellxgene_cols:
        skipped_info = ', '.join(f"'{c}' ({df[c].nunique()} categories)" for c in sorted(cellxgene_cols))
        print(f"  Skipping cellxgene label columns: {skipped_info}")

    if not candidates:
        return None, "No candidate annotation columns found."

    # Determine which candidates have at least one child among the candidates.
    # C is a child of P when: every unique value of C maps to exactly one value
    # of P, and C has strictly more unique values than P.
    nunique = {col: df[col].nunique() for col in candidates}
    has_child = set()
    for P in candidates:
        for C in candidates:
            if C == P or nunique[C] <= nunique[P]:
                continue
            pairs = df[[C, P]].dropna()
            if len(pairs) == 0:
                continue
            if _is_functional(pairs, C, P):
                has_child.add(P)
                break  # P has at least one child; no need to check further

    leaves = [col for col in candidates if col not in has_child]

    lines = ["Column hierarchy detection:"]
    lines.append(f"  Candidate columns: {candidates}")
    lines.append(f"  Columns with children (non-leaf): {sorted(has_child)}")

    if not leaves:
        # Degenerate case: every column appears to have a child (cycle or all same size)
        # Fall back to the last candidate with the most unique values.
        leaves = [max(candidates, key=lambda c: (nunique[c], candidates.index(c)))]
        lines.append(f"  No clear leaf found — falling back to most granular: '{leaves[0]}'")
    else:
        lines.append(f"  Leaf column(s): {leaves}")

    selected = max(leaves, key=lambda c: (
        bool(_REVISED_PATTERN.search(c)),
        _cell_type_score(df[c].dropna().unique().tolist()) >= 0.3,
        nunique[c],
        candidates.index(c)
    ))
    lines.append(f"  Selected: '{selected}' ({nunique[selected]} unique labels)")

    return selected, '\n'.join(lines)


# ---------------------------------------------------------------------------
# Full hierarchy (used by detect_hierarchy.py)
# ---------------------------------------------------------------------------

def detect_full_hierarchy(df):
    """Detect the full annotation column hierarchy via functional dependency.

    For each candidate column C, its immediate parent P is the valid parent
    with the most unique values (closest in granularity).

    C is a valid parent of P when:
      - for every unique value of C, ≥ _MV_THRESHOLD of its rows agree on one value of P
      - C has strictly more unique values than P

    Returns:
        candidates     — list of annotation column names (file order)
        parent_of      — dict {col: immediate_parent_col | None}
        children_of    — dict {col: [child_cols]}
        roots          — columns with no parent
        leaves         — columns with no children
        nunique        — dict {col: n_unique}
        cellxgene_cols — set of skipped cellxgene display columns
    """
    cellxgene_cols = _get_cellxgene_cols(df)

    candidates = [
        col for col in df.columns
        if col not in cellxgene_cols
        and not _is_metadata_column(col, df[col].tolist())
    ]

    # Fall back to including cellxgene cols if nothing else found
    if not candidates and cellxgene_cols:
        candidates = [
            col for col in df.columns
            if col in cellxgene_cols
            and not _is_metadata_column(col, df[col].tolist())
        ]

    if not candidates:
        return [], {}, {}, [], [], {}, cellxgene_cols

    nunique = {col: df[col].nunique() for col in candidates}

    # For each column C, find its immediate parent: the valid parent with the
    # most unique values (highest nunique among all cols where C→P is functional
    # and nunique[P] < nunique[C]).
    parent_of = {col: None for col in candidates}
    for C in candidates:
        best_parent = None
        best_nunique = -1
        for P in candidates:
            if C == P or nunique[P] >= nunique[C]:
                continue
            pairs = df[[C, P]].dropna()
            if len(pairs) == 0:
                continue
            if _is_functional(pairs, C, P):
                if nunique[P] > best_nunique:
                    best_nunique = nunique[P]
                    best_parent = P
        parent_of[C] = best_parent

    children_of = {col: [] for col in candidates}
    for col, par in parent_of.items():
        if par is not None:
            children_of[par].append(col)

    roots = [col for col in candidates if parent_of[col] is None]
    leaves = [col for col in candidates if not children_of[col]]

    return candidates, parent_of, children_of, roots, leaves, nunique, cellxgene_cols

"""Tests for generate_remap.py"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_remap import (
    build_parser,
    _is_metadata_column,
    score_column,
    detect_best_column,
    generate_remap,
    detect_parent_columns,
    check_hierarchy_consistency,
)
from evaluate import build_label_mapping


def test_generate_remap_parser_required_args():
    """Test that parser enforces required arguments."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_generate_remap_parser_with_all_required():
    """Test parser with all required arguments."""
    parser = build_parser()
    args = parser.parse_args([
        '--input', 'input.tsv',
        '--obo', 'cl.obo',
        '--output', 'remap.tsv',
    ])
    assert args.input == 'input.tsv'
    assert args.obo == 'cl.obo'
    assert args.output == 'remap.tsv'
    assert args.column is None  # default: auto-detect
    assert args.min_score == 0.10  # default


def test_generate_remap_parser_with_column():
    """Test parser with --column override."""
    parser = build_parser()
    args = parser.parse_args([
        '--input', 'input.tsv',
        '--obo', 'cl.obo',
        '--output', 'remap.tsv',
        '--column', 'annot_level_3',
    ])
    assert args.column == 'annot_level_3'


def test_is_metadata_column_numeric():
    """Test that numeric columns are detected as metadata."""
    assert _is_metadata_column('age', [30, 45, 60, 30, 45])
    assert _is_metadata_column('score', [0.5, 0.8, 0.9, 0.3, 0.7])


def test_is_metadata_column_high_cardinality():
    """Test that high-cardinality columns are detected as metadata."""
    barcodes = [f'AACG_{i:04d}' for i in range(2000)]
    assert _is_metadata_column('cell_id', barcodes)


def test_is_metadata_column_annotation():
    """Test that annotation columns are NOT flagged as metadata."""
    labels = ['neuron', 'astrocyte', 'neuron', 'oligodendrocyte', 'astrocyte']
    assert not _is_metadata_column('cell_type', labels)


def test_score_column(sample_obo_file):
    """Test column scoring by OBO resolvability."""
    cl_names, name_to_id, synonym_to_id, fuzzy_name_to_id, fuzzy_synonym_to_id = \
        build_label_mapping(sample_obo_file)

    # All resolvable
    good_vals = ['neuron', 'astrocyte', 'oligodendrocyte', 'neuron']
    score, n_unique, n_resolved = score_column(
        good_vals, name_to_id, synonym_to_id,
        fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)
    assert score == 1.0
    assert n_unique == 3
    assert n_resolved == 3

    # Partially resolvable
    mixed_vals = ['neuron', 'astrocyte', 'fake_cell_xyz', 'another_fake']
    score, n_unique, n_resolved = score_column(
        mixed_vals, name_to_id, synonym_to_id,
        fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)
    assert score == 0.5  # 2 out of 4
    assert n_unique == 4

    # Nothing resolvable
    bad_vals = ['fake1', 'fake2', 'fake3']
    score, n_unique, n_resolved = score_column(
        bad_vals, name_to_id, synonym_to_id,
        fuzzy_name_to_id, fuzzy_synonym_to_id, cl_names)
    assert score == 0.0


def test_detect_best_column(sample_obo_file, sample_multi_column_ground_truth):
    """Test auto-detection picks the best annotation column."""
    path, df = sample_multi_column_ground_truth

    best_col, report = detect_best_column(df, sample_obo_file, min_score=0.10)

    # Should pick either cell_type or author_label (both have high resolvability)
    # cell_type has 100% (neuron, oligodendrocyte) and author_label has ~80%
    # cell_type_ontology_term_id should also score high as CL IDs
    assert best_col is not None
    assert best_col in df.columns


def test_generate_remap_output(sample_obo_file, sample_multi_column_ground_truth):
    """Test that generate_remap produces correct output format."""
    path, df = sample_multi_column_ground_truth

    remap_df, summary = generate_remap(df, 'cell_type', sample_obo_file)

    # Check output columns
    assert list(remap_df.columns) == ['original_label', 'cl_term_id', 'cl_term_name', 'match_method']

    # neuron and oligodendrocyte should resolve
    resolved = remap_df[remap_df['cl_term_id'] != '']
    assert len(resolved) >= 2

    # Check that neuron maps correctly
    neuron_row = remap_df[remap_df['original_label'] == 'neuron'].iloc[0]
    assert neuron_row['cl_term_id'] == 'CL:0000540'
    assert neuron_row['match_method'] == 'exact'


def test_generate_remap_with_synonyms(sample_obo_file, sample_multi_column_ground_truth):
    """Test that generate_remap handles synonyms in the author_label column."""
    path, df = sample_multi_column_ground_truth

    remap_df, summary = generate_remap(df, 'author_label', sample_obo_file)

    # "nerve cell" is a synonym for neuron
    nerve_rows = remap_df[remap_df['original_label'] == 'nerve cell']
    assert len(nerve_rows) == 1
    assert nerve_rows.iloc[0]['cl_term_id'] == 'CL:0000540'
    assert nerve_rows.iloc[0]['match_method'] == 'synonym'

    # "zygomorphic blastocell" should be unresolved
    unresolv = remap_df[remap_df['original_label'] == 'zygomorphic blastocell']
    assert len(unresolv) == 1
    assert unresolv.iloc[0]['cl_term_id'] == ''
    assert unresolv.iloc[0]['match_method'] == 'unresolved'


def test_detect_parent_columns():
    """Test hierarchical parent column detection."""
    # Each child value must appear multiple times so that functional dependency
    # violations in 'unrelated' are detectable.
    df = pd.DataFrame({
        'level_1': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'level_2': ['A1', 'A1', 'A2', 'A2', 'B1', 'B1', 'B2', 'B2'],
        'level_3': ['A1x', 'A1y', 'A2x', 'A2y', 'B1x', 'B1y', 'B2x', 'B2y'],
        # unrelated: same child value maps to different values (A1x→X, A1x→Y)
        # — but level_3 values are unique, so we need a column that has fewer
        # unique values than level_3 but doesn't satisfy functional dependency.
        # Use level_2 as the child instead for a cleaner test.
    })
    # Test with level_2 as child (4 unique values, each appears twice)
    parents = detect_parent_columns(df, 'level_2')
    parent_names = [col for col, _ in parents]

    # level_1 is a valid parent (A1→A, A2→A, B1→B, B2→B)
    assert 'level_1' in parent_names


def test_detect_parent_columns_no_parents():
    """Test that no parents are detected when none exist."""
    df = pd.DataFrame({
        'col_a': ['X', 'Y', 'Z', 'W', 'V'],
        'col_b': ['1', '2', '3', '4', '5'],
    })
    parents = detect_parent_columns(df, 'col_a')
    assert parents == []


def test_check_hierarchy_consistency(sample_obo_file):
    """Test hierarchy consistency check with known-good mappings."""
    # Build a hierarchical dataset with 2 parent groups, each with 2+ children.
    # In the test OBO: cell → neuron → {oligodendrocyte, astrocyte}
    df = pd.DataFrame({
        'level_1': ['glial'] * 4 + ['neural'] * 4,
        'level_2': ['oligodendrocyte'] * 2 + ['astrocyte'] * 2
                    + ['neuron'] * 2 + ['oligodendrocyte'] * 2,
    })
    # Wait — oligodendrocyte appears under both 'glial' and 'neural',
    # violating functional dependency. Fix: use distinct child values per parent.
    df = pd.DataFrame({
        'level_1': ['glial'] * 4 + ['other'] * 4,
        'level_2': ['oligodendrocyte'] * 2 + ['astrocyte'] * 2
                    + ['neuron'] * 2 + ['cell'] * 2,
    })
    remap_dict = {
        'oligodendrocyte': 'CL:0000128',
        'astrocyte': 'CL:0000127',
        'neuron': 'CL:0000540',
        'cell': 'CL:0000000',
    }
    parent_cols = detect_parent_columns(df, 'level_2')
    assert len(parent_cols) >= 1

    report = check_hierarchy_consistency(
        df, 'level_2', remap_dict, parent_cols, sample_obo_file)

    # The report should contain MICA information for the 'glial' group
    assert 'MICA' in report
    assert 'glial' in report

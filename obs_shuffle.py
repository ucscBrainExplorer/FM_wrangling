"""Chunked obs reordering shared by combineEmbedding.py (--shuffle) and
apply_shuffle_index.py (re-applying a saved shuffle_index.uce.npy to a
regenerated, unshuffled obs).

Memory stays bounded for ~100M-row obs tables: columns are held in Arrow
(dictionary-encoded where low-cardinality) instead of pandas object strings.
"""
import gzip
import itertools

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Columns that are (near-)unique per cell; everything else is dictionary-encoded.
HIGH_CARDINALITY_OBS_COLS = {"new_index", "old_index"}


def write_shuffled_obs(parquet_path, shuffle_idx, out_path, block_rows=1_000_000):
    """Write the staged obs Parquet to a gzip TSV in shuffle_idx order without
    materializing the whole table in pandas.

    Loading the full obs into pandas (one Python str object per cell value) costs
    ~60 bytes/value and doubles on the .iloc reorder — far beyond RAM for a full
    ~100M-cell run. Instead each column is held in Arrow: low-cardinality columns
    as dictionary arrays (int32 codes, ~4 bytes/row), unique-per-cell columns as
    large_string (string bytes + 8-byte offset). Columns are loaded one at a time
    so the transient copy from combine_chunks() only ever covers one column.
    Output rows are then gathered with take() and written block_rows at a time.
    """
    col_names = pq.read_schema(parquet_path).names
    if len(shuffle_idx) != pq.ParquetFile(parquet_path).metadata.num_rows:
        raise ValueError(f"obs parquet row count != shuffle index length ({len(shuffle_idx)})")
    # Load high-cardinality columns first, while nothing else is resident, since
    # their combine_chunks() transient is the largest.
    load_order = sorted(col_names, key=lambda c: c not in HIGH_CARDINALITY_OBS_COLS)
    arrays = {}
    for name in load_order:
        if name in HIGH_CARDINALITY_OBS_COLS:
            col = pq.read_table(parquet_path, columns=[name]).column(0)
            # large_string: int64 offsets, so >2 GB of string data in one array is fine
            arrays[name] = col.cast(pa.large_string()).combine_chunks()
        else:
            t = pq.read_table(parquet_path, columns=[name], read_dictionary=[name])
            arrays[name] = t.unify_dictionaries().column(0).combine_chunks()

    n = len(shuffle_idx)
    with gzip.open(out_path, 'wt') as fout:
        for out_start in range(0, n, block_rows):
            idx = pa.array(shuffle_idx[out_start:out_start + block_rows])
            block = pa.table({name: arrays[name].take(idx) for name in col_names})
            df = block.to_pandas()
            if 'new_index' in df.columns:
                df = df.set_index('new_index')
            df.to_csv(fout, sep='\t', header=(out_start == 0))
            print(f"  obs shuffle: {min(out_start + block_rows, n):,}/{n:,} rows written")


def obs_tsv_to_parquet(tsv_path, parquet_path, chunk_rows=200_000):
    """Stage an obs TSV(.gz) written by combineEmbedding.py as an all-string
    Parquet file (the format write_shuffled_obs reads). Returns the row count.

    Only empty fields become null (keep_default_na=False), so every value
    round-trips to the identical TSV text: combineEmbedding writes missing
    values as empty fields, and any other string is kept verbatim.
    """
    header = pd.read_csv(tsv_path, sep='\t', nrows=0).columns.tolist()
    schema = pa.schema([(c, pa.string()) for c in header])
    n = 0
    with pq.ParquetWriter(parquet_path, schema) as w:
        for chunk in pd.read_csv(tsv_path, sep='\t', dtype=str, keep_default_na=False,
                                 na_values=[''], chunksize=chunk_rows):
            w.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
            n += len(chunk)
    return n


def compare_index_column(path_a, path_b, chunk_rows=1_000_000):
    """Stream the first (index) column of two obs TSV(.gz) files and compare row
    by row. Returns None if identical, else a description of the first difference."""
    ra = pd.read_csv(path_a, sep='\t', usecols=[0], dtype=str, keep_default_na=False, chunksize=chunk_rows)
    rb = pd.read_csv(path_b, sep='\t', usecols=[0], dtype=str, keep_default_na=False, chunksize=chunk_rows)
    offset = 0
    # zip_longest, not zip: zip would silently drop the chunk it had already pulled
    # from the longer file when the shorter one runs out
    for ca, cb in itertools.zip_longest(ra, rb):
        if ca is None or cb is None:
            return f"row count differs: {'second' if ca is None else 'first'} file has more rows after row {offset:,}"
        a, b = ca.iloc[:, 0].to_numpy(), cb.iloc[:, 0].to_numpy()
        if len(a) != len(b):
            return f"row count differs near row {offset + min(len(a), len(b)):,}"
        diff = np.flatnonzero(a != b)
        if len(diff):
            i = diff[0]
            return f"row {offset + i:,}: {a[i]!r} vs {b[i]!r}"
        offset += len(a)
    return None

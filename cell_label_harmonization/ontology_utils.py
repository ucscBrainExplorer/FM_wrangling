import logging
import math
from collections import deque
from typing import Dict, Optional, Tuple, List
import statistics
import pandas as pd

# Try checking for networkx and pronto, but don't fail at module level
# to allow inspection of code even if deps are missing.
try:
    import networkx as nx
    import pronto
except ImportError:
    nx = None
    pronto = None


def load_ontology(obo_path: str, include_relationships: bool = True) -> 'nx.DiGraph':
    """
    Load the Cell Ontology (CL) OBO file into a NetworkX DiGraph.

    Tries pronto first for parsing; falls back to a manual OBO parser when
    pronto fails (e.g. on the full cl.obo which references external ontologies
    like BFO that pronto cannot resolve).

    Args:
        obo_path (str): Path to the .obo file.
        include_relationships (bool): If True, also add edges for
            ``relationship:`` lines (e.g. develops_from, part_of) between
            CL terms, not just ``is_a`` edges.  Default False.

    Returns:
        nx.DiGraph: Directed graph representing the ontology (edges point
            to parents/superclasses).  Only non-obsolete CL: terms are
            included.
    """
    if nx is None:
        raise ImportError("networkx is required for ontology_utils.")

    g = nx.DiGraph()

    # Try pronto first (works for cl-basic.obo)
    pronto_ok = False
    if pronto is not None:
        try:
            ont = pronto.Ontology(obo_path)
            for term in ont.terms():
                g.add_node(term.id, name=term.name)
                for superclass in term.superclasses(distance=1, with_self=False):
                    g.add_edge(term.id, superclass.id)
            pronto_ok = True
        except Exception:
            g = nx.DiGraph()  # reset on failure

    # Fall back to manual OBO parsing, or re-parse for relationship edges
    if not pronto_ok or include_relationships:
        if not pronto_ok:
            # Full manual parse: is_a edges (+ relationships if requested)
            _parse_obo_into_graph(g, obo_path,
                                  include_is_a=True,
                                  include_relationships=include_relationships)
        else:
            # Pronto succeeded for is_a; only add relationship edges
            _parse_obo_into_graph(g, obo_path,
                                  include_is_a=False,
                                  include_relationships=include_relationships)

    return g


def _parse_obo_into_graph(g: 'nx.DiGraph', obo_path: str, *,
                           include_is_a: bool = True,
                           include_relationships: bool = False) -> None:
    """Parse an OBO file and add nodes/edges to an existing DiGraph.

    Reads [Term] blocks, skips obsolete terms, and restricts to CL: terms.
    Edges point from child to parent (same direction as ``is_a``).

    Args:
        g: NetworkX DiGraph to populate (may already contain nodes).
        obo_path: Path to OBO file.
        include_is_a: Add edges for ``is_a:`` lines.
        include_relationships: Add edges for ``relationship:`` lines whose
            target is a CL: term (e.g. develops_from, part_of).
    """
    current_id = None
    current_name = None
    is_a_parents = []
    rel_parents = []
    in_term = False
    is_obsolete = False

    def _flush_term():
        if not (in_term and current_id and current_id.startswith('CL:')
                and not is_obsolete):
            return
        g.add_node(current_id, name=current_name or current_id)
        if include_is_a:
            for p in is_a_parents:
                if p.startswith('CL:'):
                    g.add_node(p)  # ensure parent exists
                    g.add_edge(current_id, p)
        if include_relationships:
            for p in rel_parents:
                if p.startswith('CL:'):
                    g.add_node(p)
                    g.add_edge(current_id, p)

    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                _flush_term()
                in_term = True
                current_id = None
                current_name = None
                is_a_parents = []
                rel_parents = []
                is_obsolete = False
            elif line.startswith('[') and line.endswith(']'):
                _flush_term()
                in_term = False
            elif in_term:
                if line.startswith('id: '):
                    current_id = line[4:].strip()
                elif line.startswith('name: '):
                    current_name = line[6:].strip()
                elif line.startswith('is_a: '):
                    is_a_parents.append(line[6:].split()[0])
                elif line.startswith('relationship: '):
                    # Format: relationship: REL_ID TARGET_ID ! comment
                    parts = line[14:].split()
                    if len(parts) >= 2:
                        rel_parents.append(parts[1])
                elif line.startswith('is_obsolete: true'):
                    is_obsolete = True

    _flush_term()  # handle last term


# ---------------------------------------------------------------------------
# Method 1: Shortest undirected path distance (original)
# ---------------------------------------------------------------------------

def calculate_graph_distance(graph: 'nx.DiGraph', predicted_id: str, truth_id: str,
                             _undirected_cache: dict = {}) -> int:
    """
    Calculate the shortest-path distance between two terms in the ontology.

    Converts the directed graph to undirected and finds the shortest path.
    Note: on DAGs with multiple inheritance (like the Cell Ontology, where 33.5%
    of terms have multiple parents), shortest undirected path can shortcut across
    separate branches, potentially underestimating true semantic distance.

    Args:
        graph (nx.DiGraph): Ontology graph (edges: child -> parent).
        predicted_id (str): Predicted CL ID.
        truth_id (str): Ground truth CL ID.

    Returns:
        int: Distance (number of edges). Returns -1 if not reachable.
    """
    if predicted_id == truth_id:
        return 0

    if predicted_id not in graph or truth_id not in graph:
        return -1

    # Cache the undirected conversion to avoid rebuilding per call
    graph_id = id(graph)
    if graph_id not in _undirected_cache:
        _undirected_cache[graph_id] = graph.to_undirected()
    undirected_graph = _undirected_cache[graph_id]

    try:
        distance = nx.shortest_path_length(undirected_graph, source=predicted_id, target=truth_id)
        return distance
    except nx.NetworkXNoPath:
        return -1
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Method 2: IC-based Lin similarity using Zhou (k=0.5) weighted IC
#
# This method computes semantic similarity between ontology terms using
# Information Content (IC), which measures how specific/informative a term is.
# Unlike path-based distance, IC-based similarity is well-defined on DAGs
# with multiple inheritance and selects the Most Informative Common Ancestor
# (MICA) rather than relying on ambiguous LCA.
#
# Two key references:
#
#   Zhou, Z., Wang, Y., & Gu, J. (2008). A New Model of Information Content
#   for Semantic Similarity in WordNet. In 2008 Second International Conference
#   on Future Generation Communication and Networking Symposia, Hainan, China,
#   pp. 85-89. doi: 10.1109/FGCNS.2008.16.
#
#   Lin, D. (1998). An Information-Theoretic Definition of Similarity. In
#   Proceedings of the 15th International Conference on Machine Learning
#   (ICML 1998), Vol. 98, pp. 296-304.
#
# See IC_FORMULA_ANALYSIS.md for detailed evaluation of why Zhou k=0.5 was
# chosen over Seco, Sanchez, and depth-only IC formulas.
# ---------------------------------------------------------------------------

def _get_all_ancestors(node: str, graph: 'nx.DiGraph') -> set:
    """Get all ancestors of node (including self) by following edges toward parents."""
    ancestors = {node}
    stack = [node]
    while stack:
        current = stack.pop()
        for parent in graph.successors(current):
            if parent not in ancestors:
                ancestors.add(parent)
                stack.append(parent)
    return ancestors


def _get_all_descendants(node: str, graph: 'nx.DiGraph') -> set:
    """Get all descendants of node (including self) by following edges from children."""
    descendants = {node}
    stack = [node]
    while stack:
        current = stack.pop()
        for child in graph.predecessors(current):
            if child not in descendants:
                descendants.add(child)
                stack.append(child)
    return descendants


def _get_shortest_depth(node: str, graph: 'nx.DiGraph', root: str = 'CL:0000000') -> int:
    """Shortest path length from node to root following parent edges (BFS)."""
    if node == root:
        return 0
    visited = {node}
    queue = deque([(node, 0)])
    while queue:
        current, dist = queue.popleft()
        for parent in graph.successors(current):
            if parent == root:
                return dist + 1
            if parent not in visited:
                visited.add(parent)
                queue.append((parent, dist + 1))
    return -1  # disconnected from root


def precompute_ic(graph: 'nx.DiGraph', k: float = 0.5) -> Dict[str, float]:
    """
    Precompute Zhou (2008) weighted intrinsic IC for all terms in the ontology.

        IC(t) = k * Seco_IC(t) + (1 - k) * depth_component(t)

    where:
        Seco_IC(t)        = 1 - log(|descendants(t)| + 1) / log(N)
        depth_component(t) = log(depth(t) + 1) / log(max_depth + 1)

    This blends descendant count (how many terms are subsumed) with structural
    depth (how far from the root). With k=0.5, both contribute equally, producing
    biologically intuitive IC values on the Cell Ontology — e.g., 'neuron' correctly
    gets higher IC than 'secretory cell' despite having more descendants, because
    neuron sits deeper in the hierarchy.

    Reference:
        Zhou, Z., Wang, Y., & Gu, J. (2008). A New Model of Information Content
        for Semantic Similarity in WordNet. In 2008 Second International Conference
        on Future Generation Communication and Networking Symposia, Hainan, China,
        pp. 85-89. doi: 10.1109/FGCNS.2008.16.

    Args:
        graph (nx.DiGraph): Ontology graph (edges: child -> parent).
        k (float): Weight for Seco component (default 0.5 = equal blend).

    Returns:
        Dict[str, float]: Mapping from term ID to IC value.
    """
    nodes = list(graph.nodes)
    N = len(nodes)
    if N <= 1:
        return {n: 0.0 for n in nodes}

    log_N = math.log(N)

    # Precompute descendant counts and depths
    desc_counts = {}
    depths = {}
    for node in nodes:
        desc_counts[node] = len(_get_all_descendants(node, graph))
        depths[node] = _get_shortest_depth(node, graph)

    max_depth = max(d for d in depths.values() if d >= 0) if depths else 0
    if max_depth == 0:
        log_max_depth_plus1 = 1.0
    else:
        log_max_depth_plus1 = math.log(max_depth + 1)

    ic_values = {}
    for node in nodes:
        seco = 1.0 - math.log(desc_counts[node] + 1) / log_N
        d = depths[node] if depths[node] >= 0 else 0
        depth_comp = math.log(d + 1) / log_max_depth_plus1
        ic_values[node] = k * seco + (1.0 - k) * depth_comp

    return ic_values


def calculate_lin_similarity(graph: 'nx.DiGraph', predicted_id: str, truth_id: str,
                             ic_values: Dict[str, float]) -> float:
    """
    Calculate Lin semantic similarity between two ontology terms.

        Sim_Lin(A, B) = 2 * IC(MICA) / (IC(A) + IC(B))

    where MICA = Most Informative Common Ancestor, the common ancestor with
    the highest Information Content. Unlike LCA (Lowest Common Ancestor), MICA
    is always uniquely defined — it picks the most specific shared ancestor
    regardless of graph structure.

    Result is in [0, 1] where 1 = identical terms, 0 = completely unrelated.

    Reference:
        Lin, D. (1998). An Information-Theoretic Definition of Similarity. In
        Proceedings of the 15th International Conference on Machine Learning
        (ICML 1998), Vol. 98, pp. 296-304.

    Args:
        graph (nx.DiGraph): Ontology graph (edges: child -> parent).
        predicted_id (str): Predicted CL ID.
        truth_id (str): Ground truth CL ID.
        ic_values (Dict[str, float]): Precomputed IC values from precompute_ic().

    Returns:
        float: Lin similarity in [0, 1]. Returns -1.0 if either term is missing.
    """
    if predicted_id == truth_id:
        return 1.0

    if predicted_id not in graph or truth_id not in graph:
        return -1.0

    ic_a = ic_values.get(predicted_id, 0.0)
    ic_b = ic_values.get(truth_id, 0.0)

    if ic_a + ic_b == 0:
        return 0.0

    # Find MICA: common ancestor with highest IC
    ancestors_a = _get_all_ancestors(predicted_id, graph)
    ancestors_b = _get_all_ancestors(truth_id, graph)
    common_ancestors = ancestors_a & ancestors_b

    if not common_ancestors:
        return 0.0

    mica_ic = max(ic_values.get(a, 0.0) for a in common_ancestors)
    return (2.0 * mica_ic) / (ic_a + ic_b)


# ---------------------------------------------------------------------------
# Unified scoring functions (support both methods via 'method' parameter)
#
#   method='shortest_path'  — shortest undirected path distance (lower = more similar)
#   method='ic'    — Lin similarity with Zhou k=0.5 IC (higher = more similar)
# ---------------------------------------------------------------------------

def _compute_pairwise_score(graph, predicted_id, truth_id, method, ic_values=None):
    """Compute a single pairwise ontology score using the chosen method.

    Returns:
        float or None: The score value, or None if it could not be computed.
            For 'shortest_path': distance (int, lower = better). None when -1.
            For 'ic': Lin similarity (float in [0,1], higher = better). None when -1.
    """
    if method == 'shortest_path':
        dist = calculate_graph_distance(graph, predicted_id, truth_id)
        return dist if dist >= 0 else None
    elif method == 'ic':
        if ic_values is None:
            raise ValueError("ic_values must be provided when method='ic'")
        sim = calculate_lin_similarity(graph, predicted_id, truth_id, ic_values)
        return sim if sim >= 0 else None
    else:
        raise ValueError(f"Unknown ontology method: {method!r}. Use 'shortest_path' or 'ic'.")


def score_batch(graph: 'nx.DiGraph', predictions: List[str], ground_truth: List[str],
                method: str = 'shortest_path', ic_values: Dict[str, float] = None) -> Tuple[float, float]:
    """
    Report Mean and Median ontology score for the dataset.

    Args:
        graph (nx.DiGraph): Ontology graph.
        predictions (List[str]): List of predicted IDs.
        ground_truth (List[str]): List of ground truth IDs.
        method (str): 'shortest_path' for shortest-path distance, 'ic' for Lin similarity.
        ic_values (Dict[str, float]): Precomputed IC values (required when method='ic').

    Returns:
        Tuple[float, float]: (Mean, Median) of the scores.
            For 'shortest_path': distance values (lower = more similar).
            For 'ic': similarity values in [0,1] (higher = more similar).
    """
    scores = []

    for p, g in zip(predictions, ground_truth):
        score = _compute_pairwise_score(graph, p, g, method, ic_values)
        if score is not None:
            scores.append(score)

    if not scores:
        return float('nan'), float('nan')

    return float(statistics.mean(scores)), float(statistics.median(scores))


def calculate_per_cell_distances(graph: 'nx.DiGraph', predictions: List[str], ground_truth: List[str],
                                 method: str = 'shortest_path', ic_values: Dict[str, float] = None) -> List[float]:
    """
    Calculate ontology score for each cell individually.

    Args:
        graph (nx.DiGraph): Ontology graph.
        predictions (List[str]): List of predicted IDs.
        ground_truth (List[str]): List of ground truth IDs.
        method (str): 'shortest_path' for shortest-path distance, 'ic' for Lin similarity.
        ic_values (Dict[str, float]): Precomputed IC values (required when method='ic').

    Returns:
        List[float]: List of scores, one per cell. NaN for cells where score cannot be computed.
    """
    scores = []
    for p, g in zip(predictions, ground_truth):
        score = _compute_pairwise_score(graph, p, g, method, ic_values)
        scores.append(score if score is not None else float('nan'))
    return scores


def calculate_avg_neighbor_distances(graph: 'nx.DiGraph', neighbor_labels: List[List[str]], ground_truth: List[str],
                                     method: str = 'shortest_path', ic_values: Dict[str, float] = None) -> List[float]:
    """
    Calculate average ontology score across all neighbors for each cell.

    Args:
        graph (nx.DiGraph): Ontology graph.
        neighbor_labels (List[List[str]]): List of lists of neighbor labels per cell.
        ground_truth (List[str]): List of ground truth IDs, one per cell.
        method (str): 'shortest_path' for shortest-path distance, 'ic' for Lin similarity.
        ic_values (Dict[str, float]): Precomputed IC values (required when method='ic').

    Returns:
        List[float]: Average score across neighbors for each cell.
                    Returns np.nan if no valid neighbors found.
    """
    import numpy as np

    avg_scores = []

    for neighbors, truth in zip(neighbor_labels, ground_truth):
        neighbor_scores = []

        for neighbor_label in neighbors:
            if neighbor_label and pd.notna(neighbor_label) and str(neighbor_label).strip() != '':
                score = _compute_pairwise_score(graph, neighbor_label, truth, method, ic_values)
                if score is not None:
                    neighbor_scores.append(score)

        if len(neighbor_scores) > 0:
            avg_scores.append(float(np.mean(neighbor_scores)))
        else:
            avg_scores.append(np.nan)

    return avg_scores

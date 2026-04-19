"""Parse Cell Ontology OBO files into {term_id: name} dictionaries."""


def parse_obo_names(obo_path: str) -> dict:
    """Parse an OBO file and return a dict of {term_id: name}.

    Reads [Term] blocks and extracts the ``id:`` and ``name:`` fields.
    Non-Term sections (e.g. [Typedef]) are skipped.

    Args:
        obo_path: Path to an OBO ontology file (e.g. cl.obo).

    Returns:
        Dict mapping term IDs to canonical names,
        e.g. ``{"CL:0000540": "neuron", ...}``.
    """
    cl_map = {}
    current_id = None
    current_name = None
    in_term = False
    is_obsolete = False

    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                if in_term and current_id and current_name and not is_obsolete:
                    cl_map[current_id] = current_name
                in_term = True
                current_id = None
                current_name = None
                is_obsolete = False
            elif line.startswith('[') and line.endswith(']'):
                if in_term and current_id and current_name and not is_obsolete:
                    cl_map[current_id] = current_name
                in_term = False
                current_id = None
                current_name = None
                is_obsolete = False
            elif in_term:
                if line.startswith('id: '):
                    current_id = line[4:]
                elif line.startswith('name: ') and current_id:
                    current_name = line[6:]
                elif line == 'is_obsolete: true':
                    is_obsolete = True

    if in_term and current_id and current_name and not is_obsolete:
        cl_map[current_id] = current_name

    return cl_map


def parse_obo_replacements(obo_path: str) -> dict:
    """Parse an OBO file and return a dict of {obsolete_id: replacement_id}.

    Reads [Term] blocks and finds terms that have both ``is_obsolete: true``
    and a ``replaced_by:`` field.  Terms that are obsolete but lack a
    ``replaced_by`` field are omitted from the result.

    Args:
        obo_path: Path to an OBO ontology file (e.g. cl.obo).

    Returns:
        Dict mapping obsolete term IDs to their replacement IDs,
        e.g. ``{"CL:4023070": "CL:4023064", ...}``.
    """
    replacements = {}
    current_id = None
    is_obsolete = False
    replaced_by = None
    in_term = False

    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                # Save previous term if it was obsolete with a replacement
                if in_term and current_id and is_obsolete and replaced_by:
                    replacements[current_id] = replaced_by
                in_term = True
                current_id = None
                is_obsolete = False
                replaced_by = None
            elif line.startswith('[') and line.endswith(']'):
                # Save previous term if it was obsolete with a replacement
                if in_term and current_id and is_obsolete and replaced_by:
                    replacements[current_id] = replaced_by
                in_term = False
                current_id = None
                is_obsolete = False
                replaced_by = None
            elif in_term:
                if line.startswith('id: '):
                    current_id = line[4:]
                elif line.startswith('is_obsolete: true'):
                    is_obsolete = True
                elif line.startswith('replaced_by: '):
                    replaced_by = line[13:]

    # Handle last term in file
    if in_term and current_id and is_obsolete and replaced_by:
        replacements[current_id] = replaced_by

    return replacements

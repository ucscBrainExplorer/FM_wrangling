import json
import sys
import shutil
from datetime import datetime


def diff_entries(old, new):
    """Return a summary of changes between two dicts."""
    all_keys = sorted(set(old) | set(new))
    added, removed, changed = [], [], []
    for k in all_keys:
        if k not in old:
            added.append((k, new[k]))
        elif k not in new:
            removed.append((k, old[k]))
        elif old[k] != new[k]:
            changed.append((k, old[k], new[k]))
    return added, removed, changed


def print_diff(uuid, added, removed, changed, action):
    print(f"\n  [{action}] {uuid}")
    for k, v in added:
        print(f"    + {k}: {json.dumps(v)}")
    for k, v in removed:
        print(f"    - {k}: {json.dumps(v)}")
    for k, old_v, new_v in changed:
        print(f"    ~ {k}: {json.dumps(old_v)} -> {json.dumps(new_v)}")
    if not added and not removed and not changed:
        print("    (no changes)")


def main():
    if len(sys.argv) != 3:
        print("Usage: python merge_metadata_summary.py <data.json> <new_entry.json>")
        sys.exit(1)

    data_file = sys.argv[1]
    new_entry_file = sys.argv[2]

    # Backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{data_file}.{timestamp}.bak"
    shutil.copy2(data_file, backup_file)
    print(f"Backup created: {backup_file}")

    with open(data_file, 'r') as f:
        data = json.load(f)

    with open(new_entry_file, 'r') as f:
        new_entry = json.load(f)

    # Normalise to list
    incoming = new_entry if isinstance(new_entry, list) else [new_entry]

    # Index existing data by reference_uuid
    index = {item['reference_uuid']: i for i, item in enumerate(data)
             if 'reference_uuid' in item}

    print(f"\nChanges:")
    n_updated = n_added = 0

    for entry in incoming:
        uuid = entry.get('reference_uuid')
        if uuid and uuid in index:
            old = data[index[uuid]]
            added, removed, changed = diff_entries(old, entry)
            print_diff(uuid, added, removed, changed, action="UPDATE")
            data[index[uuid]] = entry
            n_updated += 1
        else:
            print(f"\n  [ADD] {uuid or '(no uuid)'}")
            data.append(entry)
            n_added += 1

    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nDone. {n_updated} updated, {n_added} added. '{data_file}' now has {len(data)} items.")


if __name__ == "__main__":
    main()

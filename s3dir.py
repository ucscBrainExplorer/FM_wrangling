"""Let a notebook's results_dir be a local folder or an S3 prefix.

    from s3dir import workdir, push
    results_dir = workdir("s3://latentbrain/<run>/", files=["pca.h5", "knn_30_ivfflat.npz"])
    ...
    push(results_dir)          # upload new outputs

workdir() copies files from the S3 prefix into a local folder and returns it,
so h5py, np.load, joblib and load_pumap read ordinary files. With files=,
only those names are fetched (names missing in S3 are skipped), otherwise
every file directly under the prefix. push() uploads the folder's files back
to the prefix they came from. For a local path, workdir() returns it as is and
push() does nothing.

Nothing is ever overwritten or deleted, on either side: workdir() fetches only
files missing locally and push() uploads only files missing in S3. A file
present on both sides with a different size is left alone with a warning;
rename or remove one copy to resolve it.

s3://<bucket>/<prefix>/ goes to <root>/<prefix>/. <root> is $S3DIR_ROOT
(/mnt/pvc in the NRP jupyter-pumap pod), else ~/s3cache.
"""
import json
import os
import subprocess

# local folder -> the S3 prefix workdir() filled it from, for push()
_origin = {}


def _root():
    return os.environ.get("S3DIR_ROOT") or os.path.expanduser("~/s3cache")


def _s3_files(bucket, prefix):
    """{name: size} of the files directly under s3://bucket/prefix/."""
    out = subprocess.run(
        ["aws", "s3api", "list-objects-v2", "--bucket", bucket, "--prefix", prefix + "/",
         "--delimiter", "/", "--query", "Contents[].[Key, Size]", "--output", "json"],
        check=True, capture_output=True, text=True).stdout
    return {key[len(prefix) + 1:]: size for key, size in (json.loads(out) or [])}


def _cp(src, dst):
    print(f"aws s3 cp {src} {dst}", flush=True)
    subprocess.run(["aws", "s3", "cp", src, dst, "--no-progress"], check=True)


def _differs(name, local_size, s3_size):
    print(f"WARNING: {name} differs ({local_size} bytes local, {s3_size} in S3); "
          f"left both copies alone", flush=True)


def workdir(path, files=None):
    """Local path for `path`: itself if local, else <root>/<prefix>/ with the
    S3 prefix's files (only `files` if given) copied in where missing."""
    if not path.startswith("s3://"):
        return path
    bucket, _, prefix = path[len("s3://"):].strip("/").partition("/")
    s3 = f"s3://{bucket}/{prefix}/"
    local = os.path.join(_root(), prefix) + "/"
    os.makedirs(local, exist_ok=True)
    remote = _s3_files(bucket, prefix)
    for name in (files if files is not None else remote):
        if name not in remote:
            print(f"{name}: not in {s3}, skipped", flush=True)
        elif not os.path.exists(local + name):
            _cp(s3 + name, local + name)
        elif os.path.getsize(local + name) != remote[name]:
            _differs(name, os.path.getsize(local + name), remote[name])
    _origin[os.path.abspath(local)] = (bucket, prefix)
    return local


def push(local_dir):
    """Upload the files in a workdir() folder that its S3 prefix lacks."""
    origin = _origin.get(os.path.abspath(local_dir))
    if origin is None:
        print(f"{local_dir} did not come from workdir('s3://...'); nothing to push", flush=True)
        return
    bucket, prefix = origin
    remote = _s3_files(bucket, prefix)
    for name in sorted(os.listdir(local_dir)):
        local = os.path.join(local_dir, name)
        if not os.path.isfile(local):
            continue
        if name not in remote:
            _cp(local, f"s3://{bucket}/{prefix}/{name}")
        elif os.path.getsize(local) != remote[name]:
            _differs(name, os.path.getsize(local), remote[name])

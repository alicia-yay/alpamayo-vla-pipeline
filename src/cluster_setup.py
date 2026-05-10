"""Install Alpamayo dependencies on every Ray worker.

Ray workers run from system Python (/home/ray/anaconda3), not from the venv
where this project was installed. So `physical_ai_av`, `alpamayo1_5`, and a
few other packages need to be installed on every node before any Ray task
can import them.

This script uses a SPREAD-scheduled Ray actor to run pip installs once per
node, in parallel. Each install uses --ignore-installed to override stale
versions that may have been baked into the workspace image, and --no-deps
on the alpamayo install to skip flash-attn (which fails to build in many
environments and isn't required when we use attn_implementation="eager").

Run once after starting the cluster, then again to verify all nodes are OK.

"""

import os
import subprocess
import sys
from typing import List

import ray

# Ray workers run from system Python, not the venv. Use anaconda's pip
# explicitly because the system pip in some Anyscale images has a known
# snapshot_util import error.
SYSTEM_PY = "/home/ray/anaconda3/bin/python"

# Order matters: install in this sequence to avoid pulling in conflicting
# dependency versions (e.g. transformers must be pinned BEFORE alpamayo1_5
# so alpamayo doesn't quietly upgrade it).
INSTALL_COMMANDS = [
    f"{SYSTEM_PY} -m pip install --ignore-installed --upgrade scipy",
    f"{SYSTEM_PY} -m pip install --ignore-installed torch==2.8.0",
    f"{SYSTEM_PY} -m pip install --ignore-installed physical-ai-av==0.2.0",
    f"{SYSTEM_PY} -m pip install --ignore-installed einops",
    f"{SYSTEM_PY} -m pip install --ignore-installed transformers==4.57.1",
    f"{SYSTEM_PY} -m pip install --ignore-installed --no-deps git+https://github.com/NVlabs/alpamayo1.5.git",
]

VERIFY_IMPORTS = [
    "physical_ai_av",
    "alpamayo1_5",
    "torch",
    "transformers",
    "einops",
]


@ray.remote(num_cpus=0.1, scheduling_strategy="SPREAD")
def install_on_node() -> dict:
    """Run install + verify on a single node. Returns hostname + status."""
    import socket

    hostname = socket.gethostname()
    results = {"hostname": hostname, "ip": _get_ip(), "install_ok": True, "errors": []}

    for cmd in INSTALL_COMMANDS:
        try:
            out = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if out.returncode != 0:
                results["install_ok"] = False
                results["errors"].append({
                    "cmd": cmd[:80],
                    "stderr_tail": out.stderr[-500:] if out.stderr else "",
                })
        except subprocess.TimeoutExpired:
            results["install_ok"] = False
            results["errors"].append({"cmd": cmd[:80], "stderr_tail": "TIMEOUT"})

    # Verify imports work on system Python
    verify_script = "; ".join(f"import {m}" for m in VERIFY_IMPORTS)
    out = subprocess.run(
        [SYSTEM_PY, "-c", verify_script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    results["verify_ok"] = out.returncode == 0
    if not results["verify_ok"]:
        results["verify_error"] = out.stderr[-500:]

    return results


def _get_ip() -> str:
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def main():
    os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)
    ray.init(ignore_reinit_error=True)

    # One install task per alive node. SPREAD scheduling guarantees one
    # task per node when num_cpus is tiny.
    num_nodes = len([n for n in ray.nodes() if n["Alive"]])
    print(f"Found {num_nodes} alive nodes; dispatching install task per node")

    results: List[dict] = ray.get([install_on_node.remote() for _ in range(num_nodes)])

    print()
    print("=" * 70)
    all_ok = True
    seen_hosts = set()
    for r in results:
        host_key = r.get("ip", r["hostname"])
        if host_key in seen_hosts:
            continue
        seen_hosts.add(host_key)
        status = "ok" if r["install_ok"] and r["verify_ok"] else "FAIL"
        print(f"  {r['hostname']:30s}  ip={r.get('ip', '?'):16s}  install: {'ok' if r['install_ok'] else 'FAIL':4s}  verify: {status}")
        if r["errors"]:
            for e in r["errors"]:
                print(f"    install error: {e['cmd']}")
                if e.get("stderr_tail"):
                    print(f"      {e['stderr_tail'].splitlines()[-1]}")
        if not r["verify_ok"]:
            print(f"    verify error: {r.get('verify_error', '?').splitlines()[-1]}")
            all_ok = False
        if not r["install_ok"]:
            all_ok = False

    print("=" * 70)
    if all_ok:
        print("ALL NODES OK")
        sys.exit(0)
    else:
        print("SOME NODES FAILED, see above")
        sys.exit(1)


if __name__ == "__main__":
    main()

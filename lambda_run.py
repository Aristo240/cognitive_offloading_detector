"""Lambda Cloud orchestrator: launch an instance, run an experiment, pull
results back, and terminate -- but only if everything succeeded.

Behavior on failure
-------------------
If any phase fails (launch, ssh, deploy, run, results pull), the instance is
LEFT RUNNING and the script prints the instance ID, IP, and exact commands to
SSH in or to terminate manually once you're done debugging. Nothing is
terminated unless the full run succeeds.

Default workflow
----------------
1. List available Lambda instance types and pick the cheapest with capacity
   (override with --instance-type).
2. Launch it using a registered SSH key (--ssh-key-name).
3. Wait for active state, then for sshd to accept connections.
4. Rsync the local project to ~/<remote-dir>, excluding .git, .venv, results.
5. Install requirements and run --command (default: cross_judge on synthetic).
6. Rsync results back to ./results/.
7. Terminate the instance.

Failure recovery
----------------
- To clean up an instance left running by a failed run:
    python3 lambda_run.py --terminate-only <instance-id>
- To list your running instances:
    python3 lambda_run.py --list

API
---
Uses Lambda Cloud REST API (https://cloud.lambdalabs.com/api/v1). The same
LAMBDA_API_KEY used by the Inference API works here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from base64 import b64encode
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE = "https://cloud.lambda.ai/api/v1"

PROJECT_DIR = Path(__file__).resolve().parent
RSYNC_EXCLUDES = [
    ".git/", ".venv/", "venv/", "__pycache__/", ".DS_Store",
    "results/*.jsonl", "results/*.png", "results/*.csv", "results/*.json",
    ".env",  # never push real keys via rsync; they go through .env.remote
]


# ---------------------------------------------------------------------------
# Lambda Cloud HTTP client (stdlib only)
# ---------------------------------------------------------------------------

class LambdaAPIError(RuntimeError):
    pass


def _request(method: str, path: str, body: dict | None = None) -> dict:
    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        raise RuntimeError("LAMBDA_API_KEY not set in environment.")
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    auth = b64encode(f"{api_key}:".encode()).decode()
    req.add_header("Authorization", f"Basic {auth}")
    req.add_header("Content-Type", "application/json")
    # Lambda's API is fronted by Cloudflare which blocks the default
    # urllib User-Agent ('Python-urllib/X.Y') with error 1010.
    req.add_header("User-Agent",
                   "cognitive-offloading-detector/0.1 (cloud-orchestrator)")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        raise LambdaAPIError(f"{e.code} {e.reason} on {method} {path}: {body_text}") from e
    except urllib.error.URLError as e:
        raise LambdaAPIError(f"Network error on {method} {path}: {e}") from e


def list_instance_types() -> dict:
    return _request("GET", "/instance-types")["data"]


def launch_instance(itype: str, region: str, ssh_key_names: list[str], name: str | None = None) -> str:
    body = {
        "instance_type_name": itype,
        "region_name": region,
        "ssh_key_names": ssh_key_names,
        "quantity": 1,
    }
    if name:
        body["name"] = name
    res = _request("POST", "/instance-operations/launch", body)
    ids = res["data"]["instance_ids"]
    if not ids:
        raise LambdaAPIError(f"Launch returned no instance IDs: {res}")
    return ids[0]


def get_instance(instance_id: str) -> dict:
    return _request("GET", f"/instances/{instance_id}")["data"]


def list_instances() -> list[dict]:
    return _request("GET", "/instances")["data"]


def terminate_instance(instance_id: str) -> dict:
    return _request("POST", "/instance-operations/terminate", {"instance_ids": [instance_id]})


def list_ssh_keys() -> list[dict]:
    return _request("GET", "/ssh-keys")["data"]


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def pick_instance_type(requested: str | None) -> tuple[str, str]:
    types = list_instance_types()
    available = []
    for name, info in types.items():
        regions = info.get("regions_with_capacity_available", [])
        if not regions:
            continue
        price = info["instance_type"]["price_cents_per_hour"]
        available.append((price, name, [r["name"] for r in regions]))
    available.sort()
    if not available:
        raise RuntimeError("No Lambda instance types currently have capacity. Try again later.")
    if requested:
        for price, name, regions in available:
            if name == requested:
                return name, regions[0]
        msg = "Requested instance type not available with capacity. Available cheapest 5:\n"
        for p, n, rs in available[:5]:
            msg += f"  {n}  ${p/100:.2f}/hr in {rs}\n"
        raise RuntimeError(msg)
    cheapest = available[0]
    print(f"Cheapest available: {cheapest[1]}  ${cheapest[0]/100:.2f}/hr  in {cheapest[2]}")
    return cheapest[1], cheapest[2][0]


def wait_for_active(instance_id: str, timeout: int = 900) -> dict:
    print(f"Waiting for instance {instance_id} to become active (up to {timeout}s)...")
    start = time.time()
    last_status = None
    while time.time() - start < timeout:
        info = get_instance(instance_id)
        status = info.get("status")
        ip = info.get("ip")
        if status != last_status:
            print(f"  [{int(time.time()-start)}s] status={status} ip={ip}")
            last_status = status
        if status == "active" and ip:
            return info
        if status in ("terminated", "terminating"):
            raise RuntimeError(f"Instance entered {status} state before becoming active.")
        time.sleep(15)
    raise TimeoutError(f"Instance {instance_id} did not become active within {timeout}s")


def wait_for_sshd(ip: str, ssh_key_path: str, timeout: int = 300) -> None:
    print(f"Waiting for sshd on {ip}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            subprocess.run(
                ["ssh", "-i", ssh_key_path,
                 "-o", "StrictHostKeyChecking=no",
                 "-o", "UserKnownHostsFile=/dev/null",
                 "-o", "ConnectTimeout=10",
                 f"ubuntu@{ip}", "true"],
                check=True, capture_output=True, timeout=15,
            )
            print(f"  sshd ready after {int(time.time()-start)}s")
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            time.sleep(8)
    raise TimeoutError(f"sshd on {ip} did not respond within {timeout}s")


def ssh_run(ip: str, cmd: str, ssh_key_path: str, check: bool = True) -> int:
    """Run a command on the remote via ssh, streaming output to local stdout."""
    full = ["ssh", "-i", ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=30",
            f"ubuntu@{ip}", cmd]
    summary = cmd if len(cmd) <= 120 else cmd[:117] + "..."
    print(f"  $ {summary}")
    rc = subprocess.run(full).returncode
    if check and rc != 0:
        raise RuntimeError(f"Remote command failed (rc={rc}): {summary}")
    return rc


def rsync_to(local_dir: Path, ip: str, remote_dir: str, ssh_key_path: str,
             excludes: list[str]) -> None:
    ssh_arg = (f"ssh -i {ssh_key_path} -o StrictHostKeyChecking=no "
               f"-o UserKnownHostsFile=/dev/null")
    # Note: '--info=stats1' would be nicer but macOS ships rsync 2.6.9 which
    # rejects --info. Stick to plain -az for portability.
    cmd = ["rsync", "-az", "-e", ssh_arg]
    for e in excludes:
        cmd += ["--exclude", e]
    cmd += [str(local_dir) + "/", f"ubuntu@{ip}:{remote_dir}/"]
    print(f"  $ rsync -> ubuntu@{ip}:{remote_dir}/  (excluding {len(excludes)} patterns)")
    subprocess.run(cmd, check=True)


def rsync_from(ip: str, remote_dir: str, local_dir: Path, ssh_key_path: str) -> None:
    ssh_arg = (f"ssh -i {ssh_key_path} -o StrictHostKeyChecking=no "
               f"-o UserKnownHostsFile=/dev/null")
    cmd = ["rsync", "-az", "-e", ssh_arg,
           f"ubuntu@{ip}:{remote_dir}/", str(local_dir) + "/"]
    print(f"  $ rsync <- ubuntu@{ip}:{remote_dir}/")
    subprocess.run(cmd, check=True)


def push_env_file(env_path: Path, ip: str, remote_path: str, ssh_key_path: str) -> None:
    """Securely push the .env file (containing API keys) to the remote."""
    if not env_path.exists():
        print(f"  (no local {env_path}, skipping env push)")
        return
    cmd = ["scp", "-i", ssh_key_path,
           "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null",
           str(env_path), f"ubuntu@{ip}:{remote_path}"]
    print(f"  $ scp .env -> {ip}:{remote_path}")
    subprocess.run(cmd, check=True)
    ssh_run(ip, f"chmod 600 {remote_path}", ssh_key_path)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def survival_message(instance_id: str | None, ip: str | None,
                     ssh_key_path: str | None) -> None:
    """Print clear instructions for manual debug/cleanup."""
    print("\n" + "=" * 70, file=sys.stderr)
    print("INSTANCE NOT TERMINATED.", file=sys.stderr)
    if instance_id:
        print(f"  instance_id: {instance_id}", file=sys.stderr)
    if ip:
        print(f"  ip:          {ip}", file=sys.stderr)
    if instance_id and ip and ssh_key_path:
        print(f"\nTo SSH in and debug:", file=sys.stderr)
        print(f"  ssh -i {ssh_key_path} ubuntu@{ip}", file=sys.stderr)
    if instance_id:
        print(f"\nTo terminate manually once you're done:", file=sys.stderr)
        print(f"  python3 {Path(__file__).name} --terminate-only {instance_id}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssh-key-name", default=None,
                    help="Name of an SSH key registered with Lambda Cloud. "
                         "Run --list-keys to see your registered keys.")
    ap.add_argument("--ssh-key-path", default=os.path.expanduser("~/.ssh/id_rsa"),
                    help="Local private key path matching --ssh-key-name.")
    ap.add_argument("--instance-type", default=None,
                    help="Lambda instance type. Default: cheapest with capacity.")
    ap.add_argument("--remote-dir", default="cognitive_offloading_detector",
                    help="Working directory on the remote (relative to ~).")
    ap.add_argument("--command", default=(
        "python3 cross_judge.py --source synthetic "
        "--data data/synthetic_examples.json "
        "--out-dir results/cross_judge_lambda/"
    ), help="The experiment command to run on the remote.")
    ap.add_argument("--results-remote", default="results",
                    help="Path on the remote (relative to remote-dir) to rsync back.")
    ap.add_argument("--results-local", default="results",
                    help="Local directory to receive results.")
    ap.add_argument("--no-terminate", action="store_true",
                    help="Don't terminate even on full success (useful for poking around after).")

    # Inspection / cleanup commands
    ap.add_argument("--list-types", action="store_true",
                    help="List instance types with capacity, sorted by price, and exit.")
    ap.add_argument("--list-keys", action="store_true",
                    help="List SSH keys registered with Lambda Cloud, and exit.")
    ap.add_argument("--list", action="store_true",
                    help="List currently-running Lambda instances, and exit.")
    ap.add_argument("--terminate-only", default=None,
                    help="Terminate the given instance-id and exit.")

    args = ap.parse_args()

    # Inspection paths -- no instance is created
    if args.list_types:
        for name, info in sorted(list_instance_types().items()):
            regions = info.get("regions_with_capacity_available", [])
            price = info["instance_type"]["price_cents_per_hour"]
            mark = "*" if regions else " "
            print(f"  {mark} {name:50s} ${price/100:5.2f}/hr  capacity in {[r['name'] for r in regions]}")
        return 0
    if args.list_keys:
        keys = list_ssh_keys()
        if not keys:
            print("(no SSH keys registered. Add one in the Lambda Cloud dashboard.)")
        for k in keys:
            print(f"  - {k.get('name')}  (id={k.get('id')})")
        return 0
    if args.list:
        ins = list_instances()
        if not ins:
            print("(no running instances)")
        for i in ins:
            print(f"  - {i.get('id')}  type={i.get('instance_type', {}).get('name')}  "
                  f"status={i.get('status')}  ip={i.get('ip')}")
        return 0
    if args.terminate_only:
        print(f"Terminating {args.terminate_only}...")
        terminate_instance(args.terminate_only)
        print("Done.")
        return 0

    # Full run requires ssh key name
    if not args.ssh_key_name:
        print("ERROR: --ssh-key-name is required for a full run.", file=sys.stderr)
        print("       Run with --list-keys to see registered keys.", file=sys.stderr)
        return 2
    if not Path(args.ssh_key_path).exists():
        print(f"ERROR: ssh-key-path not found: {args.ssh_key_path}", file=sys.stderr)
        return 2

    instance_id: str | None = None
    ip: str | None = None
    try:
        # Phase 1: pick + launch
        itype, region = pick_instance_type(args.instance_type)
        print(f"\n=== Phase 1: launch ===")
        instance_id = launch_instance(itype, region, [args.ssh_key_name],
                                       name=f"cog-offload-{int(time.time())}")
        print(f"  instance_id={instance_id}")
        info = wait_for_active(instance_id)
        ip = info["ip"]
        print(f"  ip={ip}")

        # Phase 2: deploy
        print(f"\n=== Phase 2: deploy ===")
        wait_for_sshd(ip, args.ssh_key_path)
        ssh_run(ip,
                "sudo apt-get update -qq && "
                "sudo apt-get install -y -qq python3-pip rsync >/dev/null",
                args.ssh_key_path)
        ssh_run(ip, f"mkdir -p ~/{args.remote_dir}", args.ssh_key_path)
        rsync_to(PROJECT_DIR, ip, args.remote_dir, args.ssh_key_path, RSYNC_EXCLUDES)
        push_env_file(PROJECT_DIR / ".env", ip, f"~/{args.remote_dir}/.env", args.ssh_key_path)

        # Phase 3: install + run
        print(f"\n=== Phase 3: install + run ===")
        ssh_run(ip,
                f"cd ~/{args.remote_dir} && "
                f"python3 -m pip install --quiet --upgrade pip && "
                f"python3 -m pip install --quiet -r requirements.txt",
                args.ssh_key_path)
        ssh_run(ip, f"cd ~/{args.remote_dir} && {args.command}", args.ssh_key_path)

        # Phase 4: pull results
        print(f"\n=== Phase 4: pull results ===")
        local_results = Path(args.results_local).resolve()
        local_results.mkdir(parents=True, exist_ok=True)
        rsync_from(ip, f"{args.remote_dir}/{args.results_remote}",
                   local_results, args.ssh_key_path)
        print(f"  Results saved locally: {local_results}")

        # Phase 5: terminate
        if args.no_terminate:
            print(f"\n--no-terminate set; instance {instance_id} (ip={ip}) left running.")
            print(f"Terminate manually with: python3 {Path(__file__).name} --terminate-only {instance_id}")
        else:
            print(f"\n=== Phase 5: terminate ===")
            terminate_instance(instance_id)
            print(f"  Terminated {instance_id}.")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        survival_message(instance_id, ip, args.ssh_key_path)
        return 130
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}", file=sys.stderr)
        survival_message(instance_id, ip, args.ssh_key_path)
        return 1


if __name__ == "__main__":
    sys.exit(main())

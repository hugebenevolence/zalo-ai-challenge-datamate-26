"""Execute generated code against one business's tables.

A spawned worker (no CUDA, no vLLM threads) loads the tables once; each job runs in a child forked
from that worker, so a crash, hang or mutation of `tables` cannot leak into the next job.

This module and helpers.py must stay importable files: `spawn` re-imports the worker function, and
the generated code's namespace is built from helpers.EXPORTS. The tables reach the worker as a plain
pickle {"frames": {key: DataFrame}, "readme": str} written by save_tables, so the notebooks are free
to define their own classes.
"""
from __future__ import annotations

import contextlib
import datetime
import io
import keyword
import math
import multiprocessing as mp
import os
import pickle
import resource
import signal
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

PRELUDE = "import pandas as pd\nimport numpy as np\nimport re, math, datetime, unicodedata\n"


def save_tables(path: Path, frames: dict, readme: str = "") -> Path:
    """Write the worker's input: plain pandas objects only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"frames": frames, "readme": readme}))
    return path


def table_variables(frames: dict) -> dict:
    """Bare names the model writes for tables: an identifier key as it is (sqlite `don_hang`), and a
    file's stem (`don_hang` for "don_hang.csv") when no other file has that stem."""
    stems: dict[str, list] = {}
    for k in frames:
        if "::" not in k and "." in k:
            stems.setdefault(k.rsplit(".", 1)[0], []).append(k)
    names = {stem: frames[keys[0]] for stem, keys in stems.items() if len(keys) == 1}
    names.update((k, v) for k, v in frames.items() if k.isidentifier())
    return {k: v for k, v in names.items() if k.isidentifier() and not keyword.iskeyword(k)}


def _cap_memory(gb: float):
    """Make a runaway program (e.g. `df[mask] & series`, which aligns into an n x n frame) fail fast with
    MemoryError instead of pushing the machine into the OOM killer. The cap is on top of what the child
    already maps (the worker's tables and libraries)."""
    if gb <= 0:
        return
    with contextlib.suppress(OSError, ValueError):
        vsize = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
        limit = vsize + int(gb * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


@dataclass
class ExecResult:
    ok: bool
    answer: dict | None
    stdout: str
    error: str | None
    elapsed: float


def _jsonable(v):
    """numpy / pandas scalars and containers -> plain JSON values."""
    import numpy as np
    import pandas as pd
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (pd.Series, np.ndarray, pd.Index)):
        return [_jsonable(x) for x in list(v)]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, datetime.date):  # includes pd.Timestamp and datetime.datetime
        return v.strftime("%Y-%m-%d")
    return v


def _child(conn, code: str, env: dict, max_stdout: int, mem_gb: float):
    _cap_memory(mem_gb)
    out = io.StringIO()
    g = dict(env)
    t = time.time()
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(PRELUDE + code, "<generated>", "exec"), g)
        ans = g.get("ANSWER")
        conn.send(dict(ok=ans is not None, answer=_jsonable(ans) if ans is not None else None,
                       stdout=out.getvalue()[-max_stdout:],
                       error=None if ans is not None else "Code finished without setting ANSWER.",
                       elapsed=time.time() - t))
    except BaseException:
        conn.send(dict(ok=False, answer=None, stdout=out.getvalue()[-max_stdout:],
                       error=traceback.format_exc(limit=4)[-2500:], elapsed=time.time() - t))
    finally:
        conn.close()


def _worker(conn, biz_pickle: str, calendar_csv: str, timeout: float, max_stdout: int):
    import pandas as pd

    from . import helpers
    payload = pickle.loads(Path(biz_pickle).read_bytes())
    calendar = pd.read_csv(calendar_csv)
    ctx = mp.get_context("fork")
    parallel = int(os.environ.get("DATAMATE_EXEC_PARALLEL", 4))
    mem_gb = float(os.environ.get("DATAMATE_EXEC_MEM_GB", 4))
    named = table_variables(payload["frames"])

    def start(code, as_of):
        env = {**named, **helpers.EXPORTS, "tables": payload["frames"], "calendar": calendar,
               "AS_OF": pd.Timestamp(as_of), "README": payload.get("readme", "")}
        r_conn, w_conn = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_child, args=(w_conn, code, env, max_stdout, mem_gb))
        p.start()
        w_conn.close()
        return p, r_conn, time.time()

    def finish(p, r_conn, t0, wait):
        res = None
        if r_conn.poll(wait):
            try:
                res = r_conn.recv()
            except EOFError:
                pass
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)
        p.join()
        if res is None:
            res = dict(ok=False, answer=None, stdout="", elapsed=time.time() - t0,
                       error=f"Timed out after {timeout:.0f}s" if time.time() - t0 >= timeout else "Process crashed")
        return res

    while True:
        job = conn.recv()
        if job is None:
            break
        if job[0] != "many":  # single job: (code, as_of)
            p, r, t0 = start(*job)
            conn.send(finish(p, r, t0, timeout))
            continue
        jobs, results, running, nxt = job[1], {}, [], 0
        while nxt < len(jobs) or running:
            while nxt < len(jobs) and len(running) < parallel:
                running.append((nxt, *start(*jobs[nxt])))
                nxt += 1
            # Collect whichever child finishes first; each job keeps its own deadline.
            idx, p, r, t0 = running[0]
            for item in running:
                if item[2].poll(0):
                    idx, p, r, t0 = item
                    break
            else:
                time.sleep(0.02)
                expired = [it for it in running if time.time() - it[3] >= timeout]
                if not expired:
                    continue
                idx, p, r, t0 = expired[0]
            running.remove((idx, p, r, t0))
            results[idx] = finish(p, r, t0, max(0.0, timeout - (time.time() - t0)))
        conn.send([results[i] for i in range(len(jobs))])


class Executor:
    def __init__(self, biz_pickle: Path, calendar_csv: Path, timeout: float = 60, max_stdout: int = 3000):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        # Not a daemon: daemonic processes cannot fork job children. close() / __exit__ shut it down.
        self.proc = ctx.Process(target=_worker, args=(child, str(biz_pickle), str(calendar_csv), timeout, max_stdout))
        self.proc.start()

    def run(self, code: str, as_of: str) -> ExecResult:
        self.conn.send((code, as_of))
        return ExecResult(**self.conn.recv())

    def run_many(self, jobs: list[tuple[str, str]]) -> list[ExecResult]:
        """Run (code, as_of) jobs, DATAMATE_EXEC_PARALLEL at a time; results in input order."""
        if not jobs:
            return []
        self.conn.send(("many", jobs))
        return [ExecResult(**r) for r in self.conn.recv()]

    def close(self):
        with contextlib.suppress(Exception):
            self.conn.send(None)
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

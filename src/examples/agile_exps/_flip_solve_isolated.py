"""Run a trajopt solve in an isolated child process with automatic retry.

Why: in this environment (conda pinocchio 3.9 / coal 3.0.2 / cyipopt 1.6), the combination
of hpp-fcl distance queries and pinocchio Jacobian evaluations inside the cyipopt callbacks
INTERMITTENTLY corrupts the CPython interpreter heap (random ``TypeError``/``AttributeError``
on healthy objects, or a plain SIGSEGV). Each individual API is fine standalone; only the
in-callback combination goes wrong, and only sometimes.

Mitigation: the whole build+solve runs in a forked child. If the child crashes (segfault or
exception), the parent — which never calls any solver — retries in a fresh child. Deterministic
IPOPT failures (infeasible NLP, etc.) still raise normally from the last attempt.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
from typing import Callable, Dict


def _worker(build_and_solve, out_queue):
    try:
        result = build_and_solve()
        out_queue.put(("ok", result))
        code = 0
    except Exception as e:  # noqa: BLE001 - forwarded to the parent
        out_queue.put(("err", f"{type(e).__name__}: {e}"))
        code = 1
    # Flush the queue feeder, then hard-exit: the solver libraries (IPOPT/MUMPS OpenMP
    # threads) deadlock in their destructors inside a forked child, which would hang
    # ``Process.join()`` in the parent forever.
    out_queue.close()
    out_queue.join_thread()
    os._exit(code)


def solve_isolated(
    build_and_solve: Callable[[], Dict],
    *,
    max_attempts: int = 4,
    tag: str = "solve",
) -> Dict:
    """Run ``build_and_solve()`` in a forked child; retry with a fresh child on crashes.

    Set ``GO2_NO_ISOLATED=1`` to bypass (solve in-process, old behaviour).
    """
    if os.environ.get("GO2_NO_ISOLATED", "").lower() in ("1", "true", "yes"):
        return build_and_solve()

    ctx = mp.get_context("fork")
    last_failure = "unknown"
    for attempt in range(1, max_attempts + 1):
        out_queue = ctx.Queue()
        proc = ctx.Process(target=_worker, args=(build_and_solve, out_queue))
        # Drain the queue concurrently with join(): results are far larger than the pipe
        # buffer, so a child blocked on write + a parent blocked on join() would deadlock.
        box: dict = {}

        def _reader():
            try:
                box["msg"] = out_queue.get()
            except Exception:  # noqa: BLE001 - child died before/while putting
                pass

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()
        proc.start()
        proc.join()
        reader.join(timeout=10.0)

        if "msg" in box:
            status, payload = box["msg"]
            if status == "ok":
                if attempt > 1:
                    print(f"[{tag}] isolated solve succeeded on attempt {attempt}")
                return payload
            last_failure = payload
            print(f"[{tag}] isolated solve attempt {attempt} failed: {payload}")
        elif proc.exitcode == 0:
            last_failure = "child exited 0 without returning a result"
            print(f"[{tag}] isolated solve attempt {attempt} failed: {last_failure}")
        else:
            last_failure = f"child exited with code {proc.exitcode} (crash)"
            print(
                f"[{tag}] isolated solve attempt {attempt} crashed "
                f"(exit {proc.exitcode}); retrying in a fresh process"
            )

    raise RuntimeError(
        f"[{tag}] isolated solve failed after {max_attempts} attempts; last failure: {last_failure}"
    )

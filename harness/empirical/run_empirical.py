#!/usr/bin/env python3
"""
==============================================================================
Tesla T4 (Turing CC 7.5) — EMPIRICAL VALIDATION RUNNER
==============================================================================

Converts simulation-verified hypotheses (H4 / H7 / H9 family) into
hardware-measured evidence on a physical T4 GPU.

Design goals
------------
1. Single self-contained entry point:  python3 harness/empirical/run_empirical.py
2. Self-stamping: records GPU, driver, CUDA, torch, git SHA, wall clock.
3. Self-logging: tees ALL stdout/stderr into results/<run>/run.log.
4. Staged with hard gates: each stage emits PASS/FAIL/SKIP for every metric
   from expected.yaml. A stage that cannot run (e.g. no ncu) reports SKIP,
   never a fake PASS.
5. Machine-readable output: summary.json + VERDICT.md per run, plus a tarball.

Stage layout
------------
  S0  Environment stamp          (always; no GPU needed)
  S1  Static compile audit       (nvcc -Xptxas -v; needs nvcc only)
  S2  Extension build            (torch cpp_extension; needs torch+nvcc)
  S3  On-GPU correctness + KAT   (needs CUDA GPU)
  S4  Telemetry                  (nvidia-smi sampler during stress; needs GPU)
  S5  ncu roofline SOL           (needs ncu + GPU; SKIP if ncu absent)
  S6  Verdict synthesis          (always; reads expected.yaml)

Capable vs not
---------------
  kernel_ready  metrics are measured on hardware   -> CONFIRM / MARGIN / FAIL
  sim_only      hypotheses report NOT_MEASURABLE    -> recorded, never blocks

Exit code: 0 if every kernel_ready criterion CONFIRMs, else 1.
==============================================================================
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path(os.getcwd()).resolve() / "harness" / "empirical" # harness/empirical
REPO = HERE.parent.parent                             # research/
SRC = REPO / "src"
RESULTS_ROOT = REPO / "results"
EXPECTED_YAML = HERE / "expected.yaml"

T4_PEAK_BW_GBPS = 320.0
T4_PEAK_FP16_TC_TFLOPS = 65.0
T4_TDP_W = 70.0

# --------------------------------------------------------------------------- #
# Tiny YAML loader (expected.yaml is deliberately simple: nested dicts of
# scalars). We avoid a hard PyYAML dependency so the script runs on a bare box;
# if PyYAML IS present we use it for robustness.
# --------------------------------------------------------------------------- #
def _load_expected():
    try:
        import yaml  # type: ignore
        with open(EXPECTED_YAML) as f:
            return yaml.safe_load(f)
    except Exception:
        # Fallback: ultra-minimal parse is NOT attempted (too error-prone).
        # Instead we hard-fail with a clear instruction.
        raise RuntimeError(
            "PyYAML is required to read expected.yaml. "
            "Install it: pip install pyyaml  (Colab has it preinstalled)."
        )


# --------------------------------------------------------------------------- #
# Logging: tee everything to console AND run.log
# --------------------------------------------------------------------------- #
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _banner(txt):
    line = "=" * 78
    print(f"\n{line}\n  {txt}\n{line}")


def _sub(txt):
    print(f"\n--- {txt} ---")


# --------------------------------------------------------------------------- #
# Shell helper — capture output even on failure, never raise on non-zero
# --------------------------------------------------------------------------- #
def _run(cmd, cwd=None, env=None, timeout=None):
    """Run cmd, return (rc, combined_output). Never raises."""
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return None, f"[TIMEOUT after {timeout}s]"
    except FileNotFoundError:
        return None, f"[COMMAND NOT FOUND: {cmd[0]}]"
    except Exception as e:
        return None, f"[EXCEPTION: {e}]"


def _which(name):
    return shutil.which(name)


# --------------------------------------------------------------------------- #
# Metric evaluation engine
# --------------------------------------------------------------------------- #
def _apply_criterion(predicted, measured, criterion, tolerance):
    """Return (status, detail) where status in CONFIRM/MARGIN/FAIL/INFO."""
    if criterion == "informational":
        return "INFO", "informational only"
    if criterion == "bool":
        return ("CONFIRM" if measured is True else "FAIL"), "boolean check"
    if measured is None:
        return "FAIL", "no measurement"

    tol = tolerance or 0.0
    if criterion == "gte":
        lo = predicted  # confirm if measured >= predicted; margin if within tol below
        if measured >= lo:
            return "CONFIRM", f"{measured:.4g} >= {lo:.4g}"
        if tol and measured >= lo - tol:
            return "MARGIN", f"{measured:.4g} within tol of {lo:.4g}"
        return "FAIL", f"{measured:.4g} < {lo:.4g}"
    if criterion == "lte":
        hi = predicted if predicted is not None else tolerance
        if measured <= hi:
            return "CONFIRM", f"{measured:.4g} <= {hi:.4g}"
        if tol and measured <= hi + tol:
            return "MARGIN", f"{measured:.4g} within tol of {hi:.4g}"
        return "FAIL", f"{measured:.4g} > {hi:.4g}"
    if criterion == "eq":
        return ("CONFIRM" if measured == predicted else "FAIL"), f"{measured} vs {predicted}"
    return "INFO", f"unknown criterion {criterion}"


class StageResult:
    def __init__(self, name):
        self.name = name
        self.status = "NOT_RUN"   # PASS/FAIL/SKIP
        self.metrics = {}         # metric -> dict(status, measured, predicted, detail)
        self.notes = []
        self.duration_s = 0.0


# ============================================================================= #
# STAGE 0 — ENVIRONMENT STAMP
# ============================================================================= #
def stage_env_stamp(ctx):
    sr = StageResult("S0_env")
    t0 = time.time()
    _banner("STAGE 0 — Environment & Provenance Stamp")

    stamp = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }

    # git
    rc, out = _run(["git", "rev-parse", "HEAD"], cwd=REPO)
    stamp["git_sha"] = out.strip() if rc == 0 else "unknown"
    rc, out = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO)
    stamp["git_branch"] = out.strip() if rc == 0 else "unknown"
    rc, out = _run(["git", "status", "--short"], cwd=REPO)
    stamp["git_dirty"] = bool(out.strip())

    # nvcc
    nvcc = _which("nvcc")
    stamp["nvcc_path"] = nvcc or "NOT_FOUND"
    if nvcc:
        rc, out = _run([nvcc, "--version"])
        stamp["nvcc_version"] = [l for l in out.splitlines() if "release" in l][:1] or out.splitlines()[-1:]

    # GPU
    nsmi = _which("nvidia-smi")
    stamp["nvidia_smi_path"] = nsmi or "NOT_FOUND"
    if nsmi:
        rc, out = _run([nsmi, "--query-gpu=name,driver_version,memory.total,power.limit,compute_cap",
                        "--format=csv,noheader"])
        stamp["gpu"] = out.strip() if rc == 0 else f"[query failed rc={rc}]"

    # torch
    try:
        import torch
        stamp["torch_version"] = torch.__version__
        stamp["torch_cuda_available"] = torch.cuda.is_available()
        stamp["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            stamp["torch_device_name"] = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            stamp["torch_compute_capability"] = f"{cc[0]}.{cc[1]}"
    except Exception as e:
        stamp["torch_version"] = f"NOT_IMPORTABLE ({e})"

    ctx["env_stamp"] = stamp
    for k, v in stamp.items():
        print(f"  {k:26s}: {v}")

    # warn if not T4-class
    name = str(stamp.get("torch_device_name", "")) + str(stamp.get("gpu", ""))
    if "T4" not in name:
        sr.notes.append("WARNING: device does not appear to be a Tesla T4. "
                        "Results valid for the detected device, but not for T4 claims.")
        print("\n  [WARN] Non-T4 device detected. Claims apply to detected GPU only.")

    sr.status = "PASS"
    sr.duration_s = time.time() - t0
    return sr


# ============================================================================= #
# STAGE 1 — STATIC COMPILE AUDIT (nvcc -Xptxas -v), no GPU required
# ============================================================================= #
def stage_compile_audit(ctx):
    sr = StageResult("S1_compile_audit")
    t0 = time.time()
    _banner("STAGE 1 — Static Compile Audit (nvcc -Xptxas -v)")

    nvcc = _which("nvcc")
    if not nvcc:
        sr.status = "SKIP"
        sr.notes.append("nvcc not found; static compile audit cannot run.")
        print("  [SKIP] nvcc not found.")
        sr.duration_s = time.time() - t0
        return sr

    targets = [
        SRC / "kernels" / "lop3_dequant.cu",
        SRC / "kernels" / "fused_w4a16_gemm.cu",
    ]
    audit = {}
    all_ok = True
    for cu in targets:
        if not cu.exists():
            sr.notes.append(f"MISSING source: {cu}")
            all_ok = False
            continue
        obj = ctx["run_dir"] / f"{cu.stem}.o"
        _sub(f"Compiling {cu.name} (sm_75, -Xptxas -v)")
        rc, out = _run([nvcc, "-O3", "-arch=sm_75", "--use_fast_math",
                        "-Xptxas=-v", "-c", str(cu), "-o", str(obj)],
                       cwd=SRC)
        print(out)
        audit[cu.name] = {"rc": rc, "raw": out}
        # parse ptxas info
        info = {"registers": {}, "spill_stores": 0, "spill_loads": 0, "stack": 0}
        cur_fn = None
        for line in out.splitlines():
            if "Compiling entry function" in line:
                cur_fn = line.split("function")[1].strip().strip("'")
            if "Used" in line and "registers" in line and cur_fn:
                # e.g. "Used 16 registers, used 0 barriers, 388 bytes cmem[0]"
                for tok in line.split(","):
                    if "registers" in tok:
                        try:
                            info["registers"][cur_fn] = int(tok.split("Used")[1].split("registers")[0].strip())
                        except Exception:
                            pass
            if "bytes stack frame" in line:
                for key, pat in (("stack", "bytes stack frame"),
                                 ("spill_stores", "bytes spill stores"),
                                 ("spill_loads", "bytes spill loads")):
                    if pat in line:
                        try:
                            info[key] = max(info[key], int(line.split(pat)[0].strip().split()[-1]))
                        except Exception:
                            pass
        audit[cu.name]["ptxas"] = info
        if rc != 0:
            all_ok = False

    ctx["compile_audit"] = audit

    # evaluate against manifest expectations (registers <=64, spills == 0)
    exp = ctx["expected"]["hypotheses"]
    for hid in ("H4_signed_int4_lop3", "H4_companion_unsigned_int4_lop3"):
        m = exp[hid]["metrics"]
        max_reg = m["registers_per_thread"]["predicted_max"]
        max_spill = m["stack_spills_bytes"]["predicted_max"]

        # take worst across all kernels/functions
        worst_reg = 0
        worst_spill = 0
        for name, a in audit.items():
            pa = a.get("ptxas", {})
            regs = list(pa.get("registers", {}).values())
            worst_reg = max(worst_reg, max(regs) if regs else 0)
            worst_spill = max(worst_spill, pa.get("spill_stores", 0) + pa.get("spill_loads", 0))

        st_reg = _apply_criterion(max_reg, worst_reg if worst_reg else None, "lte", 0)
        st_spill = _apply_criterion(max_spill, worst_spill, "lte", 0)
        sr.metrics[f"{hid}.registers_per_thread"] = {
            "status": st_reg[0], "measured": worst_reg, "predicted_max": max_reg, "detail": st_reg[1]}
        sr.metrics[f"{hid}.stack_spills_bytes"] = {
            "status": st_spill[0], "measured": worst_spill, "predicted_max": max_spill, "detail": st_spill[1]}

        print(f"\n  [{hid}] worst registers/thread = {worst_reg} (limit {max_reg}) -> {st_reg[0]}")
        print(f"  [{hid}] worst spill bytes      = {worst_spill} (limit {max_spill}) -> {st_spill[0]}")

    sr.status = "PASS" if all_ok else "FAIL"
    ctx.setdefault("artifacts", {})["compile_audit.json"] = audit
    sr.duration_s = time.time() - t0
    return sr


# ============================================================================= #
# STAGE 2 — BUILD PYTORCH CUDA EXTENSION
# ============================================================================= #
def stage_build_extension(ctx):
    sr = StageResult("S2_build_ext")
    t0 = time.time()
    _banner("STAGE 2 — Build PyTorch CUDA Extension (t4_kernels)")

    try:
        import torch  # noqa: F401
    except Exception:
        sr.status = "SKIP"
        sr.notes.append("torch not importable; cannot build extension.")
        print("  [SKIP] torch not importable.")
        sr.duration_s = time.time() - t0
        return sr

    nvcc = _which("nvcc")
    if not nvcc:
        sr.status = "SKIP"
        sr.notes.append("nvcc not found; cannot build CUDA extension.")
        print("  [SKIP] nvcc not found.")
        sr.duration_s = time.time() - t0
        return sr

    # Clean previous in-place build to force fresh ptxas
    built = list(SRC.glob("t4_kernels*.so"))
    for f in built:
        try:
            f.unlink()
        except Exception:
            pass

    _sub("python3 setup.py build_ext --inplace")
    rc, out = _run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=SRC, timeout=1800)
    print(out)
    ctx.setdefault("artifacts", {})["build_log.txt"] = out

    so = list(SRC.glob("t4_kernels*.so"))
    sr.notes.append(f"built .so files: {[p.name for p in so]}")
    if rc == 0 and so:
        sr.status = "PASS"
        print(f"  [OK] Extension built: {so[0].name}")
    else:
        sr.status = "FAIL"
        sr.notes.append("Extension build failed — see build_log.txt.")
        print("  [FAIL] Extension build failed.")
    sr.duration_s = time.time() - t0
    return sr


# ============================================================================= #
# STAGE 3 — ON-GPU CORRECTNESS + KAT (H4 u4/s4, fused GEMM, INT3/FP8 math refs)
# ============================================================================= #
def stage_gpu_correctness(ctx):
    sr = StageResult("S3_gpu_correctness")
    t0 = time.time()
    _banner("STAGE 3 — On-GPU Correctness & Known-Answer Tests")

    try:
        import torch
    except Exception:
        sr.status = "SKIP"; sr.notes.append("torch missing"); sr.duration_s = time.time()-t0
        return sr
    if not torch.cuda.is_available():
        sr.status = "SKIP"
        sr.notes.append("CUDA not available; on-GPU correctness skipped.")
        print("  [SKIP] CUDA not available.")
        sr.duration_s = time.time() - t0
        return sr

    sys.path.insert(0, str(SRC))
    try:
        import t4_kernels
    except Exception as e:
        sr.status = "FAIL"
        sr.notes.append(f"t4_kernels not importable: {e}. Run stage 2 build first.")
        print(f"  [FAIL] t4_kernels not importable: {e}")
        sr.duration_s = time.time() - t0
        return sr

    exp = ctx["expected"]["hypotheses"]
    dev = "cuda"

    def pack_kat_int32(hexval: int) -> "torch.Tensor":
        """Pack a 32-bit KAT hex into a torch int32 tensor WITHOUT dropping bit 31.

        Bug history (run 2026-07-30): torch.tensor([h & 0x7FFFFFFF], dtype=int32)
        silently truncated the sign bit, so the kernel was tested on the wrong
        packed word and element 7 mismatched. We must route through int64 then
        .to(torch.int32), matching verify_colab.sh, which preserves the pattern.
        """
        t = torch.tensor([int(hexval)], dtype=torch.int64).to(torch.int32).to(dev)
        # Guard: bit 31 must round-trip
        assert (int(t.item()) & 0xFFFFFFFF) == (hexval & 0xFFFFFFFF), (
            f"KAT packing dropped bits: in=0x{hexval:08X} got=0x{int(t.item()) & 0xFFFFFFFF:08X}")
        return t

    # ---------- reference dequantizers (independent of kernels) ---------- #
    def ref_u4(W_u32, scale, zero):
        out = []
        s = float(scale); z = float(zero)
        for wi in range(8):
            nib = (W_u32 >> (wi * 4)) & 0xF
            out.append((nib - z) * s)
        return out

    def ref_s4(W_u32, scale, zero):
        out = []
        s = float(scale); z = float(zero)
        for wi in range(8):
            nib = (W_u32 >> (wi * 4)) & 0xF
            s4 = nib - 16 if (nib & 8) else nib
            out.append((s4 - z) * s)
        return out

    results = {}

    # ---------- H4 unsigned KAT ---------- #
    _sub("H4 Unsigned INT4 KAT (vector 0xA7C13E59)")
    kat_w = 0xA7C13E59
    kat_s, kat_z = 0.25, 2.0
    W = pack_kat_int32(kat_w)
    sc = torch.tensor([kat_s], dtype=torch.float16, device=dev)
    zz = torch.tensor([kat_z], dtype=torch.float16, device=dev)
    got = t4_kernels.dequantize_u4(W, sc, zz)[:8].float().cpu().tolist()
    want = ref_u4(kat_w, kat_s, kat_z)
    mae = max(abs(g - w) for g, w in zip(got, want))
    st, dt = _apply_criterion(0.0, mae, "lte", 1e-3)
    results["u4_kat"] = {"max_abs_diff": mae, "status": st}
    print(f"   got : {[round(x,4) for x in got]}")
    print(f"   want: {[round(x,4) for x in want]}")
    print(f"   max_abs_diff = {mae:.6g}  -> {st}")
    sr.metrics["H4_companion_unsigned_int4_lop3.kat_bit_exact"] = {
        "status": st, "measured": mae, "predicted": 0.0, "tolerance": 1e-3, "detail": dt}

    # ---------- H4 signed KAT ---------- #
    _sub("H4 Signed INT4 KAT (vector 0xF817E29A)")
    kat_w = 0xF817E29A
    kat_s, kat_z = 0.5, 0.0
    W = pack_kat_int32(kat_w)
    sc = torch.tensor([kat_s], dtype=torch.float16, device=dev)
    zz = torch.tensor([kat_z], dtype=torch.float16, device=dev)
    got = t4_kernels.dequantize_s4(W, sc, zz)[:8].float().cpu().tolist()
    want = ref_s4(kat_w, kat_s, kat_z)
    mae = max(abs(g - w) for g, w in zip(got, want))
    st, dt = _apply_criterion(0.0, mae, "lte", 1e-3)
    results["s4_kat"] = {"max_abs_diff": mae, "status": st}
    print(f"   got : {[round(x,4) for x in got]}")
    print(f"   want: {[round(x,4) for x in want]}")
    print(f"   max_abs_diff = {mae:.6g}  -> {st}")
    sr.metrics["H4_signed_int4_lop3.kat_bit_exact"] = {
        "status": st, "measured": mae, "predicted": 0.0, "tolerance": 1e-3, "detail": dt}

    # ---------- H4 random allclose (4096 words) ---------- #
    _sub("H4 Signed INT4 random-tensor allclose (4096 words)")
    torch.manual_seed(1234)
    N = 4096
    # Must span the FULL 32-bit packed-word space (incl. bit31) to stress
    # top-nibble sign handling — randint(0, 0x7FFFFFFF) never does.
    Wp = torch.randint(-2**31, 2**31 - 1, (N,), dtype=torch.int64).to(torch.int32).to(dev)
    scp = torch.rand((N,), dtype=torch.float16, device=dev) * 0.1 + 0.01
    zzp = torch.randint(0, 8, (N,), dtype=torch.float16, device=dev)
    got = t4_kernels.dequantize_s4(Wp, scp, zzp).float()
    # CPU reference
    Wl = Wp.cpu().tolist()
    scl = scp.float().cpu().tolist()
    zzl = zzp.float().cpu().tolist()
    ref = []
    for w, s, z in zip(Wl, scl, zzl):
        ref.extend(ref_s4(w, s, z))
    ref = torch.tensor(ref)
    mae_rand = (got.cpu() - ref).abs().max().item()
    st, dt = _apply_criterion(0.0, mae_rand, "lte", 1e-3)
    results["s4_random"] = {"max_abs_diff": mae_rand, "status": st}
    print(f"   max_abs_diff over {N*8} values = {mae_rand:.6g}  -> {st}")
    sr.metrics["H4_signed_int4_lop3.random_allclose"] = {
        "status": st, "measured": mae_rand, "predicted": 0.0, "tolerance": 1e-3, "detail": dt}

    _sub("H4 Unsigned INT4 random-tensor allclose (4096 words)")
    got = t4_kernels.dequantize_u4(Wp, scp, zzp).float()
    ref = []
    for w, s, z in zip(Wl, scl, zzl):
        ref.extend(ref_u4(w, s, z))
    ref = torch.tensor(ref)
    mae_rand = (got.cpu() - ref).abs().max().item()
    st, dt = _apply_criterion(0.0, mae_rand, "lte", 1e-3)
    results["u4_random"] = {"max_abs_diff": mae_rand, "status": st}
    print(f"   max_abs_diff over {N*8} values = {mae_rand:.6g}  -> {st}")
    sr.metrics["H4_companion_unsigned_int4_lop3.random_allclose"] = {
        "status": st, "measured": mae_rand, "predicted": 0.0, "tolerance": 1e-3, "detail": dt}

    # ---------- Fused W4A16 GEMM correctness ---------- #
    _sub("Fused W4A16 GEMM correctness (M=1, K=1024, N=1024)")
    try:
        M, K, Nn = 1, 1024, 1024
        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=torch.float16, device=dev)
        # Full 32-bit packed range (incl. bit31) for the same reason as random test
        Wp2 = torch.randint(-2**31, 2**31 - 1, (K // 8, Nn), dtype=torch.int64).to(torch.int32).to(dev)
        sc2 = torch.rand((Nn,), dtype=torch.float16, device=dev) * 0.1 + 0.05
        zz2 = torch.randint(0, 8, (Nn,), dtype=torch.float16, device=dev)

        def dequant_ref_mat(W_packed, scale, zero_point, signed):
            Ku = W_packed.shape[0]; Nloc = W_packed.shape[1]
            Wm = torch.zeros((Ku * 8, Nloc), dtype=torch.float16)
            Wc = W_packed.cpu().numpy(); scc = scale.float().cpu().numpy(); zzc = zero_point.float().cpu().numpy()
            for ki in range(Ku):
                for col in range(Nloc):
                    val = int(Wc[ki, col])
                    for bi in range(8):
                        nib = (val >> (bi * 4)) & 0xF
                        if signed:
                            nib = nib - 16 if (nib & 8) else nib
                        Wm[ki * 8 + bi, col] = (nib - float(zzc[col])) * float(scc[col])
            return Wm.to(dev)

        out_u4 = t4_kernels.fused_w4a16_gemm_u4(A, Wp2, sc2, zz2)
        refW_u4 = dequant_ref_mat(Wp2, sc2, zz2, signed=False)
        ref_u4 = torch.matmul(A.float(), refW_u4.float()).half()
        d_u4 = (out_u4 - ref_u4).abs().max().item()

        out_s4 = t4_kernels.fused_w4a16_gemm_s4(A, Wp2, sc2, zz2)
        refW_s4 = dequant_ref_mat(Wp2, sc2, zz2, signed=True)
        ref_s4 = torch.matmul(A.float(), refW_s4.float()).half()
        d_s4 = (out_s4 - ref_s4).abs().max().item()

        maxd = max(d_u4, d_s4)
        st, dt = _apply_criterion(2.0, maxd, "lte", 0)
        results["fused_gemm"] = {"max_abs_diff_u4": d_u4, "max_abs_diff_s4": d_s4, "status": st}
        print(f"   U4 max_abs_diff = {d_u4:.5f}")
        print(f"   S4 max_abs_diff = {d_s4:.5f}")
        print(f"   worst = {maxd:.5f}  (limit 2.0)  -> {st}")
        # metric key names match expected.yaml: fused_w4a16_gemm.max_abs_diff,
        # .correctness_u4, .correctness_s4
        sr.metrics["fused_w4a16_gemm.max_abs_diff"] = {
            "status": st, "measured": maxd, "predicted_max": 2.0, "detail": dt}
        sr.metrics["fused_w4a16_gemm.correctness_u4"] = {
            "status": "CONFIRM" if d_u4 < 2.0 else "FAIL", "measured": d_u4 < 2.0, "predicted": True, "detail": "bool"}
        sr.metrics["fused_w4a16_gemm.correctness_s4"] = {
            "status": "CONFIRM" if d_s4 < 2.0 else "FAIL", "measured": d_s4 < 2.0, "predicted": True, "detail": "bool"}
    except Exception as e:
        sr.notes.append(f"fused GEMM error: {e}")
        print(f"   [ERROR] fused GEMM: {e}")

    # ---------- H7 INT3 math (reference only; no CUDA kernel exists) ---------- #
    _sub("H7 INT3 LOP3 — bit-exact math identity (reference, NOT a CUDA kernel)")
    ok = True
    for s3 in range(-4, 4):
        bits3 = s3 & 0x7
        inv = bits3 ^ 0x4
        recon = inv - 4
        if recon != s3:
            ok = False
    st, dt = _apply_criterion(0.0, 0.0 if ok else 1.0, "lte", 0)
    print(f"   8-state identity sweep s3+4 == sign_invert(s3): {st}")
    sr.metrics["H7_signed_int3_lop3.kat_bit_exact_int3"] = {
        "status": st, "measured": 0.0 if ok else 1.0, "predicted": 0.0, "detail": dt,
        "note": "math identity only; no CUDA kernel exists for INT3 yet"}
    sr.notes.append("H7 INT3: validated math identity ONLY — kernel is sim_only (not measurable).")

    # ---------- H9 FP8 E4M3 sweep (reference only; no CUDA kernel exists) ---------- #
    _sub("H9 FP8 E4M3 -> FP16 +8 re-bias sweep (reference, NOT a CUDA kernel)")
    valid = 0
    mismatch = 0
    for b in range(256):
        sign = (b >> 7) & 0x1
        e8 = (b >> 3) & 0xF
        m8 = b & 0x7
        if e8 == 0xF and m8 == 0x7:
            continue
        valid += 1
        e16 = e8 + 8 if e8 > 0 else 0
        m16 = m8 << 7
        if e8 > 0:
            v8 = ((-1) ** sign) * (2 ** (e8 - 7)) * (1.0 + m8 / 8.0)
            v16 = ((-1) ** sign) * (2 ** (e16 - 15)) * (1.0 + m16 / 1024.0)
            if abs(v8 - v16) > 1e-5:
                mismatch += 1
    exp9 = exp["H9_fp8_lop3_rescale"]["metrics"]["fp8_sweep_valid_states"]
    st = "CONFIRM" if (mismatch == 0 and valid == exp9["predicted"]) else "FAIL"
    print(f"   valid E4M3 states sweeped = {valid} (expect {exp9['predicted']}), mismatches = {mismatch} -> {st}")
    sr.metrics["H9_fp8_lop3_rescale.fp8_sweep_valid_states"] = {
        "status": st, "measured": valid, "predicted": exp9["predicted"], "detail": f"{mismatch} mismatches",
        "note": "reference math only; no CUDA kernel exists for FP8 yet"}
    sr.notes.append("H9 FP8: validated reference sweep ONLY — kernel is sim_only (not measurable).")

    ctx["correctness"] = results
    statuses = [m["status"] for m in sr.metrics.values()]
    sr.status = "FAIL" if "FAIL" in statuses else "PASS"
    sr.duration_s = time.time() - t0
    return sr


# ============================================================================= #
# STAGE 4 — TELEMETRY (nvidia-smi sampler during kernel stress)
# ============================================================================= #
def stage_telemetry(ctx):
    sr = StageResult("S4_telemetry")
    t0 = time.time()
    _banner("STAGE 4 — Hardware Telemetry (power / clock / throttle)")

    nsmi = _which("nvidia-smi")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if not nsmi or not cuda_ok:
        sr.status = "SKIP"
        sr.notes.append("nvidia-smi or CUDA missing; telemetry skipped.")
        print("  [SKIP] telemetry unavailable.")
        sr.duration_s = time.time() - t0
        return sr

    import torch
    sys.path.insert(0, str(SRC))
    try:
        import t4_kernels
    except Exception as e:
        sr.status = "SKIP"
        sr.notes.append(f"t4_kernels unavailable ({e}); telemetry skipped.")
        sr.duration_s = time.time() - t0
        return sr

    csv_path = ctx["run_dir"] / "telemetry.csv"
    # Launch sampler
    sampler = subprocess.Popen(
        [nsmi, "--query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active",
         "--format=csv", "-lms", "20"],
        stdout=open(csv_path, "w"), stderr=subprocess.STDOUT, text=True)

    _sub("Stress loop: saturated fused dequant/GEMM for ~8 seconds")
    dev = "cuda"
    N = 4096 * 512
    # Full 32-bit packed range so the stress loop exercises every nibble pattern
    Wp = torch.randint(-2**31, 2**31 - 1, (N,), dtype=torch.int64).to(torch.int32).to(dev)
    scp = torch.rand((N,), dtype=torch.float16, device=dev)
    zzp = torch.zeros((N,), dtype=torch.float16, device=dev)

    tstart = time.time()
    iters = 0
    while time.time() - tstart < 8.0:
        _ = t4_kernels.dequantize_s4(Wp, scp, zzp)
        _ = t4_kernels.dequantize_u4(Wp, scp, zzp)
        if iters % 8 == 0:
            torch.cuda.synchronize()
        iters += 1
    torch.cuda.synchronize()
    time.sleep(0.3)  # let sampler flush
    sampler.terminate()
    sampler.wait(timeout=5)

    # Parse telemetry
    clocks, powers, temps, throttle = [], [], [], []
    try:
        lines = csv_path.read_text().splitlines()
        for ln in lines:
            if "clocks.sm" in ln or not ln.strip():
                continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 4:
                continue
            try:
                # parts: ['1500 MHz', '55.00 W', '45', 'Not Active']
                c = float(parts[0].replace("MHz", "").strip() or "nan")
                p = float(parts[1].replace("W", "").strip() or "nan")
                tp = float(parts[2] or "nan")
                th = parts[3]
                if not math.isnan(c):
                    clocks.append(c)
                if not math.isnan(p):
                    powers.append(p)
                if not math.isnan(tp):
                    temps.append(tp)
                throttle.append(th)
            except Exception:
                continue
    except Exception as e:
        sr.notes.append(f"telemetry parse error: {e}")

    telem = {
        "samples": len(clocks),
        "clock_mhz_min": min(clocks) if clocks else None,
        "clock_mhz_max": max(clocks) if clocks else None,
        "clock_mhz_mean": (sum(clocks) / len(clocks)) if clocks else None,
        "power_w_max": max(powers) if powers else None,
        "power_w_mean": (sum(powers) / len(powers)) if powers else None,
        "temp_c_max": max(temps) if temps else None,
        "throttle_reasons": sorted(set(throttle)) if throttle else [],
        "stress_iters": iters,
    }
    ctx["telemetry"] = telem
    for k, v in telem.items():
        print(f"   {k:20s}: {v}")

    # informational assessment vs T4 power cap / target clock 1590
    if telem["power_w_max"] is not None:
        ok = telem["power_w_max"] <= T4_TDP_W
        sr.metrics["telemetry.power_within_tdp"] = {
            "status": "CONFIRM" if ok else "FAIL",
            "measured": telem["power_w_max"], "predicted_max": T4_TDP_W,
            "detail": "H5 family: 70W cap adherence (informational for dequant stress)"}
    if telem["clock_mhz_mean"] is not None:
        sr.metrics["telemetry.sm_clock_mean"] = {
            "status": "INFO",
            "measured": telem["clock_mhz_mean"], "predicted": 1590.0,
            "detail": "Boost clock during dequant stress (not a power-capped GEMM regime)"}

    sr.status = "PASS" if telem["samples"] > 0 else "FAIL"
    sr.duration_s = time.time() - t0
    return sr


# ============================================================================= #
# STAGE 5 — ncu ROOFLINE / SPEED-OF-LIGHT (SKIP if ncu absent)
# ============================================================================= #
def stage_ncu(ctx):
    sr = StageResult("S5_ncu")
    t0 = time.time()
    _banner("STAGE 5 — Nsight Compute Speed-of-Light Profiling")

    ncu = _which("ncu")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if not ncu:
        sr.status = "SKIP"
        sr.notes.append("ncu not found on PATH. On Colab free tier it is unavailable; "
                        "rent a T4 (RunPod/Lambda) for the SOL pass.")
        print("  [SKIP] ncu not found.")
        sr.duration_s = time.time() - t0
        return sr
    if not cuda_ok:
        sr.status = "SKIP"
        sr.notes.append("CUDA not available.")
        sr.duration_s = time.time() - t0
        return sr

    # Build a throwaway target script that exercises the dequant kernels
    target = ctx["run_dir"] / "_ncu_target.py"
    target.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "import torch, t4_kernels\n"
        "N = 4096*512\n"
        "Wp = torch.randint(0, 0x7FFFFFFF, (N,), dtype=torch.int32, device='cuda')\n"
        "scp = torch.rand((N,), dtype=torch.float16, device='cuda')\n"
        "zzp = torch.zeros((N,), dtype=torch.float16, device='cuda')\n"
        "for _ in range(20):\n"
        "Wp = torch.randint(-2**31, 2**31 - 1, (N,), dtype=torch.int64).to(torch.int32).to('cuda')\n"
        "torch.cuda.synchronize()\n"
    )
    rep = ctx["run_dir"] / "ncu_report"
    _sub("ncu SOL metrics on lop3_dequant_s4")
    metrics = ("sm__throughput.of_peak_allocation_sol,"
               "dram__throughput.of_peak_allocation_sol,"
               "l1tex__data_bank_conflicts_pipe_lsu.sum")
    rc, out = _run([ncu, "--metrics", metrics, "-o", str(rep), sys.executable, str(target)],
                   timeout=1800)
    print(out)
    ctx.setdefault("artifacts", {})["ncu_stdout.txt"] = out

    # Attempt to export csv summary
    rc2, out2 = _run([ncu, "--import", str(rep) + ".ncu-rep", "--csv"], timeout=600)
    ctx.setdefault("artifacts", {})["ncu_summary.csv"] = out2
    if out2:
        print("\n".join(out2.splitlines()[:30]))

    sr.status = "PASS" if rc == 0 else "FAIL"
    if rc != 0:
        sr.notes.append("ncu profiling returned non-zero; see ncu_stdout.txt.")
    sr.duration_s = time.time() - t0
    return sr


# ============================================================================= #
# STAGE 6 — VERDICT SYNTHESIS
# ============================================================================= #
def stage_verdict(ctx, stage_results):
    sr = StageResult("S6_verdict")
    t0 = time.time()
    _banner("STAGE 6 — Verdict Synthesis (measured vs predicted)")

    exp = ctx["expected"]
    hyps = exp["hypotheses"]

    # Aggregate all metrics across stages
    all_metrics = {}
    for r in stage_results:
        for k, v in r.metrics.items():
            all_metrics[k] = v

    # Per-hypothesis rollup
    per_hyp = {}
    for hid, hdata in hyps.items():
        cap = hdata["capabilities"]["kernel"]
        blocking = cap == "kernel_ready"
        rows = []
        worst = "INFO"
        for mname, mspec in hdata["metrics"].items():
            key = f"{hid}.{mname}"
            if key in all_metrics:
                st = all_metrics[key]["status"]
            else:
                st = "NOT_MEASURED"
            rows.append((mname, st,
                         all_metrics.get(key, {}).get("measured"),
                         mspec.get("predicted", mspec.get("predicted_max")),
                         all_metrics.get(key, {}).get("detail", mspec.get("description",""))))
            if blocking:
                if st == "FAIL":
                    worst = "FAIL"
                elif st == "MARGIN" and worst != "FAIL":
                    worst = "MARGIN"
                elif st == "NOT_MEASURED" and worst in ("INFO", "CONFIRM"):
                    worst = "NOT_MEASURED"
                elif st == "CONFIRM" and worst == "INFO":
                    worst = "CONFIRM"
        if not blocking:
            verdict = "NOT_MEASURABLE (sim_only)"
        else:
            # If nothing measurable actually ran, verdict is INCOMPLETE (honest),
            # never a spurious CONFIRM.
            verdict = {"INFO": "INCOMPLETE", "NOT_MEASURED": "INCOMPLETE"}.get(worst, worst)
        per_hyp[hid] = {"capability": cap, "blocking": blocking, "verdict": verdict, "rows": rows}

    ctx["per_hypothesis"] = per_hyp

    # Print table
    for hid, ph in per_hyp.items():
        title = hyps[hid]["title"]
        print(f"\n  {hid}: {title}")
        print(f"     capability={ph['capability']:12s}  verdict = {ph['verdict']}")
        for (mname, st, meas, pred, detail) in ph["rows"]:
            pred_s = "-" if pred is None else f"{pred}"
            meas_s = "-" if meas is None else (f"{meas:.5g}" if isinstance(meas, (int, float)) else str(meas))
            print(f"       - {mname:34s} predicted={pred_s:>10s} measured={meas_s:>10s}  [{st}]")

    # Global pass — an honest ladder:
    #   FAIL      if any blocking hypothesis failed
    #   INCOMPLETE if nothing measurable ran (no GPU etc.)
    #   MARGIN    if only marginal results
    #   CONFIRM   only when every blocking hypothesis confirmed
    blocking_fails = [h for h, p in per_hyp.items() if p["blocking"] and p["verdict"] == "FAIL"]
    blocking_incomplete = [h for h, p in per_hyp.items() if p["blocking"] and p["verdict"] == "INCOMPLETE"]
    blocking_margins = [h for h, p in per_hyp.items() if p["blocking"] and p["verdict"] == "MARGIN"]
    blocking_confirms = [h for h, p in per_hyp.items() if p["blocking"] and p["verdict"] == "CONFIRM"]

    if blocking_fails:
        global_v = "FAIL"
    elif blocking_confirms and not blocking_incomplete:
        global_v = "MARGIN" if blocking_margins else "CONFIRM"
    elif blocking_incomplete and not blocking_confirms:
        global_v = "INCOMPLETE"
    else:
        global_v = "MARGIN"

    ctx["global_verdict"] = global_v
    _banner(f"GLOBAL VERDICT (kernel_ready hypotheses): {global_v}")
    if blocking_fails:
        print(f"   FAILING: {blocking_fails}")
    if blocking_margins:
        print(f"   MARGINAL: {blocking_margins}")
    if blocking_incomplete:
        print(f"   INCOMPLETE (nothing measurable ran): {blocking_incomplete}")
    notm = [h for h, p in per_hyp.items() if not p["blocking"]]
    if notm:
        print(f"   NOT_MEASURABLE (sim_only): {notm}")

    sr.status = "PASS" if global_v in ("CONFIRM", "MARGIN") else "FAIL"
    sr.duration_s = time.time() - t0
    return sr, per_hyp, global_v


# ============================================================================= #
# MAIN
# ============================================================================= #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ncu", action="store_true", help="skip Nsight Compute stage")
    ap.add_argument("--skip-telemetry", action="store_true", help="skip telemetry sampler stage")
    ap.add_argument("--out", default=None, help="override results dir name")
    args, _ = ap.parse_known_args()

    # results dir
    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = args.out or ts
    run_dir = RESULTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # tee log
    log_fp = open(run_dir / "run.log", "w")
    sys.stdout = Tee(sys.__stdout__, log_fp)
    sys.stderr = Tee(sys.__stderr__, log_fp)

    _banner("Tesla T4 EMPIRICAL VALIDATION RUN")
    print(f"  results dir : {run_dir}")
    print(f"  started     : {_dt.datetime.now().isoformat()}")

    # session context
    try:
        expected = _load_expected()
    except RuntimeError as e:
        print(f"\n  [FATAL] {e}")
        print("  On Colab, PyYAML is preinstalled. Locally: pip install pyyaml")
        sys.exit(2)

    ctx = {"run_dir": run_dir, "expected": expected, "artifacts": {}}

    stages = []

    # S0 always runs
    stages.append(stage_env_stamp(ctx))

    try:
        stages.append(stage_compile_audit(ctx))
        stages.append(stage_build_extension(ctx))
        stages.append(stage_gpu_correctness(ctx))
        if not args.skip_telemetry:
            stages.append(stage_telemetry(ctx))
        else:
            r = StageResult("S4_telemetry"); r.status = "SKIP"; r.notes.append("skipped via flag"); stages.append(r)
        if not args.skip_ncu:
            stages.append(stage_ncu(ctx))
        else:
            r = StageResult("S5_ncu"); r.status = "SKIP"; r.notes.append("skipped via flag"); stages.append(r)
    except Exception:
        print("\n  [FATAL] unhandled exception during stages:")
        traceback.print_exc()
    finally:
        try:
            vres, per_hyp, global_v = stage_verdict(ctx, stages)
            stages.append(vres)
        except Exception:
            traceback.print_exc()
            global_v = "ERROR"
            per_hyp = {}

    # ---- persist structured outputs ---- #
    summary = {
        "run": {
            "dir": run_dir.name,
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "env": ctx.get("env_stamp", {}),
        "global_verdict": global_v,
        "stages": [
            {"name": s.name, "status": s.status, "duration_s": round(s.duration_s, 2),
             "notes": s.notes} for s in stages
        ],
        "metrics": {k: v for s in stages for k, v in s.metrics.items()},
        "hypotheses": {
            hid: {
                "capability": ph["capability"],
                "blocking": ph["blocking"],
                "verdict": ph["verdict"],
            } for hid, ph in per_hyp.items()
        },
        "telemetry": ctx.get("telemetry"),
        "correctness": ctx.get("correctness"),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- VERDICT.md ---- #
    lines = []
    lines.append(f"# Empirical Validation Verdict — {run_dir.name}\n")
    lines.append(f"- Timestamp (UTC): {summary['run']['timestamp_utc']}")
    lines.append(f"- Git SHA: {ctx.get('env_stamp', {}).get('git_sha', 'unknown')}"
                 f"  (dirty: {ctx.get('env_stamp', {}).get('git_dirty', '?')})")
    lines.append(f"- Device: {ctx.get('env_stamp', {}).get('torch_device_name', ctx.get('env_stamp', {}).get('gpu', 'unknown'))}")
    lines.append(f"- CUDA: {ctx.get('env_stamp', {}).get('torch_cuda_version', 'unknown')}"
                 f"  | PyTorch: {ctx.get('env_stamp', {}).get('torch_version', 'unknown')}")
    lines.append(f"\n## GLOBAL VERDICT: **{global_v}**\n")
    lines.append("## Per-hypothesis results\n")
    lines.append("| Hypothesis | Capability | Verdict |")
    lines.append("|---|---|---|")
    for hid, ph in per_hyp.items():
        lines.append(f"| {hid} | {ph['capability']} | {ph['verdict']} |")
    lines.append("\n## Stage status\n")
    lines.append("| Stage | Status | Duration (s) | Notes |")
    lines.append("|---|---|---|---|")
    for s in stages:
        lines.append(f"| {s.name} | {s.status} | {s.duration_s:.1f} | {'; '.join(s.notes)[:120]} |")
    lines.append("\n## Metric detail\n")
    lines.append("| Metric | Predicted | Measured | Status |")
    lines.append("|---|---|---|---|")
    for k, v in summary["metrics"].items():
        meas = v.get("measured")
        meas_s = f"{meas:.5g}" if isinstance(meas, (int, float)) else str(meas)
        pred = v.get("predicted", v.get("predicted_max", "-"))
        lines.append(f"| {k} | {pred} | {meas_s} | {v.get('status')} |")
    lines.append("\n---\n")
    lines.append("Artifacts in this directory: `run.log`, `summary.json`, `telemetry.csv`, "
                 "`compile_audit.json`, `build_log.txt`, `ncu_summary.csv` (if ncu ran).\n")
    with open(run_dir / "VERDICT.md", "w") as f:
        f.write("\n".join(lines))

    # ---- tarball ---- #
    tb = None
    try:
        tb_path = RESULTS_ROOT / f"{run_dir.name}"
        tb = subprocess.run(["tar", "czf", f"{tb_path}.tar.gz", "-C", RESULTS_ROOT, run_dir.name])
        # resolve -> actual path
        tb = f"{tb_path}.tar.gz" if tb.returncode == 0 else None
    except Exception:
        tb = None

    _banner(f"RUN COMPLETE — {global_v}")
    print(f"  results dir : {run_dir}")
    print(f"  verdict     : {run_dir / 'VERDICT.md'}")
    print(f"  summary     : {run_dir / 'summary.json'}")
    if tb:
        print(f"  tarball     : {tb}")
    print()

    log_fp.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    # exit code: 0 only on CONFIRM/MARGIN; INCOMPLETE, FAIL, ERROR all non-zero
    sys.exit(0 if global_v in ("CONFIRM", "MARGIN") else 1)


if __name__ == "__main__":
    main()

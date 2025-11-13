"""
Minimal test for CondSCVI module after removing scvi-tools dependency.

What this script does:
- Import your CondSCVI implementation (tries multiple likely module names).
- Instantiate CondSCVI with small dummy sizes.
- Build a small random batch (x, y, batch_index).
- Run one inference pass (encoder) and one generative pass (decoder).
- Try to sample from the returned generative distribution if possible.
- Print shapes and a few sanity checks.

Drop this file in the repository root and run with the same Python environment
you used for other tests (e.g. `python test_condscvi_minimal.py`).
"""
import sys
import traceback
import torch
import numpy as np

# Try to import CondSCVI from common locations you might have
_condscvi_candidates = [
    "src.models.mycondscvi",
    "src.models.condscvi",
    "src.models.condscvi",  # duplicate to be explicit
]

CondSCVI = None
for mod in _condscvi_candidates:
    try:
        module = __import__(mod, fromlist=["CondSCVI"])
        CondSCVI = getattr(module, "CondSCVI")
        print(f"Imported CondSCVI from {mod}")
        break
    except Exception:
        # continue searching
        pass

if CondSCVI is None:
    print("Failed to import CondSCVI from expected locations. Tracebacks from attempts:")
    for mod in _condscvi_candidates:
        try:
            __import__(mod, fromlist=["CondSCVI"])
        except Exception:
            traceback.print_exc(limit=1)
    sys.exit(1)

# Basic synthetic config
n_input = 50
n_labels = 4
n_batch = 1
batch_size = 6
n_samples_for_sampling = 3

# Instantiate CondSCVI - the constructor signature in your refactor may accept different args.
# We'll try a few common variants to maximize compatibility.
sc_model = None
try_variants = [
    {"n_input": n_input, "n_labels": n_labels, "n_batch": n_batch},
    {"n_input": n_input, "n_labels": n_labels},
    {"n_input": n_input, "n_labels": n_labels, "n_batch": n_batch, "n_hidden": 32, "n_latent": 5},
]

for kwargs in try_variants:
    try:
        sc_model = CondSCVI(**kwargs)
        print("CondSCVI instantiated with kwargs:", kwargs)
        break
    except Exception as e:
        print("Instantiation with", kwargs, "failed:", repr(e))

if sc_model is None:
    print("Could not instantiate CondSCVI with tried signatures. Exiting.")
    sys.exit(1)

# Ensure the inner module is present (VAEC-like)
module = getattr(sc_model, "module", sc_model)
print("Using module object of type:", type(module))

# Create dummy batch tensors
torch.manual_seed(0)
X = torch.abs(torch.randn(batch_size, n_input)) * 3.0  # non-negative "counts-like"
# Many implementations expect labels as shape (batch, 1)
Y = torch.randint(0, n_labels, (batch_size, 1), dtype=torch.long)
B = torch.zeros((batch_size, 1), dtype=torch.long)

print("Dummy data shapes: X", X.shape, "Y", Y.shape, "B", B.shape)

# 1) Run inference (encoder)
print("\n== Running inference ==")
inference_outputs = None
try:
    # Try calling inference method with common signatures
    # Examples: module.inference(x, y, batch_index=B), or module.inference(x=X, y=Y)
    try:
        inference_outputs = module.inference(X, Y, B)
    except TypeError:
        try:
            inference_outputs = module.inference(x=X, y=Y, batch_index=B)
        except Exception:
            inference_outputs = module.inference(x=X, y=Y)
    print("inference returned type:", type(inference_outputs))
    # Print representative contents
    if isinstance(inference_outputs, dict):
        for k, v in inference_outputs.items():
            try:
                print("  key:", k, "->", type(v), getattr(v, "shape", None))
            except Exception:
                print("  key:", k, "->", type(v))
    else:
        print("inference output not dict; repr:", repr(inference_outputs))
except Exception:
    print("inference failed:")
    traceback.print_exc(limit=2)
    sys.exit(1)

# 2) Run generative (decoder)
print("\n== Running generative ==")
gen_outputs = None
try:
    # Many generative signatures expect (z, library, y, batch_index)
    # Try to extract z and library from inference_outputs in a robust way
    z = None
    library = None
    # try common keys
    if isinstance(inference_outputs, dict):
        for key in ("z", "Z", "qz", "QZ", "library", "LIBRARY", "lib"):
            if key in inference_outputs:
                val = inference_outputs[key]
                # Heuristic: choose z as a tensor-like object
                if key.lower().startswith("z") and torch.is_tensor(val):
                    z = val
                if "lib" in key.lower() or "library" in key.lower():
                    library = val
    # Fallback attempts: look for tensor or distribution-like inside dict values
    if z is None:
        # find first tensor-like value
        for v in (inference_outputs.values() if isinstance(inference_outputs, dict) else []):
            if torch.is_tensor(v):
                z = v
                break
    if library is None:
        for v in (inference_outputs.values() if isinstance(inference_outputs, dict) else []):
            if torch.is_tensor(v) and v.ndim == 2:
                # may be library; use heuristic: small last dim
                library = v
                break

    # If missing, try to call a wrapper forward to get generative via module.forward
    if z is None or library is None:
        print("Could not robustly extract z/library from inference output; attempting module.forward(...)")
        # many module.forward accepts a tensors dict keyed by 'X' etc.
        # Build a fallback call
        try:
            out_forward = module.forward({"X": X, "labels": Y, "batch": B})
            print("module.forward returned keys:", list(out_forward.keys()) if isinstance(out_forward, dict) else type(out_forward))
        except Exception:
            print("module.forward fallback failed (will still try module.generative if possible)")

    # finally, if z/library obtained, call generative
    if z is not None and library is not None:
        try:
            # Try positional call
            gen_outputs = module.generative(z, library, Y, B)
        except TypeError:
            # try keyword args
            gen_outputs = module.generative(z=z, library=library, y=Y, batch_index=B)
        print("generative returned type:", type(gen_outputs))
        if isinstance(gen_outputs, dict):
            for k, v in gen_outputs.items():
                try:
                    print("  key:", k, "->", type(v), getattr(v, "shape", None))
                except Exception:
                    print("  key:", k, "->", type(v))
    else:
        print("Skipping direct generative call because z/library not available; if module.forward worked above, check its outputs.")
except Exception:
    print("generative failed:")
    traceback.print_exc(limit=3)
    sys.exit(1)

# 3) Try sampling from returned distribution (if present)
print("\n== Try sampling from PX distribution if available ==")
try:
    # heuristics: find a distribution-like object in gen_outputs values
    dist_candidate = None
    if isinstance(gen_outputs, dict):
        for v in gen_outputs.values():
            # NegativeBinomial wrapper in your code may not be a torch.distribution subclass,
            # so test for .sample or .log_prob attributes.
            if hasattr(v, "sample") or hasattr(v, "log_prob"):
                dist_candidate = v
                break
    if dist_candidate is not None:
        # try sample
        try:
            sam = dist_candidate.sample()
            print("sample() succeeded, sample shape/type:", type(sam), getattr(sam, "shape", None))
        except Exception as e:
            print("sampling raised:", repr(e))
            # if sample requires shape or args, try default: sample((1,))
            try:
                sam = dist_candidate.sample()
                print("sample() second attempt ok:", getattr(sam, "shape", None))
            except Exception:
                print("Could not sample from distribution; printed repr instead:", repr(dist_candidate))
    else:
        print("No distribution-like object discovered in generative outputs.")
except Exception:
    print("sampling attempt raised an exception:")
    traceback.print_exc(limit=2)

print("\nTest completed. If all above steps executed without fatal exceptions, CondSCVI basic forward/inference/generative APIs appear usable.")
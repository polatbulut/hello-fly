#!/usr/bin/env python3
"""What worlds can the fly actually live in? Enumerate FlyGym 1.2.1 arenas."""
import inspect
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import flygym.arena as A

names = [n for n in dir(A) if not n.startswith("_") and isinstance(getattr(A, n), type)]
print(f"flygym.arena classes ({len(names)}):\n")
for n in sorted(names):
    cls = getattr(A, n)
    try:
        sig = str(inspect.signature(cls.__init__)).replace("self, ", "")
        sig = (sig[:150] + " ...") if len(sig) > 150 else sig
    except Exception:
        sig = "(?)"
    doc = (cls.__doc__ or "").strip().split("\n")[0][:90]
    print(f"  {n}")
    print(f"      {doc}")
    print(f"      {sig}\n")

print("\nwhich of these carry an odour source?")
for n in sorted(names):
    cls = getattr(A, n)
    src = ""
    try:
        src = inspect.getsource(cls)
    except Exception:
        pass
    has = any(k in src for k in ("odor", "olfact"))
    if has:
        print(f"  {n}: YES")

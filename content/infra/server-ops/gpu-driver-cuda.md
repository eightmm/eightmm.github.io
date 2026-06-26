---
title: GPU Driver and CUDA Debugging
tags:
  - infra
  - server-ops
  - gpu
---

# GPU Driver and CUDA Debugging

Most "GPU not working" failures on a shared machine come from a mismatch between the NVIDIA driver, the CUDA runtime a framework was built against, and the toolkit on the host. The driver caps the maximum CUDA version; a newer runtime than the driver supports will fail to initialize.

The compatibility chain is:

$$
\text{kernel driver}
\rightarrow
\text{CUDA runtime}
\rightarrow
\text{framework build}
\rightarrow
\text{application kernel}
$$

Debug from the host layer upward instead of changing project dependencies first.

## Practical Checks

- Confirm the driver sees the GPUs before blaming the framework (`nvidia-smi`).
- Check the framework's bundled CUDA runtime, not just the system toolkit — they can differ.
- Treat the driver's reported CUDA version as a ceiling for runtime compatibility.
- After a driver upgrade, reboot or reload the kernel module before testing.
- Reproduce in a clean environment to separate a host problem from a project one.
- Record the error class and compatibility layer, not private node names or project paths.
- Link repeat failures to an incident note when the cause affects multiple users.

## Related

- [[infra/gpu/index|GPU]]
- [[infra/server-ops/index|Server operations]]
- [[infra/server-ops/incident-response|Incident response]]
- [[infra/hpc/slurm|Slurm]]

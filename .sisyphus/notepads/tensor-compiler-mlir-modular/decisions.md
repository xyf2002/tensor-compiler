## [Initial] Architectural Decisions

### Dialect Name: "ten" over "tensor"
- MLIR has a built-in `tensor` dialect that conflicts
- All ops use `ten.` prefix in MLIR syntax

### Fusion Strategy: TensorOps-level macro fusion
- Current implementation fuses matmul+relu at TensorOps level before lowering to Linalg
- Fusion marks matmul with `fused_relu` attribute and eliminates the relu op
- Alternative (Linalg-level fusion) would require operating on linalg.generic after lowering

### CPU Lowering: Compose existing passes
- Do NOT reimplement standard lowering
- Chain: linalg→loops → scf→cf → memref→llvm → func→llvm → reconcile-unrealized-casts

### GPU Pass: Compose existing passes  
- `convert-linalg-to-loops` → `convert-parallel-loops-to-gpu` → `convert-gpu-to-nvvm`/`convert-gpu-to-rocdl`

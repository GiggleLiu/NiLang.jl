# Reversible programming on GPU

Special Notes:
* please use `@invcheckoff` to close all reversibility check in a kernel.
* be careful about the race condition when automatic differentiating a CUDA program.

## Suggested reading order
1. `swap_gate.jl` simulates a quantum swap gate, its reversible counter part is here
http://tutorials.yaoquantum.org/dev/generated/developer-guide/2.cuda-acceleration/
2. `rotation_gate.jl` simulates a quantum rotation gate, obtaining the gradients on rotation angle would have race condition.

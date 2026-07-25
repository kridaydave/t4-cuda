# Protocol for H6: Fused Register-Level Backward GEMM with inline AdamW weight updates

## Hypothesis
Fused Register-Level Backward GEMM with inline AdamW weight updates eliminates 21.4% of GDDR6 DRAM traffic during Llama 3 8B QLoRA fine-tuning on 16GB T4 VRAM.

## Experimental Setup
- Model: Llama 3 8B
- Technique: QLoRA fine-tuning
- Hardware: 16GB T4 GPU
- Memory optimization: Fusing the backward pass (gradient computation for LoRA weights) directly with the AdamW optimizer step at the register level, avoiding the intermediate write and read of gradients to/from HBM (GDDR6 DRAM).

## Protocol
1. Calculate the theoretical GDDR6 memory traffic for standard QLoRA backward pass + optimizer step.
2. Calculate the theoretical memory traffic for the fused approach.
3. Compare the total DRAM traffic and verify the percentage of traffic eliminated.
4. Log findings to research-state.yaml and findings.md.

# Protocol: Persistent 40-block wave streaming and SMEM swizzle

## Hypothesis
Persistent 40-block wave streaming combined with 16-byte aligned LDS.U128 vector loads achieves 100% bank conflict elimination and 91.2% memory bandwidth efficiency on Turing SM 7.5.

## Methodology
1. Implement an offline bank-conflict and tile efficiency calculator in `src/verify_smem_swizzle.py`.
2. Model a warp (32 threads) performing LDS.U128 (16-byte) loads.
3. Apply SMEM swizzling to eliminate bank conflicts.
4. Calculate bandwidth efficiency based on persistent wave streaming limits.

## Expected Outcome
The simulation should report 0 bank conflicts and exactly 91.2% bandwidth efficiency.

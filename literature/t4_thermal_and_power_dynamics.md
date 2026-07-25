# Analytical Power & Thermal Modeling for 70W Passively Cooled Tesla T4 GPUs

## 1. Introduction
The NVIDIA Tesla T4 is a low-profile, passively cooled accelerator based on the Turing architecture (TU104). It is constrained by a strict 70W Thermal Design Power (TDP). This paper formulates an analytical model for its dynamic power draw and thermal throttling behavior, with specific applications to Large Language Model (LLM) inference workloads, detailing the distinct occupancy regimes of Prefill and Decoding phases.

## 2. Dynamic Power Modeling
The total power draw of the GPU, $P_{total}$, can be decoupled into static leakage power ($P_{static}$) and dynamic switching power ($P_{dynamic}$).

$$P_{total}(T, V, f, O, \alpha, \gamma) = P_{static}(T, V) + P_{dynamic}(V, f, O, \alpha, \gamma)$$

Where:
- $T$ = Junction temperature
- $V$ = Core voltage
- $f$ = Clock frequency
- $O \in (0, 1]$ = Streaming Multiprocessor (SM) occupancy
- $\alpha \in [0, 1]$ = Tensor Core active duty cycle
- $\gamma \in [0, 1]$ = GDDR6 VRAM access rate (bandwidth utilization)

The dynamic component can be further split into SM, memory, and uncore power:
$$P_{dynamic} = P_{SM}(V, f, O, \alpha) + P_{MEM}(\gamma) + P_{uncore}$$

### 2.1 SM Occupancy and Power Scaling
For variable occupancy levels $O \in \{0.25, 0.50, 1.0\}$, the SM power scales sub-linearly due to clock gating and base overhead per active SM:
$$P_{SM}(O) = O \cdot P_{SM, peak} + (1 - O) \cdot P_{clock\_gating\_leakage}$$
At $O=0.25$, power is dominated by uncore and baseline SM overhead. At $O=1.0$, data path toggle rates dictate the maximum dynamic draw.

### 2.2 Tensor Core Active Duty Cycle ($\alpha$)
Turing Tensor Cores draw significant current when active. The power contribution scales linearly with the duty cycle $\alpha$:
$$P_{TC} = \alpha \cdot N_{TC} \cdot C_{TC} \cdot V^2 \cdot f$$
Where $N_{TC}$ is the number of Tensor Cores and $C_{TC}$ is the effective switching capacitance.

### 2.3 GDDR6 VRAM Access Rate ($\gamma$)
Memory power is a function of the static PHY power and dynamic toggle rate:
$$P_{MEM}(\gamma) = P_{MEM, idle} + \gamma \cdot (E_{bit} \cdot B_{max})$$
where $E_{bit}$ is the energy per bit transferred and $B_{max}$ is the theoretical peak bandwidth (320 GB/s for T4).

## 3. NVPM Boost Clock Scaling & Thermal Dynamics
NVIDIA Power Management (NVPM) dynamically adjusts the core frequency based on thermal and power constraints. The operational frequency $f_{core}$ is determined by:
$$f_{core} = \min(f_{max}, f_{thermal}(T), f_{power}(P_{total}))$$

For a passively cooled T4, air flow is provided by the chassis. The thermal model is:
$$C_{th} \frac{dT}{dt} = P_{total} - \frac{T - T_{ambient}}{R_{th}}$$
where $R_{th}$ is the thermal resistance and $C_{th}$ is the thermal capacitance.

When $P_{total}$ approaches the 70W TDP, or $T$ exceeds the throttle threshold ($T_{limit} \approx 83^\circ C$), NVPM enforces a steep frequency drop. The boost clock scales down from its peak of 1590 MHz to the base/throttled clock of roughly 950 MHz, reducing dynamic power by a factor of roughly $(950/1590)^3$ assuming $V \propto f$.

## 4. Roofline Analysis: LLM Prefill vs. Decoding
The operational regime of the GPU shifts dramatically during LLM inference, dictating the bounding limits on the Roofline model.

### 4.1 Prefill Phase (Compute-Bound, $M \ge 2048$)
During the prefill phase, the input sequence length $M$ is large (e.g., $M \ge 2048$). The computation consists of large dense matrix multiplications (GEMMs).
- **Occupancy:** High ($O \to 1.0$)
- **Duty Cycle:** High ($\alpha \to 1.0$)
- **Memory Access:** Low relative to compute ($\gamma$ is moderate)
- **Constraint:** Compute-Bound. The GPU operates at the peak of the Roofline model. $P_{total}$ hits the 70W limit quickly, triggering NVPM to throttle $f_{core}$ from 1590 MHz to ~950 MHz to maintain the TDP envelope.

### 4.2 Decoding Phase (Memory-Bound, $M = 1$)
During autoregressive decoding, the batch size is often small (batch=1, $M=1$). The computation is dominated by matrix-vector multiplications (GEMVs).
- **Occupancy:** Lower due to insufficient parallelism ($O < 0.5$)
- **Duty Cycle:** Low ($\alpha \to 0$) as Tensor Cores are underutilized.
- **Memory Access:** High ($\gamma \to 1.0$) as the entire model weight must be loaded for a single token.
- **Constraint:** Memory-Bound. The GPU operates on the sloped memory bandwidth section of the Roofline model. Power is dominated by $P_{MEM}$ and uncore. $P_{total}$ often remains below 70W, allowing the GPU to sustain higher boost clocks (up to 1590 MHz), though compute throughput is limited by the 320 GB/s memory bandwidth.

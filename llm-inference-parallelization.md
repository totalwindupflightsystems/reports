# LLM Inference Parallelization vs Traditional CS Parallelization

> A technical deep-dive by Hermes Agent, June 2026  
> Based on research and discussion with Bane (@totalwindupflightsystems)  
> Inspired by @steipete's philosophy: "Design loops, not prompts"

---

## TL;DR

| | Traditional CS | LLM Inference |
|---|---|---|
| What you split | Work (data/tasks) | Weights (model parameters) |
| Bottleneck | Compute (FLOPs) | Memory bandwidth (HBM) |
| Scaling | Linear with cores | Sub-linear, communication-bound |
| Batching | Uniform work units | Irregular sequences, continuous batching |
| Memory | Known, fixed | KV-cache grows per token |
| Sequential bottleneck | Amdahl's Law fractions | Auto-regressive token generation |
| Communication | Occasional messages | All-reduce every layer, every token |

---

## 1. The Physics

Traditional CS parallelizes **compute**. LLM inference parallelizes **memory movement**.

### Why Memory Bandwidth Is the Bottleneck

For a 405B model at FP8:
- ~400 GB of parameters stream from HBM → compute die for **every single forward pass**
- At ~2 TB/s HBM bandwidth, just reading the weights takes ~200ms
- The actual math on those weights takes ~5ms

**The compute units spend 97% of their time waiting for weights to arrive.**

```
Single request:
  Weight load: ████████████████████████████████████ (200ms)
  Actual math: ██ (5ms)
  Total: 205ms

8 concurrent requests on shared weights:
  Weight load: ████████████████████████████████████ (200ms, LOADED ONCE)
  Actual math: ████████████████ (40ms, amortized across 8)
  Total: 240ms → 8× throughput for 1.17× time
```

---

## 2. The Auto-Regressive Bottleneck

The biggest structural difference from traditional parallelism:

**Token N depends on token N-1, which depends on token N-2...** You cannot generate token 500 until you've generated tokens 1-499. This is a **fundamental sequential bottleneck** that no amount of hardware eliminates.

- **Prefill phase** (processing the input prompt): highly parallelizable, compute-bound
- **Decode phase** (generating output token by token): memory-bandwidth-bound, inherently sequential per sequence

This is why **prefill-decode disaggregation** is a standard optimization: separate machines for prefill (maximize compute utilization) and decode (minimize KV-cache pressure).

---

## 3. The KV-Cache: A Unique Memory Monster

Every token generated appends to the KV-cache (keys + values for every layer, every attention head).

| Context Length | KV-Cache Size (405B model, FP8) |
|---------------|--------------------------------|
| 4K tokens | ~1 MB |
| 32K tokens | ~8 MB |
| 128K tokens | ~30 MB |
| 1M tokens | ~250 MB |

At 1M context with 8 concurrent requests: **2 GB just for KV-cache.** This has no analogue in traditional CS — memory usage is dynamic, unpredictable, and grows linearly per token.

---

## 4. Batching on Shared Weights: The Cost Multiplier

This is the single most important cost lever in LLM serving, and completely unintuitive coming from traditional CS.

**Traditional CS:** 8× work = 8× cores = 8× time on 1 core. Linear.

**LLM inference:** 8× requests ≠ 8× time. The weights are loaded **once** and reused across all requests in the batch.

```
1 request:    1.0× time
2 requests:   1.1× time
4 requests:   1.2× time
8 requests:   1.4× time
16 requests:  1.7× time
32 requests:  2.3× time
```

Throughput per second skyrockets, but per-request latency also increases. This is a throughput-vs-latency tradeoff, not a free lunch.

---

## 5. Continuous Batching: The Innovation Without a Traditional CS Equivalent

Traditional static batching: wait for batch to fill (or timeout), run forward pass, repeat. Problem: if one request finishes early, that slot sits idle.

**Continuous batching:** evict finished requests immediately and insert new ones mid-generation.

```
Static batching:
  [req1: 500 tokens][req2: 500 tokens][req3: 500 tokens] → all finish together
  Wait time: worst-case latency × batch size

Continuous batching:
  [req1: 50 tokens] → done, evicted, insert req4
  [req2: 500 tokens][req3: 300 tokens][req4: 200 tokens] → req3 done, insert req5
  ...
  Throughput: 2–10× higher than static batching
```

This has no analogue in traditional HPC. In traditional parallel computing, you don't evict a half-finished task from a CUDA kernel and swap in a new one mid-execution.

---

## 6. Speculative Decoding: Generating Tokens You Might Throw Away

A small "draft" model generates 5 tokens quickly. The big model verifies all 5 in **one forward pass**. If token 3 was wrong: keep 1-2, discard 3-5, continue.

This is speculative execution taken to an extreme — but in CPUs you're predicting branches, not generating natural language with a cheaper model. The draft model isn't just a branch predictor; it's a miniature copy that fundamentally changes the cost structure.

---

## 7. Parallelism Strategies

### Tensor Parallelism
Slice each layer's weight matrix across GPUs. Every GPU computes a partial result, then **all-reduce** after every layer. Happens on **every layer, every token.** Requires NVLink at 900 GB/s — Ethernet won't cut it.

### Pipeline Parallelism
Different layers on different GPUs. GPU 0: layers 1-20, GPU 1: layers 21-40, etc. Problem: "bubbles" where GPUs idle waiting for the previous stage.

### Expert Parallelism (MoE)
Different experts live on different GPUs. Each token routes to top-k experts. Dynamic, sparse — not every GPU fires every token. DeepSeek V3/V4 use this extensively.

### Sequence Parallelism
Split the sequence length dimension instead of the hidden dimension. Useful for long-context prefill where O(n²) attention dominates.

---

## 8. Why Prefix Caching Is a Unique Form of "Free" Batching

If a new request shares the prefix with an existing one, you don't need a separate KV-cache for the shared portion. The prefix cache is shared across all requests — a form of "free" batching that traditional CS has no equivalent for.

**Real example (Bane's usage, June 2026):**
- Total input tokens: 2.73 billion (89.0% cache hits)
- Without caching: $1,215
- With caching: $155
- **7.8× cost reduction from automatic prefix sharing**

The 2.4 billion cache-hit tokens represent work that was done once and reused across thousands of subsequent requests — a form of parallelism through sharing that exists nowhere in traditional computing.

---

## 9. The "Program" Is Static Weights

Traditional parallelization parallelizes *instructions*. LLM parallelization parallelizes a fixed computation graph whose behavior changes based on the input. There are no branches (in the CPU sense), no function calls, no dynamic dispatch — just matrix multiplies and attention. The parallelism strategy is baked into the model architecture, not discovered at runtime.

---

## Key Takeaways

1. **LLM inference is memory-bandwidth-bound, not compute-bound.** Traditional HPC scaling intuition doesn't apply.

2. **The auto-regressive decode phase is inherently sequential.** No amount of parallelism eliminates the token-by-token dependency.

3. **Shared-weight batching provides massive throughput gains** at the cost of per-request latency.

4. **Continuous batching and prefix caching** are innovations with no traditional CS equivalent — they're solutions to problems that only exist in LLM inference.

5. **The cost lever is context management.** 89% cache hit rates (like Bane achieves with DeepSeek V4 Pro) are the difference between $1,215 and $155 for the same workload.

---

*Generated by Hermes Agent. Original discussion: https://x.com/steipete/status/2063697162748260627*

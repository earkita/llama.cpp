# Vulkan Lightning Indexer development notes

## Current stage

Stage 3 - Vulkan implementation design complete.

Vulkan implementation work has not started.

Baseline revision: `ee3d1b54c server: abstract llama_memory calls to common_memory (#26221)`.

Development branch: `vulkan-lightning-indexer`.

## Architecture notes

`GGML_OP_LIGHTNING_INDEXER` is a fused score calculation used by GLM DSA, DeepSeek 3.2, and DeepSeek 4 graphs. It replaces this unfused sequence:

1. Matrix multiplication of every query head with every cached indexer key.
2. ReLU of each query-key dot product.
3. Multiplication by a prescaled per-head weight.
4. Sum over indexer heads.
5. Permute and make contiguous so cached-key position is dimension 0.
6. Add the attention mask.

For query token `t`, cached key `i`, and stream `s`, the output is:

```text
dst[i, t, 0, s] =
    sum(h = 0 .. H - 1, max(dot(q[:, h, t, s], k[:, 0, i, s]), 0) * w[h, t, 0, s])
    + mask[i, t, 0, s % M]
```

Here `D` is the indexer head size, `H` is the number of indexer heads, `T` is the number of query tokens per stream, `S` is the number of streams, `C` is the number of cached keys, and `M` is the mask stream count.

The operation produces scores, not top-k indices. Model graphs call `ggml_top_k` on the score tensor afterward. A Vulkan implementation of this operation therefore must produce the full `C x T x S` score tensor required by the existing graph interface. Fusing top-k or avoiding that output would be a separate graph/API change and is outside Stage 1.

Relevant implementation paths:

- Declaration and constructor: `ggml/include/ggml.h`, `ggml/src/ggml.c`
- CPU dispatch and implementation: `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cpu/ops.cpp`
- CUDA dispatch, support check, and kernels: `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/lightning-indexer.cu`
- Graph construction: `src/models/glm-dsa.cpp`, `src/models/deepseek32.cpp`, `src/models/deepseek4.cpp`
- Fused-operation resolution: `src/llama-context.cpp`, `src/llama-graph.h`
- Existing backend coverage: `tests/test-backend-ops.cpp`

## Tensor specification

GGML dimensions are listed in `ne[0]` to `ne[3]` order. Dimension 0 is the row dimension.

| Tensor | Meaning | Type | Shape |
| --- | --- | --- | --- |
| `q`, `src[0]` | Query vectors for each indexer head and query token | F32 | `[D, H, T, S]` |
| `k`, `src[1]` | Cached indexer key vectors; one key head | Backend-dependent | `[D, 1, C, S]` |
| `weights`, `src[2]` | Prescaled per-head weights for each query token | F32 | `[H, T, 1, S]` |
| `mask`, `src[3]` | Additive cached-key mask for each query token | F16 | `[C, T, 1, M]` |
| `dst` | Masked indexer score for each cached key and query token | F32 | `[C, T, 1, S]` |

Constructor invariants:

- `q.ne[0] == k.ne[0]`
- `q.ne[1] == weights.ne[0]`
- `k.ne[1] == 1`
- `mask.ne[0] == k.ne[2]`
- `mask.ne[1] == q.ne[2]`
- `q.ne[2] == weights.ne[1]`
- `weights.ne[2] == 1`
- `mask.ne[2] == 1`
- `q.ne[3] == k.ne[3] == weights.ne[3]`
- `weights.ne[3] % mask.ne[3] == 0`

There is no broadcasting in the dot product or weights. The only explicit reuse is mask stream selection by `s % mask.ne[3]`. This supports either a per-stream mask or a mask shared across multiple streams.

The CPU implementation requires `nb[0] == ggml_type_size(type)` for every input and the output, so each row must have contiguous dimension-0 elements. Higher-dimensional strides are used explicitly and may include padding or represent views. The CUDA implementation has the same row-contiguity requirement. It additionally requires an unpermuted output with monotonic strides (`nb[0] <= nb[1] <= nb[2] <= nb[3]`) and requires each higher-dimensional stride of non-quantized `q` and `k` to be 16-byte aligned. CUDA vector loads also rely on the supported `D = 128` layout.

The constructor itself does not enforce general contiguity, higher-stride ordering, alignment, or a particular `k` type. Each backend support check must reject layouts it cannot execute.

In the model graphs, `q` and `weights` may be 4D views created when a batch is divided into streams. Their higher-dimensional strides must therefore be honored; an implementation must not assume fully packed 4D tensors. Cached `k` is also addressed using its supplied `nb[2]` and `nb[3]`.

Device placement is selected by the backend scheduler. Fused-op resolution enables Lightning Indexer only if all probed instances are supported on the same device as their assigned layer. The mask is a graph input prepared on host memory and copied by the scheduler as needed. The operation does not perform cross-device transfers itself.

## Operation parameters

`GGML_OP_LIGHTNING_INDEXER` stores no values in `op_params`. All dimensions, types, strides, stream reuse, and masking data come from the tensors.

The per-head weights are prescaled by the graph before this operation by:

```text
1 / sqrt(D * H)
```

That scaling is not part of the operation contract; the operation consumes the supplied weights without further scaling.

## Position and masking behavior

The operation has no position input and no position offset parameter. Query and cached-key positions are resolved when the graph input mask is populated.

For the DSA Lightning Indexer path, graph construction forces the mask to F16 whenever the fused operation is enabled. The mask has shape `[C, T, 1, M]`.

The normal cache mask contains:

- `0` for a selectable cached key.
- `-INFINITY` for an empty cache cell, a cell belonging to another sequence, a future position under causal attention, or a position excluded by sliding-window attention.
- With ALiBi, `-abs(key_position - query_position)` for an otherwise selectable key.

M-RoPE causal ordering is also applied while constructing the mask. Lightning Indexer merely converts each F16 mask element to F32 and adds it to the calculated score. Consequently a `-INFINITY` mask remains `-INFINITY` and cannot be selected over any finite score by downstream top-k.

The backend test helper uses arbitrary finite F16 mask values as well as blocks of zero and `-INFINITY`. Therefore the general operation contract is additive masking, not only binary keep/drop masking.

## Top-k behavior

Top-k is not part of `GGML_OP_LIGHTNING_INDEXER`. The model graph computes:

```text
n_top_k = min(C, configured_indexer_top_k)
top_k = ggml_cont(ggml_top_k(indexer_score, n_top_k))
```

`ggml_top_k` returns I32 indices with shape `[n_top_k, T, 1, S]`. It returns indices only, not scores. Its semantic intent is the highest scores along dimension 0. Backend tests explicitly allow the same selected index set in a different order, and allow different indices at tied values provided selected values match and indices are not duplicated. Therefore downstream ordering and tie stability are not portable requirements. CPU `ggml_argsort` uses `std::sort` with descending score comparison, which is not stable for equal elements.

Masked entries are excluded in normal use because their scores are `-INFINITY`. If `n_top_k` exceeds the number of finite entries, the graph contract does not define an invalid-index sentinel: top-k still returns in-range indices, potentially including `-INFINITY` entries. The model chooses `n_top_k` based on `C`, not on the count of unmasked positions.

## Numerical behavior

### CPU reference

- Quantized or low-precision `k` rows are converted to F32 using the registered type conversion function.
- Each `D`-element dot product uses `ggml_vec_dot_f32`.
- ReLU is `max(qk, 0.0f)`.
- The ReLU result is multiplied by an F32 weight and accumulated over heads in an F32 scalar.
- The F16 mask is converted to F32 and added last.
- The output is F32.

The exact reduction order is embedding dimension first, then heads in increasing index order. SIMD details inside `ggml_vec_dot_f32` may change the exact rounding.

### CUDA reference

The vector kernel converts or dequantizes `k` to F32, performs F32 multiply-adds, reduces each dot product across a 32-lane warp, applies ReLU and the F32 weight, and accumulates head contributions in F32. Its reduction order differs from CPU and exact bitwise equality is not expected.

The NVIDIA WMMA path converts F32 queries and supported key types to F16 before tensor-core multiplication, with F32 accumulation. It is an optimization with lower input precision than the CPU and CUDA vector paths. It is not required algorithmically.

The existing backend test permits normalized mean squared error up to `1e-6`. A Vulkan test should initially compare F32 output using this existing bound and also inspect absolute error near zero and index agreement after top-k. Tolerances must not be relaxed without evidence.

No implementation explicitly special-cases NaN or infinity:

- A negative dot product becomes zero.
- `max(NaN, 0.0f)` in the CPU macro expression resolves according to that expression's comparison behavior; GPU ternary implementations likewise need deliberate parity testing.
- Infinite dot products, weights, or masks follow ordinary IEEE arithmetic, including possible NaN from undefined combinations such as `0 * infinity` or `infinity + -infinity`.
- Downstream top-k ordering for NaN is not specified.

NaN and infinity behavior beyond ordinary arithmetic is not a model requirement established by current tests.

## CPU and CUDA comparison

Algorithmically required behavior common to both implementations:

- Honor dimension-0 row contiguity and higher-dimensional tensor strides.
- Select `q[:, h, t, s]`, `k[:, 0, i, s]`, `weights[h, t, 0, s]`, and `mask[i, t, 0, s % M]`.
- Compute one dot product per `(h, t, i, s)`.
- Apply ReLU before weighting.
- Sum weighted head results.
- Add the F16 mask after the head sum.
- Write an F32 score tensor.

CUDA-specific choices:

- Compile-time specialization for `D = 128` and `H = 32 or 64`.
- A 32-lane warp and `float4` vectorization.
- Eight warps per block.
- Processing several cached keys per warp/block.
- Shared-memory staging of queries and weights.
- Optional NVIDIA WMMA path and F16 conversion.
- CUDA/HIP dequantization helpers.

Concepts reusable in Vulkan:

- Specialize the initial supported model shapes.
- Tile cached keys so query data and weights are reused.
- Keep head and dot-product accumulation in F32.
- Dispatch over cached-key tiles, query tokens, and streams.
- Use existing type-specific dequantization infrastructure where available.

The current CPU and CUDA implementations are each a single conceptual fused kernel. They do not allocate a full intermediate `[H, C, T, S]` tensor. CUDA directly writes `[C, T, 1, S]`. CPU uses only one temporary F32 key row per worker.

## Temporary buffers and memory complexity

The required output allocation is:

```text
4 * C * T * S bytes
```

This output is `O(C * T * S)` and is part of the current graph contract, not backend scratch memory. The unfused graph additionally materializes head-wise scores and therefore has a much larger context-proportional intermediate.

CPU scratch space is approximately:

```text
4 * D * n_threads bytes
```

plus per-thread cache-line padding.

The CUDA vector kernel uses fixed-size registers and shared memory per block for supported `D` and `H`; it has no context-length-proportional scratch allocation. The WMMA kernel likewise uses fixed-size shared-memory tiles. Its grid size grows with `C`, but per-block storage does not.

No Vulkan temporary buffers exist yet. Exact shader scratch sizes, dispatch count, and Vulkan memory estimates belong to Stage 3 after comparable backend infrastructure is evaluated.

## Likely Vulkan implementation stages

This is a Stage 1 decomposition, not a selected design:

1. Select a cached-key tile, query token, and stream.
2. Load or convert key elements and reuse query elements across the tile.
3. Reduce each query-key dot product in F32 without assuming subgroup size 32.
4. Apply ReLU and the per-head weight.
5. Reduce over heads.
6. Add the reused mask stream.
7. Write the F32 score.

Because top-k is a separate graph node, a bounded-memory fused score-plus-top-k pipeline cannot be implemented solely as support for the existing operation. Stage 3 must decide whether to implement the current score contract directly or propose a separately scoped graph fusion after correctness is established.

## Vulkan infrastructure review

The backend already provides the required integration patterns:

- Pipelines are stored per `vk_device`, compiled by `vulkan-shaders-gen.cpp`, registered with `ggml_vk_create_pipeline` or `ggml_vk_create_pipeline2`, selected in `ggml_vk_op_get_pipeline`, and dispatched with descriptor-backed tensor subbuffers and push constants.
- Capability checks live in the Vulkan `supports_op` switch and can inspect types, dimensions, contiguity, device features, workgroup limits, and whether a required pipeline was created.
- Tensor offsets are represented by Vulkan subbuffers. Operation-specific strides are normally passed in element units through push constants.
- `sum_rows.comp` shows a portable shared-memory reduction with barriers and a specialization-constant workgroup size.
- `gated_delta_net.comp` shows a fused multi-input F32 operation specialized by model dimension. It selects subgroup arithmetic, clustered operations, or a shared-memory fallback according to device capabilities.
- `topk_nary_search.comp` uses subgroup arithmetic, ballot, shuffle, full-subgroup execution, and device subgroup-size specialization when those capabilities exist.
- `topk_argsort.comp` is the shared-memory fallback for devices without the richer subgroup feature set.
- `ggml_vk_topk` performs hierarchical top-k with a double-buffered preallocated scratch buffer and explicit synchronization between passes.
- Large argsort and split-K matrix/attention paths show preallocation through `prealloc_x` or `prealloc_split_k`, descriptor reservation, storage-range checks, and synchronization of reused scratch buffers.
- Existing graph fusion is recognized by `ggml_vk_can_fuse` over consecutive graph nodes and dispatched by setting `num_additional_fused_ops`. This is the relevant mechanism for a future Lightning Indexer plus top-k fusion, but no such fusion exists today.

The existing top-k implementation already has the required downstream Vulkan operation. Initial Lightning Indexer support should reuse it as a separate graph node rather than introduce sorting code into the score shader.

## Vulkan design alternatives

Let:

- `N = T * S`, the total number of query rows across streams.
- `C`, the cached-key count.
- `D = 128`.
- `H = 32 or 64`.
- `B`, a cached-key tile size.
- `K`, the downstream top-k count.

The arithmetic cost of every exact design is `O(N * C * H * D)`.

### Design A - one score per workgroup

Dispatch one workgroup for each `(candidate, query, stream)` output. Use 128 invocations so each invocation loads one F32 query and key component. For each head:

1. Each invocation multiplies one `q[d] * k[d]`.
2. Reduce the 128 products in shared memory.
3. Invocation 0 applies ReLU, multiplies the head weight, and adds it to the score.
4. After all heads, invocation 0 adds the F16 mask and writes one F32 output.

Properties:

- Dispatches: 1.
- Workgroups: `C * N`.
- Global temporary buffer: 0 bytes.
- Shared memory: 128 F32 values, or 512 bytes per workgroup.
- Register storage: one score accumulator plus a few scalar values per invocation.
- Synchronization: workgroup barriers during each head reduction; no inter-dispatch barrier.
- Subgroup assumptions: none.
- Required features: baseline compute shader support plus F16 storage/load support already required by the backend mask representation. Arithmetic can remain F32.
- Memory complexity: required output `4 * C * N` bytes and `O(1)` backend scratch.

Advantages:

- Closest to the CPU formula.
- Simple stride and mask indexing.
- Portable across AMD and NVIDIA subgroup sizes.
- Suitable as a correctness-first implementation.

Risks:

- Each query vector is reread for every candidate.
- There are `H` shared-memory reductions and barriers per output.
- Small workgroups and repeated query loads are likely bandwidth- and synchronization-heavy.
- `C * N` workgroups must be mapped within device dispatch-count limits, potentially using shader-side loops when a dimension exceeds `maxComputeWorkGroupCount`.

This is the selected Stage 5 correctness-first design.

### Design B - tiled candidates with query reuse

Dispatch one workgroup for a tile of `B` cached keys and one query row. Stage a chunk of query heads and weights in shared memory, load key vectors cooperatively, and compute several candidate scores per workgroup. Head processing remains chunked so shared-memory use is bounded.

A portable form uses a fixed workgroup size such as 128 or 256 and shared-memory reduction. An optimized form assigns candidate reductions to subgroups, specializing for the actual device subgroup size rather than assuming 32 lanes.

Example with four staged query heads:

```text
query shared memory = 4 * 128 * 4 = 2048 bytes
weight shared memory = 4 * 4 = 16 bytes
reduction/candidate staging depends on B and workgroup mapping
```

Properties:

- Dispatches: 1.
- Workgroups: approximately `ceil(C / B) * N`.
- Global temporary buffer: 0 bytes.
- Shared memory: `O(D * head_chunk + B)` with fixed compile-time tile limits.
- Synchronization: barriers between query-head chunks and any shared reductions.
- Subgroup assumptions: the fallback requires none; optimized variants use the runtime-selected subgroup size and require capability checks.
- Memory complexity: required output `4 * C * N` bytes and `O(1)` backend scratch.

Advantages:

- Reuses queries and weights across cached keys.
- Fewer workgroups and global query loads than Design A.
- Keeps scratch independent of context length.
- Can add F16, BF16, and quantized key loaders later without changing the output contract.

Risks:

- More indexing and synchronization complexity.
- Register pressure grows with candidates per invocation.
- A subgroup implementation must handle AMD wave32/wave64 and NVIDIA warp32 correctly.
- The best `B`, head chunk, and workgroup size are device-dependent.

This is the intended optimized implementation after Design A is correct. Stage 7 should benchmark shared-memory and subgroup variants rather than replacing the portable path.

### Design C - graph-fused score and top-k

Recognize the consecutive `LIGHTNING_INDEXER -> TOP_K` graph nodes in the Vulkan fusion pass. Process cached keys in tiles, retain candidate/value pairs, and merge them hierarchically until only `K` indices per query remain. The intermediate Lightning Indexer score tensor is never written.

Possible implementations:

- A persistent workgroup per query scans all key tiles and maintains `K` candidates in shared memory. Scratch is `O(K * N)` but occupancy and latency are poor when `N` is small or `K` is large.
- A hierarchical approach emits `K` candidates per `(query, key tile)`, then repeatedly merges candidate tiles. First-pass scratch is approximately `8 * K * ceil(C / B) * N` bytes for I32 indices and F32 scores, with later passes decreasing geometrically. Double buffering can double the peak.

Properties:

- Dispatches: at least 2, normally `1 + ceil(log_B(C / K))` merge passes.
- Synchronization: one command-buffer barrier or backend scratch synchronization between passes.
- Time complexity: score calculation remains `O(N * C * H * D)`; candidate merging avoids sorting all `C` values.
- Memory complexity: `O(K * N)` for a persistent design or `O(K * ceil(C / B) * N)` for a hierarchical design.
- Subgroup assumptions: none for a shared-memory fallback; subgroup ballot/arithmetic/shuffle can accelerate compaction and merge when explicitly supported.

Advantages:

- Removes the required `4 * C * N` score output from physical memory when fusion is active.
- Provides the credible path to large context with large ubatches.
- Reuses existing Vulkan graph-fusion and hierarchical top-k concepts.

Risks:

- It is not an implementation of `GGML_OP_LIGHTNING_INDEXER` alone.
- The fusion must preserve unfused graph semantics, graph lifetime rules, diagnostics, and fallback when either node is unsupported.
- Top-k tie ordering is backend-dependent, so validation must compare allowed selected values/sets.
- `K`, tile size, and shared-memory limits can make a persistent design impractical.
- This is a larger architectural change and must not be mixed into the correctness-first patch without prior maintainer discussion.

Design C is not selected for the first implementation. It is the explicit path for eliminating context-by-ubatch score storage after Designs A and B establish correctness.

## Memory scaling estimates

The score operation itself does not materialize an `H * C * N` tensor. Designs A and B write only the required F32 result:

| Context `C` | `N = 1` | `N = 32` | `N = 512` |
| ---: | ---: | ---: | ---: |
| 1K | 4 KiB | 128 KiB | 2 MiB |
| 4K | 16 KiB | 512 KiB | 8 MiB |
| 16K | 64 KiB | 2 MiB | 32 MiB |
| 32K | 128 KiB | 4 MiB | 64 MiB |
| 64K | 256 KiB | 8 MiB | 128 MiB |
| 262K | about 1 MiB | about 32 MiB | about 512 MiB |
| 1M | about 4 MiB | about 128 MiB | about 2 GiB |

These figures exclude allocator alignment and downstream top-k scratch. They show that direct score output is reasonable for token generation and small ubatches, but becomes material at long context with large prompt-processing batches. Designs A and B do not introduce another context-proportional allocation. Design C is required if the output itself becomes the limiting allocation.

A rejected prototype is the unfused-equivalent approach that first writes all head scores. Its F32 intermediate would be:

```text
4 * H * C * N bytes
```

At `H = 64`, `C = 262K`, and `N = 32`, this is about 2 GiB before the final score output. At `N = 512`, it is about 32 GiB. This design is impractical and will not be implemented.

## Initial Vulkan support boundary

The first implementation should advertise support only for:

- `q`, `weights`, and `dst`: F32.
- `mask`: F16.
- `k`: F32 only.
- `D = 128`.
- `H = 32 or 64`.
- `k.ne[1] = 1`, `weights.ne[2] = 1`, and `mask.ne[2] = 1`.
- The constructor shape relationships documented above.
- Contiguous dimension-0 rows for every tensor.
- Higher-dimensional strides expressible in the 32-bit element-stride push constants.
- An unpermuted, contiguous F32 output.
- `S % M = 0`, with mask selection by `s % M`.
- A device workgroup size of at least 128 invocations and at least 512 bytes of compute shared memory.
- Dispatch dimensions that can be covered directly or by an explicitly implemented shader-side loop within the device's `maxComputeWorkGroupCount` limits.

No top-k limit belongs to the standalone score operation. When a future graph fusion is added, it must separately constrain `K` according to available top-k pipelines, shared memory, and storage-buffer limits.

F16, BF16, and quantized key types are deferred. Support checks must return false for them until the corresponding shader load and conversion paths are implemented and tested. Cooperative matrices, subgroup arithmetic, subgroup ballot, subgroup shuffle, subgroup-size control, and the Vulkan memory model are not required by Design A.

## AMD and NVIDIA behavior

Design A uses workgroup shared memory and barriers, so it is independent of subgroup width. A 128-invocation workgroup contains two wave64 subgroups or four wave32/warp32 subgroups, but correctness does not depend on that partition.

Design B may use subgroup operations only in a pipeline specialized for the actual device subgroup size. The backend already records subgroup basic, arithmetic, ballot, shuffle, full-subgroup, and subgroup-size-control capabilities and demonstrates capability-selected fallbacks in top-k and Gated Delta Net. No shader may assume subgroup size 32.

Expected tradeoffs:

- AMD wave64 may reduce the number of subgroups needed for a 128-element dot product but can increase inactive lanes for narrower mappings.
- AMD wave32 and NVIDIA warp32 are natural for two-stage 128-element reductions.
- Shared-memory fallback should work on both vendors but will likely be slower than subgroup reductions.
- Cooperative-matrix acceleration is optional future work and cannot be a support requirement for the first implementation.

## Chosen implementation path

1. Stage 4 adds recognition, a pipeline slot, shader registration scaffolding, dispatch plumbing, and the conservative F32-only support check. It must continue returning false until an executable kernel is present.
2. Stage 5 implements Design A and enables only the documented subset.
3. Stage 6 measures actual score output and top-k scratch scaling. Since Design A has no backend scratch, instrumentation must distinguish graph output allocation from backend preallocation.
4. Stage 7 implements and benchmarks Design B while retaining Design A as the portable fallback.
5. A Design C graph fusion is considered only after correctness and measurements demonstrate that score output is the limiting allocation, and after the larger graph change is discussed with maintainers.

This path avoids the rejected head-score matrix, has bounded backend scratch from the first kernel, and preserves a route to eliminating the context-by-ubatch output when justified.

## Supported device capabilities

No Vulkan support is implemented or advertised as of Stage 3.

CUDA currently supports only `D = 128`, `H = 32 or 64`, and key types F32, BF16, F16, Q8_0, Q5_1, Q5_0, Q4_1, and Q4_0, subject to its alignment checks. These are observations, not proposed Vulkan support claims.

## Known limitations and unresolved questions

- Existing backend tests exercise randomized score comparison but do not provide deterministic edge-case coverage for masks, ties, NaN, or infinities.
- The operation constructor accepts any key type with a CPU conversion function, while CUDA advertises a narrower list. Vulkan must advertise only implemented types.
- The current graph contract necessarily materializes the full `[C, T, 1, S]` output. Avoiding that allocation at very long context requires fusion with downstream top-k and cannot be hidden inside this operation.
- Portable top-k output order and exact tie selection are intentionally not required by existing tests.
- NaN ordering in downstream top-k is unspecified.

No critical semantic ambiguity remains for implementing the score operation. The graph-level bounded-memory objective in later stages is a scope/design issue that must be resolved explicitly in Stage 3.

## Build commands and baseline results

Configuration command:

```sh
cmake -S . -B build-vulkan-debug -DCMAKE_BUILD_TYPE=Debug -DGGML_VULKAN=ON -DGGML_NATIVE=ON
```

Result: passed. CMake found Vulkan 1.3.275, `glslc`, and `glslangValidator`. `GL_KHR_cooperative_matrix` shader compiler support was detected. OpenSSL was not found, so HTTPS support was disabled; this is unrelated to Lightning Indexer.

Build command:

```sh
cmake --build build-vulkan-debug -j"$(nproc)"
```

Result: passed; all configured targets, including `test-backend-ops`, `libggml-vulkan.so`, and `llama-server`, built successfully.

## Test commands and results

Focused baseline command:

```sh
build-vulkan-debug/bin/test-backend-ops test -o LIGHTNING_INDEXER -b CPU
```

Result: passed, 108/108 cases. Coverage included `H = 32 or 64`, `T = 1 or 512`, `S = 1 or 4`, shared and per-stream masks, and F32, F16, BF16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0, and IQ4_NL keys.

The test process reported `ggml_vulkan: No devices found`, so no Vulkan execution baseline was possible in this environment. This is expected before the operation is implemented but means later Vulkan correctness and validation criteria require access to the target GPU environment.

Stage 2 added three deterministic F32 reference cases to `test-backend-ops`:

- `scalar`: minimum dimensions with a hard-coded expected score.
- `relu_order`: `D = 128`, `H = 32`, multiple query tokens, positive and negative dot products, negative weights, equal and nearly equal key values, finite masks, and `-INFINITY` masks.
- `mask_reuse`: `C = 65`, `T = 2`, `S = 4`, and `M = 2`, covering non-divisible candidate counts, mask indexing by `s % M`, masked entries, and the maximum candidate index.

The deterministic reference independently evaluates every output score in scalar F32 order. The comparison uses a `1e-5` absolute bound; the largest difference observed while developing the test was `3.815e-6`, caused by the CPU vector dot-product reduction order differing from the scalar reference. Infinite expected values must match exactly in sign. Failure output includes the case name, dimensions, first differing `[candidate, query, stream]` coordinate, expected value, and both backend results.

Build command:

```sh
cmake --build build-vulkan-debug --target test-backend-ops -j"$(nproc)"
```

Result: passed.

Focused deterministic command:

```sh
build-vulkan-debug/bin/test-backend-ops test -o LIGHTNING_INDEXER -b CPU -p 'case='
```

Result: passed, 3/3 cases. The command was run twice with the same result.

Complete Lightning Indexer command:

```sh
build-vulkan-debug/bin/test-backend-ops test -o LIGHTNING_INDEXER -b CPU
```

Result: passed, 111/111 cases. No full model is required.

Top-k-specific cases from the original plan were not added to this operation test because Stage 1 established that `GGML_OP_TOP_K` is a separate downstream operation with existing dedicated coverage. Position offsets are represented only through additive mask values because Lightning Indexer has no position input or parameter.

## Benchmark results

No benchmarks were run in Stages 1 through 3.

## Remaining work

- Commit the Stage 3 Vulkan design.
- Begin Stage 4 scaffolding only after the Stage 3 commit.

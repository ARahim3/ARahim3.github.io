# JAX Mastery

*A Complete Guide to High-Performance Machine Learning on Modern Accelerators*

---

## Preface to this edition

The first edition of this guide, written in 2024, walked through the JAX paradigm, Flax/Haiku, basic sharding, a from-scratch Transformer, and a Kaggle migration story. That edition is still useful — but the field moved.

Between then and now (May 2026), several things shifted under our feet:

- **Flax NNX** became the recommended Flax API. Linen still works, but every new tutorial, example, and serious training stack now starts with NNX. Haiku is in maintenance.
- **`shard_map`** displaced `pmap` as the canonical explicit-SPMD primitive, and the modern parallelism story is now a three-level spectrum: *auto* (`jit` with `NamedSharding`), *explicit* (typed shardings), *manual* (`shard_map`).
- The official JAX stack diagram got two more boxes filled in: **Grain** for data loading and **Orbax** for checkpointing. The "just use `tf.data` and `flax.serialization`" advice from a few years ago no longer reflects what serious projects do.
- **Pallas** matured into a real kernel-authoring DSL with two backends (Mosaic-TPU and Mosaic-GPU/Triton). FlashAttention, paged attention, MoE dispatch, and many other production kernels now ship as Pallas reference implementations.
- The "How to Scale Your Model" book at <https://jax-ml.github.io/scaling-book/> grew a **Part 12: GPUs** that gives the GPU programming model the same treatment the original gave to TPUs.
- The modern LLM stack now assumes things the old guide didn't even mention: FlashAttention-3, paged KV caches, GQA, RoPE plus YaRN/LongRoPE for long context, MoE with expert parallelism, FP8 training, μP for hyperparameter transfer, and a serving substrate (JetStream, MaxText, AXLearn, Tunix) that is mostly Pallas underneath.
- Hardware moved too: **H100 → B200**, **TPU v5p → v6 (Trillium)**, with new tensor-core formats (FP4) and a new memory level (TMEM on Blackwell).

This edition incorporates all of that. It is also more *opinionated* about hardware. The single biggest jump from "I write JAX" to "I make JAX fast" is internalizing the **roofline model** and the architecture it sits on. So Part II is now a stand-alone hardware substrate — TPUs, GPUs, the math of collectives — that the rest of the guide leans on. Every later decision (sharding choice, dtype choice, when to drop to Pallas) traces back to that chapter.

A note about math. This edition keeps the formulas visible. Roofline expressions, attention math, RoPE rotations, communication-cost equations, parameter and FLOP counts for Transformers, mixed-precision tradeoffs — all of it. Hand-wavy prose about "memory-bound" without the formula behind it builds an opinion, not a mental model. The intent here is the mental model.

How to read this guide depends on where you are:

- **New to JAX.** Read sequentially. Parts I–III give you a working setup; Part IV is where most people actually need their first careful pass.
- **Already comfortable with JAX, want hardware intuition.** Skim Part I for vocabulary, then read Part II carefully. Parts IV and V are where the hardware insight pays off.
- **Migrating from PyTorch / TF.** Part I plus Chapter 34 (migration) plus Part III (NNX). Then Part IV when you start sharding.
- **Building or scaling LLMs.** Part V is the bulk of what you need; Part IV underneath; Part VI when XLA leaves performance on the table.
- **Writing kernels.** Part VI is the destination, but you'll need Part II (hardware) to be productive there.

Source notes appear inline as URLs. The "How to Scale Your Model" book is cited often enough that it deserves a single canonical reference here: <https://jax-ml.github.io/scaling-book/>. Where this guide reproduces equations from that book, attribution is in the citation.

---

# Part I — The JAX Paradigm

## Chapter 1. JAX as a Function-Transformation System

### 1.1 The essence of JAX

JAX is a high-performance numerical-computing library for Python, developed at Google Research, designed for modern machine learning and large-scale scientific computation. Its design braids together a familiar NumPy-style API with three transformative capabilities: automatic differentiation, just-in-time (JIT) compilation through the XLA compiler, and composable vectorization and parallelization primitives. The result lifts NumPy-style programming to operate on accelerators (GPUs, TPUs) without changing the surface code.

But "NumPy on accelerators" is the marketing line, not the design. The design is that JAX is an **extensible system for composable function transformations**. Entire Python functions are first-class objects to be analyzed, manipulated, and rewritten. Pass a function to `jax.grad` and you get a new function that computes its gradient. Pass that to `jax.jit` and you get a compiled version. Pass *that* to `jax.vmap` and you get a vectorized version. Pass *that* into `shard_map` and you get a parallel version that runs across devices. Each transformation takes a pure function in and emits a pure function out; the composition just works.

This "functions-as-data" view is the central pillar. Once you see it, the rest of JAX falls into place — JIT, autodiff, vectorization, sharding all become specialized cases of "rewrite this function under such-and-such interpretation."

### 1.2 The functional-programming imperative: purity and immutability

To enable transformations, JAX imposes one critical constraint borrowed from functional programming: it operates on **pure functions**. A pure function's output depends solely on its inputs; it relies on no external state and produces no side effects beyond its return value.

This principle is enforced through **immutability**. Unlike NumPy arrays, which are mutable and support in-place modification, JAX arrays are immutable:

```python
import jax.numpy as jnp
import numpy as np

# NumPy: mutable
numpy_array = np.arange(5)
numpy_array[0] = 99
# numpy_array is now [99 1 2 3 4]

# JAX: immutable
jax_array = jnp.arange(5)
try:
    jax_array[0] = 99
except TypeError as e:
    print(f"JAX rejects mutation: {e}")
```

To express an "update," JAX provides a side-effect-free syntax that returns a new array:

```python
original = jnp.arange(5)
updated = original.at[0].set(99)
# original is unchanged: [0 1 2 3 4]
# updated is a new array:  [99 1 2 3 4]
```

These constraints — purity and immutability — are not pedantic. They are the *prerequisites* for the transformation machinery. When a function is guaranteed pure, its behavior is fully determined by its inputs. JAX exploits this by **tracing**: it runs the function once with abstract "tracer" objects in place of real data, recording each operation those tracers participate in. The recorded program is called a **`jaxpr`** (JAX expression) — a clean, analyzable static representation, stripped of Python's dynamic complexity.

The `jaxpr` is what XLA compiles. It is also what `grad` transposes, what `vmap` lifts to batched dimensions, and what `shard_map` partitions across devices. Without purity, none of those transformations would be sound. The trade is unambiguous: functional discipline up front in exchange for full access to the transformation grammar — and through it, performance and scalability you cannot get any other way.

### 1.3 JAX vs. the incumbents

#### vs. NumPy

JAX's `jax.numpy` is a near-drop-in replacement for many NumPy use cases, but the differences are fundamental:

- **Backend.** NumPy is CPU-bound, dispatching to BLAS/LAPACK in C/Fortran. JAX uses XLA to target CPU, GPU, and TPU through a single program representation. JAX dispatch is also asynchronous: the Python interpreter returns immediately while computation runs on the device. To time anything you need `.block_until_ready()`.
- **Mutability.** Discussed above: NumPy arrays are mutable, JAX arrays are not.
- **Random numbers.** NumPy has a global RNG state (`np.random.seed(...)`), which is convenient and treacherous in parallel/transformed code. JAX requires explicit PRNG keys and split discipline:

  ```python
  key = jax.random.key(0)
  key, subkey1 = jax.random.split(key)
  x = jax.random.normal(subkey1, (10,))
  ```

  Every random draw consumes a fresh subkey. Reproducibility under transformation is a feature of the *type system*, not a runtime convention.

For micro-benchmarks of single small ops on CPU, NumPy often wins thanks to lower per-op dispatch overhead. JAX's edge appears on accelerators or when JIT can fuse long sequences of ops.

#### vs. PyTorch

PyTorch is the dominant DL framework in research; the contrast with JAX is mostly philosophical:

- **Paradigm.** PyTorch is object-oriented, with stateful `torch.nn.Module` instances and an imperative dynamic graph. JAX is functional and stateless. Model parameters live in external PyTrees and are passed explicitly to stateless functions.
- **Ecosystem.** PyTorch ships "batteries-included" optimizers, data loaders, and high-level libraries (Lightning, Transformers). JAX is more modular: Flax / Equinox for modules, Optax for optimization, Grain for data, Orbax for checkpointing. Each is small and composes with the others.
- **Compilation.** `torch.compile` brings JIT to PyTorch as of 2.0. `jax.jit` is older, deeper in the design, and almost always the path that produces the optimized version of your code.
- **Distributed.** PyTorch has `DistributedDataParallel`, FSDP, and now device meshes. JAX's `jax.sharding` API plus `shard_map` / `jit`-with-sharding cover the same ground with a single, more compositional vocabulary, and the compiler (GSPMD) is more aggressive at automatic partitioning.

#### vs. TensorFlow

TensorFlow and JAX share Google parentage and the XLA compiler, but they target different audiences:

- **Framework vs. library.** TensorFlow is an end-to-end platform with serving, mobile, and edge stories (TF Serving, TF Lite). JAX is more of a numerical-computing library favored by researchers for flexibility and Pythonic feel.
- **API.** Keras gives TensorFlow a high-level abstraction; JAX leaves the framing to libraries like Flax. Conceptually, TF's graph mode and JAX's tracing are similar (both build a graph for optimization), but JAX's user-facing model is function-first.
- **Interop.** `jax2tf` exports JAX functions to TF SavedModel for serving. `jax.experimental.export` is the modern path.

A cheat sheet that holds up well in 2026:

| Feature                | NumPy                          | PyTorch                                | TensorFlow                              | JAX                                                         |
| ---------------------- | ------------------------------ | -------------------------------------- | --------------------------------------- | ----------------------------------------------------------- |
| Primary paradigm       | Imperative array               | Imperative OO                          | Graph + Eager                           | **Functional, transformation-oriented**                     |
| Backend                | CPU (C/Fortran)                | CPU/GPU/TPU eager                      | CPU/GPU/TPU                             | **CPU/GPU/TPU via XLA**                                     |
| Autodiff               | none                           | tape (`autograd`)                      | tape / graph (`tf.GradientTape`)        | **function transformation (`jax.grad`)**                    |
| Compilation            | precompiled                    | `torch.compile` (2.0+)                 | `tf.function(jit_compile=True)`         | **core feature (`jax.jit`)**                                |
| Parallelism            | threads                        | DDP / FSDP                             | `tf.distribute.Strategy`                | **`shard_map`, sharded `jit`, GSPMD**                       |
| State                  | mutable `ndarray`              | stateful `nn.Module`                   | stateful `tf.Variable`                  | **stateless functions, external PyTrees**                   |
| Maturity               | very high                      | very high (research)                   | very high (production)                  | **growing (research + production)**                         |
| Use case               | general numeric                | rapid research                         | production deployment                   | **high-performance research, scaling, custom algorithms**   |

### 1.4 Purity, side effects, and the "control flow on values" rule

Once you accept purity, one practical consequence trips everyone exactly once:

> Inside a JIT-compiled (or vmapped, or shard-mapped, or grad-traced) function, **Python control flow cannot depend on the *value* of a traced array.**

This is because at trace time the array is a tracer with a known shape and dtype but no concrete value. JAX provides three escape hatches:

1. **Mark the argument static** with `jax.jit(..., static_argnames=...)`, so its Python value is available at trace time. The downside: every distinct value triggers a recompile.
2. **Use JAX's structured control-flow primitives**: `jax.lax.cond`, `jax.lax.switch`, `jax.lax.scan`, `jax.lax.while_loop`. These work inside traced code because they are themselves JAX primitives.
3. **Refactor** so the value-dependent branch is hoisted out of the traced region.

The "use `numpy` for what should be static, `jax.numpy` for what should be traced" rule of thumb falls out of this directly. Computing a reshape size from `x.shape` should use plain `np.prod` (the result is a Python int, baked into the compiled program); doing arithmetic on traced arrays should use `jnp.*` (the result becomes part of the graph).

---

## Chapter 2. The Five Pillars: `grad`, `jit`, `vmap`, `shard_map`, `pmap`

JAX's expressiveness rests on a small set of fundamental transformations. The first edition called this "the four pillars"; the modern story has a fifth, `shard_map`, which has displaced `pmap` as the right entry point to multi-device parallelism. We'll cover all five.

### 2.1 `jax.grad`: autodiff as a transformation

PyTorch and TensorFlow both implement autodiff with tape-based machinery: as you execute a forward pass, the framework records operations onto a tape, and `loss.backward()` plays the tape in reverse. JAX takes the function-transformation route: `jax.grad(f)` returns a *new function* that, when called, computes the gradient of `f`.

```python
import jax
import jax.numpy as jnp

def tanh_squared(x):
    return jnp.tanh(x) ** 2

grad_fn = jax.grad(tanh_squared)
print(grad_fn(1.0))  # 0.39322388...

# Analytically: d/dx [tanh(x)^2] = 2 tanh(x) * sech^2(x) = 2 tanh(x) (1 - tanh(x)^2)
analytical = 2 * jnp.tanh(1.0) * (1 - jnp.tanh(1.0) ** 2)
```

Higher-order derivatives are just composition:

```python
g2 = jax.grad(jax.grad(tanh_squared))
g3 = jax.grad(g2)
g4 = jax.grad(g3)  # arbitrarily high order
```

For loss functions of multiple arguments, `argnums` selects which to differentiate against:

```python
def mse(weights, bias, x, y):
    pred = x @ weights + bias
    return jnp.mean((pred - y) ** 2)

grad_w = jax.grad(mse, argnums=0)            # w.r.t. weights
grad_wb = jax.grad(mse, argnums=(0, 1))      # w.r.t. weights AND bias
```

`jax.value_and_grad` returns both the loss value (for logging) and gradients in a single forward+backward pass — essential, because computing them separately doubles the forward pass:

```python
loss_val, (w_grads, b_grads) = jax.value_and_grad(mse, argnums=(0, 1))(W, b, X, Y)
```

PyTrees — arbitrary nested structures of lists, tuples, dicts, and registered dataclasses — are first-class. `jax.grad` differentiates with respect to a PyTree of parameters and returns a PyTree of gradients with the same structure:

```python
def loss_pytree(params, x, y):
    pred = x @ params['W'] + params['b']
    return jnp.mean((pred - y) ** 2)

params = {'W': W, 'b': b}
grads = jax.grad(loss_pytree)(params, X, Y)
# grads is a dict with keys 'W' and 'b' and matching shapes
```

A subtler tool is `jax.vjp` (vector-Jacobian product, used for backward-mode autodiff) and `jax.jvp` (Jacobian-vector product, used for forward-mode). `jax.grad` is built on `vjp`. For per-example Jacobians of large outputs, `jax.jacrev` and `jax.jacfwd` give you full Jacobians; for the Hessian, `jax.hessian = jax.jacfwd(jax.jacrev(f))`.

### 2.2 `jax.jit`: just-in-time compilation through XLA

`jax.jit` is the workhorse. It traces a function once, lowers the resulting `jaxpr` to **HLO** (XLA's high-level optimizer IR), and asks XLA to compile it down to device kernels. The first call pays a compilation cost; subsequent calls with matching shapes/dtypes hit the cache and run the optimized binary.

```python
@jax.jit
def step(x):
    return jnp.sin(x) * jnp.cos(x) + jnp.tanh(x) / (jnp.exp(x) + 1.0)
```

The single biggest win XLA gives you is **kernel fusion**: a chain of elementwise ops becomes one kernel that keeps intermediates in registers/L1 instead of round-tripping through HBM. Each unfused elementwise op pays the full HBM round trip; fusing ten of them turns ten round trips into one. We will see in Chapter 4 *why* this matters: elementwise ops are memory-bound, so reducing bytes moved is the only lever you have.

The cost is the constraint we just met: **traced control flow can't branch on traced values.** When you really do need a value-dependent branch, `static_argnames` lets you bake a Python-level constant into the compiled version:

```python
from functools import partial

@partial(jax.jit, static_argnames=['mode'])
def f(x, mode):
    if mode == 'square':
        return x * x
    else:
        return x + 1
```

Each new value of `mode` triggers a new compilation. That's fine for a handful of modes; ruinous for an integer batch size that varies every call.

A practical mental model: **at trace time, JAX sees a `ShapedArray` (just shape + dtype), not the data.** Any code that needs to consult the data must run under `jax.lax.cond`/`scan`/`while_loop`, or be marked static.

### 2.3 `jax.vmap`: automatic vectorization

In ML, you frequently write a function on a single example and want to run it on a batch. The brute-force option is a Python `for` loop, which is slow because it dispatches one op per call. The professional option is to rewrite the function with batched dimensions, which is tedious and error-prone. `jax.vmap` automates the rewrite:

```python
def dot(v1, v2):
    return jnp.dot(v1, v2)

batched_dot = jax.vmap(dot)            # batches both args along axis 0
results = batched_dot(batch_v1, batch_v2)
```

Crucially, `vmap` does not lower to a Python loop — it pushes the mapped axis *down* into the underlying primitives. A series of matrix-vector products becomes one matrix-matrix product. The performance is identical to a hand-batched implementation.

`in_axes` controls which axis of each input gets mapped (or whether it should be broadcast):

```python
def linear(W, b, x):
    return x @ W + b

batched = jax.vmap(linear, in_axes=(None, None, 0))   # broadcast W, b; map x
```

`in_axes=None` means "broadcast this argument to all batch elements." `in_axes=0` (default) maps along axis 0 of the argument. You can also map along other axes (`in_axes=1`), or specify a PyTree of `in_axes` to map differently across the structure.

A common use of `vmap` over `grad` is computing **per-example gradients** — useful for differential privacy, meta-learning, or just checking that the mean of per-example gradients matches the gradient of the mean loss:

```python
per_example_loss = lambda params, x_i, y_i: (x_i @ params - y_i) ** 2
per_example_grad_fn = jax.grad(per_example_loss)
batched_grad_fn = jax.vmap(per_example_grad_fn, in_axes=(None, 0, 0))
per_ex_grads = batched_grad_fn(params, X, Y)
```

### 2.4 `shard_map`: explicit SPMD across devices

`shard_map` (formerly `jax.experimental.shard_map.shard_map`, now exposed as `jax.shard_map` in current JAX) is the **manual / explicit** way to write multi-device code. You declare a `Mesh` of devices, and you write the body of the function as if you are *one device's worth of work* — and then you call collectives by name when you need data from other devices.

```python
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial
import numpy as np

mesh = Mesh(np.array(jax.devices()).reshape(8), ('data',))

@partial(shard_map, mesh=mesh,
         in_specs=P('data'), out_specs=P('data'))
def f(x):
    # Inside this function, x is one shard's worth of the global array.
    # If the global x has shape (1024,) and there are 8 devices,
    # each device sees x of shape (128,).
    local_sum = x.sum()
    global_sum = jax.lax.psum(local_sum, 'data')   # all-reduce across 'data' axis
    return x / global_sum
```

The mental flip is significant. Outside `shard_map`, when you handle a sharded array, you reason about the *global* shape, and the compiler partitions for you. Inside, you reason about the *local* shape per shard, and you call collectives explicitly.

Collectives available inside `shard_map` (all in `jax.lax`, all parameterized by axis name):

- `psum(x, 'axis')` — all-reduce sum across that mesh axis.
- `pmean(x, 'axis')` — all-reduce mean.
- `pmax`, `pmin` — max / min.
- `all_gather(x, 'axis', axis=0, tiled=True/False)` — gather all shards along an array axis.
- `psum_scatter(x, 'axis', scatter_dimension=0)` — reduce-scatter (the dual of `all_gather`).
- `ppermute(x, 'axis', perm)` — point-to-point shifts; used for ring algorithms.
- `all_to_all(x, 'axis', split_axis, concat_axis)` — used in MoE expert dispatch.
- `axis_index('axis')` — this device's index along the mesh axis (the equivalent of MPI rank).

`shard_map` always composes with `jit` — wrap the call in `jax.jit` so the compiled XLA program is the actual artifact. We will return to `shard_map` in depth in Part IV.

### 2.5 `jax.pmap`: the legacy primitive

`pmap` predates the modern sharding stack. It treats the leading axis of an input array as the device axis, replicates the function across devices, and feeds each device its slice. Collectives work the same way as in `shard_map`.

```python
parallel_fn = jax.pmap(my_fn, axis_name='i')
result = parallel_fn(data)   # data.shape[0] must equal num_devices
```

It works. But:

- It composes awkwardly with the modern `jax.Array` global view.
- It does not interact cleanly with `jit`-with-sharding.
- New JAX features (typed shardings, explicit mode) target `jit` and `shard_map` and don't extend to `pmap`.

The official guidance, and ours: **for new code, use `jit` with `NamedSharding` for declarative parallelism, `shard_map` for explicit per-shard programming. Use `pmap` only for tiny pedagogical examples or to maintain existing code.** A migration recipe lives in Chapter 14.

### 2.6 Summary table

| Transformation | Effect | When to use |
| --- | --- | --- |
| `jax.grad` | autodiff | wherever you need a gradient |
| `jax.jit` | XLA compilation + fusion | wrap your hot training step; your hot inference step; almost everything |
| `jax.vmap` | batched rewrite of single-example code | when broadcasting / batching is awkward; per-example gradients |
| `jax.shard_map` | manual SPMD across devices | custom collectives, expert all-to-all, ring algorithms, migration from `pmap` |
| `jax.pmap` | legacy single-device-per-replica | maintenance, tutorials |

---

## Chapter 3. Composing Transformations: The Defining Superpower

The transformations are powerful individually. They are *transformative* together. Because each transformation takes a pure function and returns a pure function, you can chain, nest, and combine them freely.

### 3.1 The philosophy of composability

In an OO framework, a method call mutates the object, creating side effects that make composition complex. In JAX, `jax.grad(f)` does not alter `f`; it returns a new function `g` which is itself pure and can be passed to `jax.jit`, `jax.vmap`, or another transformation. The result is a "grammar" of computation — complex behaviors emerge from layering simple transformations.

### 3.2 Common composition patterns

**`jit(grad(...))` — the foundation of any training step.** First create a gradient function, then JIT-compile it so the entire forward+backward pass becomes one XLA kernel:

```python
def loss(params, x, y):
    pred = x @ params['W'] + params['b']
    return jnp.mean((pred - y) ** 2)

grad_loss = jax.jit(jax.grad(loss))
grads = grad_loss(params, X, Y)   # forward + backward, fused
```

**`vmap(grad(...))` — per-example gradients.** Compute gradients for every example in a batch independently:

```python
per_ex_loss = lambda p, xi, yi: (xi @ p['W'] + p['b'] - yi) ** 2
per_ex_grad = jax.vmap(jax.grad(per_ex_loss), in_axes=(None, 0, 0))
per_ex_grad(params, X, Y)
# Sanity check: jax.tree.map(lambda g: g.mean(0), per_ex_grad(...))
# equals jax.grad(loss)(params, X, Y).
```

**`jit(vmap(...))` — high-performance batching.** Compose `vmap` for vectorization with `jit` for compilation:

```python
proc_one = lambda x: jnp.tanh(2 * x - 1)
proc_batch = jax.jit(jax.vmap(proc_one))
```

**`jit(shard_map(jit(value_and_grad(...))))` — the multi-device training step.** The outer `jit` compiles the whole thing; `shard_map` distributes across devices; the inner `value_and_grad` computes loss and gradients per shard:

```python
@jax.jit
def train_step(state, batch):
    def step_fn(state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
        # gradient all-reduce
        grads = jax.tree.map(lambda g: jax.lax.pmean(g, 'data'), grads)
        return state.apply_gradients(grads=grads), loss
    return shard_map(
        step_fn, mesh,
        in_specs=(P(), P('data')),  # state replicated, batch sharded on data
        out_specs=(P(), P()),
    )(state, batch)
```

We'll build up to this exact pattern in Part IV. For now, the takeaway: sophisticated training pipelines are *composed* from the same five transformations applied to small, verifiable building blocks.

### 3.3 An aside: the staging mental model

When you write `jax.jit(jax.grad(jax.vmap(f)))`, what JAX actually does at compile time is:

1. Trace `f` with abstract tracers under the `vmap` interpreter, producing a `jaxpr` that has a leading batch dimension on the inputs.
2. Run the `grad` transformation on that `jaxpr`, producing a backward `jaxpr`.
3. Hand the combined forward-and-backward `jaxpr` to XLA via `jit`, which lowers it to HLO and compiles to device kernels.

Each transformation operates on `jaxpr`s, not on Python source. That is why it doesn't matter how you write the loop, the conditionals, or the data flow inside `f` — what matters is the trace. Two functions that produce the same `jaxpr` will optimize identically.

This is the "purity contract" earning its keep. Without purity, transformations would have to model side effects, and the layering would collapse.

---

# Part II — The Hardware Substrate

You can write JAX for years without ever opening a GPU whitepaper, and your code will run. But there is a wall — somewhere between "training a small Transformer" and "making a 70B model train at 50% MFU" — past which everything depends on hardware. Past that wall, your decisions about sharding, recomputation, fusion, kernel choice, and dtype are all *implicitly* arguments about bytes moved per FLOP and cycles spent waiting on memory.

This part gives you the mental model. Read it once and you will start to *predict* what will be slow before you profile.

We build the model in four chapters: the roofline (the one diagram), the TPU, the GPU, and the math of collectives. The pivot point is Chapter 4 — every later chapter in the guide leans on it.

---

## Chapter 4. The Roofline Model

A modern accelerator has, to first order, two performance ceilings:

- A **compute ceiling**: peak FLOPs/sec at the relevant precision. Call it $\pi$, in FLOP/s.
- A **bandwidth ceiling**: peak bytes/sec moved from main memory (HBM) into the chip. Call it $\beta$, in B/s.

Every kernel performs some amount of arithmetic $F$ (FLOPs) and moves some amount of data $B$ (bytes between HBM and on-chip storage). The kernel's **arithmetic intensity** is

$$
\text{AI} = \frac{F}{B} \quad [\text{FLOPs per byte}]
$$

The classical Williams–Waterman–Patterson roofline says the **maximum sustainable throughput** at intensity AI is

$$
T(\text{AI}) = \min(\pi,\ \beta \cdot \text{AI})
$$

Plot AI on the x-axis and throughput on the y-axis (log-log) and you get two straight lines meeting at the **ridge point**

$$
\text{AI}^\star = \frac{\pi}{\beta}
$$

```
throughput (FLOP/s)
   π   ┤        ┌────────────────  compute ceiling
       │       /
       │      /
       │     /
       │    /  bandwidth ceiling (slope β)
       │   /
       │  /
       │ /
       │/_______________________________ AI (FLOPs/byte)
              AI* = π / β
```

A kernel left of the ridge is **memory-bound**: not enough arithmetic per byte to feed the math units. Doubling its FLOPs costs you nothing — the chip is idle anyway, waiting on HBM. A kernel right of the ridge is **compute-bound**: math units saturated; reducing FLOPs is what speeds it up.

This model is brutally useful, because (a) it tells you *which knobs even can help you* and (b) the ridge points of every modern accelerator are hundreds of FLOPs per byte. So *almost everything that isn't a big matmul is memory-bound.*

### 4.1 The book's framing: $T_{\text{math}}$ and $T_{\text{comms}}$

The "How to Scale Your Model" book (which this guide leans on heavily) presents the roofline through two times:

$$
T_{\text{math}} = \frac{F}{\pi}, \qquad T_{\text{comms}} = \frac{B}{\beta}
$$

with bounds

$$
T_{\text{lower}} = \max(T_{\text{math}}, T_{\text{comms}}), \qquad T_{\text{upper}} = T_{\text{math}} + T_{\text{comms}}.
$$

The lower bound assumes computation overlaps perfectly with communication; the upper bound assumes they serialize. Real systems sit in between, with how close to the lower bound depending on hardware (DMA engines, async copies) and on how the kernel was written.

The two regimes:
- **Compute-bound**: $T_{\text{math}} > T_{\text{comms}}$. Math units fully utilized.
- **Communication-bound**: $T_{\text{comms}} > T_{\text{math}}$. FLOPs wasted waiting on data.

### 4.2 The dot product (worked example, memory-bound)

For two bf16 vectors of length $N$, computing $x \cdot y$:

- FLOPs: $2N - 1 \approx 2N$ (one multiply per element + a sum tree).
- Bytes: $2 \cdot 2N + 2 = 4N + 2$ (read both vectors in bf16, write one scalar).

So

$$
\text{AI(dot product)} = \frac{2N - 1}{4N + 2} \xrightarrow{N \to \infty} \frac{1}{2}
$$

Half a FLOP per byte. On any modern hardware, that is *deep* in the memory-bound regime — far below the ridge, and the dot product's throughput is therefore limited by HBM bandwidth, not by FLOPs. This is also why you don't usually write naive dot products: they have nothing to optimize.

### 4.3 The matmul (worked example, compute-bound when big)

Take $C = A \cdot B$ with shapes $A \in (m, k)$, $B \in (k, n)$, $C \in (m, n)$. The FLOP count is $2mnk$ (one multiply + one accumulate per element of $C$, summed over the contracting dim). In bf16, a read-once / write-once schedule moves $2(mk + kn + mn)$ bytes:

$$
\text{AI(matmul)} = \frac{2 m n k}{2 (mk + kn + mn)} = \frac{mnk}{mk + kn + mn}
$$

When all dims are large and equal, $m = n = k = N$:

$$
\text{AI} \approx \frac{N^3}{3 N^2} = \frac{N}{3}
$$

**Arithmetic intensity grows linearly with the matrix dimension.** That is the single most important fact in deep-learning systems. Why we batch, why we love wide hidden dims, why attention in long-context models is a problem — all roofline consequences.

Concrete numbers: at $N = 8192$ in bf16, $\text{AI} \approx 2730$ FLOPs/byte. The H100 SXM bf16 ridge point is ~295 FLOPs/byte; TPU v5p's is ~165 FLOPs/byte. We are 9–16× above the ridge — solidly compute-bound, perfect.

But change the shape. The **decode step of an LLM** is one new token times the weight matrix: $m = 1$, $n = k = N$. Now

$$
\text{AI} \approx \frac{N}{N + 2} \xrightarrow{N \to \infty} 1
$$

Asymptotically *one* FLOP per byte — two and a half orders of magnitude below the ridge. **Decode is fundamentally memory-bound.** This is the deep reason decode is slow per FLOP. The KV-cache makes it worse, not better. Tensor-parallelism actually *hurts* small-batch decode on a GPU (more on this in Part V). Everyone batches, quantizes, and pages: the only escape is amortizing HBM reads across more work.

### 4.4 The elementwise op (forever memory-bound)

For $z = a \cdot x + b \cdot y$ on $N$-element arrays, $F \sim N$ and $B \sim N$, so $\text{AI} \sim O(1)$ FLOP/byte regardless of $N$. **Elementwise ops are forever memory-bound.** This is why `jit` matters: XLA fuses chains of elementwise ops so intermediates stay in registers/L1 and never touch HBM. Each unfused op pays a full HBM round trip; ten unfused ops pay ten round trips. One fused op pays one. The roofline ceiling does not move — but $B$ shrinks by 10×.

A useful compass: if your kernel's AI is below ~200 FLOPs/byte on modern hardware, no amount of clever math will save it. You have to move fewer bytes — fuse, tile, recompute, quantize, or change the algorithm.

### 4.5 Hardware ridge points (the cheat sheet you carry)

| Hardware | Peak (bf16) | HBM bandwidth | Ridge AI* |
| --- | --- | --- | --- |
| TPU v4p | $2.75 \times 10^{14}$ FLOP/s | $1.2 \times 10^{12}$ B/s | ~229 FLOPs/B |
| TPU v5e | $1.97 \times 10^{14}$ FLOP/s | $8.1 \times 10^{11}$ B/s | ~243 FLOPs/B |
| TPU v5p | $4.59 \times 10^{14}$ FLOP/s | $2.76 \times 10^{12}$ B/s | ~166 FLOPs/B |
| TPU v6e (Trillium) | $9.20 \times 10^{14}$ FLOP/s | $1.6 \times 10^{12}$ B/s | ~575 FLOPs/B |
| NVIDIA H100 SXM | $9.89 \times 10^{14}$ FLOP/s | $3.35 \times 10^{12}$ B/s | ~295 FLOPs/B |
| NVIDIA H200 | $9.89 \times 10^{14}$ FLOP/s | $4.8 \times 10^{12}$ B/s | ~206 FLOPs/B |
| NVIDIA B200 | $2.25 \times 10^{15}$ FLOP/s | $8.0 \times 10^{12}$ B/s | ~281 FLOPs/B |

*Peaks are the matmul ceiling at bf16. FP8/INT8 doubles the FLOP ceiling and halves the byte cost, doubling AI on both axes — net effect, FP8 lets you get away with smaller matmuls before going memory-bound.*

The pattern: ridge points are 100–600 FLOPs/byte. To be compute-bound on a matmul, you need the smallest dimension of (m, n, k) to be roughly the ridge value. That is why "the batch should be at least 240" is a common TPU rule of thumb — it's just the ridge.

### 4.6 Three communication levels (not one)

The roofline as written above treats memory bandwidth $\beta$ as one number, but reality is hierarchical. For an arbitrary chip:

1. **HBM bandwidth** ($\sim$ TB/s) — main memory off-chip but on-package.
2. **On-chip SRAM bandwidth** ($\sim 10$ TB/s on TPU VMEM, much higher on GPU shared memory) — the scratchpad that kernels work out of.
3. **Inter-chip bandwidth (ICI on TPU, NVLink on GPU intra-node)** ($\sim 100\text{–}900$ GB/s) — between chips inside one slice/node.
4. **Cross-node bandwidth (DCN on TPU, InfiniBand on GPU)** ($\sim 10\text{–}50$ GB/s) — between hosts/pods.

Each level has its own roofline. A kernel that fits in VMEM with a 22× HBM-VMEM ratio gets a ridge point ~22× lower than HBM-fed operations: a TPU v5e operating purely out of VMEM goes compute-bound on matmuls with batch ~11, not ~243. This is the entire point of writing Pallas kernels — push more of the working set into VMEM/SMEM and you collapse the roofline.

We will see in the next chapters that ICI vs. DCN, and NVLink vs. InfiniBand, are why "TP within node, FSDP across nodes" is a universal mantra on GPUs but TPU pods can do TP across thousands of chips.

---

## Chapter 5. The TPU

If a GPU is "an army of small SIMT cores chasing a memory hierarchy," a TPU is "one large dataflow engine chasing a memory hierarchy." TPUs simplify dramatically — you can hold a TPU in your head with much less effort.

### 5.1 The chip

A TPU chip (v4, v5e, v5p, v6e/Trillium) contains:

- **One or two TensorCores per chip.** v4 and v5p have 2 TensorCores per chip; v5e and v6e have 1. The two-core chips are presented to XLA as one logical "Megacore" unit with 2× MXU and 2× VPU. From `jax.devices()` you see one device per chip.
- **MXU (Matrix Multiply Unit)**: a *systolic array* of 128×128 multiply-accumulate cells. v6e (Trillium) doubles to 256×256 on some configurations. One full bf16 matmul tile per cycle, which is the entire reason TPUs exist.
- **VPU (Vector Processing Unit)**: a 2D SIMD engine of shape (8, 128) with 4 ALUs per (lane, sublane) pair. Does softmax, layernorm, elementwise ops. About one-tenth the FLOP rate of the MXU.
- **VMEM (Vector Memory)**: ~32 MB of fast on-chip scratchpad, software-managed (no hardware cache eviction surprises). Bandwidth to MXU is ~22× higher than HBM bandwidth.
- **SMEM**: a smaller scalar memory for indices, sizes, and control state.
- **HBM**: the off-chip but on-package memory. v5p: 96 GB at 2.76 TB/s; v6e: 32 GB at 1.6 TB/s; v5e: 16 GB at 0.81 TB/s; v4p: 32 GB at 1.2 TB/s.

Per-chip bf16 peak FLOP/s by generation:

| Generation | bf16 FLOP/s | int8 FLOP/s |
| --- | --- | --- |
| v3 | 1.4e14 | 1.4e14 |
| v4p | 2.75e14 | 2.75e14 |
| v5e | 1.97e14 | 3.94e14 |
| v5p | 4.59e14 | 9.18e14 |
| v6e | 9.20e14 | 1.84e15 |

### 5.2 Systolic-array intuition (read this twice)

Imagine a 128×128 grid of MAC cells. Each cycle, every cell does one multiply-accumulate. A tile of $A$ slides in from the top, one row per cycle; a tile of $B$ slides in from the left, one column per cycle. Partial sums march downward through the grid. After about $128 + 128 + k$ cycles, a full 128×128 output tile of $C = A \cdot B$ exits the bottom — and the next tile starts immediately while the first is still draining.

```
        A (rows feed down)
        │ │ │ │ │
        ▼ ▼ ▼ ▼ ▼
   B → [*][*][*][*][*]    each cell:
   B → [*][*][*][*][*]      acc += a * b
   B → [*][*][*][*][*]      pass a downward
   B → [*][*][*][*][*]      pass b rightward
   B → [*][*][*][*][*]
        │ │ │ │ │
        ▼ ▼ ▼ ▼ ▼
        C (drains out)
```

In steady state, every one of the 16,384 cells is multiplying every cycle. There is no instruction fetch, no register-file read, no scheduling — operands literally walk into each cell on a wire. That is the source of the TPU's stunning throughput-per-watt.

The catch: the array is "alive" only while fully fed. Misaligned shapes — a contracting dim of 96 against a 128-wide MXU — pad to 128 and waste 25% of the array. Padding is silent in your JAX code; the TPU profiler shows you "MXU utilization." This is also why TPU shapes love multiples of 128 (or 256 on Trillium) along the contraction and output axes. If you've ever seen "padded from 96 → 128" in the XLA HLO, that's why.

A rule that falls out: for a `bf16[8, 128] @ bf16[128, 128] → f32[8, 128]` matmul on v4/v5, the MXU finishes one tile per **8 cycles**. (One tile = the smallest multiple of 8 along the lane axis × full 128 sublane / lane.) BlockSpecs on TPU should respect this 8×128 multiple structure, which is part of why writing Pallas-TPU kernels feels less like CUDA and more like building a pipeline.

### 5.3 Latency-from-fill and the small-matmul problem

The systolic array has a pipeline-fill latency of $128 + 128 = 256$ cycles before the first output appears. For a single $128 \times 128 \times 128$ tile, you pay 256 fill cycles plus 128 useful cycles — only one-third of the time was useful. Stream a long tile (e.g., $128 \times 8192 \times 128$, a wide matmul) and the fill amortizes, hitting ~99% efficiency.

This is yet another reason small matmuls are slow on TPU: the MXU spends most of its time filling and draining, not computing. It's the same roofline phenomenon we saw in §4.3, expressed in the time domain.

### 5.4 ICI: the network is part of the chip

The point that most distinguishes the TPU programming model from the GPU one:

- **ICI (Inter-Chip Interconnect)**: directly-attached optical/electrical links wiring chips into a **3D torus**. Each chip has 6 ICI ports (±X, ±Y, ±Z) on v4/v5p; 4 ports (±X, ±Y) on v5e/v6e (2D torus). Per-link bandwidth is ~9e10 B/s one-way on v5p, ~4.5e10 B/s on v5e/v4p, ~9e10 B/s on Trillium. *That is roughly an order of magnitude faster than InfiniBand and roughly half of HBM.*
- A **slice** is a contiguous rectangular subset of a pod, all connected by ICI.
- A **pod** is the full physical fabric: v4 pods are 4096 chips (16×16×16); **v5p pods are 8960 chips (16×20×28)**; Trillium pods are 256 chips wired in a 16×16 2D torus per pod, with multiple pods joining into superpods.
- The torus **wraps around at the edges**, so for a length-$N$ ring on one axis, the worst-case hop count is $N/2$, not $N-1$. (If the slice is smaller than the pod, the wrap may not be present — corners cost more.)
- A **cube** is a 4×4×4 block with optical wraparound for reconfigurable topologies.
- **DCN (Data Center Network)** connects pods/slices. DCN bandwidth is roughly an order of magnitude lower than ICI: v5p ~6.25e9 B/s/chip egress, v6e ~12.5e9 B/s/chip, v5e ~3.125e9 B/s/chip. **Multi-slice** training crosses DCN.

Per-hop latency on ICI is ~1 µs.

### 5.5 Why TPUs scale

The reason you can keep adding chips to a TPU pod and keep getting near-linear throughput, while a GPU cluster starts to fight the network past one node, is that **ICI is roughly the same speed as HBM, while InfiniBand is ~20× slower**. The TPU was designed as a network-first system from v1; the GPU cluster grew an interconnect on top of a node-local design. Both work — but ICI is what lets you do tensor parallelism across thousands of chips. That is impossible on a GPU cluster except inside an NVLink domain.

### 5.6 The TPU "everything memory" hierarchy

A v5p chip's communication ladder, fastest to slowest:

1. **HBM ↔ VMEM**: ~22× HBM bandwidth (so ~60 TB/s on v5p, but capped by VMEM size).
2. **VMEM ↔ MXU**: ~1.8e13 B/s.
3. **HBM (on-chip)**: 2.76 TB/s.
4. **ICI (inter-chip)**: 1.8e11 B/s bidirectional, only to 4–6 neighbors.
5. **PCIe (CPU↔TPU)**: 1.5e10 B/s.
6. **DCN (cross-pod)**: 6.25e9 B/s.

Communication should be proportional to bandwidth. Violations cause bottlenecks — the kind that show up as 30% MFU on a profile.

### 5.7 A worked TPU latency example

The book provides this gem. Loading 200 B of bf16 parameters of a model on 32 TPU v4p chips:

- Bytes per chip: $400 \times 10^9 / 32 = 1.25 \times 10^{10}$ B (since bf16 is 2 bytes/param).
- HBM bandwidth per chip: $1.2 \times 10^{12}$ B/s.
- Minimum latency to load parameters: $\frac{1.25 \times 10^{10}}{1.2 \times 10^{12}} \approx 10$ ms.

So a single sampling step of a 200 B parameter LLM on 32 v4p chips has a *floor* of 10 ms, just to read the weights. Decoding one token in less than 10 ms is impossible on this hardware. *That* is a roofline argument.

---

## Chapter 6. The GPU

A GPU is the most counter-intuitive accelerator if you came from CPUs, because almost nothing about its execution model is what you would guess. The right mental model is "an army of arithmetic units chasing a single memory hierarchy."

### 6.1 The execution hierarchy

```
GPU (H100 SXM)
 └─ ~132 SMs (Streaming Multiprocessors)         [~148 on B200]
     └─ 4 warp schedulers per SM
         └─ warp = 32 threads in lockstep (SIMT)
             └─ thread = a program counter + register slot
```

The GPU is partitioned into **SMs**. The H100 SXM5 has 132 active SMs (144 physical, some fused off for yield); B200 has 148. Each SM has its own register file, schedulers, math units, and a slab of fast on-chip SRAM.

Threads execute in **warps** of 32. All 32 threads in a warp share a program counter — they execute the same instruction in lockstep. If they take different control-flow paths, the warp serializes the branches ("warp divergence"); a kernel author's job is to keep warps coherent. Up to 64 warps (2048 threads) can be resident on one SM at once; the SM time-slices among them to hide memory latency, the way a CPU's out-of-order engine hides L1 misses.

### 6.2 The memory hierarchy

| Level | Size (H100) | Bandwidth | Latency |
| --- | --- | --- | --- |
| Registers | ~256 KB / SM | per-SM, ~unlimited | 1 cycle |
| Shared memory / L1 | up to 228 KB / SM | ~33 TB/s aggregate | ~30 cycles |
| L2 cache | ~50 MB (split into two ~25 MB partitions) | ~12 TB/s | ~200 cycles |
| HBM3 | 80 GB (SXM5), 141 GB (H200), 192 GB (B200) | 3.35 / 4.8 / 8.0 TB/s | ~400+ cycles |

Each level is roughly an order of magnitude smaller and faster than the next. A "well-written" GPU kernel reads each tile from HBM at most once into shared memory, does as many FLOPs as possible against that tile, then writes the result. That is the *entire* art of GPU kernel writing reduced to one sentence — and it is precisely why FlashAttention exists, because vanilla attention writes the $S \times S$ attention matrix to HBM.

Blackwell adds a new level: **TMEM** (Tensor Memory), 256 KB/SM, dedicated to feeding the larger 5th-gen tensor cores. It removes the Hopper-era constraint that the matmul accumulator had to fit in registers.

### 6.3 Tensor cores: the cult

Each SM contains four **tensor cores** in addition to its FP32 CUDA cores. A tensor core consumes a small $m \times n \times k$ tile per cycle and outputs the tile's matmul, in mixed precision: bf16/fp16 inputs accumulating into fp32, plus fp8 (Hopper) and fp4 (Blackwell). H100 SXM peak rates:

| Precision | TFLOP/s |
| --- | --- |
| TF32 | ~495 |
| bf16/fp16 | ~990 |
| fp8 (E4M3/E5M2) | ~1979 |

The non-tensor-core FP32 path is only ~67 TFLOP/s. **You leave 15× on the floor if your matmuls don't go through tensor cores.** This is why dtype matters: a model in fp32 is twice as big in memory *and* can't use the fast units. JAX/XLA generally does the right thing for matmuls produced by `jnp.matmul` / `dot_general` with bf16 inputs. If you ever write a Pallas kernel and forget to call the tensor-core MMA primitive, you will hit that 15× cliff yourself.

B200 doubles tensor-core throughput and adds fp4: dense fp8 ≈ 4.5 PFLOP/s, fp4 ≈ 9 PFLOP/s. The B200 ridge in fp8 is roughly $4500 / 8 \approx 560$ FLOPs/byte. *Even more* operations need to be big matmuls or they're memory-bound.

### 6.4 TMA: hide the loads (Hopper)

Hopper added the **Tensor Memory Accelerator (TMA)**: a dedicated DMA engine that, given a tensor descriptor (shape, dtype, strides) and a coordinate, asynchronously copies a multidimensional tile between HBM and shared memory. Before TMA, every thread issued its own load and the warp blocked. With TMA, a single thread fires the copy and the rest of the warp does math while bytes arrive.

This enables **warp specialization**: dedicate some warps to be **producers** (issue TMA loads, signal a barrier) and others to be **consumers** (wait on the barrier, do MMAs). FlashAttention-3 exploits this aggressively. CUTLASS codifies the pattern.

### 6.5 Interconnect — the bandwidth cliff

This is what dictates your sharding strategy:

- **NVLink / NVSwitch (intra-node)**: H100 SXM exposes 18 NVLink-4 lanes for ~900 GB/s of bidirectional bandwidth per GPU, all-to-all to the other 7 GPUs in an HGX/DGX node via NVSwitch. That is an order of magnitude faster than HBM-to-CPU PCIe and roughly comparable to HBM bandwidth.
- **InfiniBand / RoCE (cross-node)**: a typical H100 cluster has 8 NDR-400 NICs per node — i.e., 50 GB/s per GPU. **About 18× slower than NVLink.**

That cliff — 900 GB/s inside the box, 50 GB/s leaving it — is the entire reason for the conventional sharding mantra: **"TP within node, FSDP/DP across nodes."** Tensor parallelism requires high-bandwidth all-reduces every layer; it would die on InfiniBand. Data/FSDP reduces gradients once per step, can be overlapped, and tolerates the slower fabric.

**B200 deltas** (sketch): 192 GB HBM3e, ~8 TB/s aggregate; **NVLink 5 at ~1.8 TB/s per GPU** (twice H100); NVL72 racks scale that fabric to 72 GPUs in one coherent domain — closing the gap with TPU pods at this scale.

### 6.6 The DGX network at scale

A DGX SuperPod for 1024 H100s (the canonical published topology):

| Level | GPUs | Switch BW | Per-link BW | Per-collective BW |
| --- | --- | --- | --- | --- |
| Node (NVLink) | 8 | 6.4 TB/s | 3.6 TB/s | 450 GB/s |
| Leaf (IB) | 256 | 25.6 TB/s | 12.8 TB/s | 400 GB/s |
| Spine (IB) | 1024 | 51.2 TB/s | 51.2 TB/s | 400 GB/s |

Per-node egress: 8 × 400 Gbps IB = 400 GB/s unidirectional from each node.

GB200 NVL72 (the Blackwell rack-scale design) gives a 72-GPU NVLink domain with 900 GB/s node egress (9× H100), but node-egress to spine is still ~3.6 TB/s. The takeaway: more bandwidth inside the rack; the cross-rack story still matters.

### 6.7 GPU vs. TPU at a glance

| Aspect | GPU (H100) | TPU (v5p) |
| --- | --- | --- |
| Compute units | ~500 (132 SMs × 4 subparts) | 2 TensorCores per chip |
| Thread model | SIMT (divergent, per-thread state) | SIMD (uniform) |
| L1 / VMEM | 32 MB total SMEM (132 × 228 KB) | 32 MB VMEM per chip |
| Compiler burden | Low; HW handles much | High; pipeline manually |
| Network | Hierarchical tree (NVLink + IB) | 2D/3D torus |
| Uniform-cost scaling limit | NVLink domain (8–72 GPUs) | Up to 8960 chips |
| Matmul ridge AI | ~295 (H100 bf16) | ~165 (v5p bf16) |

The compiler-burden row is what most surprises people coming from GPU work. On a TPU, XLA does much more *for* you (because the architecture is more constrained); the upside is high single-stream MFU; the downside is that the cases where XLA falls short are exactly when you reach for Pallas.

---

## Chapter 7. The Math of Collectives

Distributed training boils down to a small handful of **collective operations**, each with a closed-form cost on a ring topology that you should commit to memory. Let $N$ = number of devices in the ring, $V$ = total payload size in bytes, $W$ = per-link bandwidth (bytes/sec).

### 7.1 AllReduce (ring algorithm)

The classic ring-allreduce decomposes into a reduce-scatter followed by an allgather, each of which sends $\frac{N-1}{N} V$ bytes per device. So:

$$
\text{bytes/device}(\text{AllReduce}) = 2 \cdot \frac{N-1}{N} \cdot V
$$

$$
T(\text{AllReduce}) \approx \frac{2V}{W} \quad \text{for large } V, N
$$

Two facts:

1. The cost in *bytes per device* asymptotes to $2V$ and **does not grow with $N$**. This is the magic of ring algorithms — adding more devices does not slow each one down. That is why allreduce scales.
2. There is also a latency term $2(N-1) \cdot \alpha$ where $\alpha$ is per-hop latency. For very small $V$, the latency term dominates. This is why fusing many small gradients into one big bucket matters — you amortize latency.

### 7.2 AllGather and ReduceScatter

Each is "half" of an allreduce:

$$
\text{bytes/device}(\text{AllGather}) = \frac{N-1}{N} V \approx V
$$

$$
\text{bytes/device}(\text{ReduceScatter}) = \frac{N-1}{N} V \approx V
$$

$V$ here is the *output* size for AllGather and the *input* size for ReduceScatter. AllGather + ReduceScatter = AllReduce in cost and (often) in scheduling.

### 7.3 AllToAll

Used in mixture-of-experts routing. Each device sends a slice to every other device; the cost is roughly $\frac{N-1}{N} V \approx V$ bytes per device, but the access pattern is a permutation (no reduction). On a ring topology it's harder to make optimal; on a 2D/3D torus you decompose along axes.

The book gives, for an $A \times B \times C$ mesh:

$$
T(\text{AllToAll}) \approx \frac{V \cdot \max(A, B, C)}{4 \cdot N \cdot W}
$$

and for a single ring of $N$ devices:

$$
T(\text{AllToAll}) \approx \frac{V}{4 W}
$$

AllToAll is why MoE training is much more sensitive to fabric quality than dense training.

### 7.4 The two collective regimes you actually see

The intuitive law:

- **Replicate weights, reduce gradients (DDP)**: one **AllReduce** of the full gradient at every step. Easy to reason about, easy to overlap with backward, but every device holds full weights and optimizer state. Memory expensive.
- **Shard weights (FSDP / ZeRO-3)**: an **AllGather** of weights on the way down (forward), and a **ReduceScatter** of gradients on the way up (backward). Same total bytes as AllReduce but you only ever materialize one shard at a time — memory cheap. Each layer pays a synchronization, so it's harder to hide.
- **Tensor parallelism**: AllReduce after each row-parallel matmul (or AllGather/ReduceScatter for column-parallel). Per-layer collectives — only feasible on fast fabric, hence "TP inside node only" on GPUs.
- **Pipeline parallelism**: point-to-point sends only, no collectives, but it leaves bubbles.

### 7.5 Why GPU clusters fight at scale, and TPU pods don't

Plug numbers in. A 70 B model in bf16 has ~140 GB of weights; FSDP across 1024 GPUs spread over 128 nodes shards them ~140 MB per device. The forward pass needs to AllGather each layer's weights — say ~100 MB per layer:

- On NVLink: $100 \text{ MB} / 900 \text{ GB/s} \approx 110\,\mu\text{s}$.
- On InfiniBand crossing 16 nodes: $100 \text{ MB} / 50 \text{ GB/s} \approx 2 \text{ ms}$.

Twenty times slower. That is the reason FSDP across nodes works only when overlapped carefully and uses hierarchical (intra-node, then inter-node) schedules — a feature called **HSDP** / hybrid sharded data parallel.

On a TPU v5p pod with 1024 chips, every chip is on ICI at ~90 GB/s/link × 6 links and the rings span the slice — you stay at NVLink-like speeds even at thousands of chips. *That* is the reason people quote "TPUs scale better."

### 7.6 How this changes the JAX decisions you actually make

You now have enough hardware vocabulary to translate every common performance question into a roofline-or-collective question.

**"Why is my elementwise chain slow?"** Each op has $\text{AI} \approx O(1)$, far below the ridge — so it's HBM-bound. Each unfused op is one HBM round trip. Fix: `jax.jit`. XLA fuses elementwise chains into a single kernel that keeps tensors in registers/L1, so you pay one round trip instead of $k$. If `jit` isn't fusing (check with `jax.jit(f).lower(...).compile().as_text()` and look at the HLO), it's usually a `dynamic_update_slice`, an unaligned `reshape`, or a sharding boundary breaking the fusion.

**"Why is my FSDP step slow at scale?"** Profile and look at AllGather time vs compute time. Three escapes:
- Increase batch / sequence length so compute grows and AllGather amortizes.
- TP inside a node, FSDP across nodes (the hybrid pattern).
- Overlap: `shard_map` with async collectives + careful scheduling lets the compiler overlap each layer's AllGather with the previous layer's compute. XLA's GSPMD does this automatically for many shapes; sometimes it needs a hint via `jax.lax.with_sharding_constraint`.

**"Why is decode so slow per FLOP?"** Because $\text{AI} \approx 1$. You are fundamentally HBM-bound — the chip spends most cycles waiting on KV-cache reads, not doing math. Fixes all reduce *bytes moved per token*:
- **Batch more requests** (continuous batching / paged attention): amortize the weight read.
- **Quantize weights** to int8 / fp8 / int4: each byte saved is bandwidth saved.
- **Compress the KV cache** (MQA / GQA / MLA): fewer KV bytes per token per layer.
- **Speculative decoding**: $k$ tokens of math per HBM round trip.
- TP *hurts* small-batch decode because the extra AllReduce per layer costs more than the FLOP savings — yet another roofline consequence.

**"Why does Pallas (or Triton) help me here?"** Because XLA, smart as it is, sometimes generates 4 kernels where 1 would suffice — particularly across reductions, scatter/gather, and custom shapes. Each kernel boundary is an HBM round trip you can sometimes elide. Pallas lets you write one kernel that holds the whole stage in VMEM (TPU) or shared memory (GPU), bringing $B$ down by a constant factor that is occasionally 2–10×. The classic case is FlashAttention.

**"Should I use bf16 or fp8?"** Two effects: peak FLOPs roughly double, and HBM bytes halve. If you were compute-bound, fp8 is ~2× faster. If you were memory-bound, fp8 is also ~2× faster (bytes halve). The catch is numerical — you need scaling, and your accumulation must stay in fp32. JAX `jax.lax.dot_general` with `preferred_element_type=jnp.float32` and bf16/fp8 inputs is the canonical pattern.

**"Will this kernel saturate the chip?"** Compute its AI. Compare to the ridge point. If $\text{AI} < \text{AI}^\star$, the answer is no, no matter how clever you are. This is the most empowering single skill in the whole field: in 30 seconds with a calculator, you can predict whether a proposed optimization is even worth attempting.

A final mental discipline. When something is slow, refuse to guess. Open the profiler (Chapter 31). Ask: is the chip *busy* (compute-bound) or *waiting* (memory-bound or comms-bound)? The roofline tells you which it should be; the profiler tells you which it is; the gap between them is your optimization opportunity. Everything else — fusion, sharding, dtype, kernel choice, recomputation — moves an op on the chart, either rightward (more arithmetic per byte) or upward (more parallelism). Once you see operations as points on the roofline plot and shardings as choices about which collective to pay, the hardware stops being a black box and starts being a tool.

---

# Part III — Building Models in the Modern Ecosystem

JAX itself is a numerical-computing library. The neural-network abstractions live in companion libraries. The first edition of this guide centered on Flax Linen and Haiku; the modern story is different. **Flax NNX** is now the recommended Flax API; **Equinox** is the principled minimalist alternative; **Haiku** is in maintenance and not where new projects start. Optimization remains **Optax**'s territory. Two new pieces have joined the canon: **Grain** for data loading and **Orbax** for checkpointing — together they fill the "Layer 1" and persistence parts of the JAX stack diagram that the first edition mostly punted on.

This part walks through the canonical 2026 ecosystem: how to define a model (Chapters 8–9), how to optimize it (10), how to feed it data (11) and persist it (12), and a complete worked end-to-end pipeline (13).

---

## Chapter 8. Flax NNX

### 8.1 Why NNX (and what changed from Linen)

Flax Linen's elegant `model = MyModule(...); params = model.init(rng, x); y = model.apply(params, x)` two-stage flow is one of the most cited examples of "functional purity at the API surface." It is also, in practice, what new users trip on most. The friction points:

- An `nn.Module` instance does **not** own its parameters — they live in a separate `FrozenDict`.
- `@nn.compact` vs. `setup()` — two ways to declare submodules with subtly different semantics around lazy initialization.
- The variable-collections system (`{'params': ..., 'batch_stats': ...}`) and the `mutable=` kwarg dance for stateful layers.
- Threading `train=True/False` and `rngs={...}` through `apply` for dropout / BN.

NNX reorganizes this around real Python objects. An `nnx.Module` instance is mutable, holds its own parameters as attributes, and looks much more like `torch.nn.Module`. Under the hood, NNX stays JAX-pure: every parameter is wrapped in an `nnx.Variable` (a tracked reference object) and the framework provides explicit `split`/`merge` to convert a module to/from the `(graphdef, state)` pure-functional pair that JAX transforms need.

The official rationale, paraphrased: Linen achieves functional purity *at the API surface*; NNX achieves it *at the transform boundary*. Users get a familiar Pythonic object model; JAX still sees pure functions where it has to.

Linen is **not deprecated** — long-term support is committed — but new features (newer parallelism-aware transforms, the `nnx.bridge` interop) target NNX, and the official `flax.readthedocs.io` examples (MNIST, ResNet, Gemma) are now NNX-first.

### 8.2 The reference / Variable / split-merge model

A minimal NNX module:

```python
from flax import nnx
import jax, jax.numpy as jnp

class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, (din, dout)) * 0.01)
        self.b = nnx.Param(jnp.zeros((dout,)))

    def __call__(self, x):
        return x @ self.w + self.b

model = Linear(10, 4, rngs=nnx.Rngs(0))
y = model(jnp.ones((1, 10)))                 # call directly; no .apply()
print(model.w.value.shape)                   # (10, 4) — a real attribute
```

Three things to notice:

1. `nnx.Rngs(0)` replaces Linen's `init(rng, x)`. You pass RNGs to the constructor; the constructor *runs initializers immediately*, so by the time `__init__` returns, the module is fully initialized. There is no separate "init phase."
2. `self.w` is an `nnx.Param`, an `nnx.Variable` subclass. The actual array sits at `self.w.value`; reads/writes go through the wrapper so NNX can track them.
3. Calling the module is just calling it — no `apply`, no params dict.

To use `jax.jit`, `jax.grad`, etc. you convert the module into a pure pair:

```python
graphdef, state = nnx.split(model)             # state: pytree of arrays
                                               # graphdef: static structure

@jax.jit
def loss_fn(state, x, y):
    model = nnx.merge(graphdef, state)         # rebuild inside jit
    return jnp.mean((model(x) - y) ** 2)

grads = jax.grad(loss_fn)(state, x, y)
```

`graphdef` is hashable, traced once, and identifies the module's structure. `state` is the PyTree of `Variable` values and is what you actually differentiate / jit over. After the transform, `nnx.update(model, new_state)` writes back in place. This is the **functional core, imperative shell** pattern — same JAX semantics, much friendlier surface.

NNX also supports `nnx.split(model, nnx.Param, nnx.BatchStat, ...)` to fan state out by Variable subtype — replacing Linen's "variable collections" with regular Python type filters.

### 8.3 NNX-aware transforms

NNX ships transform wrappers (`nnx.jit`, `nnx.grad`, `nnx.vmap`, `nnx.scan`, `nnx.pmap`, `nnx.shard_map`) that hide the split/merge boilerplate:

```python
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        return jnp.mean((model(batch['x']) - batch['y']) ** 2)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)             # mutates model & opt state
    return loss
```

Rule of thumb:

- Use `nnx.*` transforms when transforming a function that takes/returns NNX modules or `nnx.Optimizer` directly.
- Use raw `jax.*` transforms when you've already split the module to `(graphdef, state)` — i.e., when you want full control and explicit state plumbing (good for distributed / checkpointing code). Under the hood, `nnx.jit` is just `jax.jit` plus split/merge.
- `nnx.scan` is particularly nice for layer-stacking (BERT-style trunks): write one layer, scan a stacked variant, get `lax.scan` performance with a real Pythonic module.

### 8.4 Stateful layers: BatchNorm, Dropout, RNGs

In Linen these required `mutable=['batch_stats']`, a separate RNG stream named `'dropout'`, and the `train=True/False` flag through `apply`. NNX makes them attributes:

```python
class Block(nnx.Module):
    def __init__(self, dim, *, rngs):
        self.lin = nnx.Linear(dim, dim, rngs=rngs)
        self.bn  = nnx.BatchNorm(dim, rngs=rngs)
        self.drop = nnx.Dropout(0.1, rngs=rngs)

    def __call__(self, x):
        x = self.lin(x)
        x = self.bn(x)            # uses internal running stats
        x = nnx.relu(x)
        x = self.drop(x)          # uses self.drop.rngs internally
        return x

model = Block(64, rngs=nnx.Rngs(params=0, dropout=1))
model.train()                     # toggles BN/Dropout to train mode
# model.eval() for eval
```

`BatchNorm` stores its running mean/var as `nnx.BatchStat` Variables; the optimizer naturally won't touch them because you `nnx.split(model, nnx.Param, ...)` and only update the `Param` sub-state. Dropout's RNG advances inside the module.

### 8.5 Side-by-side: an MLP and a Transformer block, Linen vs NNX

**MLP (Linen):**

```python
import flax.linen as nn

class MLP(nn.Module):
    hidden: int
    out: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        return nn.Dense(self.out)(x)

model = MLP(64, 10)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28*28)))
y = model.apply(params, x)
```

**MLP (NNX):**

```python
from flax import nnx

class MLP(nnx.Module):
    def __init__(self, din, hidden, out, *, rngs):
        self.fc1 = nnx.Linear(din, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, out, rngs=rngs)

    def __call__(self, x):
        return self.fc2(nnx.relu(self.fc1(x)))

model = MLP(28*28, 64, 10, rngs=nnx.Rngs(0))
y = model(x)
```

Three lines shorter, no `init/apply`, no `params` dict, and `model.fc1.kernel.value` is inspectable in a debugger.

**Transformer block (NNX):**

```python
class TransformerBlock(nnx.Module):
    def __init__(self, dim, heads, *, rngs):
        self.ln1  = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(num_heads=heads, in_features=dim, rngs=rngs)
        self.ln2  = nnx.LayerNorm(dim, rngs=rngs)
        self.ff1  = nnx.Linear(dim, 4*dim, rngs=rngs)
        self.ff2  = nnx.Linear(4*dim, dim, rngs=rngs)
        self.drop = nnx.Dropout(0.1, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop(self.attn(self.ln1(x), self.ln1(x)))
        x = x + self.drop(self.ff2(nnx.gelu(self.ff1(self.ln2(x)))))
        return x
```

`model.train()` / `model.eval()` flips Dropout mode globally. There is no `train` boolean to thread.

### 8.6 Migration guidance

- **Stay on Linen** if: you have a large existing codebase (T5X-style), depend on Linen-only third-party libraries, or are locked to a checkpoint format. Linen still works, and `nnx.bridge.ToNNX` / `nnx.bridge.ToLinen` (in `flax.nnx.bridge`) let you embed Linen modules inside NNX models and vice versa during a gradual migration.
- **Use NNX** for: new projects, teaching, anything where state-mutation ergonomics matter (LoRA adapters, hot-swappable layers, RL with multiple agents, model surgery).
- The migration guide at <https://flax.readthedocs.io/en/latest/guides/linen_to_nnx.html> walks the bridge patterns.

---

## Chapter 9. Equinox: the Module-as-Pytree Tradition

Equinox (<https://docs.kidger.site/equinox/>) is the principled minimalist alternative to Flax. It predates NNX and remains the favorite of many researchers, especially in scientific computing (its author also maintains diffrax for ODEs/SDEs and lineax for linear solvers).

### 9.1 The model-as-pytree philosophy

In Equinox, an `eqx.Module` is *literally* a registered JAX PyTree. There is no `Variable` wrapper, no `split`/`merge`. The module **is** the parameter container, and JAX transforms see it directly:

```python
import equinox as eqx, jax, jax.numpy as jnp

class MLP(eqx.Module):
    layers: list

    def __init__(self, sizes, key):
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [eqx.nn.Linear(i, o, key=k)
                       for i, o, k in zip(sizes[:-1], sizes[1:], keys)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

model = MLP([784, 64, 10], jax.random.PRNGKey(0))
# model is a pytree. jax.tree.leaves(model) is the parameters.
```

You can pass `model` straight to `jax.jit`, `jax.grad`, `jax.vmap`. There is no separate "state" to plumb.

### 9.2 Filtered transforms

The catch: a model contains both arrays (parameters) and non-array Python objects (e.g., activation functions, ints). Raw `jax.grad(f)(model)` would try to differentiate everything. Equinox's solution is **filtered transforms**:

```python
@eqx.filter_jit
def loss_fn(model, x, y):
    return jnp.mean((jax.vmap(model)(x) - y) ** 2)

grads = eqx.filter_grad(loss_fn)(model, x, y)
```

`eqx.filter_jit` / `eqx.filter_grad` automatically partition the PyTree into "arrays we differentiate / trace" and "static Python objects we leave alone." For freezing layers in fine-tuning:

```python
filter_spec = jax.tree.map(lambda _: True, model)
filter_spec = eqx.tree_at(lambda m: m.layers[0], filter_spec, replace=False)
grads = eqx.filter_grad(loss_fn, arg=filter_spec)(model, x, y)
```

### 9.3 When to prefer Equinox over NNX

- You want **no framework magic** beyond bare JAX semantics. An Equinox module is a PyTree, period.
- You're doing **scientific computing** — ODEs (diffrax), implicit layers (optimistix), linear solves (lineax). Equinox composes naturally with all of Patrick Kidger's stack.
- You want **PEP-style typing / dataclass syntax** without metaclasses; `eqx.Module` is a thin subclass of `dataclass`.
- You're writing **research code** that you might later need to fold/scan/checkpoint manually.

NNX is more ergonomic for big stateful models with lots of moving parts (LoRA, KV-cache, RL); Equinox is leaner, more transparent, very stable. Both are valid. Pick one and commit; mixing causes friction.

---

## Chapter 10. Optax: Optimization

Optax (<https://optax.readthedocs.io/>) is mature and largely stable. It is the standard optimizer library for JAX.

### 10.1 The core model: gradient transformations

An "optimizer" in Optax is a **gradient transformation** — a pair `(init_fn, update_fn)` where

```
update_fn(grads, state, params=None) -> (new_grads, new_state)
```

Everything composes:

```python
import optax

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),                 # grad clip
    optax.adamw(learning_rate=schedule, weight_decay=0.1, b1=0.9, b2=0.95),
)
opt_state = optimizer.init(params)
updates, opt_state = optimizer.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

### 10.2 Schedules

Schedules are `step → lr` functions. Modern recipes almost always combine warmup with cosine decay:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=2000,
    decay_steps=200_000,
    end_value=3e-5,
)
```

Other schedules: `linear_schedule`, `cosine_decay_schedule`, `exponential_decay`, `piecewise_constant_schedule`, `join_schedules`. Pass any callable as the `learning_rate` to use it.

### 10.3 The canonical AdamW + cosine + warmup + grad-clip recipe

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-4,
    warmup_steps=2_000, decay_steps=200_000, end_value=3e-5,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
        learning_rate=schedule,
        b1=0.9, b2=0.95, eps=1e-8,
        weight_decay=0.1,
        # only decay weight matrices, not biases / norms
        mask=lambda p: jax.tree.map(lambda x: x.ndim > 1, p),
    ),
)
```

### 10.4 Notable building blocks

- **Gradient accumulation:** `optax.MultiSteps(optimizer, every_k_schedule=k)` — virtually larger batch via accumulation.
- **EMA / Polyak averaging:** `optax.ema(decay=0.999)`.
- **`optax.contrib.schedule_free_adamw`:** Schedule-Free Adam (Defazio et al. 2024), popular for ablations because it removes the cosine schedule.
- **Lion, Adafactor, Sophia, Shampoo:** all available; Lion (`optax.lion`) has become a common LLM choice.
- **Multi-objective transforms:** `optax.multi_transform({'wd': adamw_with_wd, 'no_wd': adam}, param_labels)` — different optimizers per parameter group.
- **`optax.losses`:** `softmax_cross_entropy_with_integer_labels`, `sigmoid_binary_cross_entropy`, `cosine_similarity`, etc.

### 10.5 Adjacent libraries (one-liners)

- **chex** (<https://chex.readthedocs.io/>): testing utilities for JAX (`chex.assert_shape`, `chex.assert_trees_all_close`, fake `pmap`).
- **jaxtyping** (<https://docs.kidger.site/jaxtyping/>): named/typed array annotations, e.g. `Float[Array, "batch seq dim"]`. Now de-facto convention.
- **einops / einshape**: `rearrange(x, 'b n (h d) -> b h n d', h=heads)` — much clearer than `reshape + transpose`.
- **lineax**: JAX-native linear solvers (CG, GMRES, LU, QR), differentiable through the solve.
- **diffrax**: ODE/SDE/CDE solvers with adjoints, for neural ODEs / score-based diffusion.
- **MaxText** (<https://github.com/google/maxtext>): Google's reference open-source LLM training stack. Pure JAX/Flax NNX with `shard_map`.
- **Levanter** (<https://github.com/stanford-crfm/levanter>): Stanford's Haliax-based LLM trainer with named tensors and bit-exact reproducibility.
- **AXLearn** (<https://github.com/apple/axlearn>): Apple's open-source large-model training framework on JAX.
- **Tunix** (<https://github.com/google/tunix>): JAX-native post-training (SFT, RLHF, DPO/GRPO, LoRA), NNX-native.
- **Penzai** (<https://penzai.readthedocs.io/>): interpretability and model surgery — selectors, named axes, the `treescope` pretty-printer.
- **kfac-jax**: K-FAC second-order optimizer.

---

## Chapter 11. Grain: Deterministic Data Pipelines

### 11.1 What it is and why

The first edition of this guide endorsed the old advice: "use `tensorflow_datasets` with `tf.data`, or PyTorch's `DataLoader`." Reasonable in 2022; obsolete now. The official JAX stack diagram positions **Grain** (<https://github.com/google/grain>) as the canonical Layer 1 — designed around JAX's distribution model, with no TF dependency.

The single design contract that distinguishes Grain from `tf.data` and PyTorch's `DataLoader` is **exact, bit-reproducible determinism keyed on a single integer index.** If you record "we are at global step 47,238," Grain can rehydrate the exact same batch on a fresh process, on different hardware, with a different number of hosts, as long as the underlying `DataSource` hasn't changed. This is what makes restart-from-checkpoint *correct* in distributed training.

### 11.2 Core concepts

Grain has five composable abstractions:

**`DataSource`** — a random-access store of records. Contract: `__len__` and `__getitem__(int) → record`. Built-ins: `grain.ArrayRecordDataSource`, `grain.RangeDataSource`, `grain.InMemoryDataSource`, `grain.sources.parquet.ParquetIterableDataSource`. Custom in 5 lines:

```python
class JsonlSource(grain.RandomAccessDataSource):
    def __init__(self, path):
        self._offsets = build_offset_index(path)
        self._fh = open(path, "rb")
    def __len__(self): return len(self._offsets)
    def __getitem__(self, i):
        self._fh.seek(self._offsets[i])
        return json.loads(self._fh.readline())
```

**`Sampler`** — given a `(record_index, epoch)` pair, returns `RecordMetadata(index, record_key, rng)` for the next item. The default `IndexSampler` produces a deterministic permutation per epoch from a base seed. Critically, the sampler is *the* source of randomness; nothing downstream needs to consult global RNG. *That is what makes the pipeline checkpointable as just an integer.*

**`Operations`** — pure functions over records or batches. `MapTransform`, `RandomMapTransform` (gets a per-record `np.random.Generator` derived from `(seed, index)`), `FilterTransform`, `Batch`. Compose as a list:

```python
ops = [
    Tokenize(vocab),
    grain.Batch(batch_size=128, drop_remainder=True),
    Pad(seq_len=2048),
]
```

**`DataLoader`** — orchestrator. Owns multiple worker processes, fans out indices, applies operations, emits batches. Workers are stateless per-record; all state lives in the sampler position.

```python
loader = grain.DataLoader(
    data_source=src,
    sampler=grain.IndexSampler(
        num_records=len(src),
        shuffle=True,
        seed=42,
        num_epochs=None,
        shard_options=grain.ShardOptions(
            shard_index=jax.process_index(),
            shard_count=jax.process_count(),
            drop_remainder=True),
    ),
    operations=ops,
    worker_count=8,
    worker_buffer_size=2,
)
```

**`RecordMetadata`** — small struct (`index`, `record_key`, `rng`) flowing alongside each record.

### 11.3 The determinism story

`tf.data` is deterministic *if* you avoid `tf.random` calls, set `reshuffle_each_iteration=False`, disable nondeterministic optimizations. PyTorch's `DataLoader` is deterministic if you set worker init correctly, pin generators, never call `random.random()` from workers. Both put determinism on the user.

Grain inverts this. The default is deterministic; non-determinism takes effort. The mechanism:

1. **Per-record indices.** The sampler emits `(global_record_index, epoch)`. That tuple plus the seed uniquely identifies the record's position in the universe.
2. **Seeded shuffling via permutations.** Grain generates a per-epoch permutation of $[0, N)$ using a counter-based PRNG (Philox-style) seeded by `(base_seed, epoch)`. There is no in-memory shuffle buffer; the $i$-th record of epoch $e$ is computed in $O(1)$.
3. **State = one integer.** `loader.checkpoint()` returns a small dict containing the next index per shard. `loader.from_checkpoint(state)` restores. No fast-forwarding.

This matters for **straggler-recovery**: if one host crashes at step 100,000 in a 64-host job, you restart and every host resumes at exactly the same record stream they would have produced. With a stateful shuffle buffer (tf.data) this is not generally possible.

### 11.4 Multi-host pipelines

```python
shard_opts = grain.ShardOptions(
    shard_index=jax.process_index(),
    shard_count=jax.process_count(),
    drop_remainder=True,
)
```

Grain partitions the index space $[0, N)$ into `shard_count` ranges and only emits indices in this host's range. The *global* shuffle permutation is still computed; sharding selects a subset. Change `shard_count` between runs (resume on fewer hosts) and you still cover every record exactly once per epoch.

For data parallelism with global batch size $B$ across $H$ hosts, each host's loader is configured with `per_host_batch_size = B / H`, and you concatenate per-host batches into a globally sharded `jax.Array`:

```python
global_batch = jax.make_array_from_process_local_data(
    sharding=NamedSharding(mesh, P("data")),
    local_data=local_batch,
    global_shape=(B, *feature_shape),
)
```

### 11.5 File formats

- **ArrayRecord** (`pip install array_record`) — the format Grain was designed for. Binary container with $O(1)$ random access, built on Riegeli compression. The spiritual successor to TFRecord, without the `tf.train.Example` proto layer. Convert from TFRecord in one line: `array_record.tfrecord_to_array_record(in_path, out_path)`.
- **TFRecord** — supported sequentially, but you give up $O(1)$ access (no checkpointable shuffling).
- **Parquet** — `grain.sources.parquet.ParquetIterableDataSource`. Random access at row-group granularity.
- **JSONL** — DIY with offset indexing.
- **WebDataset / tar shards** — community sources exist.

Rule of thumb: if you control the data, convert to ArrayRecord once.

### 11.6 Comparison

| Property | Grain | tf.data | PyTorch DataLoader |
| --- | --- | --- | --- |
| Backend deps | None | TF | PyTorch |
| Output type | NumPy / `jax.Array` | `tf.Tensor` (needs conv) | `torch.Tensor` (needs conv) |
| Determinism by default | Yes | No (opt-in, fragile) | No (opt-in, fragile) |
| Checkpointable state | One int per shard | Iterator save (heavy) | None built-in |
| Multi-host sharding | First-class | Via `experimental_distribute_dataset` | Via `DistributedSampler` |
| Random access required | Yes | No | Recommended |

**When to use which:**
- **Default for new projects: Grain.**
- **tf.data:** if you already have a `tf.data` pipeline you don't want to rewrite, or need TF-specific ops. Feed to JAX via `tfds.as_numpy()`.
- **PyTorch DataLoader:** when your data lives behind PyTorch-native readers (HF `datasets` streaming, `torchvision`, `torchaudio`) and rewriting is more expensive than the determinism cost. Use a `collate_fn` that returns NumPy arrays.

---

## Chapter 12. Orbax: Sharded, Async Checkpointing

### 12.1 What it is

Orbax (<https://orbax.readthedocs.io/>) is the canonical JAX persistence library, replacing the old `flax.serialization` / `msgpack` path. Two products under one umbrella:

- **`orbax.checkpoint`** — general-purpose checkpointing of arbitrary PyTrees containing `jax.Array`s, NumPy arrays, Python scalars, optimizer state, metadata.
- **`orbax.export`** — export a JAX function to TensorFlow SavedModel for serving. Out of scope here.

### 12.2 The `CheckpointManager` API

The high-level entry point. Owns a directory, manages step-numbered subdirectories, applies retention policy, provides async writes:

```python
import orbax.checkpoint as ocp

options = ocp.CheckpointManagerOptions(
    max_to_keep=3,
    save_interval_steps=1000,
    keep_period=10000,
    create=True,
    cleanup_tmp_directories=True,
    enable_async_checkpointing=True,
)

mgr = ocp.CheckpointManager(
    directory="gs://my-bucket/runs/run42",
    options=options,
    item_names=("state", "data_iter", "config"),
)

# In the training loop
if mgr.should_save(step):
    mgr.save(step, args=ocp.args.Composite(
        state=ocp.args.StandardSave(state),
        data_iter=ocp.args.JsonSave(loader.checkpoint()),
        config=ocp.args.JsonSave(cfg_dict),
    ))

mgr.wait_until_finished()  # before resume / shutdown
```

### 12.3 Sharded checkpointing — the killer feature

When a `state` has leaves that are `jax.Array`s with `NamedSharding`, naive `pickle`/`msgpack` would `device_get` the whole thing onto host 0 — OOM for any model that doesn't fit in one host's RAM.

Orbax's `StandardSave` (or `PyTreeCheckpointHandler`) uses **TensorStore** under the hood. Each leaf is written as a TensorStore Zarr/N5 array; **each host writes the slices it owns directly to the storage backend** (GCS, local FS, S3) in parallel. There is *never* a global gather.

On restore, each host reads only the slices its devices need. The reshape is automatic because Zarr supports arbitrary chunked reads. This is what enables a topology change:

```python
# Trained on 64 chips, mesh=Mesh(64, ('model',))
# Resuming on 32 chips, mesh=Mesh(32, ('model',))

abstract_state = jax.eval_shape(init_fn, jax.random.key(0))
abstract_state = jax.tree.map(
    lambda s: jax.ShapeDtypeStruct(
        s.shape, s.dtype, sharding=NamedSharding(new_mesh, P('model'))),
    abstract_state,
)

restored = mgr.restore(
    step=mgr.latest_step(),
    args=ocp.args.Composite(
        state=ocp.args.StandardRestore(abstract_state),
        data_iter=ocp.args.JsonRestore(),
    ),
)
state = restored.state
```

The `abstract_state` (PyTree of `ShapeDtypeStruct` with `sharding=` populated) is the contract: "give me back arrays of these shapes/dtypes laid out under this new sharding." Orbax handles the chunk math.

### 12.4 Async checkpointing

Synchronous saves block training. For a 70 B model, that's 30–60 seconds of dead time per checkpoint. Async saves do the device→host transfer synchronously (cheap, ms) and the host→storage write in a background thread (slow, seconds).

Mechanics:

1. `mgr.save(step, ...)` snapshots device buffers to host pinned memory (sync, ~ms).
2. A background thread writes to TensorStore (async, seconds).
3. The directory is renamed to its final `<step>` name only after the write completes — atomic durability.
4. `mgr.wait_until_finished()` blocks until pending writes drain.

Don't call `save` faster than your write bandwidth — `save_interval_steps` enforces a minimum gap.

### 12.5 Integrating with TrainState (Linen) and NNX

**Linen / `flax.training.TrainState`:**

```python
from flax.training import train_state
import optax

state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=optax.adamw(3e-4))

mgr.save(step, args=ocp.args.Composite(
    state=ocp.args.StandardSave(state)))

abstract = jax.eval_shape(
    lambda: train_state.TrainState.create(
        apply_fn=model.apply, params=init_params, tx=optax.adamw(3e-4)))
state = mgr.restore(step, args=ocp.args.Composite(
    state=ocp.args.StandardRestore(abstract))).state
```

**NNX:**

```python
from flax import nnx

abstract_model = nnx.eval_shape(lambda: MyModel(rngs=nnx.Rngs(0)))
graphdef, abstract_state = nnx.split(abstract_model)
restored_state = mgr.restore(step,
    args=ocp.args.StandardRestore(abstract_state))
model = nnx.merge(graphdef, restored_state)
```

### 12.6 Failure-mode tour

- **Skipping determinism (use `tf.data` shuffle buffer, restart):** at restart the buffer warms from a new offset; you replay records you already trained on for ~`buffer_size` steps and skip records you should have seen. Loss curves spike. *Fix:* Grain.
- **Non-sharded checkpoint on a 70 B model across 64 hosts:** host 0 OOMs. *Fix:* Orbax `StandardSave`.
- **Synchronous checkpoint writes:** every `save_interval_steps`, the train step blocks. On a 405 B model with 1.6 TB of state and 2 GB/s GCS bandwidth, that's ~13 minutes of idle TPU per checkpoint. *Fix:* `enable_async_checkpointing=True`.
- **Restoring on a different topology with `pickle`:** saved arrays have committed shardings baked into their device-memory layout. *Fix:* Orbax + abstract `ShapeDtypeStruct`.
- **Forgetting to checkpoint the data iterator:** model state restores at step 47,238; loader starts from index 0. *Fix:* include `loader.checkpoint()` in the `Composite` save.
- **Calling `mgr.save` from only host 0:** Orbax's distributed save coordinates across hosts; calling from one hangs the others. *Fix:* every host calls `mgr.save`.
- **`worker_count=0` in Grain:** loader runs in main process; train step blocks on data prep. *Fix:* `worker_count = 4 × num_local_devices`, `worker_buffer_size=2`.

---

## Chapter 13. The End-to-End Training Pipeline

We synthesize Chapters 8–12 into a complete, modernized end-to-end skeleton. Replace the CIFAR-10 + Linen pipeline of the first edition.

### 13.1 The skeleton

```python
import jax, jax.numpy as jnp, numpy as np, optax
import grain.python as grain
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.distributed.initialize()                 # no-op on single host
devices = np.array(jax.devices()).reshape(-1, 1)
mesh = Mesh(devices, ('data', 'model'))

# ---- 1. Data: Grain ----
src = grain.ArrayRecordDataSource(DATA_GLOB)
sampler = grain.IndexSampler(
    num_records=len(src), shuffle=True, seed=SEED, num_epochs=None,
    shard_options=grain.ShardOptions(
        jax.process_index(), jax.process_count(), True))
loader = grain.DataLoader(
    data_source=src, sampler=sampler,
    operations=[Decode(), grain.Batch(LOCAL_BS, True)],
    worker_count=8)

# ---- 2. Model + state ----
model = MyTransformer(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(3e-4))
graphdef, opt_state = nnx.split(optimizer)

# ---- 3. Checkpoint manager ----
mgr = ocp.CheckpointManager(
    CKPT_DIR,
    options=ocp.CheckpointManagerOptions(
        max_to_keep=3, save_interval_steps=1000,
        enable_async_checkpointing=True),
    item_names=('state', 'data_iter'))

# ---- 4. Resume if available ----
start_step = 0
loader_iter = iter(loader)
if mgr.latest_step() is not None:
    abstract_state = nnx.eval_shape(lambda: nnx.split(
        nnx.Optimizer(MyTransformer(rngs=nnx.Rngs(0)),
                      optax.adamw(3e-4)))[1])
    abstract_state = jax.tree.map(
        lambda s: jax.ShapeDtypeStruct(
            s.shape, s.dtype, sharding=NamedSharding(mesh, P())),
        abstract_state)
    restored = mgr.restore(mgr.latest_step(),
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            data_iter=ocp.args.JsonRestore()))
    opt_state = restored.state
    loader = grain.DataLoader.from_checkpoint(loader, restored.data_iter)
    loader_iter = iter(loader)
    start_step = mgr.latest_step() + 1

# ---- 5. Train step ----
@jax.jit
def train_step(opt_state, batch):
    optimizer = nnx.merge(graphdef, opt_state)
    def loss_fn(model):
        logits = model(batch['x'])
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['y']).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    return nnx.split(optimizer)[1], loss

# ---- 6. Loop ----
for step in range(start_step, NUM_STEPS):
    local_batch = next(loader_iter)
    global_batch = jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(
            NamedSharding(mesh, P('data')), x,
            (GLOBAL_BS, *x.shape[1:])), local_batch)
    opt_state, loss = train_step(opt_state, global_batch)
    if mgr.should_save(step):
        mgr.save(step, args=ocp.args.Composite(
            state=ocp.args.StandardSave(opt_state),
            data_iter=ocp.args.JsonSave(loader.checkpoint())))

mgr.wait_until_finished()
mgr.close()
```

The shape of this skeleton *is* the canonical 2026 JAX training loop: **Grain → sharded `jit` step → Orbax**. Every line earns its place; every primitive composes deterministically with the others. We will spend Part IV understanding the sharding lines (`Mesh`, `NamedSharding`, `make_array_from_process_local_data`) in detail.

### 13.2 What's still missing (and why)

- **Mixed precision.** We assume bf16 throughout the model; the loss may need fp32 promotion. Chapter 24 covers mixed precision in detail.
- **Gradient checkpointing.** For models that don't fit in HBM with full activations, wrap the layer with `nnx.remat` (or `jax.checkpoint`). Chapter 25 covers this in the parallelism context.
- **Logging and eval.** Both run-of-the-mill — write to TensorBoard / wandb every $k$ steps; run a separate eval step on a held-out shard. Not covered in depth here.
- **Multi-host details.** This skeleton works on multi-host; the only host-aware lines are `jax.distributed.initialize()` and the sharding bits. Chapter 17 is the multi-host deep dive.

---

# Part IV — Sharded Computation

JAX's parallelism story has matured into a coherent three-level spectrum:

1. **Auto / compiler-driven** — `jit` + `NamedSharding` + GSPMD. You declare data layouts; the compiler picks collectives.
2. **Explicit / typed-sharding** — the same, but with mesh axes typed `AxisType.Explicit` so shardings become part of each `jax.Array`'s static type.
3. **Manual SPMD** — `shard_map`. You write per-shard code and call collectives explicitly.

The legacy fourth level — `pmap` — exists, works for tutorials, but is no longer where new code starts.

This part teaches each level, the math behind sharded matmul (with the four cases), and the multi-host story that ties them all together.

---

## Chapter 14. Mesh, Sharding, NamedSharding, PartitionSpec

### 14.1 The `Mesh`

A `jax.sharding.Mesh` is a logical N-D grid of physical devices, with named axes:

```python
import numpy as np
from jax.sharding import Mesh

devices = np.array(jax.devices()).reshape(2, 4)
mesh = Mesh(devices, axis_names=('data', 'model'))
```

The `axis_names` are the vocabulary you use when describing shardings. A 1-D mesh `Mesh(devices, ('data',))` is appropriate for pure data parallelism / FSDP; a 2-D mesh adds a model-parallel axis; a 3-D mesh adds pipeline. Modern practice rarely goes above 4 axes (`('data', 'fsdp', 'tensor', 'expert')` for MoE) — the compiler can express anything you want, but readability suffers.

Newer JAX exposes `jax.make_mesh(axis_shapes, axis_names)` as a shortcut.

### 14.2 `PartitionSpec`

A `jax.sharding.PartitionSpec` (idiomatically `P`) describes how a single array's dimensions map onto the mesh axes. It is a tuple of length `array.ndim`:

```python
from jax.sharding import PartitionSpec as P

P('data', None)               # axis 0 sharded over 'data', axis 1 replicated
P(None, 'model')              # axis 1 sharded over 'model', axis 0 replicated
P('data', 'model')            # axis 0 over 'data', axis 1 over 'model'
P(('data', 'fsdp'), None)     # axis 0 sharded over BOTH 'data' and 'fsdp'
P()                           # fully replicated (rank-0 spec, ok for scalars; for higher rank,
                              # use P(None, None, ...) of the right length)
```

`None` means "replicated across this mesh axis." A name means "sharded across this mesh axis." A tuple of names means "sharded jointly across multiple mesh axes."

### 14.3 `NamedSharding` and `device_put`

A `NamedSharding(mesh, spec)` ties a `PartitionSpec` to a `Mesh`. `jax.device_put(x, sharding)` distributes the array accordingly:

```python
from jax.sharding import NamedSharding

mesh = Mesh(np.array(jax.devices()).reshape(8), ('data',))
x = jnp.arange(1024).reshape(128, 8)
x_sharded = jax.device_put(x, NamedSharding(mesh, P('data', None)))

# Visualize
jax.debug.visualize_array_sharding(x_sharded)
```

### 14.4 The book's notation: subscripted axes

The "How to Scale Your Model" book uses a compact notation that's worth borrowing for thinking. An array `A[I, J]` of logical shape `(I, J)`, sharded with subscripts:

- $A[I_X, J]$ — axis $I$ sharded over mesh axis $X$, axis $J$ replicated.
- $A[I_X, J_Y]$ — axes sharded over $X$ and $Y$ respectively.
- $A[I_{XY}, J]$ — axis $I$ sharded over the joint $X \times Y$ mesh axes.
- $A[I, J]$ with no subscripts — fully replicated.

This is just shorthand for `PartitionSpec`s, but it's easier to read in equations involving multiple sharded operands.

### 14.5 `shard_map`-style mental model

When an array `x` is sharded `P('data', None)` across an 8-device mesh `('data',)`, *outside* sharded-aware regions:
- `x.shape == (128, 8)` (the global shape).
- The compiler treats it as a logical global array.
- Inside `jit`, GSPMD partitions ops over its actual layout.

*Inside* a `shard_map(in_specs=P('data', None))` region:
- `x.shape == (16, 8)` (the local-per-shard shape).
- You write code as if you are one device.
- Cross-device data needs explicit collectives.

This is the single most important mental model in JAX parallelism. Get it once and the rest is bookkeeping.

---

## Chapter 15. Auto-Sharded `jit` (GSPMD)

### 15.1 The "computation follows data" principle

Modern JAX shifts the responsibility of orchestrating parallel execution from the user to the compiler. You declare the desired layout of inputs and outputs; the JIT compiler analyzes the global graph and partitions the computation, inserting communication collectives as needed. This is sometimes called **"computation follows data."**

The essence:

```python
@jax.jit
def matmul(x, w):
    return x @ w

x = jax.device_put(x_global, NamedSharding(mesh, P('data', None)))
w = jax.device_put(w_global, NamedSharding(mesh, P(None, 'model')))
y = matmul(x, w)   # GSPMD picks collectives; result is sharded P('data', 'model')
```

GSPMD ("General and Scalable Parallelization for ML Computation Graphs") is the algorithm in XLA that performs this propagation. It also powers TF's mesh, and its newer evolution **Shardy** is becoming the default.

### 15.2 Sharding constraints: tickling the compiler

For intermediate tensors, you can pin a sharding with `jax.lax.with_sharding_constraint`:

```python
def layer(x, w1, w2):
    h = x @ w1
    h = jax.lax.with_sharding_constraint(h, NamedSharding(mesh, P('data', 'model')))
    return h @ w2
```

This is a **hint**, not a hard requirement — GSPMD may still reshard around it if cheaper. But it lets you guide the compiler when its automatic choices are suboptimal.

### 15.3 FSDP in 6 lines

The canonical 2026 idiom for FSDP is striking in its simplicity:

```python
mesh = Mesh(jax.devices(), ('data',))
param_sharding = jax.tree.map(
    lambda p: NamedSharding(mesh, P('data')), params)
params = jax.device_put(params, param_sharding)
batch_sharding = NamedSharding(mesh, P('data'))

@jax.jit
def step(params, opt_state, batch):
    ...   # standard single-device step
```

That's the entire FSDP setup. The compiler inserts `AllGather` for the forward, `ReduceScatter` for the backward, automatically. Compared to PyTorch's FSDP wrapper, the contrast is striking.

### 15.4 Explicit sharding mode (typed shardings)

The most important conceptual change in JAX parallelism since `shard_map`: shardings can become part of each `jax.Array`'s **type**.

```python
import jax.sharding as shd

mesh = jax.make_mesh(
    axis_shapes=(2, 2), axis_names=('X', 'Y'),
    axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16).reshape(8, 2), P('X', 'Y'))

@jax.jit
def f(x):
    print(jax.typeof(x))      # f32[8@X, 2@Y] — sharding visible as type
    return jnp.square(x)
```

When two operands have incompatible shardings, you get a Python-level error at trace time rather than a silent reshard. For matmuls / einsums where the output sharding is ambiguous, you specify it explicitly:

```python
return jnp.einsum('bd,df->bf', jnp.square(x), w,
                  out_sharding=P('X', 'Y'))
```

Why this matters:

1. Sharding stops being a compiler-side concern and becomes a first-class part of the program's semantics.
2. Eliminates a class of "the compiler did something I didn't expect" bugs.
3. Makes `shard_map` boundaries crisp: explicit-mode arrays at the boundary have a typed sharding that must match `in_specs` / `out_specs`.

Status (May 2026): Explicit mode is the recommended path for new code in the official tutorials, but Auto mode remains the default for backward compatibility and is fully supported. Some advanced features (sharding rules for custom primitives, certain dynamic shapes) are still better-supported in Auto mode.

---

## Chapter 16. Manual SPMD with `shard_map`

### 16.1 API

```python
from jax.experimental.shard_map import shard_map  # also available as jax.shard_map

shard_map(f, mesh, in_specs, out_specs, check_rep=True, auto=frozenset())
```

- `mesh`: a `Mesh` (or `AbstractMesh`).
- `in_specs`, `out_specs`: PyTree-structured `PartitionSpec`s describing how each input/output is sharded.
- `check_rep`: enables static replication-tracking analysis to verify `out_specs` claiming replication.
- `auto`: a frozenset of mesh axes that should remain in *auto* mode — lets `shard_map` be a *partial* manual region.

### 16.2 The mental flip

**Outside `shard_map`** with input sharded `P('data', None)` across 8 devices: the array's `.shape` is `(1024, D)`, the compiler partitions ops for you.

**Inside `shard_map`** with `in_specs=P('data', None)`: the same value arrives in your function with shape `(128, D)` — the local per-shard shape. You write code as if on one device, calling collectives explicitly when you need data from peers.

Same mental model as MPI/NCCL programming, but with traceable JAX ops and `jit` compilation.

### 16.3 Worked example: tensor-parallel matmul, auto vs manual

```python
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial
import numpy as np

devices = jax.devices()    # assume 8
mesh = Mesh(np.array(devices).reshape(2, 4), ('data', 'model'))

B, D, F = 1024, 4096, 4096
x = jnp.ones((B, D))
w = jnp.ones((D, F))

# ---- AUTO ----
xs = NamedSharding(mesh, P('data', None))
ws = NamedSharding(mesh, P(None, 'model'))

@jax.jit
def auto_tp_matmul(x, w):
    return x @ w

y_auto = auto_tp_matmul(jax.device_put(x, xs), jax.device_put(w, ws))
# Output is sharded (data, model). GSPMD picked collectives automatically.

# ---- MANUAL (shard_map) ----
@jax.jit
def manual_tp_matmul(x, w):
    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None), P(None, 'model')),
             out_specs=P('data', 'model'))
    def kernel(x_local, w_local):
        # x_local: (B/2, D); w_local: (D, F/4)
        return x_local @ w_local        # contracting dim NOT sharded → no collective
    return kernel(x, w)

y_manual = manual_tp_matmul(jax.device_put(x, xs), jax.device_put(w, ws))
```

Same answer. The auto version trusts GSPMD; the manual version is unambiguous. Switch `w` to `P('model', None)` (shard the *contracting* axis) and the manual version requires an explicit `psum('model')`:

```python
@partial(shard_map, mesh=mesh,
         in_specs=(P('data', None), P('model', None)),
         out_specs=P('data', None),
         check_rep=False)
def tp_matmul_contract(x, w):
    # w sharded on D → each device computes a partial sum, then psum
    partial_sum = x @ w
    return jax.lax.psum(partial_sum, 'model')
```

### 16.4 Common patterns

**Manual ring all-reduce** (the textbook example of `ppermute`):

```python
@partial(shard_map, mesh=mesh, in_specs=P('data'), out_specs=P('data'))
def ring_allreduce(x):
    n = jax.lax.axis_size('data')
    perm = [(i, (i + 1) % n) for i in range(n)]
    acc = x
    for _ in range(n - 1):
        x = jax.lax.ppermute(x, 'data', perm)
        acc = acc + x
    return acc   # equivalent to jax.lax.psum(x, 'data') but written explicitly
```

**Expert all-to-all (MoE)** — the canonical case where `shard_map` shines and auto-mode struggles:

```python
@partial(shard_map, mesh=mesh,
         in_specs=(P('expert', None), P('expert', None)),
         out_specs=P('expert', None))
def moe_dispatch(tokens, routes):
    # Reorder local tokens by destination expert, then all_to_all.
    sent = reorder_by_destination(tokens, routes)
    return jax.lax.all_to_all(sent, 'expert',
                              split_axis=0, concat_axis=0, tiled=True)
```

**Custom collective overlap (collective matmul):**

You can explicitly overlap an `all_gather` with a matmul by chunking the work:

```python
@partial(shard_map, mesh=mesh, in_specs=(P('data', None), P('model', None)),
         out_specs=P('data', None))
def collective_matmul(x, w):
    n = jax.lax.axis_size('model')
    perm = [(i, (i + 1) % n) for i in range(n)]
    acc = x @ w   # local part
    # Pipeline: each iteration shifts w one step around the ring,
    # accumulating into acc, while previous matmul is still finishing.
    for _ in range(n - 1):
        w = jax.lax.ppermute(w, 'model', perm)
        acc = acc + x @ w
    return acc
```

The book reports this kind of overlap closes the gap between sharded and unsharded matmul: ~244 µs (overlapped) vs 311 µs (blocking AllGather + matmul) vs 224 µs (unsharded baseline).

### 16.5 `check_rep` replication tracking

`shard_map` does some type-level reasoning to track which mesh axes each intermediate is **replicated** across. If you say `out_specs=P(None, 'model')` (axis 0 replicated across `model`), `check_rep` verifies that the value you actually return is provably replicated along `model`. Mismatches fail at trace time:

```
out_specs claims replication over axis 'model' but the value is not provably replicated
```

The fix is usually to insert a `jax.lax.psum(x, 'model')` (forcing replication) or to relax the spec.

`check_rep=False` disables the analysis when you're sure the partial-replication claim is correct and the analysis is too conservative.

### 16.6 Decision tree

```
Q: Do you have a single-device working model and want to scale it?
├── Q: Standard data parallelism / FSDP, no exotic comms?
│   └── jit + NamedSharding. Param sharding P('data') (FSDP) or replicate (DP).
│       Shard the batch on the data axis. Done.
│
├── Q: TP + FSDP (transformer at scale)?
│   └── 2D mesh ('data', 'model'). Param spec P('model', None) or
│       P(None, 'model') depending on layer; activation spec P('data', 'model')
│       on the relevant axis. Still jit + NamedSharding.
│
├── Q: 3D parallelism with pipeline?
│   └── Mostly jit + NamedSharding for DP/TP, with shard_map regions for
│       pipeline stage boundaries (explicit send/recv via ppermute).
│       Or use a library (MaxText, Levanter, AXLearn).
│
├── Q: MoE with expert parallelism?
│   └── shard_map. The all_to_all pattern is the canonical case for
│       manual SPMD; auto mode struggles to express it cleanly.
│
├── Q: Custom collective (ring-attention, sequence parallelism with
│     overlapping comms, ZeRO++ variants)?
│   └── shard_map. You want full control over the schedule.
│
├── Q: Legacy pmap code?
│   └── Migrate to shard_map. Mechanical translation:
│       pmap(f, axis_name='i') with in_axes=0  ↔
│       shard_map(f, mesh, in_specs=P('i'), out_specs=P('i'))
│
└── Q: Single-host tutorial / quickstart?
    └── jit + NamedSharding. There is no longer a reason to start with pmap.
```

### 16.7 Migration recipe (pmap → shard_map)

```python
# Before:
@partial(jax.pmap, axis_name='i')
def f(x):
    return jax.lax.pmean(x ** 2, 'i')

x = jnp.arange(8 * 16).reshape(8, 16)   # leading axis = device
y = f(x)

# After:
mesh = Mesh(jax.devices(), ('i',))

@jax.jit
@partial(shard_map, mesh=mesh, in_specs=P('i', None), out_specs=P(None, None))
def f(x):
    return jax.lax.pmean(x ** 2, 'i')

x = jax.device_put(jnp.arange(8 * 16).reshape(8, 16),
                   NamedSharding(mesh, P('i', None)))
y = f(x)
```

The collectives are identical; sharding is explicit; the array is a normal `jax.Array` with shape `(128, 16)` rather than a leading-device-axis special case.

---

## Chapter 17. Multi-Host Training

JAX's multi-host model is fundamentally simple once the sharding picture is in place: every host holds the **shards of `jax.Array`s on that host's local devices**, and `jit` + `shard_map` do the cross-host communication via the same collectives as cross-device on a single host.

### 17.1 `jax.distributed.initialize`

Every process calls this exactly once before any other JAX work:

```python
import jax
jax.distributed.initialize(
    coordinator_address="10.0.0.1:1234",
    num_processes=8,
    process_id=int(os.environ["PROCESS_ID"]),
)
```

On Cloud TPU and SLURM/MPI clusters, `jax.distributed.initialize()` (no args) auto-detects the environment. On Kubernetes/GKE, pass the coordinator address explicitly.

After `initialize`:

- `jax.process_count()` — total number of hosts.
- `jax.process_index()` — this host's rank (0..N-1).
- `jax.devices()` — **global** device list (all devices on all hosts).
- `jax.local_devices()` — only devices physically attached to this host.
- `jax.local_device_count()` — number of accelerators on this host.

### 17.2 The global view: `jax.Array` is the global addressable thing

The single most important thing to internalize about multi-host JAX, and where it differs sharply from PyTorch DDP / TF MirroredStrategy:

- A `jax.Array` is a **logical, global** array that may be physically distributed across devices on multiple hosts.
- Each host's process **only stores the shards on its local devices**; it can address the rest of the array logically (participate in collectives) but cannot directly read remote bytes.
- Inside a `jit`, every host runs the *same* program. Because shardings are part of the type signature, the compiler produces a single XLA HLO partitioned across all devices on all hosts. Hosts do not need to coordinate Python execution — they just need to all hit the same `jit` boundaries with consistent shardings.
- `arr.addressable_shards` — local shards.
- `arr.global_shards` — conceptual list of all shards (with metadata for non-local ones).
- `np.asarray(arr)` raises if the array is not fully addressable on this host. Use `jax.experimental.multihost_utils.process_allgather(arr)` to bring a global array to every host (expensive — for logging only).

### 17.3 Multi-host data ingestion

Each host needs to feed *its own* shards into the global `jax.Array`. The right pattern is `jax.make_array_from_process_local_data`:

```python
batch_sharding = NamedSharding(global_mesh, P('data'))

def get_local_batch():
    # This host's slice of the global batch.
    # Shape: (per_host_batch, ...); per_host_batch = global_batch / num_hosts
    return ...

local = get_local_batch()
global_batch = jax.make_array_from_process_local_data(
    sharding=batch_sharding,
    local_data=local,
    global_shape=(global_batch_size,) + local.shape[1:],
)
```

Each host produces the slice its devices are responsible for, and `make_array_from_process_local_data` stitches the global `jax.Array` together without cross-host data movement.

For input pipelines, the recommended library is **Grain** (Chapter 11) with `ShardOptions(shard_index=jax.process_index(), shard_count=jax.process_count())`.

### 17.4 What can go wrong

- **Different Python decisions on different hosts.** A `jit` compiled on host 0 with one sharding and on host 1 with another: hang. Every host must execute the same Python path leading to the same `jit` boundary. If you do conditional logic, condition only on values every host agrees on (`jax.process_index() == 0` is fine for "log on host 0 only," but it's *not* fine for "skip a step on host 0").
- **Accidentally forcing a fully-addressable array.** `np.asarray(global_arr)` works fine on single host; on multi-host, only host 0 has the data needed. Use `process_allgather` if you really need it.
- **Per-host file paths.** `Orbax` writes correctly across hosts with the right setup (Section 12.3); don't manually `pickle.dump` per-host files for sharded state.

---

## Chapter 18. Sharded Matmul: The Four Cases

The matmul is the building block of every model. How it shards is the building block of every parallelism strategy. The book classifies four cases by which axes of the operands are sharded; each case has its own collective and its own cost.

Notation: let $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$, $C = A B \in \mathbb{R}^{m \times n}$. The contracting dim is $k$. Mesh axis names $X, Y$.

### 18.1 Case 1 — neither operand sharded on the contracting dim

$$A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$$

Each device has the full $J$ axis of both operands; the local matmul is correct as is. **No collective.** This is the cheapest case and what you get from FSDP-only or "outer" parallelism.

### 18.2 Case 2 — one operand sharded on the contracting dim

$$A[I, J_X] \cdot B[J, K] \to ?$$

Each device has only a slice of the contracting axis on $A$. Two strategies:

**(a) AllGather $A$ first, then matmul.** Reassemble $A[I, J]$ on every device, then run the local matmul. Cost: $\frac{N-1}{N} \cdot |A| \approx |A|$ bytes per device.

**(b) Local matmul, then AllReduce.** Each device computes $A[I, J_X^{(d)}] \cdot B[J^{(d)}, K]$ (its partial sum). AllReduce across $X$ to combine. Cost: $2 \cdot |C| \cdot \frac{N-1}{N} \approx 2 |C|$ bytes per device.

Which is cheaper depends on the relative sizes of $A$ and $C$.

### 18.3 Case 3 — both operands sharded on the contracting dim

$$A[I, J_X] \cdot B[J_X, K] \to C[I, K]\{U_X\}$$

Each device has matched slices of $J_X$. The local matmul produces a partial sum (the book's notation: $\{U_X\}$ marks "unreduced over $X$"). **AllReduce** across $X$ to finish. Cost: $2 |C| \cdot \frac{N-1}{N}$.

This is the canonical *tensor parallel* layer: row-parallel on the first matmul (Case 3), column-parallel on the second (Case 1), and you recover an $A \cdot W_1 \cdot W_2$ result with one AllReduce per layer.

### 18.4 Case 4 — both operands sharded on the same non-contracting dim

$$A[I_X, J] \cdot B[J, K_X] \to ?$$

This is invalid as written — you can't combine an $A$ shard's row of $C$ with the $B$ shard's column without first gathering one of them. AllGather one operand (Case 2), then proceed.

### 18.5 The collective costs (book equations, verbatim)

For ring algorithms on $N$ devices with payload $V$ bytes and per-link bandwidth $W_{\text{ici}}$ B/s:

$$T_{\text{AllGather}} = \frac{V (N - 1)}{N \cdot W_{\text{ici}}}$$

$$T_{\text{ReduceScatter}} = \frac{V (N - 1)}{N \cdot W_{\text{ici}}}$$

$$T_{\text{AllReduce}} = 2 \cdot \frac{V (N - 1)}{N \cdot W_{\text{ici}}} \approx \frac{2 V}{W_{\text{ici}}}$$

For an $A \times B \times C$ mesh AllToAll: $T \approx \frac{V \max(A,B,C)}{4 N W}$.

The asymptotic forms ($N \to \infty$) are what you keep in your head.

### 18.6 The asymmetry: communication doesn't grow with $N$

For a ring all-reduce in the bandwidth-bound regime, the time is

$$T_{\text{AllReduce}} \approx \frac{2V}{W_{\text{ici}}}$$

— independent of $N$! That is the secret behind ring algorithms scaling. The book calls this out explicitly: "when performing an AllGather (or ReduceScatter or AllReduce) in a throughput-bound regime, the actual communication time depends only on the size of the array and the available bandwidth, not the number of devices."

The latency floor does grow with $N$ (it's $\sim N \alpha$ for per-hop latency $\alpha$), so for tiny payloads you do see scaling overhead. Bucket your gradients to live in the bandwidth-bound regime.

### 18.7 Worked example (book): AllGather on TPU v5e

$E = 2048$, $F = 8192$, mesh $\{X: 8, Y: 4\}$, sharding $A[E_Y, F]$ in bf16.

- Each device holds $\text{bf16}[512, 8192] = 8.4$ MB.
- Total array size: $34$ MB.
- Communication time $\approx 34 \times 10^6 / 4.5 \times 10^{10} \approx 377\,\mu\text{s}$ (bandwidth-bound).
- For tiny arrays ($256 \times 256$), latency-bound at $\sim 3\,\mu\text{s}$ per hop.

### 18.8 What this means for your model

Every Transformer training step is a sequence of these four matmul cases, each with its own collective. The art of "designing a parallelism strategy" reduces to:

1. Pick a mesh.
2. Assign each weight matrix a `PartitionSpec`.
3. Compute the dominating collectives per layer.
4. Check that they're below the compute time so the chip stays compute-bound.

We do exactly this for a 70 B Transformer in Chapter 25.

---

# Part V — The Modern LLM Stack

The first edition of this guide implemented a vanilla decoder-only Transformer in Flax Linen and stopped there. Past that point, an enormous amount of accumulated craft is what separates "a Transformer that works" from "a Transformer that trains efficiently and serves with reasonable economics." This part walks through what a serious 2026 practitioner needs.

We start with notation and the operational FLOP/parameter math (Chapter 19). Then attention beyond the textbook — FlashAttention (20) and KV-cache management (21). Position embeddings (22). MoE (23). Numerics and stability (24). Training parallelism end-to-end with a concrete LLaMA-3-70B worked example (25). And inference, where everything you learned about training hits a different reality (26).

---

## Chapter 19. The Transformer at Scale

### 19.1 Notation (the book's, which we adopt)

A Transformer has many dimensions and using consistent letters for them is half the battle:

| Symbol | Meaning |
| --- | --- |
| $B$ | batch size |
| $T$ | sequence length (query side) |
| $S$ | sequence length (key/value side); $S = T$ in self-attention |
| $D$ | model dimension (`d_model`) |
| $F$ | FFN hidden dimension (typically $\sim 4D$) |
| $H$ | head dimension |
| $N$ | number of attention heads |
| $K$ | number of KV heads (for grouped-query attention; $K = N$ for MHA, $K=1$ for MQA) |
| $L$ | number of layers |
| $V$ | vocabulary size |

For a contraction of two arrays $C$ and $D$ where some dimensions are *contracting* and others are *batching*, the FLOP cost is **twice** the product of the array dimensions (counting batching/contracting only once).

### 19.2 Per-layer parameter count

A single Transformer block has:

- **Attention projections** (Q, K, V, O):
  - Q: $D \times N H$ params; K: $D \times K H$; V: $D \times K H$; O: $N H \times D$.
  - Total: $2 D (N + K) H$.
- **FFN with gating** (e.g. SwiGLU):
  - Up projection ($W_{\text{in}}$, $W_{\text{gate}}$): $2 D F$.
  - Down projection ($W_{\text{out}}$): $D F$.
  - Total: $3 D F$.
- **Layer norms**: $\sim D$ per norm, two per block.

So per layer:

$$
\text{params/layer} \approx 2 D (N + K) H + 3 D F
$$

For "standard" MHA + 4× FFN with $N = K$, $H = D/N$: per-layer params $\approx 4D^2 + 12 D^2 = 16 D^2$.

For LLaMA-3-70B with $D = 8192$, $L = 80$, $N = 64$, $K = 8$, $H = 128$, $F = 28672$:

- FFN per layer: $3 \cdot 8192 \cdot 28672 \approx 7.05 \times 10^8$ params.
- Attention per layer: $2 \cdot 8192 \cdot (64 + 8) \cdot 128 = 1.51 \times 10^8$ params.
- Per layer: $\approx 8.56 \times 10^8$.
- Total layers: $80 \cdot 8.56 \times 10^8 \approx 6.85 \times 10^{10}$.
- Vocab head ($V \approx 128256$): $2 \cdot 128256 \cdot 8192 = 2.1 \times 10^9$.
- **Total: $\approx 7.04 \times 10^{10}$ params** — the "70 B" rounds.

The MLP block dominates the parameter count. As long as $T < 8D$, the MLP also dominates the FLOPs budget — a fact we'll re-derive.

### 19.3 Per-layer FLOP count (training, forward + backward)

Backward of a matmul is two more matmuls (input grad + weight grad), so training FLOPs $\approx 3 \times$ forward FLOPs. The forward FLOP for $X[B, T, D] \cdot W[D, F] \to Y[B, T, F]$ is $2 B T D F$. With backward, $6 B T D F$.

Per layer, summing over QKV projection, attention scores, output, FFN:

$$
\text{FLOPs/layer/step} \approx 6 B T \big(2 D (N + K) H + 3 D F\big) + 12 B T^2 N H \cdot (\text{factor for backward})
$$

Or, more usably, the rule of thumb the book makes precise:

$$
\boxed{\text{Total training FLOPs per token} \approx 6 \cdot \text{params}}
$$

The factor of 6 comes from: 2× for the matmul itself, multiplied by 3× for forward + 2× backward (input/output grad).

For LLaMA-3-70B:
- FLOPs per token: $6 \cdot 7 \times 10^{10} = 4.2 \times 10^{11}$.
- Full training run on 15 T tokens: $4.2 \times 10^{11} \cdot 1.5 \times 10^{13} = 6.3 \times 10^{24}$ FLOPs.
- On 8960 v5p chips at 40% MFU: $6.3 \times 10^{24} / (8960 \cdot 4.59 \times 10^{14} \cdot 0.4) \approx 3.83 \times 10^6$ s $\approx 44$ days.

### 19.4 When does attention dominate?

Attention's FLOPs scale as $O(B T^2 N H)$, while FFN scales as $O(B T D F)$. The ratio:

$$
\frac{\text{attention FLOPs}}{\text{MLP FLOPs}} = \frac{12 B T^2 N H}{18 B T D F} = \frac{T \cdot N H}{1.5 \cdot D F}
$$

With $F = 4D$ and $N H = D$: $\frac{T \cdot D}{6 D^2} = T/(6D)$, which crosses 1 when $T \sim 6D$. The book gives a tighter estimate of $T > 8D$ for attention to dominate during training. For LLaMA-3 with $D = 8192$, that's $T > 65{,}536$ — still longer than typical pretraining context. Once you fine-tune for very long context (128 k+), attention dominates and you start to need exotic kernels and parallelism.

### 19.5 KV cache size

The KV cache stores keys and values for every position generated so far, every layer, every KV head:

$$
\boxed{\text{KV cache size} = 2 \cdot S \cdot L \cdot K \cdot H \cdot \text{bytes}_{\text{dtype}}}
$$

The leading 2 accounts for both K and V.

LLaMA-3-70B with $S = 8192$, $L = 80$, $K = 8$, $H = 128$, in bf16:

$$
2 \cdot 8192 \cdot 80 \cdot 8 \cdot 128 \cdot 2 = 2.68 \text{ GiB per request}.
$$

With $K = 64$ (full MHA, no GQA): 21.5 GiB per request — *enormous*. This is why GQA exists and why every modern LLM uses it.

### 19.6 Activation memory and gradient checkpointing

Naively storing all forward activations for backward is infeasible at scale. With ~20 intermediate nodes per layer that the autograd needs:

$$
\text{Activation memory} \approx 2 \cdot 20 \cdot B T D L \quad (\text{bf16})
$$

For $B \cdot T = 4$ M tokens, $L = 64$, $D = 8192$: $\approx 84$ TB. Way too much.

**Gradient checkpointing** (rematerialization): store only selected activations during forward; recompute the rest during backward. Two common strategies:

- **Block remat** — save only each layer's input. Recompute everything else. Cuts activation memory by ~5× (in the example above, to ~4.2 TB) at the cost of one extra forward pass during backward — i.e. $\sim 33\%$ extra FLOPs.
- **Big-matmul-only** — save matmul outputs (typically 7 per layer) but recompute small ops. Better trade-off for most cases.

In JAX, this is `jax.checkpoint` (alias `jax.remat`) or its NNX wrapper. Apply at layer granularity:

```python
class TransformerBlock(nnx.Module):
    ...

class CheckpointedBlock(nnx.Module):
    def __init__(self, ...):
        self.block = nnx.remat(TransformerBlock)(...)
    def __call__(self, x):
        return self.block(x)
```

---

## Chapter 20. FlashAttention: Online Softmax from First Principles

### 20.1 The problem

Standard scaled dot-product attention computes

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{H}}\right) V
$$

with $Q \in \mathbb{R}^{T \times H}$, $K, V \in \mathbb{R}^{S \times H}$.

Implemented naively: form $S = QK^\top \in \mathbb{R}^{T \times S}$, write to HBM, apply softmax (row-wise reduction), write to HBM, multiply by $V$, write out. Memory cost is $O(T S)$ for the attention matrix — quadratic in sequence length. More importantly, **bandwidth cost is $O(T S)$** — every byte of $S$ goes to HBM and comes back for the next op.

For long contexts this is the dominant cost: not the FLOPs (though they're quadratic too), but the *bandwidth*. The attention arithmetic intensity is

$$
\text{AI(attention)} \approx \frac{T \cdot S}{T + S}
$$

For prefill ($S = T$): $\text{AI} \approx T/2$. For decode ($T = 1$): $\text{AI} \approx 1$. We saw both in Chapter 4.

### 20.2 The insight

The key insight (Dao et al., FlashAttention 2022): **softmax can be computed incrementally**, so you never have to materialize the full $T \times S$ matrix. Process attention in tiles, keeping the running statistics needed to correctly normalize.

For a row of $S$, the standard softmax is

$$
\text{softmax}(s)_i = \frac{e^{s_i - m}}{\sum_j e^{s_j - m}}, \quad m = \max_j s_j
$$

(the $-m$ is the "max trick" for numerical stability).

Suppose we process the row in tiles. After tile $i$, we have running max $m^{(i)}$ and running sum $\ell^{(i)} = \sum_{j \in \text{tiles 1..i}} e^{s_j - m^{(i)}}$. When tile $i+1$ arrives with local max $\hat m$ and partial exponential sum $\hat \ell$:

$$
m^{(i+1)} = \max(m^{(i)}, \hat m)
$$
$$
\ell^{(i+1)} = e^{m^{(i)} - m^{(i+1)}} \cdot \ell^{(i)} + e^{\hat m - m^{(i+1)}} \cdot \hat \ell
$$

The accumulator $O$ updates the same way (with rescale-and-add):

$$
O^{(i+1)} = e^{m^{(i)} - m^{(i+1)}} \cdot O^{(i)} + e^{\hat m - m^{(i+1)}} \cdot \tilde O^{(i+1)}
$$

where $\tilde O^{(i+1)} = e^{s_{i+1} - \hat m} V_{i+1}$. After all tiles, divide by $\ell^{\text{final}}$ for the proper softmax denominator.

This gives **exact** softmax-attention output without ever forming the full $T \times S$ matrix. The whole kernel is a doubly-nested loop: outer over $Q$ tiles (size $B_q$), inner over $K, V$ tiles (size $B_k$), accumulating in fp32 registers, writing only $O$ and per-row log-sum-exp $\ell$ (used by backward).

### 20.3 The kernel structure (Pallas sketch)

```python
def fa2_kernel(q_ref, k_ref, v_ref, o_ref, lse_ref, *, BQ, BK, D):
    # Grid: (batch, num_heads, q_blocks). Each program handles ONE Q tile
    # for ONE (batch, head) pair, sweeping over ALL K tiles inside.

    q = q_ref[...]                                 # (BQ, D), in VMEM/SMEM
    m_i = jnp.full((BQ,), -jnp.inf)                # running row-max
    l_i = jnp.zeros((BQ,))                         # running normalizer
    acc = jnp.zeros((BQ, D), jnp.float32)          # running output

    num_k_blocks = k_ref.shape[0] // BK
    def body(k_idx, carry):
        m_i, l_i, acc = carry
        k = pl.load(k_ref, (pl.ds(k_idx * BK, BK), slice(None)))   # (BK, D)
        v = pl.load(v_ref, (pl.ds(k_idx * BK, BK), slice(None)))
        s = (q @ k.T) * (1.0 / jnp.sqrt(D))                        # (BQ, BK)
        s = jnp.where(causal_mask(k_idx), s, -jnp.inf)
        m_new = jnp.maximum(m_i, s.max(-1))
        p = jnp.exp(s - m_new[:, None])
        l_new = jnp.exp(m_i - m_new) * l_i + p.sum(-1)
        acc = acc * jnp.exp(m_i - m_new)[:, None] + p @ v
        return m_new, l_new, acc

    m_i, l_i, acc = jax.lax.fori_loop(0, num_k_blocks, body, (m_i, l_i, acc))
    o_ref[...] = (acc / l_i[:, None]).astype(o_ref.dtype)
    lse_ref[...] = m_i + jnp.log(l_i)              # save for backward
```

Key invariants:

- **$Q$ tile lives in registers** for the whole inner loop. The whole point of FlashAttention is that we never re-read $Q$ from HBM.
- **$K, V$ tiles stream through SMEM/VMEM** via a pipelined DMA. On Hopper this is one `cp.async.bulk` (TMA) per tile.
- **Online softmax** keeps $(m_i, \ell_i)$ for correct accumulator updates without ever materializing the full $S$ matrix.
- **Two matmuls per inner iteration**: $Q K^\top$ and $P V$. On H100 these are WGMMA; on TPU they are MXU.
- **Logsumexp $\ell$ saved** so backward can recompute attention weights without storing them.

The forward kernel composes with `jax.custom_vjp` for `grad`; the backward uses Pallas too. See `jax.experimental.pallas.ops.attention`.

### 20.4 v2 vs v3

- **v2** rearranged parallelism: parallelize over `(batch, head, q-tile)` outer loop (instead of v1's `batch * head` only), and reduced non-matmul FLOPs by deferring the $1/\ell$ rescale to the end. Brought H100 forward to ~70% of theoretical peak.
- **v3** is Hopper-specific: **warp specialization** (separate producer warps issue TMA loads, consumer warps run WGMMA), 2-stage pipelining of GEMM and softmax so tensor cores stay busy while special-function units compute exp, and **FP8 (e4m3) matmuls with FP32 accumulation**. Hits ~75% of FP16 peak and ~1.2 PFLOPS in FP8 on H100. The B200 port adds asynchronous TMA stores and 5th-gen tensor cores.

### 20.5 Where it lands in JAX

- `jax.nn.dot_product_attention(query, key, value, ..., implementation="cudnn"|"xla"|"pallas")` — added 2024, stable in 2025–2026. On NVIDIA dispatches to cuDNN's fused FlashAttention-3 via the XLA `cuDnnFusedMHA` custom call. On TPU uses the splash-attention Pallas kernel.
- `jax.experimental.pallas.ops.attention` — reference Pallas implementation (Triton/GPU and Mosaic/TPU). Useful when you need to fork the kernel (custom score modifications, etc.).
- **Splash attention** (`jax.experimental.pallas.ops.tpu.splash_attention`) — TPU-native sparse-mask FlashAttention; the canonical fused attention on TPU v4 / v5p / v6e.
- **NNX**: `flax.nnx.MultiHeadAttention` calls `jax.nn.dot_product_attention` under the hood; MaxText and AXLearn have their own thin wrappers.

### 20.6 Sliding window / local attention

For very long contexts, full quadratic attention is wasteful when most useful information is local. A sliding window of size $W$ caps per-query work at $O(N \cdot W)$ — restoring linearity in time with unchanged constant memory. Used by Longformer / BigBird (with global tokens) and by Mistral / Gemma 2 (alternating with full attention per layer).

In a Pallas kernel this is just *block skipping*: KV tiles outside $[i_{\text{start}} - W, i_{\text{start}}]$ are never loaded. `jax.nn.dot_product_attention` accepts a `local_window_size=(left, right)` argument; on TPU `splash_attention` accepts `MultiHeadMask([CausalMask(...), LocalMask(window=...)])` compositions.

---

## Chapter 21. KV Cache: Prefill, Decode, Paging, GQA

### 21.1 The two-phase nature of inference

A decoder-only LLM has fundamentally different arithmetic intensity in its two phases:

- **Prefill** processes the prompt of length $N$ in one shot. Attention work: $O(N^2 H)$. MLP work: $O(N D^2)$. **Compute-bound.** FLOPs scale as $2 \cdot \text{params} \cdot N$; bandwidth as $\text{params} + N^2$.
- **Decode** generates one token per step using cached $K, V$ of length $N$. Attention work: $O(N H)$. MLP work: $O(D^2)$. **Memory-bandwidth-bound.** You read **all parameters and the entire KV cache** for every single token.

Roofline gives a hard ceiling: $\text{tokens/s} \le \beta / \text{bytes per step}$.

For a 70 B model in bf16, that's 140 GB of weights per token. On an H100 (3.35 TB/s HBM3) the absolute ceiling is

$$
\frac{3.35 \times 10^{12}}{1.4 \times 10^{11}} \approx 24 \text{ tok/s/GPU}
$$

before any KV cache reads. This single fact dictates the entire inference stack: batching, paging, quantization, speculative decoding all exist to push past this wall.

### 21.2 Critical batch size

The book gives the **critical batch size** for compute-bound decode:

$$
\boxed{B_{\text{crit}} = \frac{\pi}{\beta} \cdot \frac{\text{bits}_{\text{param}}}{\text{bits}_{\text{activation}}}}
$$

For TPU v5e bf16: $B_{\text{crit}} \approx 240$ tokens. For int8 weights with bf16 activations: $B_{\text{crit}} \approx 120$ — *quantization is also a roofline lever*.

Prefill prompts typically exceed 240 tokens, so prefill is naturally compute-bound. Decode requires 240 *concurrent requests* to be compute-bound, which is why batching is crucial for decode throughput.

Attention's arithmetic intensity in this framing:

$$
\text{AI(attention)} = \frac{S T}{S + T}
$$

For prefill ($S = T$): $T/2$. For decode ($T = 1$): $\approx 1$. **Decode attention is always bandwidth-bound.**

### 21.3 KV cache shape and size

Per layer, per request:

$$
\text{KV cache shape}: (2, S_{\max}, K, H)
$$

Total bytes per request:

$$
2 \cdot L \cdot S_{\max} \cdot K \cdot H \cdot \text{dtype\_bytes}
$$

LLaMA-3-70B at 8 K context, batch 1, bf16:

$$
2 \cdot 80 \cdot 8192 \cdot 8 \cdot 128 \cdot 2 = 2.68 \text{ GiB}
$$

At batch 32: 86 GiB. The KV cache can dwarf the parameters at modest batch sizes. This is *the* lever in inference systems.

### 21.4 Paged attention

Instead of allocating a contiguous $S_{\max}$-long buffer per request (which fragments memory and forces over-allocation), split the KV cache into fixed-size **blocks** (typically 16 or 32 tokens). Each request holds a **block table**: `block_table[b, i]` is the physical block id storing logical positions $[i \cdot \text{block}, (i+1) \cdot \text{block})$. The attention kernel takes $(Q, K_{\text{blocks}}, V_{\text{blocks}}, \text{block\_tables}, \text{seq\_lens})$ and gathers the right blocks.

Memory waste drops from ~60% (contiguous over-allocation) to <4%. Originally introduced by vLLM (Kwon et al. 2023, PagedAttention).

**Ragged paged attention** generalizes further: the kernel handles a *batch of variable-length sequences* in a single dispatch, with separate `cu_seq_lens` cumulative offsets — exactly what continuous batching needs.

JAX landings:

- **JetStream** (TPU-first inference server): paged KV via `ragged_paged_attention` Pallas kernel.
- **MaxText** added paged attention in 2024.
- On GPU, `jax.experimental.pallas.ops.paged_attention` (Triton-style kernel). vLLM-jax (community port) exposes the same scheduler.

### 21.5 GQA / MQA / MLA

The decode bottleneck is *KV cache bandwidth*, not parameter bandwidth. **Multi-Query Attention** (Shazeer 2019) shares one $(K, V)$ head across all query heads, cutting KV cache size by $N$. **Grouped-Query Attention (GQA)** (Ainslie et al. 2023) is the practical compromise: $K$ groups, where $K \mid N$, with $N/K$ typically 4–8. LLaMA-3 uses 8:1 ($N=64, K=8$).

In Pallas, you broadcast on the head axis when loading from HBM — one KV tile feeds $N/K$ Q tiles in registers, which is also a *compute* win because it amortizes KV loads.

**MLA (Multi-head Latent Attention)** in DeepSeek-V3 takes this further: project KV into a low-rank "latent" form before caching, then expand back at use time. Cuts KV cache to ~$1/16$ of MHA at minimal quality cost.

All major JAX attention implementations accept `num_kv_heads`. `jax.nn.dot_product_attention` infers GQA from shape mismatch.

---

## Chapter 22. Position Embeddings: RoPE, YaRN, ALiBi

### 22.1 The why

Sinusoidal absolute positions don't compose well with attention; learned absolute positions don't extrapolate. **Rotary Position Embeddings** (Su et al. 2021) encode position by *rotating* $Q$ and $K$ by an angle that depends on absolute position, so that $\langle q_m, k_n \rangle$ becomes a function of $m - n$ only — a clean relative-position signal that respects the inner-product structure of attention.

### 22.2 RoPE

Pair adjacent feature dims into 2-vectors. For dim pair $(2i, 2i+1)$ and base $\theta_i = \text{base}^{-2i/D}$ (canonically $\text{base} = 10000$), define the rotation matrix

$$
R_m^i = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}
$$

Then $q_m = R_m \cdot q$, $k_n = R_n \cdot k$, so

$$
q_m^\top k_n = q^\top R_{n - m} k
$$

which depends only on the **relative** position $n - m$. Magic.

In JAX:

```python
def apply_rope(x, positions, base=10000.0):
    # x: (..., S, D); D must be even
    D = x.shape[-1]
    inv_freq = 1.0 / (base ** (jnp.arange(0, D, 2) / D))   # (D/2,)
    angles = positions[..., None] * inv_freq               # (..., S, D/2)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return jnp.stack([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1).reshape(x.shape)
```

### 22.3 Extending context: PI / NTK / YaRN / LongRoPE

LLMs trained at context $N_{\text{train}}$ break above their training length unless RoPE is adjusted. The relevant tricks:

- **Position Interpolation (PI)** (Chen et al. 2023): rescale positions by $s = N_{\text{train}} / N_{\text{target}}$ so the model sees the same angular range. Cheap, but degrades fine-grained locality (high frequencies get squished).
- **NTK-aware scaling** (bloc97 2023): change the *base* instead: $\text{base}' = \text{base} \cdot s^{D/(D-2)}$. Intuition (from Neural Tangent Kernel theory): high frequencies are barely scaled (locality preserved); low frequencies stretch (long-range positions stay distinct).
- **YaRN** (Peng et al. 2023): mixes both per-frequency-band — high frequencies untouched, low frequencies linearly interpolated, mid frequencies smoothly transitioned. Adds a *temperature* $1/t = 0.1 \ln(s) + 1$ on logits to compensate for entropy increase from longer sequences. State-of-the-art for long-context fine-tuning; used by Qwen2.5-1M, DeepSeek-V3.
- **LongRoPE / LongRoPE2** (Microsoft 2024–25): per-dimension search of optimal scale factors via evolutionary optimization on a held-out long-context perplexity signal. Pushes Phi-3.5/4 to >2 M tokens.

In JAX, all of these are pure-Python preprocessing of `inv_freq` — no kernel changes. MaxText's `MultiHeadAttention` exposes `rope_type ∈ {default, llama3, yarn, longrope}`.

### 22.4 ALiBi

ALiBi (Press et al. 2021) drops position embeddings entirely and adds a *linear, head-specific bias* to attention logits proportional to $-|i - j|$:

$$
S_{ij} \mathrel{+}= -m_h \cdot |i - j|
$$

with $m_h = 2^{-8h/N}$ for head $h \in [0, N)$.

Trivially extrapolates. Used by BLOOM, MPT. Implement in a Pallas kernel by adding the bias inside the inner loop before softmax.

### 22.5 When to choose what

RoPE is the default in every serious 2025–2026 model — it's the only family with a developed ecosystem of context extension techniques (YaRN, LongRoPE) and works with FlashAttention's online softmax. ALiBi is simpler and extrapolates "for free" but underperforms long-context fine-tuned RoPE. Sinusoidal absolute is dead.

---

## Chapter 23. Mixture of Experts

### 23.1 The why

A dense FFN of width $4D$ activates all $4D^2$ parameters per token. An MoE replaces it with $E$ experts, each of width $4D/k$ (or full $4D$), and routes each token to $k$ of them ($k=1$ in Switch Transformer; $k=2$ in Mixtral, GPT-OSS, DeepSeek-V2; $k=8$ in DeepSeek-V3 fine-grained).

FLOPs scale with $k$, parameters with $E$: **you decouple capacity from compute.** DeepSeek-V3 has 671 B params but activates 37 B per token.

### 23.2 The algorithm

1. **Router**: $\text{logits} = x \cdot W_{\text{router}}$, then $\text{gates}, \text{idx} = \text{top\_k}(\text{softmax}(\text{logits}), k)$.
2. **Dispatch**: send token $i$ to its $k$ chosen experts. With expert parallelism this is an **all-to-all** collective.
3. **Expert FFN**: each expert is a SwiGLU/GEGLU MLP; expert $e$ sees only its assigned tokens.
4. **Combine**: weighted sum of expert outputs by gates, scattered back to original positions.

### 23.3 Capacity factor

Each expert's max input length is

$$
C = \lceil \text{cf} \cdot k \cdot N / E \rceil
$$

for some capacity factor $\text{cf} \ge 1$ (typical 1.0–1.5). Tokens routed to a full expert are *dropped* (residual passes through); that's the price of static-shape XLA. DeepSeek-V3's "auxiliary-loss-free" routing uses a per-expert bias updated by EMA of load, sidestepping drops.

### 23.4 Auxiliary losses

**Load-balancing loss** (Switch Transformer):

$$
L_{\text{aux}} = \alpha \cdot E \cdot \sum_e f_e \cdot P_e
$$

where $f_e$ = fraction of tokens routed to expert $e$, $P_e$ = mean router probability for $e$. Penalizes routing concentration.

**Router z-loss** (Zoph et al. 2022):

$$
L_z = \beta \cdot \text{mean}\!\left(\big(\text{logsumexp}(\text{logits})\big)^2\right)
$$

Keeps router logits small so softmax is stable in bf16.

### 23.5 Expert parallelism

A modern training run combines:

- **DP/FSDP**: replicate computation, shard weights across data axis.
- **TP**: shard weight matrices column- and row-wise within a layer.
- **PP**: split layers across stages.
- **EP** (expert parallelism): shard *experts* across devices. Requires **all-to-all** in dispatch and combine.
- **CP** (context parallelism / Ring Attention): shard the sequence dimension; ring-exchange KV blocks. Critical for >100 K context training.

A typical `(data, fsdp, tensor, expert)` mesh in MaxText:

```python
mesh = jax.make_mesh((dp, fsdp, tp, ep), ('data', 'fsdp', 'tensor', 'expert'))
expert_weights_spec = P('expert', None, 'tensor')   # (E, d_in, d_out_shard)
hidden_spec         = P(('data', 'fsdp'), None, 'tensor')
```

The all-to-all is `jax.lax.all_to_all(tokens, axis_name='expert', split_axis=0, concat_axis=0)`. JAX's compiler (XLA + Shardy) overlaps this with the expert MLP through pipeline-style chunking.

### 23.6 JAX implementations

- **MaxText** has production MoE in `layers/moe.py` with capacity-factor dispatch and dropless variants.
- **Pallas grouped matmul (`gmm`)**: a single kernel that does $[B, d] \times [E, d, d'] \to [B, d']$ with token-to-expert indices, eliminating the explicit all-to-all on TPU when $E$ fits in one mesh axis. JetStream uses this for Mixtral inference.
- **megablocks-jax** (community port of Gale et al.'s block-sparse expert kernels) implements dropless MoE via grouped GEMM.
- **AXLearn** ships full `MoEFeedForward` modules with EP/TP composition.

NNX sketch:

```python
class MoE(nnx.Module):
    def __init__(self, d, d_ff, E, k, mesh, *, rngs):
        self.W_router = nnx.Param(...)                       # (d, E)
        self.W_in   = nnx.Param(jnp.zeros((E, d, d_ff)),
                                sharding=P('expert', None, 'tensor'))
        self.W_out  = nnx.Param(jnp.zeros((E, d_ff, d)),
                                sharding=P('expert', 'tensor', None))
        self.k, self.E = k, E

    def __call__(self, x):
        logits = x @ self.W_router                              # (B*S, E)
        gates, idx = jax.lax.top_k(jax.nn.softmax(logits), self.k)
        h = pl.experimental.gmm(x, self.W_in,  idx)             # dispatch + matmul
        h = jax.nn.swish(h)
        y = pl.experimental.gmm(h, self.W_out, idx)
        return jnp.einsum('bki,bk->bi', y, gates)
```

---

## Chapter 24. Numerics: bf16, fp16, fp8, mixed precision, μP

### 24.1 The dtype zoo

| dtype | exp bits | mantissa | range | precision (eps) |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | $1.2 \times 10^{-7}$ |
| bf16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ (= fp32) | $7.8 \times 10^{-3}$ |
| fp16 | 5 | 10 | $\pm 6.5 \times 10^{4}$ | $9.8 \times 10^{-4}$ |
| fp8 e4m3 | 4 | 3 | $\pm 448$ | $1.95 \times 10^{-3}$ |
| fp8 e5m2 | 5 | 2 | $\pm 5.7 \times 10^{4}$ | $0.125$ |

The key insight: **bf16 has fp32's range with fp16's storage.** That is why every modern training stack is bf16-by-default — gradients, activations, weights live in bf16, no loss scaling needed.

**fp16 needs loss scaling** because gradients underflow below $\sim 6 \times 10^{-5}$; in bf16 the same value is representable.

**fp8** (H100 / B200, TPU v5p+): two formats. **e4m3** has more precision but range only ±448; used for forward activations and weights (which can be tensor-scaled). **e5m2** has fp16-like range; used for backward gradients (which have outliers). FP8 GEMMs use **per-tensor or per-row scaling**: a fp32 scale is computed (often $\text{amax}/\text{fp8\_max}$), the operand is cast to fp8, the matmul runs on tensor cores in fp8 with fp32 accumulation, the output is rescaled.

NVIDIA Transformer Engine and JAX's `jax.experimental.fp8` provide the scaling-aware GEMM wrapper. On B200 the new **MXFP8 / MXFP4** microscaling (block of 32 elements with shared 8-bit exponent) is supported natively.

### 24.2 Mixed precision recipe

The recipe:

- **Master weights in fp32** (or bf16 for very large models with high-quality optimizers like Lion/Shampoo).
- **Forward + backward activations in bf16.**
- **Optimizer states (Adam $m, v$) in fp32** (or bf16 with Kahan summation; some recent results show fp32 $v$ and bf16 $m$ is enough).
- **Upcast to fp32** inside: layernorm/RMSNorm reduction (variance), softmax in attention (handled by FlashAttention which accumulates fp32 LSE), final loss, optimizer update.
- **Stochastic rounding** for bf16 weight updates avoids the "weight update silently zero" failure mode for tiny gradients.

**Loss scaling**: $L_{\text{scaled}} = L \cdot S$ to lift gradients out of fp16 underflow, then unscale before optimizer. JAX has `jmp` (DeepMind's mixed precision policy library) and Optax's `optax.apply_if_finite`. **Not needed for bf16.**

JAX implementation pattern: cast inputs to bf16 at the embedding output; rely on `jax.nn.dot_product_attention` and Pallas kernels to internally accumulate in fp32. For norms, write them with explicit fp32:

```python
def rms_norm(x, weight, eps=1e-6):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(var + eps)).astype(x.dtype) * weight
```

### 24.3 Quantization in JAX

**Inference quantization:**

- **int8 weights, bf16 activations (W8A16)**: per-channel symmetric scales. Halves weight bandwidth → halves decode latency. ~zero quality loss.
- **int4 weights (W4A16)**: GPTQ (Frantar 2023), AWQ (Lin 2023). Group-wise scales (group size 64–128). Requires a custom dequant-fused matmul. On TPU, MaxText uses Pallas `int4_matmul`; on GPU, `jax.experimental.pallas.ops.gpu.gptq_matmul`. Inference-only.
- **fp8 e4m3 weights + e4m3 activations (W8A8)**: full fp8 GEMM. Modest bandwidth win, but the *latency* win comes from 2× tensor-core throughput on H100.
- **MXFP4 / NVFP4** (block of 16/32 with shared microexponent): emerging on B200; ~int4 quality at fp4 throughput.

**Training quantization (AQT and qwix):**

- **AQT** (Accurate Quantized Training, <https://github.com/google/aqt>): replaces `jnp.einsum` / `lax.dot_general` with a wrapper that fakes int8 quantization on the forward, uses a straight-through estimator on the backward, learns calibration scales. Used to train MaxText models in int8 end-to-end with <0.5% loss.
- **qwix** (<https://github.com/google/qwix>): the 2025 successor: declarative quantization rules per-layer for both PTQ and QAT, supports int4/int8/fp8, integrates with Flax NNX's parameter-tree introspection.
- **Native fp8 training**: `jax.experimental.fp8` exposes `fp8_einsum` with delayed scaling. MaxText `--quantization=fp8` enables this; H100 training of LLaMA-3-equivalent models hits ~1.5× throughput vs bf16.

AQT-style forward sketch:

```python
def aqt_dot(lhs, rhs, calibration):
    s_l = calibration.lhs_scale(lhs)
    s_r = calibration.rhs_scale(rhs)
    lhs_q = jnp.round(lhs / s_l).astype(jnp.int8)
    rhs_q = jnp.round(rhs / s_r).astype(jnp.int8)
    out = jax.lax.dot_general(lhs_q, rhs_q, ...,
                              preferred_element_type=jnp.int32)
    return out.astype(jnp.float32) * s_l * s_r
```

### 24.4 Stability tricks

**RMSNorm vs LayerNorm.** LayerNorm: $(x - \mu)/\sigma \cdot \gamma + \beta$. RMSNorm (Zhang & Sennrich 2019):

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{D}\sum_i x_i^2 + \epsilon}} \cdot \gamma
$$

Drops the mean-centering and bias. ~30% faster (one less reduction, no bias term), and empirically equally good or better at scale. Used by LLaMA, Mistral, DeepSeek, Gemma — RMSNorm is the default in 2026.

**qk-norm** (Henry et al. 2020; revived by Gemma 2, Chameleon, OLMo 2). Apply RMSNorm to $Q$ and $K$ independently before attention: $Q' = \text{RMSNorm}(Q)$, $K' = \text{RMSNorm}(K)$. Bounds attention logits $q' \cdot k' \in [-d, d]$ (since both are unit-RMS), so softmax can never blow up. The single most reliable cure for "attention logit explosion."

**z-loss** (PaLM 2022):

$$
L_z = 10^{-4} \cdot \text{mean}\!\left(\big(\text{logsumexp}(\text{logits})\big)^2\right)
$$

on the LM head. Prevents the partition function from drifting up, which empirically prevents loss spikes and bf16 softmax overflow. Cheap and free improvement.

**Residual scaling.** Two flavors:
- **Pre-norm with $\alpha$**: $x = x + \alpha \cdot f(\text{LN}(x))$, with $\alpha = 1/\sqrt{2L}$ (DeepNorm) keeps activation norms stable at depth.
- **Skip-init / $\beta$-LayerNorm**: initialize the residual branch to zero so each block is identity at step 0.

### 24.5 μP (Maximal Update Parameterization)

**The why.** Hyperparameters tuned at small scale do not transfer to large scale under standard parameterization. Optimal LR, init scale, output multiplier all change with width. Yang & Hu 2021 showed that a specific **width-dependent rescaling** of init, LR, and output makes training dynamics width-invariant: tune at 64 M params, transfer to 70 B with the same LR.

**The recipe.** For a layer of fan-in $n$ and base width $n_0$:

- **Init**: $W \sim \mathcal{N}(0, \sigma^2/n)$ (standard) for input/hidden, $\mathcal{N}(0, \sigma^2/n_0)$ (constant in $n$) for output layer.
- **LR**: $\eta \cdot n_0/n$ for hidden weights (Adam), $\eta$ (constant) for input/output and biases.
- **Output multiplier**: $1/n$ on the LM head logits.

The result: as width scales, (a) feature updates have $\Theta(1)$ magnitude, (b) the *function* changes by $\Theta(1)$, (c) the optimal LR is the same constant. In practice you do a small-scale sweep (e.g. width=256) over LR, then train at full scale with that LR.

Cerebras-GPT, Microsoft Phi, MiniCPM all reported successful transfer. JAX has `mup` (community port, <https://github.com/microsoft/mup>) exposing `MuReadout` and per-layer LR scaling that integrates with `optax.chain`. MaxText has `--use_mup=True`.

---

## Chapter 25. Training-Time Parallelism — The LLaMA-3 70B Worked Example

We now have the vocabulary to do the full worked example. This follows Part 6 of the scaling book.

### 25.1 Setup

- **Model**: LLaMA-3-70B. $L = 80$, $D = 8192$, $F = 28672$, $N = 64$, $K = 8$, $H = 128$.
- **Hardware**: TPU v5p pod, 8960 chips (16×20×28 topology).
- **Batch**: 4 M tokens per step (1024 sequences × 4096 length).
- **Precision**: bf16 weights, fp32 optimizer state.
- **Goal**: pick a parallelism strategy that keeps the chips compute-bound.

### 25.2 Parameter and FLOP budgets

From Chapter 19:
- Total params: $\sim 7 \times 10^{10}$.
- FLOPs/token: $6 \cdot \text{params} = 4.2 \times 10^{11}$.
- Total FLOPs for 15 T tokens: $6.3 \times 10^{24}$.
- At 40% MFU: $\sim 44$ days on the full pod.

Memory:
- Params (bf16): 140 GB.
- Optimizer state (fp32 Adam, two accumulators + bf16 master in some setups, or full fp32): 560 GB.
- Gradient checkpoints (4/layer): ~21 TB.
- Total: 21.6 TB. Per chip on 8960 chips: 2.4 GB.

That's well below the 96 GB v5p HBM. Memory is not the constraint here — communication is.

### 25.3 The four parallelism strategies (book equations)

**Data Parallelism (DP) and FSDP (ZeRO-3)**: shard the batch across devices. Compute-bound when

$$
\boxed{\frac{B}{X} > \frac{C}{W_{\text{ici}}}}
$$

where $B$ is total batch (tokens), $X$ is number of FSDP shards, $C$ is FLOP/s/chip, $W_{\text{ici}}$ is bidirectional ICI bandwidth.

For TPU v5p: $C = 4.6 \times 10^{14}$, $W_{\text{ici}} = 1.8 \times 10^{11}$, ratio = **2550**. Min per-device batch ≈ 2550 tokens.

**Tensor Parallelism (TP)** (Megatron): shard weight matrices within a layer. Compute-bound when

$$
\boxed{F > Y \cdot \frac{C}{W_{\text{ici}}}}
$$

where $Y$ is the TP degree and $F$ is the FFN hidden dim. For TPU v5p: $F > 2550 Y$, so $Y < F/2550 = 28672/2550 \approx 11$. Practical TP degree caps at ~8.

**Pipeline Parallelism (PP)**: shard along the layer dimension. Per-layer comms:

$$
T_{\text{PP}} \approx 1.5 \cdot \frac{2 B D}{W \cdot N_{\text{layers}}}
$$

Theoretically cheap (scales with $N_{\text{layers}}$). Practical issues: pipeline bubbles, ZeRO-3 incompatibility, code complexity.

**Expert Parallelism (EP)**: shard experts; all-to-all dominates. Two regimes depending on intra-/inter-node bandwidths.

### 25.4 Mixed FSDP + TP

A common 2D strategy with mesh axes $X$ (FSDP) and $Y$ (TP). The book gives the optimal split:

$$
\boxed{X_{\text{opt}} = \sqrt{\frac{B}{F} \cdot \frac{M_X}{M_Y} \cdot N}}
$$

where $N$ is total chips, $M_X / M_Y$ are mesh axes used for each.

Compute-bound condition for FSDP+TP:

$$
\frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}, \quad \alpha = C / W_{\text{ici}} = 2550
$$

Min per-device batch: $\sim 100$ tokens/chip — **8× smaller than pure FSDP/DP**. This is what enables training on 18,000+ chips with modest batch sizes.

### 25.5 Plugging in for LLaMA-3-70B

$B = 4.19 \times 10^6$, $N = 8960$, $F = 28672$:

$$
X_{\text{opt}} = \sqrt{\frac{4.19 \times 10^6 \cdot 8960}{28672} \cdot \frac{2}{1}} \approx 1618
$$

Round to power-of-2: $X = 1024$, $Y = 4$ on a pod with 4096-chip slice (or $X = 2048$, $Y = 4$ on the full 8960).

The book reports the actual run: **1024-way FSDP + 2-way sequence + 4-way tensor parallelism** on a v5p pod, chosen for the best compute-bound margin. The $X_{\text{opt}}$ formula is what guides this choice.

### 25.6 Cross-pod scaling (DCN)

For training across multiple pods over DCN at $W_{\text{dcn}} \sim 6 \times 10^9$ B/s/chip:

$$
\frac{B}{\text{slice}} > \frac{C}{W_{\text{dcn}}} \approx 71{,}000 \text{ tokens/slice}
$$

Use pure DP across slices when each slice's batch ≥ 71 K tokens.

### 25.7 The decision recipe

The whole game, distilled:

1. Compute total FLOPs needed: $6 \cdot \text{params} \cdot \text{tokens}$.
2. Estimate per-chip batch under pure FSDP/DP: $B / N$.
3. If per-chip batch $> 2550$: pure FSDP/DP works.
4. If between 100 and 2550: use FSDP+TP, pick $X_{\text{opt}}$.
5. If below 100: stack PP or EP on top.
6. Verify per-chip memory < HBM (96 GB on v5p).
7. Verify cross-pod DP works (per-slice batch > 71 K).

### 25.8 A note on profiles vs. theory

The math gives you a *target*. Real runs hit MFUs of 30–55% (TPUs sometimes higher, GPUs sometimes lower with FSDP across nodes). The gap between theoretical and actual is exactly what profiling (Chapter 31) is for.

---

## Chapter 26. Inference: Prefill, Decode, Continuous Batching, Speculative Decoding

### 26.1 The two-phase nature dictates everything

To restate from Chapter 21 in system terms: prefill is *arithmetic-intensity-rich* (the prompt processes in batch over its sequence dim, so GEMMs are large and tensor-core-bound); decode is *arithmetic-intensity-poor* (one token per request × batch, GEMVs that are HBM-bound). A serving stack thus has two regimes:

- **Prefill**: maximize batch sequence-length dim; use FlashAttention-3 in fp8; chunked prefill (split a 32 K prompt into 2 K chunks so prefill doesn't starve concurrent decodes).
- **Decode**: maximize batch *count*, paged KV, quantized weights, speculative decoding, multi-token prediction.

**Disaggregated serving** (DistServe, Splitwise, Mooncake): run prefill and decode on *different* hardware pools — prefill on H100s with high FLOPs, decode on B200s or TPU v6e with high HBM bandwidth. The KV cache is RDMA-transferred between them. JetStream and SGLang both support disaggregation in 2025+.

### 26.2 The book's inference equations

Theoretical minimum step time (memory-bandwidth-bound):

$$
T_{\text{min}} = \frac{B \cdot |\text{KV}| + |\text{params}|}{N \cdot \beta_{\text{HBM}}}
$$

Theoretical maximum throughput:

$$
\text{tokens/s}_{\max} = \frac{B \cdot N \cdot \beta_{\text{HBM}}}{B \cdot |\text{KV}| + |\text{params}|}
$$

General step time (compute + memory):

$$
T_{\text{step}} = \underbrace{\frac{B \cdot |\text{KV}|}{N \beta_{\text{HBM}}}}_{\text{attention, BW-bound}} + \max\!\left(\underbrace{\frac{2 B \cdot \text{params}}{N \pi}}_{\text{MLP compute}}, \underbrace{\frac{|\text{params}|}{N \beta_{\text{HBM}}}}_{\text{MLP BW}}\right)
$$

### 26.3 Concrete example: LLaMA-3-70B on TPU v5e

$L = 80$, $D = 8192$, $K = 8$, $H = 128$, int8 weights and KV.

- KV cache per token: $2 \cdot 8 \cdot 128 \cdot 80 \cdot 1 = 160$ KB/token (int8 1 byte).
- Params (int8): 70 GB.
- Hardware: v5e 4×2 = 8 chips, 128 GB HBM total, 6.5 TB/s aggregate.

| Topology | Dtype | Max batch | Step time | Throughput/chip |
| --- | --- | --- | --- | --- |
| 4×2 | int8 | 43 | 17 ms | 235 tok/s |
| 4×4 | int8 | 140 | 19 ms | 0.90 QPS |
| 2×2 | int4 | 43 | 19 ms | 1.11 QPS |

Prefill (8 K tokens, B=8): 0.91 s at 40% MFU.

The inference example confirms the Chapter 21 thesis: KV-cache memory bandwidth dominates.

### 26.4 Sharding during inference

Fundamentally different from training:

- **Prefill**: nearly identical to training — TP up to ICI bandwidth limit (~4–8-way), then sequence parallelism.
- **Decode**:
  - **FSDP impossible**: memory-bound in parameter loading; moving weights via ICI defeats the purpose.
  - **DP useless**: replicates parameters, no speedup; better to run separate model instances.
  - **Sequence parallelism infeasible**: only one token per step.
  - **Only option: TP** with activations replicated, weights sharded.

The book gives the threshold for when extending TP beyond the compute roofline still helps:

$$
Y > \frac{F}{B \cdot \beta_{\text{ratio}}}, \quad \beta_{\text{ratio}} = \beta_{\text{HBM}} / \beta_{\text{ICI}} \approx 8
$$

For $F = 16384$, $B = 32$: up to 64-way TP without throughput regression.

KV cache sharding: Megatron-shard along head dimension (up to $K$-way), then batch-shard. Requires two AllToAlls per attention layer to shift Q activations and shift output back.

### 26.5 Continuous batching

Static batching wastes cycles whenever any request finishes. **Continuous batching** (Yu et al., Orca 2022) refills the batch *every step*: as soon as a request finishes, its slot is replaced with a new prefill chunk. Combined with paged KV, throughput rises 5–10×.

Algorithm sketch:

1. Scheduler maintains a queue of pending requests and an "in-flight" batch with paged KV.
2. Each step: combine *one prefill chunk* (if any pending) with all in-flight decodes into a single ragged-batch dispatch.
3. The kernel (`ragged_paged_attention`) handles mixed prefill+decode in one call.
4. Finished requests evicted; pages freed.

JetStream's scheduler is in `jetstream/core/orchestrator.py`; SGLang-jax mirrors vLLM's structure.

### 26.6 Speculative decoding

Decode is HBM-bound — so the chip has spare FLOPs. **Speculative decoding** (Leviathan et al. 2022, Chen et al. 2023) uses them by running a **small draft model** for $k$ steps, then **verifying with the big model in one parallel forward pass**. Whichever prefix the big model accepts (under rejection-sampling that keeps the output distribution exact) is committed; on a reject, the big model samples one replacement token.

Effective speedup:

$$
\text{speedup} \approx \frac{1 + \alpha + \alpha^2 + \cdots + \alpha^k}{1 + c} = \frac{1 - \alpha^{k+1}}{(1-\alpha)(1+c)}
$$

where $\alpha$ is the per-token acceptance rate and $c = T_{\text{draft}} / T_{\text{target}}$.

Variants:

- **Tree speculation** (Medusa, EAGLE, EAGLE-2/3): the draft is a *tree* of candidate tokens; the verifier accepts whichever path is consistent. Acceptance rates of 4–7 tokens per step.
- **Multi-token prediction** (Gloeckle et al. 2024; DeepSeek-V3): train the model itself to predict the next $k$ tokens from extra heads — at inference, those heads are the draft.
- **Lookahead decoding** (Fu et al. 2024): draft-free; uses Jacobi iteration over an n-gram window. Works when no draft exists.

JetStream supports speculative decoding via `--speculative_decoding.draft_path`. MaxText has `inference/speculative.py`.

### 26.7 The JAX inference stack (2026)

- **JetStream** (<https://github.com/AI-Hypercomputer/JetStream>): TPU-first throughput-optimized inference server. Continuous batching, paged KV, prefill chunking, fp8/int8 quantized serving, speculative decoding. Separates *engine* (model forward, JAX/Pallas) from *orchestrator* (scheduler, gRPC). Reference engines: Pax (legacy), MaxText (current).
- **MaxText**: training and inference reference; LLaMA, Gemma, Mixtral, DeepSeek configs. Pure JAX/Flax NNX.
- **AXLearn** (Apple): training+serving framework on JAX, custom serving on TPU and GPU.
- **SAX** (Server for Accelerators in XLA): older, still used internally at Google for ultra-low-latency.
- **vLLM-jax / SGLang-jax**: community ports of popular GPU servers, useful when you need vLLM's UX on TPU.
- **Tunix** (Google, 2025): JAX-native post-training (RLHF, DPO, GRPO) library that interoperates with JetStream for the rollout stage.

### 26.8 Quantized serving — the typical numbers ladder

For a 70 B model on a single H100 node (8 × H100):

| Configuration | tok/s/GPU at B=1 | aggregate tok/s |
| --- | --- | --- |
| bf16 dense | ~25 | — |
| int8 W8A16 | ~50 | — |
| int4 W4A16 GPTQ | ~95 | — |
| fp8 W8A8 + paged KV + continuous batching | — | ~30 K (B=256) |
| + speculative (draft 1B, $\alpha$=0.75, $k$=4) | — | ~80 K |

Each layer of this stack is a Pallas kernel (or a cuDNN call from `jax.nn`) plus a scheduler change. The lesson: in 2026, you do not write attention; you compose kernels and shardings.

---

# Part VI — Pallas: Custom Kernels in JAX

XLA is excellent for the common case. It generates near-peak kernels for matmuls, fuses chains of elementwise ops, picks reasonable layouts. But "near-peak for the common case" is not the same as "peak for your case." If you write FlashAttention-3 with `jnp.einsum`, you'll get a multi-kernel decomposition that materializes the $T \times S$ score matrix in HBM. If you write a paged-KV attention with `vmap` and indexing, you'll get correct code at decode-time speeds that are 2–5× off the theoretical floor. The fix is to drop a level: **write the kernel yourself.**

**Pallas** (<https://docs.jax.dev/en/latest/pallas/>) is JAX's kernel-authoring DSL. It gives you the controls — block sizes, memory placement, manual DMAs, tensor-core invocation — without leaving the JAX ecosystem. A `pallas_call` slots into your model code as a regular callable that composes with `jit`, `vmap`, `shard_map`, and (with `custom_vjp`) `grad`.

This part covers the programming model (Chapter 27), the two main backends — Mosaic-TPU and Mosaic-GPU/Triton (28–29) — and worked kernels for the canonical cases (30).

---

## Chapter 27. The Pallas Programming Model

### 27.1 What Pallas is

Pallas is a JAX-embedded kernel DSL. You write a Python function that looks like ordinary `jax.numpy`, wrap it in `pl.pallas_call`, and JAX lowers — *not* to XLA HLO, but to one of three lower-level kernel compilers:

| Backend | Target | Compiler path |
| --- | --- | --- |
| **Mosaic-TPU** | TPU v4 / v5e / v5p / v6 (Trillium) | Pallas IR → Mosaic dialect → LLO |
| **Mosaic-GPU** | NVIDIA Hopper (H100) and Blackwell (B100/B200) | Pallas IR → MLIR → PTX |
| **Triton** | NVIDIA Ampere / Hopper (older path) | Pallas IR → Triton IR → PTX |

You stay inside JAX (the kernel composes with all transforms), but you control tiling, memory placement, and the inner loop the way you would in CUDA, Triton, or hand-written XLA custom calls.

### 27.2 The grid + BlockSpec + Ref mental model

A Pallas program has three pieces:

1. **Grid** — a tuple of integers naming the iteration space. Pallas executes your kernel body once per grid index, conceptually in parallel. On GPU each grid index becomes a CUDA program (block); on TPU each grid index becomes a pipeline iteration that drives a DMA + compute step.
2. **BlockSpec** — for each input/output, a `pl.BlockSpec(block_shape, index_map)` tells Pallas *which tile of the global array* to expose for grid index $(i, j, ...)$. The `index_map` is a Python function from grid coordinates to tile coordinates.
3. **Refs** — inside the kernel body, you don't see `jax.Array` values. You see `pl.Ref` objects — mutable, addressable handles to a slice of fast memory (VMEM on TPU, SMEM on GPU) that has been pre-staged. Read with `x_ref[...]`, write with `o_ref[...] = ...`.

The mental shift from JAX to Pallas: **you stop writing values and start writing memory transactions.** Pallas turns "compute output[i,j]" into "stage tile $(i,j)$ of input into VMEM, run the body, DMA the result tile back to HBM."

### 27.3 An add kernel

```python
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    # Refs into VMEM/SMEM that already hold one tile.
    o_ref[...] = x_ref[...] + y_ref[...]

@jax.jit
def add(x, y):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // 128,),
        in_specs=[pl.BlockSpec((128, x.shape[1]), lambda i: (i, 0)),
                  pl.BlockSpec((128, y.shape[1]), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((128, x.shape[1]), lambda i: (i, 0)),
    )(x, y)
```

The grid has $N/128$ iterations. For iteration $i$, Pallas DMAs rows $[128 i : 128(i+1)]$ of `x` and `y` into VMEM (TPU) or SMEM (GPU), binds them to the refs. Your body runs in registers/VPU. The result tile is DMA'd back to HBM.

### 27.4 A matmul kernel

```python
def matmul_kernel(x_ref, y_ref, o_ref):
    # 2D grid: (i, j) over output tiles. K-axis is reduced inside the body.
    @pl.when(pl.program_id(2) == 0)
    def _():
        o_ref[...] = jnp.zeros_like(o_ref)
    o_ref[...] += x_ref[...] @ y_ref[...]   # one K-tile MAC

@jax.jit
def matmul(x, y, *, bm=128, bn=128, bk=128):
    M, K = x.shape; _, N = y.shape
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        grid=(M // bm, N // bn, K // bk),
        in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                  pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
    )(x, y)
```

Three points:

1. The K loop is *inside* the grid. Pallas accumulates by passing the same `o_ref` tile across all K iterations sharing $(i, j)$ — the input-output aliased accumulator pattern. On TPU the accumulator lives in VMEM; on GPU it lives in registers.
2. `pl.program_id(axis)` returns the current grid coordinate (the equivalent of `tl.program_id` in Triton or `blockIdx` in CUDA).
3. `pl.when(...)` is structured if/else for first-iteration zero-init.

### 27.5 Composition with JAX transforms

- **`jit`**: a `pallas_call` is a JAX primitive. It traces, lowers, and caches. You always wrap the call site in `jit`.
- **`vmap`**: prepends a grid dimension and updates each `BlockSpec.index_map` to thread the new axis. Cleanest way to batch a kernel.
- **`grad`**: Pallas calls are *not automatically differentiable* — the kernel body is opaque to autodiff. To use a Pallas kernel inside an autodiff path, wrap with `jax.custom_vjp` and write the backward kernel by hand. FlashAttention's `mha_forward` + `mha_backward` glued by `custom_vjp` is the canonical pattern.
- **`shard_map` / `pjit`**: clean — each device runs the kernel on its local shard, and you compose collectives outside. There is also a TPU story for collectives *inside* the kernel via remote DMAs.

### 27.6 Memory hierarchy

Pallas exposes three levels through the `pl.MemorySpace` enum:

- **HBM** (`pl.ANY` / `pl.HBM` / GPU global memory) — where inputs/outputs live.
- **VMEM (TPU) / SMEM (GPU shared memory)** — where the staged tiles live during a grid iteration. VMEM is megabytes per core on TPU; SMEM is ~228 KB per SM on H100.
- **Registers / sublanes** — scratch for actual compute. You don't address these explicitly.

The key abstraction is that **`BlockSpec` + grid implicitly schedules the DMAs.** Pallas's pipeline emitter inserts double-buffered async copies: while iteration $i$ computes, iteration $i+1$'s tiles are already being prefetched. On TPU this is the canonical "pipelined" code generation; on GPU the Mosaic-GPU backend uses TMA (Tensor Memory Accelerator) loads on Hopper.

For *manual* DMA control (paged attention, ragged inputs, gather-scatter), Pallas exposes `pl.emit_pipeline(...)` (TPU), `pltpu.make_async_copy(src_ref, dst_ref, sem)` (TPU), `plgpu.copy_smem_to_gmem` / `plgpu.copy_gmem_to_smem` (Mosaic-GPU), and barrier semaphores for cross-iteration sync.

The general rule on TPU: if your access pattern fits a strided `BlockSpec`, let Pallas emit the pipeline. If it doesn't — paged KV cache, dynamic shapes — drop to manual DMAs and `emit_pipeline`.

### 27.7 Block-level vs warp/thread-level

Pallas is deliberately **block-level**: every operation inside the kernel acts on tiles, not on individual threads. You write `x_ref[...] @ y_ref[...]`, not "thread $t$ loads element $[t, k]$." The Pallas compiler decides how to schedule across warps / VPU lanes / MXU.

Triton is also block-level on the surface (`tl.dot`, `tl.load(ptr + offsets)`) but exposes more thread-level primitives (`tl.atomic_add`, `tl.where` over arbitrary tiles, direct pointer arithmetic). The Pallas-Triton backend reuses Triton's lowering, so on GPU you can think of Pallas as "Triton with JAX semantics on the outside." Mosaic-GPU is more aggressive: lowers directly to MLIR and uses WGMMA/TMA Hopper-class primitives by name.

The practical implication: in Pallas you essentially never write a lane-level loop. If you need "for each thread in this warp do X," you are using the wrong tool — drop to raw Triton via `jax.ffi`.

---

## Chapter 28. Mosaic-TPU: Programming the Systolic Array

### 28.1 The TPU compute resources

The TPU has three relevant compute units accessible from Pallas:

- **MXU** — the systolic array (128×128 on v4/v5; 256×256 on v6). Does dense matmul in bf16/int8/fp8.
- **VPU** — the vector unit. Elementwise, reductions, transposes, lane permutes over (8, 128) sublanes.
- **VMEM** — scratchpad SRAM, 16–32 MB per TensorCore.
- **SMEM** — small scalar memory for indices, sizes, control state.

### 28.2 BlockSpec on TPU

A TPU `BlockSpec` block shape **must be a multiple of (8, 128)** for VMEM tiles in the typical case — these are the sublane × lane dimensions of the VPU. If you write block shape (32, 256), you get four sublane groups by two lane groups; Pallas lays it out to maximize VPU throughput. Off-multiple shapes cost pad-out and slow paths.

### 28.3 Using the systolic array

You almost never invoke the MXU directly. You write `x_ref[...] @ y_ref[...]` (or `jnp.dot(...)`), and Mosaic decides whether it goes to the MXU or stays on the VPU. Dispatch criterion: "is this a contracting matmul on bf16/fp8/int8 with shapes the MXU can tile cleanly?" Tall-skinny matmuls and very small inner dims may stay on the VPU. To force MXU you generally need shapes that are multiples of 128 on both contracted and output axes.

The accumulator pattern matters: MXU accumulators are kept in **MXU-resident state** while you sweep through K tiles. Pallas detects K-loop input-output aliasing on `o_ref` (the matmul example in §27.4). If you split the accumulator across grid iterations, you fall off the MXU fast path.

### 28.4 Common TPU patterns

- **Pipelined elementwise / fused kernels** — default `pallas_call` with simple `BlockSpec`. Used for `gelu(x) + bias`, layer-norm, RMSNorm, rotary, etc.
- **Matmul + epilogue** — matmul with fused activation/bias/quantization, accumulating into `o_ref` and applying the activation in the last K-tile via `pl.when(k == K_total - 1)`.
- **`emit_pipeline` with custom DMAs** — paged attention, ragged batching, dynamic-shaped MoE. Build a list of "work items" in SMEM, then loop with manual `make_async_copy` calls, double-buffered with two semaphores.
- **Cross-core collectives via remote DMAs** — `pltpu.async_remote_copy` lets you emit AllGather/ReduceScatter inside a kernel for tensor-parallel attention.

### 28.5 Alignment to (8, 128) — why it matters

The VPU reads/writes in (8, 128)-shaped tiles natively. A kernel reading 16-byte cache lines is happy; a kernel reading 32-element vectors slots cleanly. A kernel reading (5, 100)-shaped tiles incurs padding to (8, 128), wasting 30%+ of bandwidth. The XLA HLO will show `pad[...]` ops; the profiler will show "VMEM utilization" below expectations.

---

## Chapter 29. Mosaic-GPU and Triton: Programming Tensor Cores

GPU-side Pallas has two backends. Pick at `pallas_call` time via `compiler_params=plgpu.GPUCompilerParams(...)` for Mosaic-GPU, or by default for Triton.

### 29.1 Triton backend

Older, more battle-tested. Lowers Pallas IR to Triton IR and reuses OpenAI's compiler. Works on Ampere (A100), Hopper (H100), Blackwell.

**Strengths**: handles arbitrary dynamic shapes well; mature autotuning via Triton heuristics.

**Weaknesses**: cannot fully exploit Hopper's TMA + WGMMA without going through Mosaic-GPU; less control over exact memory layout.

### 29.2 Mosaic-GPU backend

Designed specifically for **Hopper-class** features: TMA (Tensor Memory Accelerator) async copies, WGMMA (warpgroup matmul) on tensor cores, distributed shared memory, and the new asynchronous proxy on Blackwell.

```python
def attn_kernel(q_ref, k_ref, v_ref, o_ref, smem_scratch, barrier):
    plgpu.copy_gmem_to_smem(k_ref, smem_scratch.k, barrier)   # async TMA load
    plgpu.barrier_wait(barrier)
    qk = plgpu.wgmma(smem_scratch.q, smem_scratch.k)          # tensor-core matmul
    ...
```

The key shift vs. Triton: you explicitly orchestrate **async barriers**, because TMA and WGMMA are asynchronous on Hopper. This is what FlashAttention-3 exploits.

### 29.3 When to pick which

- **Hopper / Blackwell + max performance** → Mosaic-GPU. TMA and WGMMA for free.
- **Cross-vendor or older GPUs (Ampere)** → Triton.
- **Highly dynamic shapes / dispatch logic** → Triton.
- **One source for both TPU and GPU (simple kernels)** → write against the abstract Pallas API; both backends accept the simple cases. Performance won't be optimal on either.

### 29.4 GPU memory hierarchy in Pallas

- **GMEM** (global / HBM3) — `pl.ANY`. 80 GB on H100, 192 GB on B200.
- **SMEM** (shared memory) — per-SM scratchpad, ~228 KB.
- **Registers** — per-thread, ~256 × 32-bit per thread.
- **TMEM** (tensor memory, Blackwell only) — 256 KB / SM, feeds new tensor cores. Pallas / Mosaic-GPU TMEM support continues to evolve.

---

## Chapter 30. Worked Kernels: FlashAttention and Ragged Paged Attention

### 30.1 FlashAttention forward (revisited)

We sketched the structure in §20.3. Here are the invariants that distinguish a good Pallas FlashAttention from a slow one:

- **Q tile in registers** for the entire inner loop. Never re-read from HBM.
- **K, V tiles streamed via pipelined DMA**. On Hopper: TMA. On TPU: BlockSpec auto-pipelines.
- **Online softmax** keeps $(m_i, \ell_i)$ statistics; never materialize the $T \times S$ matrix.
- **fp32 accumulators** for the partial sums. The matmul itself runs in bf16/fp8 on tensor cores; the accumulator (`acc`) is fp32 to avoid drift.
- **Logsumexp saved** so backward can recompute attention weights without storing them.
- **Causal mask via block skipping**: if the K block is entirely above the Q block on the diagonal, skip it entirely (don't even DMA the tile).

### 30.2 Ragged paged attention sketch

For LLM inference with continuous batching, the KV cache is *paged*: each sequence's KV tokens live in fixed-size blocks scattered across HBM, indexed by a per-sequence "block table."

```python
def paged_attn_kernel(q_ref, k_pages_ref, v_pages_ref,
                      block_tables_ref, seq_lens_ref,
                      o_ref, *, page_size, block_q):
    # Grid: (batch, num_kv_heads). One Q vector per program for decode.
    b = pl.program_id(0)
    h = pl.program_id(1)

    seq_len = seq_lens_ref[b]                           # scalar in SMEM
    num_pages = pl.cdiv(seq_len, page_size)

    q = q_ref[...]                                      # (n_q_per_kv, d_head)
    m_i, l_i = -jnp.inf, 0.0
    acc = jnp.zeros_like(q, jnp.float32)

    def body(page_idx, carry):
        m_i, l_i, acc = carry
        # Look up which physical block holds logical page `page_idx`.
        phys_block = block_tables_ref[b, page_idx]      # scalar in SMEM

        # Manual async DMA — block table is dynamic, so static BlockSpec
        # with index_map doesn't suffice.
        k_tile = pl.load(k_pages_ref, (phys_block, h, slice(None), slice(None)))
        v_tile = pl.load(v_pages_ref, (phys_block, h, slice(None), slice(None)))

        # Mask out positions past seq_len in the last page.
        valid = jnp.arange(page_size) + page_idx * page_size < seq_len

        s = q @ k_tile.T
        s = jnp.where(valid[None, :], s, -jnp.inf)
        m_new = jnp.maximum(m_i, s.max(axis=-1))
        alpha = jnp.exp(m_i - m_new)
        p = jnp.exp(s - m_new[:, None])
        l_new = alpha * l_i + p.sum(axis=-1)
        acc = alpha[:, None] * acc + p @ v_tile
        return m_new, l_new, acc

    m_i, l_i, acc = jax.lax.fori_loop(0, num_pages, body, (m_i, l_i, acc))
    o_ref[...] = acc / l_i[:, None]
```

The novel pieces vs. dense FlashAttention:

- `block_tables_ref` lives in SMEM (TPU) or shared memory (GPU). Scalar lookup is cheap.
- The DMA target is *dynamic* (`phys_block` is a runtime value). This is why a static `BlockSpec` with `index_map` doesn't work; manual loads or a "dynamic BlockSpec" pattern are required.
- `seq_lens_ref` enables masking so partial last pages don't pollute the softmax.

The actual reference TPU implementation is in `jax/experimental/pallas/ops/tpu/paged_attention/`; the GPU version mirrors vLLM's PagedAttention closely.

### 30.3 Pallas vs. Triton, side-by-side

If you wrote the same FlashAttention-2 forward in raw Triton, you'd write:

- Explicit `tl.load` with `mask=(offsets < N_CTX)` for tail handling.
- Manual `tl.dot(q, k, trans_b=True)`.
- `tl.where`, `tl.exp`, `tl.maximum` on tiles.
- `@triton.autotune` with a list of `(BLOCK_Q, BLOCK_K, num_warps, num_stages)` configs.
- A separate Python launcher computing `grid = (batch * num_heads, cdiv(N_CTX, BLOCK_Q))`.

Pallas differences:

1. **No Python launcher** — `pallas_call` is the launcher; JAX handles batching via `vmap`.
2. **No autotune story baked in** — wrap `pallas_call` in `jax.jit` over different block sizes and pick at compile time.
3. **Composes with `shard_map`, `pjit`, mesh** — Triton can't.
4. **One source for TPU and GPU** — same kernel body lowers to MXU on TPU and WGMMA on GPU, with backend tweaks via `compiler_params`.
5. **Slightly less control** — can't reach into warp-level primitives. For 95% of kernels this doesn't matter.

A reasonable rule of thumb: a well-written Pallas FlashAttention on H100 reaches ~90–95% of the official Triton FlashAttention-2's throughput. On TPU there is no Triton, so Pallas is the only sane choice.

### 30.4 When (not) to write Pallas

The decision rule:

> Profile first; XLA is excellent for the common case; only write Pallas if XLA gives you a kernel that is leaving performance on the table for a structural reason.

Reasons XLA falls short:

1. **Custom attention variants.** Causal, sliding-window, ALiBi, sparse, paged, ragged batched. XLA can express these but produces multiple kernels with HBM round-trips.
2. **Sparse / structured-sparse kernels.** Block-sparse attention, top-k routing for MoE, segment-sum on ragged sequences.
3. **Fused custom ops with weird epilogues.** Matmul + dequantize + RoPE + scatter, all fused.
4. **Specialized layouts.** int4/fp4 weights packed in a specific way; speculative decoding with custom KV layout.
5. **Decode-time inference kernels.** Tiny matmuls where XLA's dispatch overhead dominates.
6. **Communication-fused kernels** (TPU). AllReduce-fused matmul for tensor parallelism.

For plain training-step kernels — embedding lookup, dense matmul, GELU, softmax, layernorm — XLA is at or very near peak. Don't write Pallas there.

### 30.5 Gotchas

- **Compile times.** Mosaic-TPU compiles can take tens of seconds for non-trivial kernels because the autoscheduler explores many tile/pipeline configurations. Cache aggressively; persistent compilation cache helps.
- **Error messages.** A shape mismatch deep inside a `BlockSpec.index_map` shows up as an MLIR verifier failure, not a Python traceback. The `pl.debug_print` macro helps; so does `JAX_TRACEBACK_FILTERING=off`.
- **No autotune.** Roll your own grid search via `jax.jit` with static argnames over `block_q, block_k`.
- **Dynamic block sizes** are not supported. Dynamic outer shapes are fine.
- **No autodiff.** Write the backward kernel by hand; glue with `custom_vjp`.

### 30.6 Debugging Pallas

The toolkit:

1. **`pl.debug_print("x = {}", x_ref[...])`** — printf from inside a kernel. Slow; remove before benchmarking.
2. **`interpret=True` flag on `pallas_call`** — runs the kernel in a pure-Python interpreter instead of lowering. ~100× slower, but you can use `pdb`, regular Python prints, and standard JAX errors. Indispensable for logic bugs.
3. **HLO/jaxpr inspection** — `jax.jit(fn).lower(...).compiler_ir()` shows the HLO around the `pallas_call`. Backend-specific dump flags (`MLIR_ENABLE_DUMP=1`) show the kernel IR.
4. **`jax.profiler` + perfetto** — DMA + compute overlap on TPU; Nsight on GPU.

The recommended workflow: write the kernel, get it correct under `interpret=True`, *then* turn off `interpret` and benchmark / autotune block shapes.

---

# Part VII — Mastering the Craft

Becoming proficient with JAX is more than knowing APIs and building models. It requires diagnosing performance issues, debugging compiled and parallel code, knowing the production stacks, and being able to translate from the imperative world. This final part covers the practical craft.

---

## Chapter 31. Profiling and the MFU Mindset

### 31.1 The MFU question

Whenever a training job is running, the question to ask is: **what fraction of theoretical peak FLOPs am I achieving?** This is the **Model FLOPs Utilization (MFU)**:

$$
\boxed{\text{MFU} = \frac{\text{achieved FLOPs/s}}{N \cdot \pi}}
$$

where $\pi$ is per-chip peak FLOPs. Achieved FLOPs/s for one training step:

$$
\text{achieved FLOPs/s} = \frac{6 \cdot \text{params} \cdot \text{tokens per step}}{T_{\text{step}}}
$$

Healthy MFUs in 2026: **40–55% on TPU v5p for dense Transformers; 30–45% on H100 for FSDP+TP**. If you're below 25%, something is leaving 2× on the table — and you should profile.

### 31.2 Capturing traces

```python
import jax

with jax.profiler.trace("/tmp/jax_profile"):
    state, loss = train_step(state, batch)
    loss.block_until_ready()
```

`jax.profiler.start_trace(...)` / `stop_trace()` is the alternative idiom for non-context use.

For TPU, the trace files visualize in **TensorBoard's profiler plugin** (XProf):

```bash
tensorboard --logdir=/tmp/jax_profile
```

For GPU, **Nsight Compute** is the canonical tool, with `jax.profiler.start_server()` exposing trace data via a network port.

### 31.3 The XProf views

Three main views:

- **Trace Viewer** — chronological timeline of operations on each device. Shows idle gaps, communication overlapping (or not) with compute, kernel durations.
- **Op Stats** / **Graph Viewer** — aggregate stats per HLO op: total time, FLOPs, memory traffic. Compares achieved vs theoretical.
- **Memory Profile** — memory usage over time. Useful for OOM debugging and identifying where peak memory is spent.

### 31.4 What healthy vs. sick looks like

**Healthy** training profile:

- Compute kernels (matmuls, attention) are back-to-back with minimal gaps.
- Communication ops overlap with compute (look for AllGather and matmul kernels running in parallel).
- HBM bandwidth utilization is 60–80% during memory-bound ops.
- MXU utilization is 80%+ during matmuls.
- Per-step time is roughly $T_{\text{math}}$ from the roofline, not significantly above.

**Sick** training profile:

- Long "host preprocessing" gaps between steps → data pipeline bottleneck. Use Grain with more workers; profile dataloader independently with `time.perf_counter`.
- Long single-kernel times that blow past theoretical → check the HLO for missed fusion, layout retiling, or padding.
- Communication kernels are not overlapping with compute → use `with_sharding_constraint` to hint better placement; consider hybrid sharding strategies.
- "MXU utilization 30%" while compute kernels look long → padding / shape mismatch. The VMEM-to-MXU pipe is the bottleneck.
- "ReduceScatter taking 5× longer than expected" → DCN bandwidth is the cap; can the strategy shift to pure intra-pod?

### 31.5 The roofline check

The most empowering profiling exercise: pick the slowest kernel in the trace, compute its theoretical roofline time, divide measured by theoretical. If the ratio is close to 1, the kernel is healthy and the slowness is *intrinsic*. If the ratio is 2× or more, there's a fixable structural issue.

Worked example from the book: a 32×1024 by 1024×8192 matmul on TPU v2-8.

$$
T_{\text{math}} = \frac{2 \cdot 32 \cdot 1024 \cdot 8192}{23 \times 10^{12} \cdot 8} = 95.6 \text{ ms}
$$

Measured: **96 ms**. The kernel is at peak — no further optimization possible without changing shapes or precision.

### 31.6 Live memory inspection

For interactive memory monitoring (especially on TPU where `nvidia-smi` doesn't apply), `jax-smi` periodically dumps memory profiles you can inspect from the CLI. Useful for "is this OOM about to happen?"

---

## Chapter 32. Debugging JIT-compiled and Parallel Code

Standard Python debuggers (`pdb`, `print`) are largely ineffective inside JIT-compiled or parallelized functions. The Python code only runs once during tracing, and numerical values are not available then. JAX provides specialized utilities.

### 32.1 `jax.debug.print`

Use in place of `print` inside JIT-compiled functions. Embeds into the compiled graph so it prints actual device values at runtime:

```python
@jax.jit
def f(x):
    y = x * 2
    jax.debug.print("Intermediate y: {y}", y=y)
    return y / x
```

Caveats:
- Introduces synchronization; can alter performance.
- Order of prints from different devices under `pmap` / `shard_map` is not guaranteed.
- Prints from inside `vmap` interleave per batch element.

For host-side callbacks (Python function call from inside JAX, useful for logging to wandb), use `jax.debug.callback`.

### 32.2 `jax.debug.breakpoint`

For interactive debugging *inside* a compiled function:

```python
@jax.jit
def f(x):
    z = x / (x - 1.0)
    if jnp.any(jnp.isinf(z)):
        jax.debug.breakpoint()    # opens a pdb-like prompt
    return z
```

### 32.3 NaN debugger

```python
jax.config.update("jax_debug_nans", True)
```

Raises an error immediately upon NaN/Inf creation inside JIT code. Slow; use only when chasing.

### 32.4 The "double where" trick for stable backwards

A common NaN source: `jnp.log(x)` where $x \le 0$ at certain inputs, even if those inputs end up zeroed out in the forward (because the backward still computes the gradient through the bad branch).

```python
# BAD: even though we zero out x<=0, the grad through log(x) for x<=0 is NaN
y = jnp.where(x > 0, jnp.log(x), 0.0)

# GOOD: replace x with a safe value before taking log
safe_x = jnp.where(x > 0, x, 1.0)
y = jnp.where(x > 0, jnp.log(safe_x), 0.0)
```

The second form has a finite gradient everywhere because `log(safe_x)` never hits a problematic value.

### 32.5 Tracer errors

```
TracerBoolConversionError: Attempted boolean conversion of traced array
```

Means Python `if`/`while` is branching on a traced value. Fix:

- Use `jax.lax.cond`, `jax.lax.scan`, `jax.lax.while_loop`.
- Mark the offending argument `static_argnames` if its values are few.
- Refactor.

### 32.6 OOM debugging

When you OOM:

1. **Sanity-check memory.** Total = params + grads + opt state + activations + KV cache. Estimate per-chip with the Chapter 19/25 formulas.
2. **Add gradient checkpointing.** Wrap layers in `nnx.remat` (or `jax.checkpoint`).
3. **Reduce per-chip activation by sharding.** Sequence parallelism; FSDP+TP.
4. **Reduce optimizer-state precision.** bf16 master + Kahan, or shard optimizer state with ZeRO-1.
5. **Reduce batch.** Last resort; impacts MFU.

---

## Chapter 33. Production Stacks

You usually don't write a 70 B-model training stack from scratch. The 2026 reference stacks:

### 33.1 MaxText

<https://github.com/google/maxtext>

Google's reference open-source LLM training codebase. Pure JAX/Flax NNX. Designed to scale from a single TPU host to thousands of chips with `shard_map` and high MFU. Models: LLaMA 1/2/3, Gemma, Mixtral, DeepSeek, GPT-OSS.

What you get:
- A clean training loop with checkpointing (Orbax), data (Grain), and metrics (TensorBoard).
- All major sharding strategies (DP, FSDP, TP, EP, sequence-parallel).
- Mixed precision (bf16 / fp8) and quantization (AQT int8, fp8).
- Inference path via JetStream integration.

Best as a reference: read the configs (`MaxText/configs/`) to see how a production training run is structured, even if you don't use it directly.

### 33.2 Levanter

<https://github.com/stanford-crfm/levanter>

Stanford CRFM's Haliax-based LLM trainer. Bit-exact reproducibility, named tensors, very clean code. Especially valuable if you want named-axis tensor programming (Haliax is a sibling of Penzai's named arrays).

### 33.3 AXLearn

<https://github.com/apple/axlearn>

Apple's open-source large-model training framework. Mixture-of-experts, evaluation harnesses, custom serving on TPU and GPU. A third industrial-strength reference alongside MaxText and Levanter.

### 33.4 JetStream

<https://github.com/AI-Hypercomputer/JetStream>

TPU-first inference server (from Google). Continuous batching, paged KV, prefill chunking, fp8/int8 quantized serving, speculative decoding. Reference engines: Pax (legacy) and MaxText (current).

### 33.5 Tunix

<https://github.com/google/tunix>

JAX-native post-training: SFT, RLHF, DPO/GRPO, LoRA. NNX-native. Integrates with MaxText and JetStream for the rollout stage.

### 33.6 T5X (legacy)

<https://github.com/google-research/t5x>

The earlier (Linen-based) Google research training framework. Effectively retired. A lot of public checkpoints (T5, UL2, PaLM-derivatives) and tutorial code still reference it; mention it exists, but **don't point new readers there.**

### 33.7 The decision

For a typical 2026 project:

- **Pretraining a 1B–70B model from scratch** → MaxText. Configure, fork only what you must.
- **A novel architecture / research** → Roll your own with NNX + Optax + Grain + Orbax (Chapter 13's skeleton).
- **Fine-tuning an open-weights model** → Tunix or HuggingFace's PEFT-with-JAX backend.
- **Serving** → JetStream (TPU) or vLLM-jax / SGLang-jax (GPU).

---

## Chapter 34. Migrating from PyTorch

Migrating a project from PyTorch to JAX is a common task for those seeking JAX's performance and parallelism story. The process involves a shift in mindset from object-oriented stateful code to functional stateless code.

### 34.1 Step-by-step

**1. Model definition.** Convert `torch.nn.Module` to `flax.nnx.Module`:

- Submodules go in `__init__`, exactly like PyTorch.
- The forward pass is a regular `__call__`, but uses `jax.nn` ops and `jnp` arrays instead of `torch.nn.functional` and `torch.Tensor`.
- The model holds parameters as `nnx.Param` attributes; access with `.value` for the underlying array.

**2. State management.** PyTorch's implicit state becomes explicit:

- An `nnx.Optimizer` wraps `(model, optax_optimizer)` and tracks both. Use `optimizer.update(grads)` to apply gradients in place.
- Or: split the optimizer into `(graphdef, state)` for full functional control with `nnx.split` / `nnx.merge`.
- BatchNorm / Dropout state lives in the module itself; toggle with `model.train()` / `model.eval()`.

**3. Training loop.** The imperative PyTorch loop:

```python
optimizer.zero_grad()
loss = model(x, y)
loss.backward()
optimizer.step()
```

becomes a pure `train_step` function:

```python
@jax.jit
def train_step(opt_state, batch):
    optimizer = nnx.merge(graphdef, opt_state)
    def loss_fn(model):
        return loss(model(batch['x']), batch['y'])
    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    return nnx.split(optimizer)[1], loss
```

**4. Data loading.** The easiest part. Reuse the existing PyTorch `DataLoader`. The only change: write a custom `collate_fn` that returns NumPy arrays:

```python
def numpy_collate(batch):
    return {k: np.stack([item[k] for item in batch]) for k in batch[0]}

loader = DataLoader(dataset, batch_size=B, collate_fn=numpy_collate)
```

Or migrate to Grain (Chapter 11) for determinism and multi-host correctness.

**5. Checkpoints.** Replace `torch.save` / `torch.load` with Orbax (Chapter 12). Conversion of existing PyTorch checkpoints requires mapping parameter names — typically a one-time script.

### 34.2 Side-by-side cheat sheet

| Aspect | PyTorch | JAX/Flax NNX |
| --- | --- | --- |
| Model def | `class Model(nn.Module): def __init__(self): self.layer = nn.Linear(...)` | `class Model(nnx.Module): def __init__(self, *, rngs): self.layer = nnx.Linear(..., rngs=rngs)` |
| Instantiation | `model = Model().to(device)` | `model = Model(rngs=nnx.Rngs(0))` |
| Forward | `logits = model(batch)` | `logits = model(batch)` |
| Optimizer | `optimizer = optim.Adam(model.parameters())` | `optimizer = nnx.Optimizer(model, optax.adam(...))` |
| Train step | `loss.backward(); optimizer.step()` | `loss, grads = nnx.value_and_grad(loss_fn)(model); optimizer.update(grads)` |
| Distributed | `DistributedDataParallel(model)` | `jax.tree.map(lambda p: NamedSharding(mesh, P('data')), params); jax.device_put(...)` |
| Checkpoint | `torch.save(model.state_dict(), f)` | `mgr.save(step, args=ocp.args.StandardSave(state))` |

### 34.3 Less-common interop

For direct interoperability with PyTorch models inside JAX transforms, libraries like `torch2jax` convert PyTorch modules and tensors into JAX-compatible objects that work under `jit` and `grad`. Useful for incrementally migrating, less performant than a full JAX rewrite.

---

## Chapter 35. The Universal Algorithm-to-Performance Recipe

The ultimate JAX skill: implementing any algorithm or formula at high performance. The recipe is the same regardless of complexity.

### 35.1 Five steps

**1. Express the core logic as a pure function.** Write the math in `jnp` ops, operating on a single example or the smallest logical unit. Don't think about batching, gradients, or distribution yet.

**2. Vectorize with `vmap`.** If the operation runs on batches, apply `jax.vmap` with appropriate `in_axes`. The function reads identically; only the input shapes change.

**3. Differentiate with `grad`.** If part of optimization, wrap with `jax.grad` or `jax.value_and_grad`. The math becomes the computation; gradients are automatic.

**4. Compile with `jit`.** Wrap the final computation in `jax.jit` for XLA fusion and kernel optimization.

**5. Distribute with `shard_map` or `jit`-with-sharding.** For multi-device, put your inputs under a `NamedSharding` and let GSPMD partition; or write per-shard with `shard_map`.

### 35.2 Worked example: a simple physics-informed loss

A neural ODE with a regularizer that penalizes divergence of the learned vector field:

```python
def vector_field(params, x):
    """Single-point neural network: x ∈ R^d → R^d."""
    return mlp(params, x)

def divergence(params, x):
    """∇·f at point x — sum of partial derivatives."""
    f = lambda x: vector_field(params, x)
    jac = jax.jacrev(f)(x)
    return jnp.trace(jac)

def loss(params, batch_x):
    """Mean of divergence² over batch."""
    div = jax.vmap(divergence, in_axes=(None, 0))(params, batch_x)
    return jnp.mean(div ** 2)

@jax.jit
def train_step(params, opt_state, batch_x):
    grads = jax.grad(loss)(params, batch_x)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state
```

We walked through every transformation:
- `vector_field` is the pure single-point logic.
- `divergence` builds on it with `jax.jacrev`.
- `loss` lifts to a batch with `vmap`.
- `train_step` adds `grad` for gradients and `jit` for compilation.

For multi-device, add: `params = jax.device_put(params, NamedSharding(mesh, P()))` (replicated) and `batch_x = jax.device_put(batch_x, NamedSharding(mesh, P('data')))` (sharded). The compiler handles the rest.

### 35.3 The pattern, over and over

This compositional pattern — small pure pieces, layered transformations — is JAX's defining strength. It is what makes "this is the math" and "this is fast distributed code" feel like the same statement, just composed differently.

That is the lesson of the entire guide. Master the parts. Compose them. The rest is detail.

---

# Conclusion

JAX is a paradigm shift in high-performance numerical computing. It moves away from imperative, object-oriented frameworks and toward an explicit, functional, transformation-oriented model. The shift requires investment — purity, immutability, explicit state — and rewards it with composability and performance no imperative system can match.

The five pillars (`grad`, `jit`, `vmap`, `shard_map`, `pmap`) are not isolated tools but a *grammar* for computation. Every advanced technique in this guide — FSDP, FlashAttention, paged KV cache, MoE expert parallelism, Pallas kernels — emerges from layering those primitives on small, verifiable building blocks.

The hardware substrate (Part II) is what gives JAX its edge, because JAX's transformations are aware of the substrate in a way other frameworks aren't. Roofline thinking. Communication-cost reasoning. Sharding as a first-class type. These aren't just optimizations; they're a *language* for talking about modern accelerators.

The modern ecosystem (Part III) — Flax NNX, Equinox, Optax, Grain, Orbax — has matured into something coherent. The "story, flow, depth" the original ask asked for is real now. You can build, train, and ship a 70 B-parameter model with code that is shorter and more readable than what PyTorch + accelerate + DeepSpeed gives you.

The LLM stack (Part V) is where the field has moved fastest. FlashAttention, paged KV, GQA, RoPE/YaRN, MoE, FP8, μP — these were all research in 2023 and are all production by 2026. JAX has first-class support for every one of them, often as Pallas reference implementations.

Pallas (Part VI) is the escape hatch. When XLA can't generate the kernel you need, you write it. The `pallas_call` slots into your model the same as any other op — composing with `jit`, `vmap`, `shard_map`. Triton on the GPU. Mosaic on the TPU. One source.

For the ambitious practitioner, mastering this stack offers a durable advantage. You can express ideas more directly than in PyTorch, scale them further than in TensorFlow, and reason about their performance in ways that aren't possible without seeing through the abstraction. The investment pays compound returns.

---

## Resources

### Canonical references

- **JAX docs** — <https://docs.jax.dev/en/latest/>
- **How to Scale Your Model** (the book this guide leans on) — <https://jax-ml.github.io/scaling-book/>
- **Flax (NNX)** — <https://flax.readthedocs.io/>
- **Optax** — <https://optax.readthedocs.io/>
- **Grain** — <https://github.com/google/grain>
- **Orbax** — <https://orbax.readthedocs.io/>
- **Pallas** — <https://docs.jax.dev/en/latest/pallas/>

### Hardware references

- **NVIDIA H100 whitepaper** — <https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper>
- **NVIDIA Hopper Tuning Guide** — <https://docs.nvidia.com/cuda/hopper-tuning-guide/>
- **NVIDIA Blackwell architecture brief** — <https://resources.nvidia.com/en-us-blackwell-architecture>
- **TPU v5p docs** — <https://cloud.google.com/tpu/docs/v5p>
- **Trillium (v6e) docs** — <https://cloud.google.com/tpu/docs/v6e>
- **TPU v4 paper** (Jouppi et al., ISCA 2023) — arXiv:2304.01433

### Production stacks

- **MaxText** — <https://github.com/google/maxtext>
- **Levanter** — <https://github.com/stanford-crfm/levanter>
- **AXLearn** — <https://github.com/apple/axlearn>
- **JetStream** — <https://github.com/AI-Hypercomputer/JetStream>
- **Tunix** — <https://github.com/google/tunix>

### Key papers

- FlashAttention v1/v2/v3: arXiv:2205.14135, 2307.08691, 2407.08608
- PagedAttention / vLLM: arXiv:2309.06180
- RoPE: arXiv:2104.09864; YaRN: arXiv:2309.00071; LongRoPE: arXiv:2402.13753
- GQA: arXiv:2305.13245
- Switch Transformer: arXiv:2101.03961; Mixtral: arXiv:2401.04088; DeepSeek-V3: arXiv:2412.19437
- μP: arXiv:2203.03466
- Speculative decoding: arXiv:2211.17192; EAGLE: arXiv:2401.15077

### Conceptual references

- Williams, Waterman & Patterson, "Roofline" (CACM 2009)
- Horace He, "Making Deep Learning Go Brrrr From First Principles" — <https://horace.io/brrr_intro.html>
- Patarasuk & Yuan, "Bandwidth Optimal All-Reduce Algorithms" (JPDC 2009)
- Stanford CS336 "Large Language Models" — course materials online
- HuggingFace Ultra-Scale Playbook — <https://huggingface.co/spaces/nanotron/ultrascale-playbook>

### JAX adjacent libraries

- **Equinox** — <https://docs.kidger.site/equinox/>
- **Penzai** + **Treescope** — <https://penzai.readthedocs.io/>, <https://treescope.readthedocs.io/>
- **chex** — <https://chex.readthedocs.io/>
- **jaxtyping** — <https://docs.kidger.site/jaxtyping/>
- **diffrax** (ODEs) — <https://docs.kidger.site/diffrax/>
- **lineax** (linear solvers) — <https://docs.kidger.site/lineax/>
- **AQT** (quantization) — <https://github.com/google/aqt>
- **qwix** (quantization) — <https://github.com/google/qwix>

---

*End of guide.*



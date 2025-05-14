# Flex RNN
Flex RNN compiles linear attention mechanisms like RWKV and Mamba from simple pytorch code. For example, the Delta Rule kernel can be implemented as
```python
@flex_rnn.jit
def delta_rule(q, k, v, beta, S):
    S = S + beta * k * (v - (S*k).sum(-1,True))
    return (S*q).sum(-1,True), S
```

See [tests/examples.py](tests/examples.py) for example implementations of RWVK-7, RWKV-6, RWKV-4, Mamba-2, Mamba, Delta Rule, Gated Delta Rule, Retention, GLA, GSA, HGRN, Longhorn, mLSTM and S4D.

The compiler traces the code, generates the backward pass, compiles it to cuda code, and returns a callable function equivalent to `naive` defined below. Regardless of input and output dtypes, all internal calculations are performed using 32-bit floats. The backward pass fully recalculates all states, so precision should be similar to torch autograd. See below for how the speed compares to other kernels.

## Usage
For batch size `B`, sequence length `T`, number of heads `H` and state dimensions `M`x`N`, the resulting function expects inputs with shapes `[B,T,H,M,N]` followed by initial state(s) with shape(s) `[B,H,M,N]`. Inputs with dimensions of size 1 will be broadcast, so for the Delta Rule, q should have shape `[B,T,H,1,N]` and v have shape `[B,T,H,M,1]`. Mamba-2's `A` is a headwise parameter with no batch or time dependence, so it would have shape `[1,1,H,1,1]`. Initial states are not broadcast.

Let's see how to implement an equivalent to `fla.ops.delta_rule.chunk_delta_rule(q, k, v, beta, scale = 1, initial_state = S, output_final_state = True)`. That function expects `q`, `k` and `v` to have shape `[B,T,H,M]`, `beta` to have shape `B,T,H` and the initial state to have shape `[B,T,H,M,N]`. Hence, we can implement it by unsqueezing as follows:
```python
def matching_delta_rule(q, k, v, beta, S):
    q, k, v, beta = [i.unsqueeze(-2) for i in (q, k, v, beta.unsqueeze(-1))]
    y, S = delta_rule(q, k, v.mT, beta, S.mT)
    return y, S.mT 
```

In general, the compiled function behaves like `jit.naive` from [flex_rnn/jit.py](flex_rnn/jit.py). It is often helpful to first debug using the naive function (for example, use `y, S = delta_rule.naive(q, k, v.mT, beta, S.mT)` instead of `y, S = delta_rule(q, k, v.mT, beta, S.mT)` in the code above), before working with the fast compiled kernels.

When there's a single state variable and a single output variable, and disregarding strides of outputs, the compiled function behaves like
```python
def simplified_naive(step, *args):
    inputs, state = args[:-1], args[-1]
    T = max(i.shape[1] for i in inputs)
    output = []
    for t in range(T):
        inputs_t = [i.expand(-1,T,-1,-1,-1)[:,t].float() for i in inputs]
        output_t, state = step(*inputs_t, state)
        output.append(output_t)
    return torch.stack(output, dim=1).squeeze(-1), state
```

## Installation
```bash
git clone https://github.com/johanwind/flex_rnn
cd flex_rnn
pip install .
```

You should then be able to run examples like
```python
python tests/examples.py --op rwkv7 --check-flex
```

## Speed
Batch size `B = 8`, sequence length `T = 1024`, number of heads `H = 4096 / 128 = 32` and state dimensions `128`x`128`. NVIDIA 4070 mobile GPU.

| Op | Reference | flex_rnn / reference time | flex_rnn / reference memory |
|:---|:---|---:|---:|
| RWKV-4 | [fla](https://github.com/fla-org/flash-linear-attention) fused_recurrent_rwkv4 | 3 ms / 6 ms | 0.7 GB / 0.9 GB |
| RWKV-6 | [fla](https://github.com/fla-org/flash-linear-attention) chunk_rwkv6 | 44 ms / 34 ms | 3.2 GB / 2.5 GB |
| RWKV-7 | [fla](https://github.com/fla-org/flash-linear-attention) chunk_rwkv7 | 55 ms / 82 ms | 3.2 GB / 3.4 GB |
| HGRN | [fla](https://github.com/fla-org/flash-linear-attention) chunk_hgrn | 3 ms / 11 ms | 0.6 GB / 0.9 GB |
| Retention | [fla](https://github.com/fla-org/flash-linear-attention) chunk_retention | 31 ms / 13 ms | 1.9 GB / 1.4 GB |
| GLA | [fla](https://github.com/fla-org/flash-linear-attention) chunk_gla | 32 ms / 32 ms | 2.2 GB / 2.4 GB |
| GSA | [fla](https://github.com/fla-org/flash-linear-attention) chunk_gsa [^1] | 105 ms / 67 ms | 3.1 GB / 2.5 GB |
| Delta Rule | [fla](https://github.com/fla-org/flash-linear-attention) chunk_delta_rule | 45 ms / 19 ms | 2.8 GB / 1.2 GB |
| Gated Delta Rule | [fla](https://github.com/fla-org/flash-linear-attention) chunk_gated_delta_rule | 50 ms / 24 ms | 2.8 GB / 1.3 GB |
| Mamba | [mamba_ssm](https://github.com/state-spaces/mamba) selective_scan_fn | 32 ms / 17 ms | 1.5 GB / 0.8 GB |
| Mamba2  | [mamba_ssm](https://github.com/state-spaces/mamba) mamba_chunk_scan_combined | 39 ms / 10 ms | 2.5 GB / 0.6 GB |
| Longhorn | [github.com/Cranial-XIX/longhorn_cuda](https://github.com/Cranial-XIX/longhorn_cuda) | 120 ms / 115 ms | 3.2 GB / 0.8 GB |
| S4D | [github.com/state-spaces/s4](https://github.com/state-spaces/s4) (with log_vandermonde_cuda) | 40 ms / 60 ms | 1.6 GB / 2.1 GB |
| mLSTM | [mlstm_kernels](https://github.com/nx-ai/mlstm_kernels) mlstm_chunkwise__xl_chunk [^2] | 49 ms / 12 ms | 3.3 GB / 1.1 GB |

All inputs and initial states in bfloat16 except S4D and Mamba (because their reference implementations only support float32).

[^1]: Gives large gradient errors (~4%). fused_recurrent_gsa is more accurate, but slower.
[^2]: Gives incorrect gradients, i.e. several of mlstm_chunkwise__xl_chunk's gradients are completely different from their references like mlstm_chunkwise__native_autograd.

## Features
- Compatible with torch.compile without graph breaks
- Handles non-contiguous input strides without copying
- float32, bfloat16, float16 input and output types
- High precision because of float32 internal calculations

## Limitations
- The backend computes the state row-wise. Specifically, that means we only support reductions in the last dimension, i.e. `torch.sum(x, dim=-1, keepdim=True)`.
- The supported operations are essentially `+, -, *, /, <, >, exp, log, sqrt` and `x.sum(dim=-1, keepdim=True)`.
- Input dimensions have only been tested for powers of 2.
- Speed and memory usage are often worse than specialized implementations (see speed comparison above).

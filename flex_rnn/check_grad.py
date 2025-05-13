import torch as th
import triton.testing

def myclone(x):
    #return x.clone() # Doesn't preserve stride if it's not dense and non-overlapping
    # The following preserves stride of th.randn(1).expand(10), unlike something like x.clone(), but has other issues
    r = th.empty_strided(x.shape,x.stride(),dtype=x.dtype,device=x.device)
    r.copy_(x)
    #r.untyped_storage().copy_(x.untyped_storage())
    return r

def check_grad(f1, f2, inputs, backward = True, aux=(), tol = 1e-2, verbose = True):
    inputs = [myclone(p) for p in inputs]
    if backward:
        inputs = [p.requires_grad_() for p in inputs]
    y1 = f1(*inputs,*aux)
    y2 = f2(*inputs,*aux)
    if type(y1) != tuple: y1 = (y1,)
    if type(y2) != tuple: y2 = (y2,)
    def rel(a,b): return (a-b).norm()/max(b.norm(),1e-30)
    max_err = 0
    if verbose: print('Forward rel. error'+'s'*(len(y1)>1))
    for a,b in zip(y1,y2):
        assert a.shape == b.shape, 'Shape mismatch '+str([(a.shape,b.shape) for a,b in zip(y1,y2)])
        max_err = max(max_err, rel(a,b))
        if verbose: print(f'{rel(a,b):.2e}  ({b.norm():.0e})')
    if not backward: return
    dy = tuple(th.randn_like(i) for i in y1)

    d1 = th.autograd.grad(y1, inputs, grad_outputs=tuple(myclone(i) for i in dy), allow_unused=True)
    for p in inputs:
        if p.grad is not None:
            p.grad.random_() # So th.empty doesn't recover the gradient
        p.grad = None
    d2 = th.autograd.grad(y2, inputs, grad_outputs=dy, allow_unused=True)
    if verbose: print('Gradient rel. errors')
    for a,b,c in zip(d1,d2,inputs):
        if a is None: a = th.zeros_like(c)
        if b is None: b = th.zeros_like(c)
        max_err = max(max_err, rel(a,b))
        if verbose: print(f'{rel(a,b):.2e}  ({b.norm():.0e})')

    assert max_err < tol, f'Large grad_check error: {max_err*100:.1f}%!'
    return max_err


def benchmark(f, params, backward = True, aux=()):
    if backward:
        for p in params: p.requires_grad_()
    dy = None
    def wrap():
        y = f(*params,*aux)
        if not backward: return
        nonlocal dy
        if type(y) != tuple: y = (y,)
        if dy is None: dy = tuple(th.randn_like(i) for i in y)
        return th.autograd.grad(y, params, grad_outputs=dy, allow_unused=True)

    wrap() # Warmup (compile triton)
    th.cuda.synchronize()
    th.cuda.reset_peak_memory_stats()
    wrap() # Measure memory
    th.cuda.synchronize()
    print(f'memory {th.cuda.max_memory_allocated()/2**30:.2f} GB')
    ms, min_ms, max_ms = triton.testing.do_bench(wrap, quantiles=[0.5,0.2,0.8], warmup=500,rep=1000)
    print(f'{ms:.2f} ms ({min_ms:.2f} - {max_ms:.2f})')
    return ms

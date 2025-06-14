import argparse, flex_rnn, torch as th
th.manual_seed(0)
th.set_default_device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--batchsz', type=int, default=8)
parser.add_argument('--modeldim', type=int, default=None)
parser.add_argument('--headsz', type=int, default=128)
parser.add_argument('--seqlen', type=int, default=None)
parser.add_argument('--op', type=str, default='rwkv7')
parser.add_argument('--benchmark-flex', action=argparse.BooleanOptionalAction)
parser.add_argument('--benchmark-ref', action=argparse.BooleanOptionalAction)
parser.add_argument('--check-flex', action=argparse.BooleanOptionalAction)
parser.add_argument('--check-ref', action=argparse.BooleanOptionalAction)
parser.add_argument('--torch-compile', action=argparse.BooleanOptionalAction)
parser.add_argument('--forward', action=argparse.BooleanOptionalAction) # Forward pass only
cmd_args = parser.parse_args()

if cmd_args.modeldim is None:
    cmd_args.modeldim = (256 if (cmd_args.check_flex or cmd_args.check_ref) else 4096)
if cmd_args.seqlen is None:
    cmd_args.seqlen = (256 if (cmd_args.check_flex or cmd_args.check_ref) else 1024)

op = cmd_args.op
B,T,M,C = cmd_args.batchsz, cmd_args.seqlen, cmd_args.modeldim, cmd_args.headsz
H = M//C

do_backward = not cmd_args.forward
need_ref = not (cmd_args.benchmark_flex or cmd_args.check_flex)

print('Testing', op)
print(f'Input sizes {B=}, {T=}, {H=}, {C=}')

assert op in ['rwkv7', 'rwkv6', 'gla', 'retention', 'delta_rule', 'gated_delta_rule', 'mamba2', 'longhorn', 'rwkv4', 'hgrn', 'mamba', 's4d', 'gsa', 'mlstm']


if op == 'rwkv7':
    q,w,k,v,a,b = [th.randn(B,T,H,C, dtype=th.bfloat16) for i in range(6)]
    w = -th.sigmoid(w)
    a = th.nn.functional.normalize(a, p=2,dim=-1)
    b = -a*th.sigmoid(b)
    s = th.randn(B,H,C,C)
    inputs = (q,w,k,v,a,b,s)

    if need_ref:
        from fla.ops.rwkv7 import chunk_rwkv7
        def ref(q,w,k,v,a,b,s):
            return chunk_rwkv7(q,w,k,v,a,b,1,s,True)

    @flex_rnn.jit
    def step(q,w,k,v,a,b,s):
        s = s*w.exp() + (s*a).sum(-1,True)*b + v*k
        return (s*q).sum(-1,True).to(q.dtype), s

    def flex(q,w,k,v,a,b,s):
        q,w,k,v,a,b = [i.unsqueeze(-2) for i in (q,w,k,v,a,b)]
        y,s = step(q,w,k,v.mT,a,b,s.mT)
        return y,s.mT


elif op == 'rwkv6':
    q,w,k,v = [th.randn(B,T,H,C) for i in range(4)]
    w = -th.sigmoid(w)
    u = th.randn(H,C)
    s = th.randn(B,H,C,C)
    inputs = (q,k,v,w,u,s)

    if need_ref:
        from fla.ops.rwkv6 import chunk_rwkv6
        def ref(q,k,v,w,u,s):
            return chunk_rwkv6(q,k,v,w,u,1,s,True)

    @flex_rnn.jit
    def step(q,k,v,w,u,s):
        y = ((s+v*(u*k))*q).sum(-1,True)
        s = s*w.exp() + v*k
        return y.to(q.dtype), s

    def flex(q,k,v,w,u,s):
        q,k,v,w,u = [i.unsqueeze(-2) for i in (q,k,v,w,u[None,None])]
        y,s = step(q,k,v.mT,w,u,s.mT)
        return y,s.mT


elif op == 'gla':
    q,k,v,g = [th.randn(B,T,H,C) for i in range(4)]
    g = -th.sigmoid(g)
    s = th.randn(B,H,C,C)
    inputs = (q,k,v,g,s)

    if need_ref:
        from fla.ops.gla import chunk_gla
        def ref(q,k,v,g,s):
            return chunk_gla(q,k,v,g,1,s,True)

    @flex_rnn.jit
    def step(q,k,v,g,s):
        s = s*g.exp() + v*k
        return (s*q).sum(-1,True).to(q.dtype), s

    def flex(q,k,v,g,s):
        q,k,v,g = [i.unsqueeze(-2) for i in (q,k,v,g)]
        y,s = step(q,k,v.mT,g,s.mT)
        return y,s.mT


elif op == 'retention':
    q,k,v = [th.randn(B,T,H,C) for i in range(3)]
    s = th.randn(B,H,C,C)
    inputs = (q,k,v,s)

    if need_ref:
        from fla.ops.retention import chunk_retention
        def ref(q,k,v,s):
            return chunk_retention(q,k,v,1,s,True)

    @flex_rnn.jit
    def step(w,q,k,v,s):
        s = w*s + v*k
        return (s*q).sum(-1,True).to(q.dtype), s

    w = (1-2**(-5-th.arange(H).float())).view(1,1,H,1,1).expand(B,T,H,1,1).contiguous()
    def flex(q,k,v,s):
        q,k,v = [i.unsqueeze(-2) for i in (q,k,v)]
        y,s = step(w,q,k,v.mT,s.mT)
        return y,s.mT


elif op == 'delta_rule':
    q,k,v = [th.randn(B,T,H,C,dtype=th.bfloat16) for i in range(3)]
    k = th.nn.functional.normalize(k, p=2,dim=-1)
    beta = th.randn(B,T,H,dtype=th.bfloat16)
    s = th.randn(B,H,C,C)
    inputs = (q,k,v,beta,s)

    if need_ref:
        from fla.ops.delta_rule import chunk_delta_rule
        def ref(q,k,v,beta,s):
            return chunk_delta_rule(q,k,v,beta,1,s,True)

    @flex_rnn.jit
    def step(q,k,v,beta,s):
        s = s + beta * k * (v - (s*k).sum(-1,True))
        return (s*q).sum(-1,True).to(q.dtype), s

    def flex(q,k,v,beta,s):
        q,k,v,beta = [i.unsqueeze(-2) for i in (q,k,v,beta.unsqueeze(-1))]
        y,s = step(q,k,v.mT,beta,s.mT)
        return y,s.mT


elif op == 'gated_delta_rule':
    q,k,v = [th.randn(B,T,H,C,dtype=th.bfloat16) for i in range(3)]
    k = th.nn.functional.normalize(k, p=2,dim=-1)
    g = -th.sigmoid(th.randn(B,T,H,dtype=th.bfloat16))
    beta = th.randn(B,T,H,dtype=th.bfloat16)
    s = th.randn(B,H,C,C)
    inputs = (q,k,v,g,beta,s)

    if need_ref:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        def ref(q,k,v,g,beta,s):
            return chunk_gated_delta_rule(q,k,v,g,beta,1,s,True)

    @flex_rnn.jit
    def step(q,k,v,g,beta,s):
        s = g.exp() * s
        s = s + beta * k * (v - (s*k).sum(-1,True))
        return (s*q).sum(-1,True).to(q.dtype), s

    def flex(q,k,v,g,beta,s):
        q,k,v,g,beta = [i.unsqueeze(-2) for i in (q,k,v,g.unsqueeze(-1),beta.unsqueeze(-1))]
        y,s = step(q,k,v.mT,g,beta,s.mT)
        return y,s.mT


elif op == 'mamba2':
    x = th.randn(B,T,H,C)
    dt = th.sigmoid(th.randn(B,T,H))
    A = -th.rand(H)
    b = th.randn(B,T,1,C)
    c = th.randn(B,T,1,C)
    s = th.randn(B,H,C,C)
    inputs = (x,dt,A,b,c,s)

    if need_ref:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        def ref(x,dt,A,b,c,s):
            return mamba_chunk_scan_combined(x,dt,A,b,c, chunk_size=256, initial_states=s, return_final_states=True)

    @flex_rnn.jit
    def step(x,dt,A,b,c,s):
        s = (A*dt).exp() * s + dt * x * b
        return (s*c).sum(-1,True).to(x.dtype), s

    def flex(x,dt,A,b,c,s):
        x,dt,A,b,c = [i.unsqueeze(-2) for i in (x,dt.unsqueeze(-1),A[None,None,:,None],b,c)]
        y,s = step(x.mT,dt,A,b,c,s)
        return y,s


elif op == 'longhorn':
    u = th.randn(B,H*C,T)
    q = th.randn(B,C,T)
    k = th.randn(B,C,T)
    dt = th.randn(B,H*C,T)
    inputs = (u,q,k,dt)

    if need_ref:
        import warnings
        warnings.simplefilter("ignore", FutureWarning)
        from longhorn.ops.selective_scan_interface import selective_scan_online7_fn # from https://github.com/Cranial-XIX/longhorn_cuda
        ref = selective_scan_online7_fn

    @flex_rnn.jit
    def step(u,q,k,dt,s):
        k2 = k*k
        dt = 1/(1+(-dt).exp() + k2.sum(-1,True))
        s = s - dt*s*k2 + dt*u*k
        return (s*q).sum(-1,True).to(q.dtype), s

    def flex(u,q,k,dt):
        u,q,k,dt = [i.mT.view(B,T,-1,1,C) for i in (u,q,k,dt)]
        s = th.zeros(B,H,C,C)
        return step(u.mT,q,k,dt.mT,s)[0].view(B,T,H*C).mT


elif op == 'rwkv4':
    w = th.randn(M)
    u = th.randn(M)
    k = th.randn(B,T,M)
    v = th.randn(B,T,M)
    s = th.randn(B,3,1,M)
    s[:,2] = -1e30
    inputs = (w,u,k,v,s)

    if need_ref:
        from fla.ops.rwkv4 import fused_recurrent_rwkv4
        ref = fused_recurrent_rwkv4

    @flex_rnn.jit
    def step(w,u,k,v, sa,sb,sc):
        x = k+u-sc
        A = (-x*(x>0)).exp()
        B = (x*(x<0)).exp()
        y = (A*sa + B*v) / (A*sb + B)

        w = -w.exp()
        x = k-w-sc
        A = (-x*(x>0)).exp()
        B = (x*(x<0)).exp()
        sa_ = A*sa + B*v
        sb_ = A*sb + B
        sc_ = k-x*(x<0)
        return y.to(v.dtype), (sa_,sb_,sc_)

    def flex(w,u,k,v,s):
        w,u,k,v = [i[:,:,None,:,None] for i in [w[None,None],u[None,None],k,v]]
        sa,sb,sc = s.unsqueeze(-1).unbind(1)
        y,sa,sb,sc = step(w,u,k,v, sa,sb,sc)
        return y.squeeze(2), th.stack((sa,sb,sc), dim=1).squeeze(-1)


elif op == 'hgrn':
    x = th.randn(B,T,M)
    g = -th.sigmoid(th.randn(B,T,M))
    inputs = (x,g)

    if need_ref:
        from fla.ops.hgrn import chunk_hgrn
        def ref(x,g):
            return chunk_hgrn(x,g)[0]

    @flex_rnn.jit
    def step(x,g,s):
        s = g.exp() * s + x
        return s.to(x.dtype),s

    def flex(x,g):
        s = th.zeros(B,1,M,1)
        x,g = [i[:,:,None,:,None] for i in [x,g]]
        return step(x,g,s)[0].squeeze(2)


elif op == 'mamba':
    N = 16
    u = th.randn(B,M,T)
    dt = th.sigmoid(th.randn(B,M,T))
    A = -th.sigmoid(th.randn(M,N))
    b = th.randn(B,N,T)
    c = th.randn(B,N,T)
    inputs = (u,dt,A,b,c)

    if need_ref:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        ref = selective_scan_fn

    @flex_rnn.jit
    def step(u,dt,A,b,c,s):
        s = (dt*A).exp() * s + (dt*u)*b
        return (s*c).sum(-1,True).to(u.dtype), s

    def flex(u,dt,A,b,c):
        u,dt,b,c = [i.mT[:,:,None,:,None] for i in [u,dt,b,c]]
        A = A[None,None,None]
        s = th.zeros(B,1,M,N)
        return step(u,dt,A,b.mT,c.mT,s)[0].squeeze(2).mT


elif op == 's4d':
    import sys, os
    sys.path.append(os.environ['S4_HOME'])
    from models.s4.s4 import FFTConv # from https://github.com/state-spaces/s4

    model = FFTConv(M, d_state=C, mode='s4d', activation=None, transposed=False)
    keys = ['kernel.inv_dt', 'kernel.A_real', 'kernel.A_imag', 'kernel.B', 'kernel.C', 'D']
    inv_dt, A_real, A_imag, b, c, D = [dict(model.named_parameters())[i] for i in keys]
    u = th.randn(B,T,M)
    inputs = (u,inv_dt,A_real,A_imag,b,c,D)

    def ref(u, inv_dt, A_real, A_imag, b, c, D):
        return th.func.functional_call(model, dict(zip(keys, (inv_dt, A_real, A_imag, b, c, D))), (u,))[0]

    @flex_rnn.jit
    def step(u, Ar,Ai, Qr,Qi, D, Sr,Si):
        Sr_ = Ar*Sr - Ai*Si + u
        Si_ = Ar*Si + Ai*Sr
        y = th.sum(Sr_*Qr - Si_*Qi, dim=-1,keepdim=True) + D * u
        return y.to(u.dtype), (Sr_,Si_)

    def flex(u, log_dt, A_real, A_imag, b, c, D):
        dt = log_dt.exp()
        A = -(A_real.exp() + 1j*A_imag)
        dA = (dt*A).exp()
        Q = th.view_as_complex(c[0]) * th.view_as_complex(b[0]) * (dA-1.) / A * 2

        u = u[:,:,None,:,None]
        Ar,Ai,Qr,Qi,D = [i[None,None,None] for i in [dA.real,dA.imag,Q.real,Q.imag,D]]
        Si,Sr = [th.zeros(B,1,M,C//2) for i in range(2)]
        return step(u,Ar,Ai,Qr,Qi,D.mT,Si,Sr)[0].squeeze(2)


elif op == 'gsa':
    q,k,v,x,g = [th.randn(B,T,H,C,dtype=th.bfloat16) for i in range(5)]
    g = -th.sigmoid(g)
    sa,sb = [th.randn(B,H,C,C) for i in range(2)]
    inputs = (q,k,v,x,g,sa,sb)

    if need_ref:
        from fla.ops.gsa import chunk_gsa # Quite inaccurate (~4% error)
        #from fla.ops.gsa import fused_recurrent_gsa # More accurate, but slower
        def ref(q,k,v,x,g,sa,sb):
            #return fused_recurrent_gsa(q,k,v,x,g,1,(sa,sb))[0] # Can't backprop with output_final_state
            y,(sa,sb) = chunk_gsa(q,k,v,x,g,1,(sa,sb),True)
            return y,sa,sb

    @flex_rnn.jit
    def step1(q,k,x,g,s):
        s = g.exp()*s + x*k
        return (s*q).sum(-1,True), s
    @flex_rnn.jit
    def step2(q,x,v,g,s):
        s = g.exp()*s + v*x
        return (s*q).sum(-1,True).to(v.dtype), s

    def flex(q,k,v,x,g,sa,sb):
        q,k,v,x,g = [i.unsqueeze(-2) for i in [q,k,v,x,g]]
        qv,sa = step1(q,k,x.mT,g.mT,sa.mT)
        qv = qv.softmax(-1).unsqueeze(-2)
        y,sb = step2(qv,x,v.mT,g,sb.mT)
        return y, sa.mT, sb.mT


elif op == 'mlstm':
    q,k,v = [th.randn(B,H,T,C) for i in range(3)]
    i = th.randn(B,H,T)
    f = th.randn(B,H,T)
    sa = th.randn(B,H,C,C)
    sb = th.randn(B,H,C)
    sc = th.randn(B,H,1)
    inputs = (q,k,v,i,f,sa,sb,sc)

    if need_ref:
        if 1: # Fast, but gives wrong gradients
            from mlstm_kernels.torch.chunkwise.triton_xl_chunk import mlstm_chunkwise__xl_chunk
            def ref(q,k,v,i,f,sa,sb,sc):
                y,(sa,sb,sc) = mlstm_chunkwise__xl_chunk(q,k,v,i,f, sa,sb,sc, return_last_states=True, chunk_size=256)
                return y,sa,sb,sc
        else:
            #from mlstm_kernels.torch.chunkwise.native import mlstm_chunkwise__native_autograd
            from mlstm_kernels.torch.recurrent import mlstm_recurrent_sequence__native_fw
            def ref(q,k,v,i,f,sa,sb,sc):
                y,(sa,sb,sc) = mlstm_recurrent_sequence__native_fw(q,k,v,i,f, sa,sb,sc, return_last_states=True)
                return y,sa,sb,sc

    def maximum(a,b): return a+(b-a)*(b>a).float()

    @flex_rnn.jit
    def step1(i,f, sc):
        f = -th.log(1+th.exp(-f))
        sc_ = maximum(f+sc, i)
        i = (i-sc_).exp()
        f = (f+sc-sc_).exp()
        return (i,f,sc_), sc_

    @flex_rnn.jit
    def step2(k,q,i,f, sb):
        sb_ = f*sb + i*k
        y = (sb_*q).sum(-1,True)
        return y,sb_

    @flex_rnn.jit
    def step3(q,k,v,i,f,norm, sa):
        sa_ = f*sa + i*v*k
        y = (norm*q*sa_).sum(-1,True)
        return y, sa_

    def flex(q,k,v,i,f,sa,sb,sc):
        i,f,sb,sc = [x.unsqueeze(-1) for x in [i,f,sb,sc]]
        q,k,v,i,f = [x.transpose(1,2).unsqueeze(-2) for x in [q,k,v,i,f]]
        i,f,sc, scT = step1(i,f, sc)
        i, f = i.unsqueeze(-1), f.unsqueeze(-1)
        sb_q, sbT = step2(k,q,i,f, sb.mT)
        norm = 1/(th.maximum(th.abs(sb_q), (-sc).exp()*C**.5) + 1e-6*C**.5).unsqueeze(-1)
        y, saT = step3(q,k,v.mT,i,f,norm, sa.mT)
        return y.transpose(1,2), saT.mT,sbT.squeeze(-2),scT.squeeze(-1)


else:
    assert 0, f'Unknown op "{op}"'


if cmd_args.benchmark_flex:
    print('Benchmarking flex compiled kernel')
    if not op in ['s4d','mamba']:
        inputs = tuple(i.to(th.bfloat16) for i in inputs)
    if cmd_args.torch_compile: flex = th.compile(flex)
    flex_rnn.benchmark(flex, inputs, backward=do_backward)

elif cmd_args.benchmark_ref:
    print('Benchmarking reference implementation')
    if not op in ['s4d','mamba']:
        inputs = tuple(i.to(th.bfloat16) for i in inputs)
    if cmd_args.torch_compile: ref = th.compile(ref)
    flex_rnn.benchmark(ref, inputs, backward=do_backward)

elif cmd_args.check_flex:
    def ref(*args):
        for step in [v for k,v in globals().items() if k.startswith('step')]:
            step.apply = step.naive_fp64
        return flex(*args)
    print('Verifying flex compiled kernel')
    flex_ = th.compile(flex) if cmd_args.torch_compile else flex
    flex_rnn.check_grad(flex_, ref, inputs, backward=do_backward)

elif cmd_args.check_ref:
    print('Checking precision of reference implementation')
    for step in [v for k,v in globals().items() if k.startswith('step')]:
        step.apply = step.naive_fp64
    if cmd_args.torch_compile: ref = th.compile(ref)
    flex_rnn.check_grad(ref, flex, inputs, backward=do_backward)

else:
    print('Comparing flex compiled kernel to reference')
    if cmd_args.torch_compile: flex = th.compile(flex)
    flex_rnn.check_grad(flex, ref, inputs, backward=do_backward)

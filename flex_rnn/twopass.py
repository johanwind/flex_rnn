# Copyright (c) 2025, Johan Sokrates Wind

import torch as th
from flex_rnn.to_cuda import to_cuda, parse_cuda, build_cuda
from flex_rnn.graph import Graph
from flex_rnn.trace_isolated import trace_fwbw, trace_fwbw_isolated

def split_list(args, lens):
    l = []
    off = 0
    for i in lens:
        l.append(args[off:off+i])
        off += i
    assert off == len(args)
    return l

def split_dim(x,dim,K):
    r = ()
    for i in x:
        shape = list(i.shape) if type(i) == th.Tensor else i
        shape = shape[:dim]+([shape[dim]//K,K] if shape[dim]>1 else [1,1])+shape[dim:][1:]
        r += (i.view(*shape) if type(i) == th.Tensor else shape,)
    return r

def transpose_list(x, d0,d1):
    r = ()
    for i in x:
        if type(i) == list:
            i = th.empty(*i, device='meta')
        r += (i.transpose(d0,d1),)
    return r

def get_meta(l):
    return tuple((i.shape,i.stride(),i.dtype) for i in l)

def verify_meta(a, b, msg = ''):
    for i, ((a_shape,a_stride,a_dtype), (b_shape,b_stride,b_dtype)) in enumerate(zip(get_meta(a),get_meta(b))):
        assert a_shape == b_shape, msg + f'#{i+1}: {a_shape=} != {b_shape=}'
        a_stride = [stride if shape > 1 else 0 for stride,shape in zip(a_stride,a_shape)]
        b_stride = [stride if shape > 1 else 0 for stride,shape in zip(b_stride,b_shape)]
        assert a_stride == b_stride, msg + f'#{i+1}: {a_stride=} != {b_stride=}'
        assert a_dtype == b_dtype, msg + f'#{i+1}: {a_dtype=} != {b_dtype=}'

def build_twopass(step, inputs, state, dryrun = False):
    B,T,H = [max(i.shape[j] for i in inputs) for j in range(3)]
    M = max(i.shape[-2] for i in inputs+state)
    N = max(i.shape[-1] for i in inputs+state)

    for i,x in enumerate(inputs):
        assert x.ndim == 5, f'invalid shape {list(x.shape)} of argument #{i+1}'
        for j in range(5):
            assert x.shape[j] in [1, [B,T,H,M,N][j]], f'shape {list(x.shape)} of argument #{i+1} is incompatible with {[B,T,H,M,N]}'
        assert 1 in x.shape[-2:] or x.shape[1] == 1, f'shape {list(x.shape)} of argument #{i+1} is both a matrix for each head, and depends on time. This is not supported'
        assert x.dtype in [th.float, th.bfloat16, th.float16, th.double]

    for i,x in enumerate(state):
        assert list(x.shape) == [B,H,M,N], f'shape {list(x.shape)} of argument #{i+1+len(inputs)} != {[B,H,M,N]}'
        assert x.dtype in [th.float, th.bfloat16, th.float16, th.double]

    atomic_indices = [i for i in range(len(inputs)) if list(inputs[i].shape[:3]) != [B,T,H]]

    scalar_indices, col_indices, row_indices, mat_indices = [], [], [], []
    for i in range(len(inputs)):
        if list(inputs[i].shape[-2:]) == [1,1]:
            scalar_indices.append(i)
        elif list(inputs[i].shape[-2:]) == [M,1]:
            col_indices.append(i)
        elif list(inputs[i].shape[-2:]) == [1,N]:
            row_indices.append(i)
        else:
            assert list(inputs[i].shape[-2:]) == [M,N]
            mat_indices.append(i)

    inputs_t = [i[:,0] for i in inputs]
    output_t, nstate = step(*[i.to('meta') for i in [*inputs_t, *state]])
    output_t, nstate = [i if type(i) == tuple else (i,) for i in [output_t,nstate]]
    ninputs, noutputs, nstates = len(inputs_t), len(output_t), len(nstate)

    assert len(nstate) == len(state)
    assert all(a.shape == b.shape for a,b in zip(nstate,state))
    assert all(list(i.shape) == [B,H,M,1] for i in output_t), [i.shape for i in output_t]
    assert all(i.dtype in [th.float, th.bfloat16, th.float16, th.double] for i in output_t)

    shapes = tuple(list(j[0,0].shape) for j in tuple(inputs_t)+state+output_t+state)

    for i in scalar_indices:
        inputs_t[i] = inputs_t[i].expand(-1,-1,-1,N)

    trace_fn = (trace_fwbw if th._guards.detect_fake_mode() is None else trace_fwbw_isolated)
    code = trace_fn(step, tuple(j[0,0].clone().to('meta') for j in tuple(inputs_t)+state)).code.strip()
    graph = Graph('graph', code)

    output_names, nstate_names, dinputs_names, dpstate_names = split_list(graph.outputs, [noutputs, nstates, ninputs, nstates])

    if 1: # Check that forward pass sums are supported
        fw = graph.copy('step')
        fw.outputs = output_names+nstate_names
        fw.prune()
        assert not any(fw.inputs_mask[ninputs+nstates:]) # Forward pass shouldn't depend on gradients
        fw.inputs = fw.inputs[:ninputs+nstates]

        for node in fw.node:
            if node.op == 'sum.dim_IntList':
                assert node.other_args == ['[-1]','True'], 'Only torch.sum(x, dim=-1, keepdim=True) is supported'

    # sum(x, dim=0, keepdim=True) is only used to accumulate gradients, that is done elsewhere instead
    for node in graph.node:
        if node.op == 'sum.dim_IntList':
            if node.other_args[0] == '[-1]':
                node.other_args[0] = '[1]'
            if node.other_args[0] == '[0]':
                node.op = 'clone.default'
                node.other_args = []
    graph.compute_shapes(shapes)

    scalar_needs_reduce = [graph.shape[dinputs_names[i]][1] > 1 for i in scalar_indices]
    #print(graph.code(shapes=True));exit()


    restep = graph.copy('restep')
    restep.outputs = nstate_names
    restep.prune()
    aux = [i.name for i in restep.node if i.op == 'sum.dim_IntList']
    restep.inputs += aux
    restep.prune()
    assert not any(restep.inputs_mask[ninputs+nstates:len(restep.inputs)-len(aux)])
    restep.inputs = restep.inputs[:ninputs+nstates]+restep.inputs[len(restep.inputs)-len(aux):]


    row_restep = restep.copy('row_restep')
    row_restep.transpose()

    step = graph.copy('step')
    step.outputs = output_names+nstate_names+aux
    step.prune()
    assert not any(step.inputs_mask[ninputs+nstates:])
    step.inputs = step.inputs[:ninputs+nstates]

    drow = graph.copy('drow')
    drow.inputs += aux
    drow.outputs = dpstate_names + [dinputs_names[i] for i in row_indices+scalar_indices+mat_indices]
    drow.prune()
    daux = [i.name for i in drow.node if i.op == 'sum.dim_IntList' and i.other_args[0] == '[1]']
    drow.inputs += daux
    drow.prune()
    drow.transpose()

    dstep = graph.copy('dstep')
    dstep.inputs += aux
    dstep.outputs = dpstate_names + [dinputs_names[i] for i in col_indices] + daux
    dstep.prune()


    def max_pow2(n): # Largest power of 2 less than n
        return 2**(n.bit_length()-1)

    dM_bw1 = dM_fw = max_pow2(min(200//(1+len(state)+len(mat_indices)),M,N)) # Largest power of 2 which ~fits in registers
    dM_bw2 = min(N,256)
    dT = max_pow2(min(256//(len(inputs)+len(row_indices)+len(scalar_indices)+len(state)+len(aux)),T))
    dT_bw1 = 1

    if N == 1:
        dT_bw1 = dT = min(T,32)
        dM_fw = dM_bw1 = min(M,64)
    elif any(dstep.inputs_mask[ninputs:ninputs+nstates]): # dcol depends directly on the full state, which requires a lot of shared memory
        dM_bw1 = min(M,N,16)
        dT = max_pow2(min(M,N,256//(dM_bw1*(len(state)+len(mat_indices)))))
        dT_bw1 = dT
        #print('flex_rnn: A column vector\'s gradient depends directly on the full state, this uses a lot of shared memory')

    #print(f'flex_rnn: {dT=}, {dM_fw=}, {dM_bw1=}, {dM_bw2=}')

    assert T%dT == 0

    output = tuple(th.empty(i.shape[0],T,*i.shape[1:], dtype=i.dtype, device='meta') for i in output_t)
    output_out = tuple(i.squeeze(-1) for i in output)
    state_store = tuple(th.empty(B,H,(T-1)//dT+1,i.shape[-1],i.shape[-2], dtype=th.float32, device='meta').mT for i in state)
    dstate = tuple(th.empty_like(i, device='meta') for i in state)
    dstateT = tuple(i.contiguous() for i in dstate)
    dinputs_out = tuple(th.empty_like(i, device='meta') for i in inputs)
    dinputs = tuple((x.to(th.float32) if i in atomic_indices else x) for i,x in enumerate(dinputs_out))
    aux_out = (th.empty(len(aux), B,H,T,M,1, dtype=th.float32, device='meta'),)

    dcol_inputs = tuple(dinputs[i] for i in col_indices)
    drow_inputs = tuple(dinputs[i] for i in row_indices+scalar_indices)
    dmat_inputs = tuple(dinputs[i] for i in mat_indices)

    forward_meta = get_meta([*output_out, *dstate, *state_store, *aux_out])
    backward_meta = get_meta([*dinputs_out, *dstate])

    if dryrun:
        return forward_meta, backward_meta

    aux = tuple(th.empty(B,H,T,M,1, dtype=th.float32, device='meta') for i in range(len(aux)))
    daux = tuple(th.empty(B,H,T,M,1, dtype=th.float32, device='meta') for i in range(len(daux)))

    K = dM_fw
    step.reshape(M = K)

    stateK, dstateK, inputsK, outputK, state_storeK, auxK = [split_dim(i, -2, K) for i in [state, dstate, inputs, output, state_store, aux]]

    src = f'''
__global__ void fw(T* __restrict__ inputs[i]_, T* __restrict__ state0_[i]_, T* output[i]_, T* stateT_[i]_, T* state_store[i]_, T* aux[i]_) {{ # inputs, state, output, dstate, state_store, aux
  [[maybe_unused]] int bi = blockIdx.z, hi = blockIdx.y, mi = blockIdx.x, rowi = threadIdx.x%{K}, basej = threadIdx.x/{K}*{K};
  state[i] = state0_[i]_[bi,hi,mi,:,:] # stateK
  for (int ti = 0; ti < {T}; ti++) {{
    if (ti%{dT} == 0) {{
      state_store[i]_[bi,hi,ti/{dT},mi,:,:] = state[i] # state_storeK
    }}
    inputs[i] = inputs[i]_[bi,ti,hi,mi,:,:] # inputsK
    output[i] # outputK
    aux[i] # auxK
    nstate[i] # stateK
    step(inputs[i], state[i], output[i], nstate[i], aux[i]); # inputs, state, output, state, aux
    state[i] = nstate[i] # stateK
    output[i]_[bi,ti,hi,mi,:,:] = output[i] # outputK
    aux[i]_[bi,hi,ti,mi,:,:] = aux[i] # auxK
  }}
  stateT_[i]_[bi,hi,mi,:,:] = state[i] # dstateK
}}'''
    fw_src = to_cuda(step) + parse_cuda(src)
    fw = (fw_src, (M//K,H,B), max(N,K))

    def forward(inputs_, state0):
        verify_meta(inputs_, inputs, 'input Tensor metadata doesn\'t match compilation\n')
        verify_meta(state0, state, 'initial state Tensor metadata doesn\'t match compilation\n')
        device = inputs_[0].device
        output_ = tuple(th.empty_like(i, device=device) for i in output)
        aux_ = th.empty_like(aux_out[0], device=device)
        state_store_ = tuple(th.empty_like(i, device=device) for i in state_store)
        stateT_ = tuple(th.empty_like(i, device=device) for i in state0)
        fw_kernel(*inputs_, *state0, *output_, *stateT_, *state_store_, *aux_.unbind())
        verify_meta(output_, output)
        verify_meta(stateT_, dstate)
        verify_meta(state_store_, state_store)
        verify_meta((aux_,), aux_out)
        return *tuple(i.squeeze(-1) for i in output_), *stateT_, *state_store_, aux_


    K = dM_bw1
    restep.reshape(M = K)
    dstep.reshape(M = K)
    stateK, dstateK, dstateTK, inputsK, outputK, auxK, dauxK, state_storeK, dcol_inputsK = [split_dim(i, -2, K) for i in [state, dstate, dstateT, inputs, output, aux, daux, state_store, dcol_inputs]]

    src = f'''
__global__ void bw1(T* __restrict__ inputs[i]_, T* __restrict__ doutput[i]_, T* __restrict__ dstateT_[i]_, T* __restrict state_store[i]_, T* __restrict__ aux[i]_, T* dstate0_[i]_, T* dstate_store[i]_, T* dcol_inputs[i]_, T* daux[i]_) {{ # inputs, output, state, state_store, aux, dstate, state_store, dcol_inputs, daux
  [[maybe_unused]] int bi = blockIdx.z, hi = blockIdx.y, mi = blockIdx.x, rowi = threadIdx.x%{K}, basej = threadIdx.x/{K}*{K};
  dstate[i] = dstateT_[i]_[bi,hi,mi,:,:] # dstateTK
  for (int t0 = {T-dT_bw1}; t0 >= 0; t0 -= {dT_bw1}) {{
    if (t0%{dT} == {dT-dT_bw1}) {{
      dstate_store[i]_[bi,hi,t0/{dT},mi,:,:] = dstate[i] # state_storeK
    }}\n'''
    for dt in range(dT_bw1):
        src += f'inputs_{dt}_[i] = inputs[i]_[bi,t0+{dt},hi,mi,:,:] # inputsK\n'
        src += f'aux_{dt}_[i] = aux[i]_[bi,hi,t0+{dt},mi,:,:] # auxK\n'
    src += f'state_0_[i] = state_store[i]_[bi,hi,t0/{dT},mi,:,:] # state_storeK\n'
    for dt in range(dT_bw1-1):
        src += f'state_{dt+1}_[i] # stateK\n'
        src += f'restep(inputs_{dt}_[i], state_{dt}_[i], aux_{dt}_[i], state_{dt+1}_[i]); # inputs, state, aux, state\n'
    src += f'dcol_inputs[i] # dcol_inputsK\n'
    src += f'dpstate[i] # stateK\n'
    for dt in range(dT_bw1-1,-1,-1):
        src += f'doutput_{dt}_[i] = doutput[i]_[bi,t0+{dt},hi,mi,:,:] # outputK\n'
        src += f'daux_{dt}_[i] # dauxK\n'
        src += f'dstep(inputs_{dt}_[i], state_{dt}_[i], doutput_{dt}_[i], dstate[i], aux_{dt}_[i], dpstate[i], dcol_inputs[i], daux_{dt}_[i]); # inputs, state, output, state, aux, state, dcol_inputs, daux\n'
        src += f'dstate[i] = dpstate[i] # stateK\n'
        src += f'daux[i]_[bi,hi,t0+{dt},mi,:,:] = daux_{dt}_[i] # dauxK\n'
        for i in range(len(dcol_inputs)):
            plus = '+'*(col_indices[i] in atomic_indices)
            src += f'dcol_inputs{i}_[bi,t0+{dt},hi,mi,:,:] {plus}= dcol_inputs{i} # dcol_inputsK[{i}]\n'
    src += '  }\n'
    src += '  dstate0_[i]_[bi,hi,mi,:,:] = dstate[i] # dstateK\n'
    src += '}\n'

    bw1_src = to_cuda(restep) + '\n' + to_cuda(dstep) + parse_cuda(src)
    bw1 = (bw1_src, (M//K,H,B), max(K,N))


    dM,dN = dM_bw2,1
    drow.reshape(dM,dN)
    row_restep.reshape(dM,dN)

    stateK, dstateK, inputsK, outputK, state_storeK, auxK, dauxK, drow_inputsK, dmat_inputsK = [transpose_list(split_dim(split_dim(transpose_list(i,-1,-2),-2,dM),-1,dN),-3,-2) for i in [state, dstate, inputs, output, state_store, aux, daux, drow_inputs, dmat_inputs]]

    src = f'''
__device__ float reduce(float x) {{
    __shared__ float share[{dM}];
    __syncthreads();
    share[threadIdx.x] = x;
    __syncthreads();
    if (threadIdx.x == 0) {{
        float sum = 0;
        for (int i = 0; i < {dM}; i++)
            sum += share[i];
        share[0] = sum;
    }}
    __syncthreads();
    float ret = share[0];
    __syncthreads();
    return ret;
}}
__global__ void bw2(T* __restrict__ inputs[i]_, T* __restrict__ doutput[i]_, T* __restrict__ state_store[i]_, T* __restrict__ dstate_store[i]_, T* __restrict__ aux[i]_, T* __restrict__ daux[i]_, T* drow_inputs[i]_, T* dmat_inputs[i]_) {{ # inputs, output, state_store, state_store, aux, daux, drow_inputs, dmat_inputs
  [[maybe_unused]] int bi = blockIdx.z/{H}, hi = blockIdx.z%{H}, t0 = blockIdx.y, mi = blockIdx.x, ti = t0*{dT}, rowi = threadIdx.x;\n'''
    for dt in range(dT):
        src += f'drow_inputs_{dt}_[i] = 0 # drow_inputsK\n'
        for i in row_indices+scalar_indices:
            src += f'inputs_{dt}_{i} = inputs{i}_[bi,ti+{dt},hi,mi,?,:,:] # inputsK[{i}]\n'
    src += f'for (int ni = 0; ni < {M//dN}; ni++) {{\n'
    src += f'dstate[i] = dstate_store[i]_[bi,hi,t0,mi,ni,:,:] # state_storeK\n'
    for dt in range(dT):
        for i in col_indices+mat_indices:
            src += f'inputs_{dt}_{i} = inputs{i}_[bi,ti+{dt},hi,mi,ni,:,:] # inputsK[{i}]\n'
        src += f'aux_{dt}_[i] = aux[i]_[bi,hi,ti+{dt},mi,ni,:,:] # auxK\n'
    src += f'state_0_[i] = state_store[i]_[bi,hi,ti/{dT},mi,ni,:,:] # state_storeK\n'
    for dt in range(dT-1):
        src += f'state_{dt+1}_[i] # stateK\n'
        src += f'row_restep(inputs_{dt}_[i], state_{dt}_[i], aux_{dt}_[i], state_{dt+1}_[i]); # inputs, state, aux, state\n'
    src += f'drow_inputs[i] # drow_inputsK\n'
    src += f'dmat_inputs[i] # dmat_inputsK\n'
    src += f'dmat_inputs_acc[i] = 0 # dmat_inputsK\n'
    src += f'dpstate[i] # stateK\n'
    for dt in range(dT-1,-1,-1):
        src += f'doutput_{dt}_[i] = doutput[i]_[bi,ti+{dt},hi,mi,ni,:,:] # outputK\n'
        src += f'daux_{dt}_[i] = daux[i]_[bi,hi,ti+{dt},mi,ni,:,:] # dauxK\n'
        src += f'drow(inputs_{dt}_[i], state_{dt}_[i], doutput_{dt}_[i], dstate[i], aux_{dt}_[i], daux_{dt}_[i], dpstate[i], drow_inputs[i], dmat_inputs[i]); # inputs, state, output, state, aux, daux, state, drow_inputs, dmat_inputs\n'
        src += f'dstate[i] = dpstate[i] # dstateK\n'
        src += f'drow_inputs_{dt}_[i] += drow_inputs[i] # drow_inputsK\n'
        src += f'dmat_inputs_acc[i] += dmat_inputs[i] # dmat_inputsK\n'
    src += f'dmat_inputs[i]_[bi,?,hi,mi,ni,:,:] += dmat_inputs_acc[i] # dmat_inputsK\n'
    src += '}\n'
    for dt in range(dT):
        for i in range(len(drow_inputs)):
            if i >= len(row_indices):
                if scalar_needs_reduce[i-len(row_indices)]:
                    src += f'drow_inputs_{dt}_{i} = reduce(drow_inputs_{dt}_{i});\n'
                src += f'if (threadIdx.x == 0)\n  '
            plus = '+'*((row_indices+scalar_indices)[i] in atomic_indices)
            src += f'drow_inputs{i}_[bi,ti+{dt},hi,mi,?,:,:] {plus}= drow_inputs_{dt}_{i} # drow_inputsK[{i}]\n'
    src += '}'
    bw2_src = to_cuda(drow) + '\n' + to_cuda(row_restep) + parse_cuda(src)
    bw2 = (bw2_src, (N//dM,T//dT,H*B), dM)

    def backward(doutput, dstateT_, inputs_, state_store_, aux_):
        doutput,dstateT_ = [tuple(i.contiguous() for i in j) for j in [doutput,dstateT_]]
        verify_meta(inputs_, inputs)
        verify_meta(doutput, tuple(i.squeeze(-1) for i in output))
        verify_meta(dstateT_, dstateT)
        verify_meta(state_store_, state_store)
        verify_meta(aux_, aux_out)
        device = inputs_[0].device
        aux_ = aux_[0].unbind()
        daux_ = tuple(th.empty_like(i, device=device) for i in daux)
        dstate_store_ = tuple(th.empty_like(i, device=device) for i in state_store)
        dinputs_ = [th.empty_like(i, device=device) for i in dinputs]
        for i in atomic_indices: dinputs_[i].zero_()
        dstate0 = tuple(th.empty_like(i, device=device) for i in dstate)
        bw1_kernel(*inputs_, *doutput, *dstateT_, *state_store_, *aux_, *dstate0, *dstate_store_, *[dinputs_[i] for i in col_indices], *daux_)
        bw2_kernel(*inputs_, *doutput, *state_store_, *dstate_store_, *aux_, *daux_, *[dinputs_[i] for i in row_indices+scalar_indices+mat_indices])
        for i in atomic_indices:
            dinputs_[i] = dinputs_[i].to(inputs_[i])
        verify_meta(dinputs_, dinputs_out)
        verify_meta(dstate0, dstate)
        return tuple(dinputs_), dstate0

    fw_kernel, bw1_kernel, bw2_kernel = build_cuda(*(fw, bw1, bw2))
    return forward, backward, forward_meta, backward_meta

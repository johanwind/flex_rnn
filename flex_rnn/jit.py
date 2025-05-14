# Copyright (c) 2025, Johan Sokrates Wind

import torch as th
import inspect, uuid
from flex_rnn.twopass import build_twopass, split_list, verify_meta

class jit:
    def __init__(self, step):
        self.step = step
        self.forward_func = self.backward_func = None

        n_step_args = len(inspect.signature(self.step).parameters)
        y,s = self.step(*[th.empty(1,1, device='meta') for i in range(n_step_args)])
        self.noutputs, self.nstates = len(y), len(s)
        self.ninputs = n_step_args - self.nstates

        self.apply = self.build_call()

    def __call__(self, *args):
        return self.apply(*args)

    def naive(self, *args, dtype=th.float32):
        inputs, state0 = split_list(args, [self.ninputs, self.nstates])
        state = state0
        T = max(i.shape[1] for i in inputs)
        for t in range(T):
            inputs_t = tuple(i.expand(-1,T,-1,-1,-1)[:,t].to(dtype) for i in inputs)
            output_t, state = self.step(*inputs_t, *state)
            output_t, state = [i if type(i) == tuple else (i,) for i in [output_t,state]]
            if t == 0:
                output = tuple(th.empty(i.shape[0],T,i.shape[1],i.shape[2], dtype=i.dtype, device=i.device) for i in output_t)
            for a,b in zip(output,output_t):
                a[:,t] = b.squeeze(-1)
        state = tuple(th.empty_like(x).copy_(y) for x,y in zip(state0, state)) # Match strides and dtypes of state0 if it is dense and non-overlapping in memory, else contiguous
        return *output, *state

    def naive_fp64(self, *args): return self.naive(*args, dtype=th.double)

    def build_call(self): # This is complicated since we want to be compatible with th.compile without graph breaks
        id = uuid.uuid4().hex

        naux = 1 # aux tensors are packed into a single tensor to make the schema predictable
        schema = '('+', '.join(f'Tensor x{i}' for i in range(self.ninputs+self.nstates))+') -> ('+', '.join(['Tensor']*(self.noutputs+self.nstates*2+naux))+')'
        @th.library.custom_op("flex_rnn::fw_"+id, mutates_args=(), schema=schema)
        def apply(*args):
            inputs, state = split_list(args, [self.ninputs, self.nstates])
            if self.forward_func is None:
                self.forward_func, self.backward_func, self.fw_meta, self.bw_meta = build_twopass(self.step, inputs, state)
            r = self.forward_func(inputs, state)
            meta = tuple(th.empty_strided(shape,stride,dtype=dtype,device='meta') for shape,stride,dtype in self.fw_meta)
            verify_meta(r, meta)
            return r
        @apply.register_fake
        def _(*args):
            inputs, state = split_list(args, [self.ninputs, self.nstates])
            if not hasattr(self, 'fw_meta'):
                self.fw_meta, self.bw_meta = build_twopass(self.step, inputs, state, dryrun = True)
            return tuple(th.empty_strided(shape,stride,dtype=dtype,device=args[0].device) for shape,stride,dtype in self.fw_meta)

        schema = '('+', '.join(f'Tensor x{i}' for i in range(self.noutputs+self.nstates*2+self.ninputs+naux))+') -> ('+', '.join(['Tensor']*(self.ninputs+self.nstates))+')'
        @th.library.custom_op("flex_rnn::bw_"+id, mutates_args=(), schema=schema)
        def backward_inner(*args):
            doutput, dstateT, inputs, state_store, aux = split_list(args, [self.noutputs, self.nstates, self.ninputs, self.nstates, naux])
            dinputs, dstate0 = self.backward_func(doutput, dstateT, inputs, state_store, aux)
            verify_meta(dinputs+dstate0, tuple(th.empty_strided(shape,stride,dtype=dtype,device='meta') for shape,stride,dtype in self.bw_meta))
            return *dinputs, *dstate0
        def backward_inner_fake(*args):
            return tuple(th.empty_strided(shape,stride,dtype=dtype,device=args[0].device) for shape,stride,dtype in self.bw_meta)
        backward_inner.register_fake(backward_inner_fake)

        def setup_context(ctx, inputs, output):
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(*inputs[:self.ninputs], *output[self.noutputs+self.nstates:])
            ctx.output_meta = tuple(i.to('meta') for i in output[:self.noutputs+self.nstates])
        def backward(ctx, *args):
            args = list(args)
            for i,meta in enumerate(ctx.output_meta):
                if args[i] is None:
                    args[i] = th.zeros_like(meta, device=ctx.saved_tensors[0].device)
            return backward_inner(*args[:self.noutputs+self.nstates], *ctx.saved_tensors)

        apply.register_autograd(backward, setup_context=setup_context)
        return lambda *args : apply(*args)[:self.noutputs+self.nstates]

def simplified_naive(step, *args):
    inputs, state = args[:-1], args[-1]
    T = max(i.shape[1] for i in inputs)
    output = []
    for t in range(T):
        inputs_t = [i.expand(-1,T,-1,-1,-1)[:,t].float() for i in inputs]
        output_t, state = step(*inputs_t, state)
        output.append(output_t)
    return th.stack(output, dim=1).squeeze(-1), state

# Copyright (c) 2025, Johan Sokrates Wind

import sys, hashlib, re, os, torch as th
from torch.utils.cpp_extension import load
from types import SimpleNamespace

def to_cuda(graph):
    M,N = [max(i[k] for i in graph.shape.values()) for k in [0,1]]
    assert N%M == 0 or N == 1

    # Make sure outputs aren't duplicates of inputs or each other
    for i in range(len(graph.outputs)):
        old = graph.outputs[i]
        seen = graph.inputs+graph.outputs[:i]
        if old in seen:
            while graph.outputs[i] in seen: graph.outputs[i] += '_'
            graph.node.append(SimpleNamespace(name=graph.outputs[i], op='clone.default', named_args=[old], other_args=[]))
            graph.shape[graph.outputs[i]] = graph.shape[old]

    src = f'__device__ void {graph.func_name}('+', '.join('float'+['&','*'][min(graph.shape[i])>1]+' '+i for i in graph.inputs+graph.outputs)+') {\n'
    if M < N:
        src +=f'[[maybe_unused]] int basej = threadIdx.x/{M}*{M};\n'
        #if any(node.op == 'sum.dim_IntList' for node in graph.node) and N > 1:
            #src +=f'__shared__ float share[{N}];\n'
    else:
        src += '[[maybe_unused]] int basej = 0;\n'
    src +=f'[[maybe_unused]] __shared__ float share[{N}];\n'

    shared = set()

    for node in graph.node:
        lhs = node.name
        m,n = graph.shape[lhs]
        if not lhs in graph.outputs:
            if min(m,n) > 1:
                src +=f'float {lhs}[{M}];\n'
            else:
                src +=f'float {lhs};\n'

        op = node.op.split('.')[0]
        unary = {'exp': '__expf', 'clone': '', 'neg': '-', 'reciprocal': '1.f/', 'log': '__logf', 'zeros_like': '0*', 'sqrt': '__fsqrt_rn'}
        binary = {'mul': '*', 'add': '+', 'sub': '-', 'div': '/', 'lt': '<', 'gt': '>'}
        if op in unary:
            opname = unary[op]
            if min(m,n) > 1:
                src += '#pragma unroll\n'
                src +=f'for (int j = 0; j < {M}; j++)\n'
                src +=f'  {lhs}[j] = {opname}({node.named_args[0]}[j]);\n'
            else:
                src += f'{lhs} = {opname}({node.named_args[0]});\n'
        elif op in binary:
            opsign = binary[op]
            a,b = node.named_args+node.other_args
            sa = graph.shape[a]
            sb = graph.shape[b] if len(node.named_args) == 2 else [1,1]
            if min(max(sa[i],sb[i]) for i in [0,1]) == 1:
                src +=f'  {lhs} = {a} {opsign} {b};\n'
            else:
                l = []
                for x,s in [(a,sa),(b,sb)]:
                    if min(s) > 1:
                        x += '[j]'
                    elif s == [1,N]:
                        if not x in shared:
                            src +=f'__shared__ float {x}_share[{N}];\n'
                            src += '__syncthreads();\n'
                            src +=f'{x}_share[threadIdx.x] = {x};\n'
                            src += '__syncthreads();\n'
                            shared.add(x)
                        x += '_share[basej+j]'
                    else: assert s in [[1,1],[M,1]], str(s)
                    l += [x]
                src += '#pragma unroll\n'
                src +=f'for (int j = 0; j < {M}; j++)\n'
                src +=f'  {lhs}[j] = {l[0]} {opsign} {l[1]};\n'
                src += '\n'
        elif node.op == 'sum.dim_IntList':
            assert node.other_args == ['[1]','True']
            assert len(node.named_args) == 1
            if graph.shape[node.named_args[0]][1] == 1:
                src +=f'{lhs} = {node.named_args[0]};\n'
            elif min(graph.shape[node.named_args[0]]) > 1:
                src +=f'{lhs} = {node.named_args[0]}[0];\n'
                src += '#pragma unroll\n'
                src +=f'for (int j = 1; j < {M}; j++)\n'
                src +=f'  {lhs} += {node.named_args[0]}[j];\n'

                if M < N:
                    src +=f'''{{
  int i = threadIdx.x%{M}, j = threadIdx.x/{M};
  __syncthreads();
  share[j+i*{N//M}] = {lhs};
  __syncthreads();
  if (j == 0) {{
    float sum = {lhs};
    #pragma unroll
    for (int l = 1; l < {N//M}; l++) sum += share[l+i*{N//M}];
    share[i*{N//M}] = sum;
  }}
  __syncthreads();
  {lhs} = share[i*{N//M}];
  __syncthreads();
}}\n'''
            else:
                assert graph.shape[node.named_args[0]] == [1,N]
                src +=f'''{{
  __syncthreads();
  share[threadIdx.x] = {node.named_args[0]};
  __syncthreads();
  if (threadIdx.x == 0) {{
    float sum = 0;
    for (int i = 0; i < {N}; i++)
      sum += share[i];
    share[0] = sum;
  }}
  __syncthreads();
  {lhs} = share[0];
  __syncthreads();
}}\n'''
        else:
            assert 0, "to_cuda doesn't support "+str(node)
    src = '\n'.join('  '+i for i in src.split('\n'))[2:-2] # indent
    src += '}\n'
    return src


def parse_cuda_line(s0):
    if not ' # ' in s0: return s0
    try:
        s, v = s0.replace('[i]','{i}').split(' # ')
        locals = sys._getframe(3).f_locals # TODO: unreliable
        v = eval(v, locals)
        if type(v) != tuple: v = (v,)
        if ',:' in s: # Global mem
            indent = ' '*(len(s)-len(s.lstrip()))
            l,r = s.strip().split(' += ' if ' += ' in s else ' = ')
            load = (not ',:' in l)
            var,lookup = (l,r) if load else (r,l)
            var0 = var.replace('.mT','')
            ptr = lookup.split('[')[0]
            inds = lookup.split('[')[1].split(']')[0].split(',')
            mi,ni = [i for i,v in enumerate(inds) if v == ':']
            assert mi == len(inds)-2 and ni == len(inds)-1

            ret = ''
            for i,meta in enumerate(v):
                var = var0
                strides = meta.stride()
                shape = meta.shape

                assert len(inds) == len(shape)

                m,n = shape[mi],shape[ni]
                if m > 1 and n > 1:
                    if load:
                        ret += indent +f'float {var}[{m}];\n'.format(i=i)
                    ret += indent +'#pragma unroll\n'
                    ret += indent +f'for (int j = 0; j < {m}; j++)\n  '
                    inds[mi],inds[ni] = [f'rowi',f'basej+j']
                    var += '[j]'
                else:
                    if load:
                        ret += indent +f'float {var};\n'.format(i=i)
                    elif m > 1 and n == 1:
                        ret += indent +f'if (threadIdx.x < {m})\n  '
                    inds[mi],inds[ni] = ['rowi','0'] if m > 1 else ['0','threadIdx.x']
                ind = ' + '.join(f'({i})*({stride})' for i,stride,shapei in zip(inds, strides, shape) if shapei > 1) or '0'
                if load:
                    d = {th.float: '', th.bfloat16: '__bfloat162float', th.float16: '__half2float', th.double: ''}
                    ret += indent +f'{var} = {d[meta.dtype]}({ptr}[{ind}]);\n'.format(i=i)
                else:
                    d = {th.float: '', th.bfloat16: '__float2bfloat16', th.float16: '__float2half', th.double: ''}
                    var = d[meta.dtype]+f'({var})'
                    if '+=' in s:
                        ret += indent + f'atomicAdd(&{ptr}[{ind}], {var});\n'.format(i=i)
                    else:
                        ret += indent +f'{ptr}[{ind}] = {var};\n'.format(i=i)
            if len(v) == 1: ret = ret.rstrip('\n')
            return ret
        elif len(s.split()) == 1 or ' = 0' in s: # Declare
            indent = ' '*(len(s)-len(s.lstrip()))
            var = s.split()[0]
            ret = ''
            for i,shape in enumerate(v):
                shape = shape.shape
                ret += indent +f'float {var}'.format(i=i)+f'[{shape[-2]}]'*(min(shape[-2:])>1)+' = {}'*(' = 0' in s)+';\n'
            return ret
        elif ' = ' in s or ' += ' in s: # Assign
            indent = ' '*(len(s)-len(s.lstrip()))
            a,eq,b = s.split()
            ret = ''
            for i,shape in enumerate(v):
                shape = shape.shape
                if min(shape[-2:]) > 1:
                    ret += indent +f'#pragma unroll\n'
                    ret += indent +f'for (int j = 0; j < {shape[-2]}; j++) {{\n'
                    ret += indent +f'  {a}[j] {eq} {b}[j];\n'.format(i=i)
                    ret += indent +f'}}\n'
                else:
                    ret += indent +f'{a} {eq} {b};\n'.format(i=i)
            return ret
        else: # Expand list
            v = iter(v)
            def repl(match):
                l = []
                for i,t in enumerate(next(v)):
                    s = match.group(0).format(i=i)
                    if 'T*' in s:
                        d = {th.float: 'float', th.bfloat16: '__nv_bfloat16', th.float16: '__half', th.double: 'double'}
                        s = s.replace('T*',d[t.dtype]+'*')
                    l.append(s)
                return ', '.join(l) if l else '@'
            r = re.sub(r'[\w&* ]+\{i\}\w*', repl, s).replace(',@','').replace('@','')
            assert list(v) == []
            return r
    except Exception as e:
        print('Error, failed to parse:')
        print(s0)
        print()
        raise

def parse_cuda(src):
    return '\n'.join(parse_cuda_line(s) for s in src.split('\n'))

def dump(fn, txt):
    if not os.path.isfile(fn) or open(fn).read() != txt:
        with open(fn, "w") as f: f.write(txt)

nbuilds = 0
def build_cuda(*args):
    cu = ''
    cpp = '#include <torch/extension.h>\n'
    cpp_lib = ''
    cpp_lib_impl = ''
    func_names = []
    for src,grid,threads in args:
        args,mutable,types = [],[],[]
        for i in src.split('void ')[-1].split('(')[1].split(')')[0].split(', '):
            mutable.append('__restrict__ ' not in i)
            args.append(i.split()[-1])
            types.append(i.split()[0])
        func_name = src.split('void ')[-1].split('(')[0]
        cu += src
        cu += '\n'
        cu +=f'void {func_name}_('+', '.join('void* '+i for i in args)+') {\n'
        cu +=f'  {func_name}<<<dim3({", ".join(map(str,grid))}),dim3({threads})>>>('+', '.join(f'({t}){i}' for t,i in zip(types,args))+');\n'
        cu += '}\n'

        cpp +=f'void {func_name}_('+', '.join('void* '+i for i in args)+');\n'
        cpp +=f'void {func_name}('+', '.join('torch::Tensor& '+i for i in args)+') {\n'
        cpp +=f'  {func_name}_('+', '.join(f'(void*){i}.data_ptr()' for i in args)+');\n'
        cpp += '}\n'

        schema = []
        for var,mut in zip(args,mutable):
            if mut: var = '('+var+'!) '+var
            schema.append('Tensor '+var)
        cpp_lib +=f'  m.def("{func_name}('+', '.join(schema)+') -> ()");\n'
        func_names.append(func_name)

    if '__nv_bfloat16' in cu: cu = '#include <cuda_bf16.h>\n'+cu
    if '__half' in cu: cu = '#include <cuda_fp16.h>\n'+cu

    id = '_'.join(func_names) + '_' + hashlib.md5((cu+cpp).encode('utf-8')).hexdigest()

    cpp +=f'TORCH_LIBRARY({id}, m) {{\n'
    cpp += cpp_lib
    cpp += '}\n'
    cpp +=f'TORCH_LIBRARY_IMPL({id}, CUDA, m) {{\n'
    for func_name in func_names:
        cpp +=f'  m.impl("{func_name}", &{func_name});\n'
    cpp += '}\n'

    os.makedirs('/tmp/flex_rnn', exist_ok=True)
    dump(f'/tmp/flex_rnn/{id}.cu', cu)
    dump(f'/tmp/flex_rnn/{id}.cpp', cpp)

    CUDA_FLAGS = [] #['-res-usage']#, ["--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name=id, sources=[f"/tmp/flex_rnn/{id}.cu", f"/tmp/flex_rnn/{id}.cpp"], is_python_module=False, verbose=False, extra_cuda_cflags=CUDA_FLAGS)
    return tuple(getattr(getattr(th.ops, id), func_name) for func_name in func_names)

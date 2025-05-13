import torch, functorch.compile, subprocess, sys, pickle

def trace_fwbw_isolated(f, args):
    #print('flex_rnn: tracing in a separate process to avoid interfering with torch.compile')
    def flat(*args):
        y = f(*args)
        return sum([i if type(i) == tuple else (i,) for i in y], ())
    flat = trace_fw(flat, args)
    args = (flat,[i.shape for i in args])
    #return run(args)
    proc = subprocess.Popen([sys.executable, __file__], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = proc.communicate(input=pickle.dumps(args))[0]
    assert proc.returncode == 0
    return pickle.loads(out)

def run(args):
    f,shapes = args
    args = tuple(torch.empty(shape, device='meta') for shape in shapes)
    return trace_fwbw(f,args)

def trace_fw(f, args):
    ret = None
    def fw(gm, sample_inputs):
        nonlocal ret
        assert ret is None
        ret = gm
        return gm.forward
    functorch.compile.aot_function(f, fw_compiler=fw)(*args)
    return ret

def trace_fwbw(f, args):
    y = f(*args)
    nargs = len(args)
    def fwbw(*args):
        x,dy = args[:nargs], args[nargs:]
        y, dfunc = torch.func.vjp(f, *x)
        dx = dfunc(dy)
        return *y, *dx
    return trace_fw(fwbw, args+y)

if __name__ == '__main__':
    args = pickle.loads(sys.stdin.buffer.read())
    gm = run(args)
    sys.stdout.buffer.write(pickle.dumps(gm))

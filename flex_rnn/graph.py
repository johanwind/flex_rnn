# Copyright (c) 2025, Johan Sokrates Wind

from types import SimpleNamespace
from copy import deepcopy
import ast

class Graph:
    def __init__(self, func_name, code):
        self.inputs = code.split('forward(self, ')[1].split(')')[0].split(', ')
        code = code.split('\n')
        self.func_name = func_name
        self.node = []
        alias = {}
        seen = set(self.inputs)
        for i in code[1:-1]:
            lhs = i.strip().split(';')[0].split(' = ')[0]
            rhs = i.strip().split(';')[0].split(' = torch.ops.aten.')[1]
            op = rhs.split('(')[0]

            tree = ast.parse(rhs)
            args = [ast.get_source_segment(rhs, node) for node in tree.body[0].value.args + tree.body[0].value.keywords]
            args = [(alias[i] if i in alias else i) for i in args]

            named = 1
            named_args, other_args = [], []
            for i in args:
                if not i in seen:
                    named = 0
                    other_args.append(i)
                else:
                    assert named
                    named_args.append(i)

            if op in ('clone.default', '_to_copy.default', 'expand.default'):
                alias[lhs] = named_args[0]
                continue

            seen.add(lhs)
            self.node.append(SimpleNamespace(name=lhs, op=op, named_args=named_args, other_args=other_args))
        self.outputs = [(alias[i] if i in alias else i) for i in code[-1].split('return ')[-1][1:-1].split(', ')]
        self.prune()

    def compute_shapes(self, shapes):
        assert len(shapes) == len(self.inputs)
        assert all(type(i) == list and len(i) == 2 for i in shapes), shapes
        self.shape = dict(zip(self.inputs,shapes))
        for node in self.node:
            h,w = [max(self.shape[i][k] for i in node.named_args) for k in [0,1]]
            if node.op == 'sum.dim_IntList':
                if node.other_args[0] == '[1]': w = 1
                elif node.other_args[0] == '[0]': h = 1
                else: assert 0, node.other_args
            self.shape[node.name] = [h, w]

    def prune(self): # Remove nodes which don't contribute to outputs
        need = set(self.outputs)
        r = []
        for node in reversed(self.node):
            if node.name in need and not node.name in self.inputs:
                need |= set(node.named_args)
                r.append(node)
        self.node = list(reversed(r))
        self.inputs_mask = [(i in need) for i in self.inputs]

    def copy(self, name = None):
        r = deepcopy(self)
        if not name is None: r.func_name = name
        return r

    def transpose(self):
        for k in self.shape: self.shape[k].reverse()
        for node in self.node:
            if node.op == 'sum.dim_IntList':
                if   node.other_args[0] == '[0]': node.other_args[0] = '[1]'
                elif node.other_args[0] == '[1]': node.other_args[0] = '[0]'
                else: assert 0

    def reshape(self, M = None, N = None):
        for v in list(self.shape.values()):
            if not M is None and v[0] > 1: v[0] = M
            if not N is None and v[1] > 1: v[1] = N

    def code(self, shapes = False):
        r =  f'def {self.func_name}('+', '.join(self.inputs)+'):\n'
        for node in self.node:
            r += f'    {node.name} = {node.op}('+', '.join(node.named_args+node.other_args)+')'+(f' # {self.shape[node.name]}' if shapes else '')+f'\n'
        r += '    return '+', '.join(self.outputs)
        return r


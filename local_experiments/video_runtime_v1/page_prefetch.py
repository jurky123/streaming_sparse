"""SM80 page pipeline for the anchored, column-mass fused attention kernel."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import triton
from triton.runtime.jit import JITFunction
from triton.runtime.cache import get_cache_manager
from triton.compiler.compiler import CompiledKernel, make_backend
from ..review_v1.tuned_attention import _tail


def stagger_copies(text):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if '= scf.for' in line), None)
    if start is None:
        return None
    end = next(i for i in range(start + 1, len(lines)) if 'scf.yield' in lines[i])
    body = lines[start + 1:end]
    # Match the tested loop including operand dependencies and tensor layouts.
    # Normalize SSA numbering, the bound for the last page, and debug locations.
    names = {}
    pattern = re.sub(r' loc\(#loc\d+\)', '', '\n'.join(body))
    pattern = re.sub(r'%\w+', lambda m: names.setdefault(m[0], '%' + str(len(names))), pattern)
    if hashlib.sha256(pattern.encode()).hexdigest() != 'c1a0f3382ec80165c9013ac851c00d064cacbb59713a9956fb8069500d0a1bec':
        return None
    vload = next(i for i, line in enumerate(body) if 'local_load' in line and 'mutable' in line and '64x128xbf16' in line.split(' -> ')[0])
    address = next(i for i, line in enumerate(body) if '= arith.addi' in line)
    kcommit = next(i for i, line in enumerate(body) if 'async_commit_group' in line)
    wait = next(i for i, line in enumerate(body) if 'async_wait' in line)
    body[wait] = body[wait].replace('num = 0', 'num = 1')
    body = (body[:vload] + ['      %waitv = triton_gpu.async_wait  {num = 0 : i32}', body[vload]]
            + body[address:kcommit + 1] + body[vload + 1:address] + body[kcommit + 1:])
    return '\n'.join(lines[:start + 1] + body + lines[end:]) + '\n'


class PagePipeline(JITFunction):
    def create_binder(self, backend):
        super().create_binder(backend)
        compile_native = self.compile

        def compile_page(src, target, options):
            kernel = compile_native(src, target=target, options=options)
            if target.backend != 'cuda' or target.arch != 80 or triton.__version__ != '3.2.0':
                return kernel
            scheduled = stagger_copies(kernel.asm['ttgir'])
            if scheduled is None:
                return kernel
            key = hashlib.sha256((kernel.hash + scheduled).encode()).hexdigest()
            cache = get_cache_manager(key)
            group = cache.get_group('page.json')
            if group:
                return CompiledKernel(src, group, key)

            from triton._C.libtriton import ir

            backend = make_backend(target)
            context = ir.context()
            ir.load_dialects(context)
            backend.load_dialects(context)
            with TemporaryDirectory() as temp:
                path = Path(temp) / 'page.ttgir'
                path.write_text(scheduled)
                module = ir.parse_mlir_module(str(path), context)
            module.context = context
            metadata = kernel.metadata._asdict()
            stages = {}
            backend.add_stages(stages, backend.parse_options(options))
            group = {'page.ttgir': cache.put(scheduled, 'page.ttgir')}
            for ext in ('llir', 'ptx', 'cubin'):
                module = stages[ext](module, metadata)
                name = 'page.' + ext
                group[name] = cache.put(module, name)
            metadata.update(hash=key, target=asdict(target))
            group['page.json'] = cache.put(json.dumps(metadata), 'page.json')
            cache.put_group('page.json', group)
            return CompiledKernel(src, group, key)

        self.compile = compile_page


page_tail = PagePipeline(_tail.fn)

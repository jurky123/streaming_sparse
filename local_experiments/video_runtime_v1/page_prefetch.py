"""SM80 page pipeline for the anchored, column-mass fused attention kernel."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import triton
from triton.runtime.jit import JITFunction
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


def align_coefficients(text):
    # Each lane reduces both adjacent groups of four representatives locally.
    # Match both sides of the no-copy 64x64 -> 64x16x4 reshape.
    shape = 'tt.reshape'
    mapping = 'tensor<64x64xf32, #blocked7> -> tensor<64x16x4xf32, #blocked6>'
    if not any(shape in line and mapping in line for line in text.splitlines()):
        return text
    layouts = (
        ('#blocked6', '[4, 1, 4]', '[2, 16, 1]', '[1, 2, 4]', '[8, 4, 1]', '[4, 1, 1]', '[2, 1, 0]'),
        ('#blocked7', '[4, 4]', '[2, 16]', '[1, 8]', '[8, 4]', '[4, 1]', '[1, 0]'),
    )
    result = text
    for name, size, lanes, new_size, new_lanes, warps, order in layouts:
        prefix = f'{name} = #triton_gpu.blocked<{{'
        suffix = f', warpsPerCTA = {warps}, order = {order}}}>'
        old = prefix + f'sizePerThread = {size}, threadsPerWarp = {lanes}' + suffix
        new = prefix + f'sizePerThread = {new_size}, threadsPerWarp = {new_lanes}' + suffix
        if old not in result:
            return text
        result = result.replace(old, new)
    return result


class PagePipeline(JITFunction):
    def __init__(self, fn, *, aligned_coefficients=False):
        super().__init__(fn)
        self.aligned_coefficients = aligned_coefficients

    def create_binder(self, backend):
        super().create_binder(backend)
        compile_native = self.compile

        def compile_page(src, target, options):
            kernel = compile_native(src, target=target, options=options)
            if target.backend != 'cuda' or target.arch != 80:
                return kernel
            scheduled = stagger_copies(kernel.asm['ttgir'])
            if scheduled is None:
                return kernel
            if self.aligned_coefficients:
                scheduled = align_coefficients(scheduled)
            from triton.runtime.cache import get_cache_manager
            from triton.compiler.compiler import CompiledKernel, make_backend

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


if triton.__version__ == "3.2.0":
    page_tail = PagePipeline(_tail.fn)
    coeff_tail = PagePipeline(_tail.fn, aligned_coefficients=True)
else:
    page_tail = coeff_tail = _tail

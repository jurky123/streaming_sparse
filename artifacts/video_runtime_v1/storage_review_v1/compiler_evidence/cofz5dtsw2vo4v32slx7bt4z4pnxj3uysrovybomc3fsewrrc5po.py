# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/qi/cqidsavawpr4uz4nw4sm2dn3m7igr4fyavno5366a5wao35sljcl.py
# Topologically Sorted Source Nodes: [stats], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   stats => full_default_14
# Graph fragment:
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 288), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_stack_0 = async_compile.triton('triton_poi_fused_stack_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (), 'tt.equal_to': (1,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.full([1], 288, tl.int64)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/qj/cqjxnuqfe4bkdrpgcfy3sv5jxoizzp5bb4mtrcsgc2hn5jywvslc.py
# Topologically Sorted Source Nodes: [valid_1, sum_8], Original ATen: [aten.expand, aten.sum]
# Source node to ATen node mapping:
#   sum_8 => sum_10
#   valid_1 => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([12, 24, %arg5_1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%full_default, [-1]), kwargs = {})
triton_red_fused_expand_sum_1 = async_compile.triton('triton_red_fused_expand_sum_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_expand_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_expand_sum_1(out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    x0 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        tmp0 = tl.full([1, 1], 1, tl.int64)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/vf/cvfy3ikozxx3msw7oyogqxejtoxrktdkx62p4fbuf5clhr2fgzxy.py
# Topologically Sorted Source Nodes: [arange, active], Original ATen: [aten.arange, aten.lt]
# Source node to ATen node mapping:
#   active => lt_2
#   arange => iota
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota, %unsqueeze_27), kwargs = {})
triton_poi_fused_arange_lt_2 = async_compile.triton('triton_poi_fused_arange_lt_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i1', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_lt_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_lt_2(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 128
    x0 = (xindex % 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = libdevice.floor(tl.full([], 0.100000000000000, tl.float64)*ks0.to(tl.float64)).to(tl.int32)
    tmp2 = triton_helpers.minimum(tmp1, tmp0)
    tmp3 = x0
    tmp4 = tmp3 < tmp2
    tl.store(out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xn/cxnsucvm7ii2jtms6nwjqtibulsh4izmyoujqjbggikrh5jsmrw6.py
# Topologically Sorted Source Nodes: [zeros_like_1, arange, active, mask], Original ATen: [aten.zeros_like, aten.arange, aten.lt, aten.scatter]
# Source node to ATen node mapping:
#   active => lt_2
#   arange => iota
#   mask => scatter_1
#   zeros_like_1 => full_default_3
# Graph fragment:
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([12, 24, %arg5_1], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota, %unsqueeze_27), kwargs = {})
#   %scatter_1 : [num_users=4] = call_function[target=torch.ops.aten.scatter.src](args = (%full_default_3, -1, %getitem_1, %lt_2), kwargs = {})
triton_poi_fused_arange_lt_scatter_zeros_like_3 = async_compile.triton('triton_poi_fused_arange_lt_scatter_zeros_like_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_lt_scatter_zeros_like_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_lt_scatter_zeros_like_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], False, tl.int1)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/le/cledl5uqk7bkmwehgefeoigvqwawubra2cjjt42zol2xraodn73p.py
# Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   sum_2 => sum_3
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_5, [-1]), kwargs = {})
triton_per_fused_sum_4 = async_compile.triton('triton_per_fused_sum_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_4(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 16)
    r4 = rindex
    x0 = (xindex % 8)
    x2 = ((xindex // 128) % 12)
    x3 = xindex // 1536
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, RBLOCK], 64, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 64), "index out of bounds: 0 <= tmp4 < 64")
    tmp6 = tl.load(in_ptr1 + (r4 + 16*x0 + 128*x2 + 1536*tmp4 + 98304*x3), None)
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.sum(tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/oe/coehkz3mnwyxhmmupp3zqj7bievcqk5dn5srq52qdxd57ttzv3ut.py
# Topologically Sorted Source Nodes: [q2], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   q2 => div_2
# Graph fragment:
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, 128), kwargs = {})
triton_poi_fused_div_5 = async_compile.triton('triton_poi_fused_div_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 24)
    x2 = xindex // 3072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 1536*x1), None)
    tmp1 = 0.0078125
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ev/cevsngpdut4ualknjiloqu2epuimdf6ook7e5jbhkgbkznrjkvon.py
# Topologically Sorted Source Nodes: [mapping, mapping_1], Original ATen: [aten.full_like, aten.scatter]
# Source node to ATen node mapping:
#   mapping => full_default_8
#   mapping_1 => scatter_3
# Graph fragment:
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([288, %arg5_1], -1), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_3 : [num_users=1] = call_function[target=torch.ops.aten.scatter.src](args = (%full_default_8, 1, %view_13, %expand_1), kwargs = {})
triton_poi_fused_full_like_scatter_6 = async_compile.triton('triton_poi_fused_full_like_scatter_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_full_like_scatter_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_full_like_scatter_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], -1, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/5w/c5wfobi44ikmirw3x4rgdlaiv7pypug35hq4cgf7sj2nhqsft4ey.py
# Topologically Sorted Source Nodes: [st], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   st => convert_element_type_3
# Graph fragment:
#   %convert_element_type_3 : [num_users=6] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.float32), kwargs = {})
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/6x/c6xjxo4fard6zqfmdbjpivhu3sxoibkpbi35ljkfquxigtc525zm.py
# Topologically Sorted Source Nodes: [amax], Original ATen: [aten.amax]
# Source node to ATen node mapping:
#   amax => amax_1
# Graph fragment:
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%select_2, [-1], True), kwargs = {})
triton_red_fused_amax_8 = async_compile.triton('triton_red_fused_amax_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_amax_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_amax_8(in_ptr0, out_ptr0, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 36
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 3)
    x1 = xindex // 3
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + x0*((2 + ks0) // 3)
        tmp1 = ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (23 + 48*r2 + 48*ks0*x1 + 48*x0*((2 + ks0) // 3)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/vo/cvoh5m33fr7wvlnca74dnm6ruxxb2pitjfn7iubj4lapkamgth5t.py
# Topologically Sorted Source Nodes: [amax], Original ATen: [aten.amax]
# Source node to ATen node mapping:
#   amax => amax_1
# Graph fragment:
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%select_2, [-1], True), kwargs = {})
triton_per_fused_amax_9 = async_compile.triton('triton_per_fused_amax_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_amax_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_amax_9(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 3*x0), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/x3/cx3d5wsce2elc42rrgzlzqu2ohobtgioupcfjbywint2ixgnm3ry.py
# Topologically Sorted Source Nodes: [tensor_bmm_default], Original ATen: [streaming_sparse.tensor_bmm]
# Source node to ATen node mapping:
#   tensor_bmm_default => tensor_bmm
# Graph fragment:
#   %tensor_bmm : [num_users=1] = call_function[target=torch.ops.streaming_sparse.tensor_bmm.default](args = (%view, %permute), kwargs = {})
triton_poi_fused_tensor_bmm_10 = async_compile.triton('triton_poi_fused_tensor_bmm_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tensor_bmm_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_tensor_bmm_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 384)
    x2 = xindex // 49152
    tmp0 = tl.load(in_ptr0 + (((((x3 // 128) % 384)) % 16)), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 64, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 64), "index out of bounds: 0 <= tmp4 < 64")
    tmp6 = tl.load(in_ptr1 + (x0 + 128*x2 + 1536*tmp4 + 98304*(x1 // 16)), None)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/cw/ccwy35wptn3zu7dotqnvqyvnvupbx23jtzia7w2sqlr3ymka3agj.py
# Topologically Sorted Source Nodes: [probes, norm], Original ATen: [aten.index_select, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   norm => pow_2, sum_4
#   probes => index
# Graph fragment:
#   %index : [num_users=4] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [None, None, %arg4_1]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%index, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [-1]), kwargs = {})
triton_red_fused_index_select_linalg_vector_norm_11 = async_compile.triton('triton_red_fused_index_select_linalg_vector_norm_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_select_linalg_vector_norm_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_index_select_linalg_vector_norm_11(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4608
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    x1 = ((xindex // 16) % 12)
    x2 = xindex // 192
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x6 = xindex
    x4 = ((xindex // 16) % 24)
    x5 = xindex // 384
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp1 = tl.full([XBLOCK, RBLOCK], 64, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 64)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 64")
        tmp6 = tl.load(in_ptr1 + (r3 + 128*x1 + 1536*tmp4 + 98304*x2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp6 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.load(in_ptr1 + (r3 + 128*x5 + 1536*tmp4 + 98304*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tl.store(out_ptr1 + (r3 + 128*x6), tmp11, rmask & xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x6), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/2v/c2vliqcz2hpejio465jx3jr6j6og3scpeprmk7vnpkbv742dmpem.py
# Topologically Sorted Source Nodes: [tensor_bmm_default], Original ATen: [streaming_sparse.tensor_bmm]
# Source node to ATen node mapping:
#   tensor_bmm_default => tensor_bmm
# Graph fragment:
#   %tensor_bmm : [num_users=1] = call_function[target=torch.ops.streaming_sparse.tensor_bmm.default](args = (%view, %permute), kwargs = {})
triton_poi_fused_tensor_bmm_12 = async_compile.triton('triton_poi_fused_tensor_bmm_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tensor_bmm_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_tensor_bmm_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/mp/cmp2wgmhnchgazqwpvqbeketr2b2c2yfvtz6hnuz4vdyztzaalpm.py
# Topologically Sorted Source Nodes: [z, softmax], Original ATen: [aten.div, aten._softmax]
# Source node to ATen node mapping:
#   softmax => amax, exp, sub_8, sum_1
#   z => div
# Graph fragment:
#   %div : [num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_2, 11.313708498984761), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_3, [-1], True), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_3, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_8,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
triton_red_fused__softmax_div_13 = async_compile.triton('triton_red_fused__softmax_div_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_div_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_div_13(in_out_ptr0, out_ptr0, out_ptr1, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.08838834764831843
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tl.store(in_out_ptr0 + (r1 + 4*ks0*x0), tmp2, rmask & xmask)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_out_ptr0 + (r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp6 - tmp4
        tmp8 = tl_math.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/k3/ck3em7eoezdnaijwhmfa4hetxozl4fafhqcbly7u3pxoxd736yew.py
# Topologically Sorted Source Nodes: [zz, lm, logsumexp_1], Original ATen: [aten._to_copy, aten.logsumexp]
# Source node to ATen node mapping:
#   lm => abs_2, add_503, amax_2, eq_118, exp_1, full_default_4, log, sub_95, sum_12, where_3
#   logsumexp_1 => amax_3
#   zz => convert_element_type_6
# Graph fragment:
#   %convert_element_type_6 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float64), kwargs = {})
#   %amax_2 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_6, [-1], True), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax_2,), kwargs = {})
#   %eq_118 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_2, inf), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_118, %full_default_4, %amax_2), kwargs = {})
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %where_3), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_95,), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1]), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_12,), kwargs = {})
#   %add_503 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %squeeze), kwargs = {})
#   %amax_3 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%add_503, [-1], True), kwargs = {})
triton_red_fused__to_copy_logsumexp_14 = async_compile.triton('triton_red_fused__to_copy_logsumexp_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_logsumexp_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_logsumexp_14(in_ptr0, out_ptr0, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 3)
    x1 = xindex // 3
    _tmp35 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float64)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + x0*((2 + ks0) // 3)
        tmp1 = ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (4*r2 + 4*ks0*x1 + 4*x0*((2 + ks0) // 3)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp3.to(tl.float64)
        tmp5 = tl.load(in_ptr0 + (1 + 4*r2 + 4*ks0*x1 + 4*x0*((2 + ks0) // 3)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5.to(tl.float64)
        tmp7 = triton_helpers.maximum(tmp4, tmp6)
        tmp8 = tl.load(in_ptr0 + (2 + 4*r2 + 4*ks0*x1 + 4*x0*((2 + ks0) // 3)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp8.to(tl.float64)
        tmp10 = triton_helpers.maximum(tmp7, tmp9)
        tmp11 = tl.load(in_ptr0 + (3 + 4*r2 + 4*ks0*x1 + 4*x0*((2 + ks0) // 3)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp11.to(tl.float64)
        tmp13 = triton_helpers.maximum(tmp10, tmp12)
        tmp14 = tl_math.abs(tmp13)
        tmp15 = tl.full([1, 1], float("inf"), tl.float64)
        tmp16 = tmp14 == tmp15
        tmp17 = tl.full([1, 1], 0.0, tl.float64)
        tmp18 = tl.where(tmp16, tmp17, tmp13)
        tmp19 = tmp4 - tmp18
        tmp20 = libdevice.exp(tmp19)
        tmp21 = tmp6 - tmp18
        tmp22 = libdevice.exp(tmp21)
        tmp23 = tmp20 + tmp22
        tmp24 = tmp9 - tmp18
        tmp25 = libdevice.exp(tmp24)
        tmp26 = tmp23 + tmp25
        tmp27 = tmp12 - tmp18
        tmp28 = libdevice.exp(tmp27)
        tmp29 = tmp26 + tmp28
        tmp30 = libdevice.log(tmp29)
        tmp31 = tmp30 + tmp18
        tmp32 = tl.full(tmp31.shape, float("-inf"), tmp31.dtype)
        tmp33 = tl.where(tmp2, tmp31, tmp32)
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
        tmp36 = triton_helpers.maximum(_tmp35, tmp34)
        _tmp35 = tl.where(rmask & xmask, tmp36, _tmp35)
    tmp35 = triton_helpers.max2(_tmp35, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp35, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/he/che3ig76x6petixdtknf7avdjbsbcgjq34f7t3hi7rdqjatz2lyz.py
# Topologically Sorted Source Nodes: [zz, lm, logsumexp_1], Original ATen: [aten._to_copy, aten.logsumexp]
# Source node to ATen node mapping:
#   lm => abs_2, add_503, amax_2, eq_118, exp_1, full_default_4, log, sub_95, sum_12, where_3
#   logsumexp_1 => amax_3
#   zz => convert_element_type_6
# Graph fragment:
#   %convert_element_type_6 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float64), kwargs = {})
#   %amax_2 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_6, [-1], True), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax_2,), kwargs = {})
#   %eq_118 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_2, inf), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_118, %full_default_4, %amax_2), kwargs = {})
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %where_3), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_95,), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1]), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_12,), kwargs = {})
#   %add_503 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %squeeze), kwargs = {})
#   %amax_3 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%add_503, [-1], True), kwargs = {})
triton_per_fused__to_copy_logsumexp_15 = async_compile.triton('triton_per_fused__to_copy_logsumexp_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_logsumexp_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_logsumexp_15(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 3*x0), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/u4/cu4tiafkshhsikhlgdfchmy47oflle7fww4k4zjypeujmsisrykq.py
# Topologically Sorted Source Nodes: [alpha, truediv_3, sub, abs_1, mul_2, sum_3, u, sqrt, mul_3, mul_4, x1, beta, mul_5, sum_5, square_1, add_2, mul_7, sum_6, zz, lm, logsumexp_1, lm_1], Original ATen: [aten.sum, aten.div, aten.sub, aten.abs, aten.mul, aten.clamp_min, aten.sqrt, aten.pow, aten.add, aten._to_copy, aten.logsumexp]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   add_2 => add_413
#   alpha => sum_2
#   beta => mul_140
#   lm => abs_2, add_503, amax_2, eq_118, exp_1, full_default_4, log, sub_95, sum_12, where_3
#   lm_1 => sub_98
#   logsumexp_1 => abs_3, add_509, eq_121, exp_2, full_default_5, log_1, sub_97, sum_13, where_4
#   mul_2 => mul_237
#   mul_3 => mul_256
#   mul_4 => mul_289
#   mul_5 => mul_308
#   mul_7 => mul_342
#   sqrt => sqrt
#   square_1 => pow_4
#   sub => sub_44
#   sum_3 => sum_5
#   sum_5 => sum_7
#   sum_6 => sum_8
#   truediv_3 => div_4
#   u => clamp_min
#   x1 => sum_6
#   zz => convert_element_type_6
# Graph fragment:
#   %sum_2 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_4, [-1]), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%unsqueeze_10, 4), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %div_4), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_44,), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%abs_1, %select_5), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_237, [-1]), kwargs = {})
#   %clamp_min : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%view_9, 0), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min,), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %sqrt), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %select_6), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_289, [-1]), kwargs = {})
#   %mul_140 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %select), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %mul_140), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_308, [-1]), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_140, 2), kwargs = {})
#   %add_413 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min, %pow_4), kwargs = {})
#   %mul_342 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %add_413), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_342, [-1]), kwargs = {})
#   %convert_element_type_6 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float64), kwargs = {})
#   %amax_2 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_6, [-1], True), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax_2,), kwargs = {})
#   %eq_118 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_2, inf), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_118, %full_default_4, %amax_2), kwargs = {})
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %where_3), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_95,), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1]), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_12,), kwargs = {})
#   %add_503 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %squeeze), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax_3,), kwargs = {})
#   %eq_121 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_3, inf), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_121, %full_default_5, %amax_3), kwargs = {})
#   %sub_97 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_503, %where_4), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_97,), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [-1], True), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_13,), kwargs = {})
#   %add_509 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log_1, %where_4), kwargs = {})
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_503, %add_509), kwargs = {})
triton_red_fused__to_copy_abs_add_clamp_min_div_logsumexp_mul_pow_sqrt_sub_sum_16 = async_compile.triton('triton_red_fused__to_copy_abs_add_clamp_min_div_logsumexp_mul_pow_sqrt_sub_sum_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr7': '*fp64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_abs_add_clamp_min_div_logsumexp_mul_pow_sqrt_sub_sum_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 28, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_abs_add_clamp_min_div_logsumexp_mul_pow_sqrt_sub_sum_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr7, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    x3 = xindex // 384
    x4 = (xindex % 16)
    x5 = ((xindex // 16) % 24)
    tmp46 = tl.load(in_ptr4 + (x4 + 16*x3 + 192*x5), xmask, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    _tmp140 = tl.full([XBLOCK, RBLOCK], 0, tl.float64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr0 + (1 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr0 + (2 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr0 + (3 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr3 + (9 + 48*r1 + 48*ks0*x3), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr3 + (21 + 48*r1 + 48*ks0*x3), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (33 + 48*r1 + 48*ks0*x3), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp42 = tl.load(in_ptr3 + (45 + 48*r1 + 48*ks0*x3), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp50 = tl.load(in_ptr5 + (8 + 48*r1 + 48*ks0*x3), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp53 = tl.load(in_ptr5 + (20 + 48*r1 + 48*ks0*x3), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp57 = tl.load(in_ptr5 + (32 + 48*r1 + 48*ks0*x3), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp61 = tl.load(in_ptr5 + (44 + 48*r1 + 48*ks0*x3), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp65 = tl.load(in_ptr6 + (4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp70 = tl.load(in_ptr5 + (10 + 48*r1 + 48*ks0*x3), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp72 = tl.load(in_ptr6 + (1 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp76 = tl.load(in_ptr5 + (22 + 48*r1 + 48*ks0*x3), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp79 = tl.load(in_ptr6 + (2 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp83 = tl.load(in_ptr5 + (34 + 48*r1 + 48*ks0*x3), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp86 = tl.load(in_ptr6 + (3 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp90 = tl.load(in_ptr5 + (46 + 48*r1 + 48*ks0*x3), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = tl_math.exp(tmp2)
        tmp5 = tmp3 / tmp4
        tmp7 = tmp6 - tmp1
        tmp8 = tl_math.exp(tmp7)
        tmp9 = tmp8 / tmp4
        tmp10 = tmp5 + tmp9
        tmp12 = tmp11 - tmp1
        tmp13 = tl_math.exp(tmp12)
        tmp14 = tmp13 / tmp4
        tmp15 = tmp10 + tmp14
        tmp17 = tmp16 - tmp1
        tmp18 = tl_math.exp(tmp17)
        tmp19 = tmp18 / tmp4
        tmp20 = tmp15 + tmp19
        tmp21 = 0.25
        tmp22 = tmp20 * tmp21
        tmp23 = tmp5 - tmp22
        tmp24 = tl_math.abs(tmp23)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp28 = tmp9 - tmp22
        tmp29 = tl_math.abs(tmp28)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 * tmp31
        tmp33 = tmp27 + tmp32
        tmp34 = tmp14 - tmp22
        tmp35 = tl_math.abs(tmp34)
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 * tmp37
        tmp39 = tmp33 + tmp38
        tmp40 = tmp19 - tmp22
        tmp41 = tl_math.abs(tmp40)
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp41 * tmp43
        tmp45 = tmp39 + tmp44
        tmp47 = libdevice.sqrt(tmp46)
        tmp48 = 0.08838834764831843
        tmp49 = tmp47 * tmp48
        tmp51 = tmp49 * tmp50
        tmp52 = tmp5 * tmp51
        tmp54 = tmp49 * tmp53
        tmp55 = tmp9 * tmp54
        tmp56 = tmp52 + tmp55
        tmp58 = tmp49 * tmp57
        tmp59 = tmp14 * tmp58
        tmp60 = tmp56 + tmp59
        tmp62 = tmp49 * tmp61
        tmp63 = tmp19 * tmp62
        tmp64 = tmp60 + tmp63
        tmp66 = 0.0
        tmp67 = triton_helpers.maximum(tmp65, tmp66)
        tmp68 = libdevice.sqrt(tmp67)
        tmp69 = tmp5 * tmp68
        tmp71 = tmp69 * tmp70
        tmp73 = triton_helpers.maximum(tmp72, tmp66)
        tmp74 = libdevice.sqrt(tmp73)
        tmp75 = tmp9 * tmp74
        tmp77 = tmp75 * tmp76
        tmp78 = tmp71 + tmp77
        tmp80 = triton_helpers.maximum(tmp79, tmp66)
        tmp81 = libdevice.sqrt(tmp80)
        tmp82 = tmp14 * tmp81
        tmp84 = tmp82 * tmp83
        tmp85 = tmp78 + tmp84
        tmp87 = triton_helpers.maximum(tmp86, tmp66)
        tmp88 = libdevice.sqrt(tmp87)
        tmp89 = tmp19 * tmp88
        tmp91 = tmp89 * tmp90
        tmp92 = tmp85 + tmp91
        tmp93 = tmp51 * tmp51
        tmp94 = tmp67 + tmp93
        tmp95 = tmp5 * tmp94
        tmp96 = tmp54 * tmp54
        tmp97 = tmp73 + tmp96
        tmp98 = tmp9 * tmp97
        tmp99 = tmp95 + tmp98
        tmp100 = tmp58 * tmp58
        tmp101 = tmp80 + tmp100
        tmp102 = tmp14 * tmp101
        tmp103 = tmp99 + tmp102
        tmp104 = tmp62 * tmp62
        tmp105 = tmp87 + tmp104
        tmp106 = tmp19 * tmp105
        tmp107 = tmp103 + tmp106
        tmp108 = tmp0.to(tl.float64)
        tmp109 = tmp6.to(tl.float64)
        tmp110 = triton_helpers.maximum(tmp108, tmp109)
        tmp111 = tmp11.to(tl.float64)
        tmp112 = triton_helpers.maximum(tmp110, tmp111)
        tmp113 = tmp16.to(tl.float64)
        tmp114 = triton_helpers.maximum(tmp112, tmp113)
        tmp115 = tl_math.abs(tmp114)
        tmp116 = tl.full([1, 1], float("inf"), tl.float64)
        tmp117 = tmp115 == tmp116
        tmp118 = tl.full([1, 1], 0.0, tl.float64)
        tmp119 = tl.where(tmp117, tmp118, tmp114)
        tmp120 = tmp108 - tmp119
        tmp121 = libdevice.exp(tmp120)
        tmp122 = tmp109 - tmp119
        tmp123 = libdevice.exp(tmp122)
        tmp124 = tmp121 + tmp123
        tmp125 = tmp111 - tmp119
        tmp126 = libdevice.exp(tmp125)
        tmp127 = tmp124 + tmp126
        tmp128 = tmp113 - tmp119
        tmp129 = libdevice.exp(tmp128)
        tmp130 = tmp127 + tmp129
        tmp131 = libdevice.log(tmp130)
        tmp132 = tmp131 + tmp119
        tmp134 = tl_math.abs(tmp133)
        tmp135 = tmp134 == tmp116
        tmp136 = tl.where(tmp135, tmp118, tmp133)
        tmp137 = tmp132 - tmp136
        tmp138 = libdevice.exp(tmp137)
        tmp139 = tl.broadcast_to(tmp138, [XBLOCK, RBLOCK])
        tmp141 = _tmp140 + tmp139
        _tmp140 = tl.where(rmask & xmask, tmp141, _tmp140)
        tl.store(out_ptr0 + (r1 + ks0*x0), tmp20, rmask & xmask)
        tl.store(out_ptr1 + (r1 + ks0*x0), tmp45, rmask & xmask)
        tl.store(out_ptr2 + (r1 + ks0*x0), tmp64, rmask & xmask)
        tl.store(out_ptr3 + (r1 + ks0*x0), tmp92, rmask & xmask)
        tl.store(out_ptr4 + (r1 + ks0*x0), tmp107, rmask & xmask)
    tmp140 = tl.sum(_tmp140, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp142 = tl.load(in_ptr0 + (4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp144 = tl.load(in_ptr0 + (1 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp147 = tl.load(in_ptr0 + (2 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp150 = tl.load(in_ptr0 + (3 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp143 = tmp142.to(tl.float64)
        tmp145 = tmp144.to(tl.float64)
        tmp146 = triton_helpers.maximum(tmp143, tmp145)
        tmp148 = tmp147.to(tl.float64)
        tmp149 = triton_helpers.maximum(tmp146, tmp148)
        tmp151 = tmp150.to(tl.float64)
        tmp152 = triton_helpers.maximum(tmp149, tmp151)
        tmp153 = tl_math.abs(tmp152)
        tmp154 = tl.full([1, 1], float("inf"), tl.float64)
        tmp155 = tmp153 == tmp154
        tmp156 = tl.full([1, 1], 0.0, tl.float64)
        tmp157 = tl.where(tmp155, tmp156, tmp152)
        tmp158 = tmp143 - tmp157
        tmp159 = libdevice.exp(tmp158)
        tmp160 = tmp145 - tmp157
        tmp161 = libdevice.exp(tmp160)
        tmp162 = tmp159 + tmp161
        tmp163 = tmp148 - tmp157
        tmp164 = libdevice.exp(tmp163)
        tmp165 = tmp162 + tmp164
        tmp166 = tmp151 - tmp157
        tmp167 = libdevice.exp(tmp166)
        tmp168 = tmp165 + tmp167
        tmp169 = libdevice.log(tmp168)
        tmp170 = tmp169 + tmp157
        tmp171 = libdevice.log(tmp140)
        tmp172 = tl_math.abs(tmp133)
        tmp173 = tmp172 == tmp154
        tmp174 = tl.where(tmp173, tmp156, tmp133)
        tmp175 = tmp171 + tmp174
        tmp176 = tmp170 - tmp175
        tl.store(out_ptr7 + (r1 + ks0*x0), tmp176, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/7w/c7wxsliaut3m4hyzaz3ertvdc5gdoj4jjr6aebvgwgqlm6tx46gj.py
# Topologically Sorted Source Nodes: [f], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   f => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_19, %unsqueeze_20, %unsqueeze_21, %unsqueeze_22], -1), kwargs = {})
triton_poi_fused_stack_17 = async_compile.triton('triton_poi_fused_stack_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x4 = xindex // 4
    x1 = ((xindex // 4) % ks0)
    x3 = xindex // ks1
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (11 + 48*x1 + 48*ks0*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tmp9 = tl.load(in_ptr2 + (x4), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 2, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tmp13 & tmp15
    tmp17 = tl.load(in_ptr3 + (x4), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp0 >= tmp14
    tmp19 = tl.full([1], 3, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tmp18 & tmp20
    tmp22 = tl.load(in_ptr4 + (23 + 48*x1 + 48*ks0*x3), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr5 + (x3), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.load(in_ptr6 + (x4), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 * tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp21, tmp26, tmp27)
    tmp29 = tmp0 >= tmp19
    tmp30 = tl.full([1], 4, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tl.load(in_ptr4 + (23 + 48*x1 + 48*ks0*x3), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr5 + (x3), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tl.load(in_ptr7 + (x4), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp34 * tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp29, tmp36, tmp37)
    tmp39 = tl.where(tmp21, tmp28, tmp38)
    tmp40 = tl.where(tmp16, tmp17, tmp39)
    tmp41 = tl.where(tmp4, tmp12, tmp40)
    tl.store(out_ptr0 + (x5), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/bq/cbqysbfglgdtmqlums4lo47x5wka4wnhy4qknsb43vriamyriwb5.py
# Topologically Sorted Source Nodes: [invert, masked_fill, mul_9, sum_7, risk, ranked], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.mul, aten.sum, aten.mean, aten.scatter]
# Source node to ATen node mapping:
#   invert => full_default_1
#   masked_fill => full_default_2, where
#   mul_9 => mul_362
#   ranked => scatter
#   risk => mean
#   sum_7 => sum_9
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([12, 24, %arg5_1], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_362 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_25), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_362, [-1]), kwargs = {})
#   %mean : [num_users=5] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_9, [2]), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_1, %full_default_2, %mean), kwargs = {})
#   %scatter : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%where, -1, %unsqueeze_26, inf), kwargs = {})
triton_per_fused_bitwise_not_masked_fill_mean_mul_scatter_sum_18 = async_compile.triton('triton_per_fused_bitwise_not_masked_fill_mean_mul_scatter_sum_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_bitwise_not_masked_fill_mean_mul_scatter_sum_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_bitwise_not_masked_fill_mean_mul_scatter_sum_18(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = (xindex % ks0)
    x5 = xindex // ks0
    x2 = xindex // ks1
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0 + 4*ks0*r3 + 64*ks0*x5), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (4*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x0 + 4*ks0*r3 + 64*ks0*x5), xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x0 + 4*ks0*r3 + 64*ks0*x5), xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x0 + 4*ks0*r3 + 64*ks0*x5), xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr1 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 16.0
    tmp20 = tmp18 / tmp19
    tmp21 = tl.full([1, 1], False, tl.int1)
    tmp22 = float("-inf")
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp20, xmask)
    tl.store(out_ptr0 + (x4), tmp23, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/co/ccojm64mg7ytzaur64g7cxxin4kas2wtuoqnojwifp3nikwmm6jl.py
# Topologically Sorted Source Nodes: [invert, masked_fill, ranked], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.scatter]
# Source node to ATen node mapping:
#   invert => full_default_1
#   masked_fill => full_default_2, where
#   ranked => scatter
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([12, 24, %arg5_1], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_1, %full_default_2, %mean), kwargs = {})
#   %scatter : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%where, -1, %unsqueeze_26, inf), kwargs = {})
triton_poi_fused_bitwise_not_masked_fill_scatter_19 = async_compile.triton('triton_poi_fused_bitwise_not_masked_fill_scatter_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_not_masked_fill_scatter_19', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_not_masked_fill_scatter_19(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp0 < ks0")
    tmp2 = float("inf")
    tl.store(out_ptr0 + (tmp0 + ks0*x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/va/cvacan66k7mpe3at75id5lk6g6y2hermrv6ynsr3l4cmvi3zsurj.py
# Topologically Sorted Source Nodes: [removable], Original ATen: [aten.scatter]
# Source node to ATen node mapping:
#   removable => scatter_2
# Graph fragment:
#   %scatter_2 : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%scatter_1, -1, %unsqueeze_28, False), kwargs = {})
triton_poi_fused_scatter_20 = async_compile.triton('triton_poi_fused_scatter_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scatter_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_scatter_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.int1)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/j2/cj2e3kaw4cnz7uqblp2s5ra6havlpmaz5hur3jdiwabevo5xbiil.py
# Topologically Sorted Source Nodes: [removable], Original ATen: [aten.scatter]
# Source node to ATen node mapping:
#   removable => scatter_2
# Graph fragment:
#   %scatter_2 : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%scatter_1, -1, %unsqueeze_28, False), kwargs = {})
triton_poi_fused_scatter_21 = async_compile.triton('triton_poi_fused_scatter_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i1', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scatter_21', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_scatter_21(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp0 < ks0")
    tmp2 = tl.full([1], False, tl.int1)
    tl.store(out_ptr0 + (tmp0 + ks0*x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/gk/cgkxpoivi2sz5qkhvyl3qg3qpiuk25htetjd3rgzdmunixzv275x.py
# Topologically Sorted Source Nodes: [invert_3, masked_fill_3, masked_fill_4, rr, invert_4, result], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.gather, aten.scatter]
# Source node to ATen node mapping:
#   invert_3 => bitwise_not_3
#   invert_4 => bitwise_not_4
#   masked_fill_3 => full_default_6, where_5
#   masked_fill_4 => full_default_7, where_6
#   result => scatter_5
#   rr => gather
# Graph fragment:
#   %bitwise_not_3 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%scatter_2,), kwargs = {})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not_3, %full_default_6, %mean), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%scatter_1, %full_default_7, %mean), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view_22, 1, %unsqueeze_42), kwargs = {})
#   %bitwise_not_4 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze_44,), kwargs = {})
#   %scatter_5 : [num_users=1] = call_function[target=torch.ops.aten.scatter.src](args = (%view_21, 1, %gather, %bitwise_not_4), kwargs = {})
triton_poi_fused_bitwise_not_gather_masked_fill_scatter_22 = async_compile.triton('triton_poi_fused_bitwise_not_gather_masked_fill_scatter_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_not_gather_masked_fill_scatter_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_not_gather_masked_fill_scatter_22(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.int1)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask).to(tl.int1)
    tmp1 = tmp0 == 0
    tmp3 = float("inf")
    tmp4 = tl.where(tmp1, tmp3, tmp2)
    tmp6 = float("-inf")
    tmp7 = tl.where(tmp5, tmp6, tmp2)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tl.store(out_ptr2 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ka/cka6ak7dr5qdjomj7ntaz7knmknl5zl4nlwypw2xwme4fl4gxxye.py
# Topologically Sorted Source Nodes: [mapping, mapping_1], Original ATen: [aten.full_like, aten.scatter]
# Source node to ATen node mapping:
#   mapping => full_default_8
#   mapping_1 => scatter_3
# Graph fragment:
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([288, %arg5_1], -1), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_3 : [num_users=1] = call_function[target=torch.ops.aten.scatter.src](args = (%full_default_8, 1, %view_13, %expand_1), kwargs = {})
triton_poi_fused_full_like_scatter_23 = async_compile.triton('triton_poi_fused_full_like_scatter_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i32', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_full_like_scatter_23', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_full_like_scatter_23(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x0 + ks0*x1), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp0 < ks0")
    tmp2 = x0
    tl.store(out_ptr0 + (tmp0 + ks0*x1), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/du/cdunankaytyoyloiogho4xeqhi7mxjvlm2gylknyf4ktodz7quj2.py
# Topologically Sorted Source Nodes: [mapping_2], Original ATen: [aten.scatter]
# Source node to ATen node mapping:
#   mapping_2 => scatter_4
# Graph fragment:
#   %scatter_4 : [num_users=1] = call_function[target=torch.ops.aten.scatter.src](args = (%scatter_3, 1, %view_14, %expand_2), kwargs = {})
triton_poi_fused_scatter_24 = async_compile.triton('triton_poi_fused_scatter_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scatter_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_scatter_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/wz/cwzj73qfah6f2v4oigwtc6466lni3cdzahjqperfsh3ou2d7g56o.py
# Topologically Sorted Source Nodes: [mapping_2], Original ATen: [aten.scatter]
# Source node to ATen node mapping:
#   mapping_2 => scatter_4
# Graph fragment:
#   %scatter_4 : [num_users=1] = call_function[target=torch.ops.aten.scatter.src](args = (%scatter_3, 1, %view_14, %expand_2), kwargs = {})
triton_poi_fused_scatter_25 = async_compile.triton('triton_poi_fused_scatter_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i32', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scatter_25', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_scatter_25(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + (x0 + ks0*x1), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp0 < ks0")
    tmp2 = 16 + x0
    tl.store(out_ptr0 + (tmp0 + ks0*x1), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/o7/co7ihzpdv7q4fzjv3z4mnmstplxmv2naais5h7d53dyrn6eqlsin.py
# Topologically Sorted Source Nodes: [lb], Original ATen: [aten.logsumexp]
# Source node to ATen node mapping:
#   lb => abs_4, add_625, amax_5, eq_155, exp_4, full_default_9, log_2, sub_125, sum_16, where_7
# Graph fragment:
#   %amax_5 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%select_10, [2], True), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax_5,), kwargs = {})
#   %eq_155 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_4, inf), kwargs = {})
#   %full_default_9 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_7 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_155, %full_default_9, %amax_5), kwargs = {})
#   %sub_125 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_10, %where_7), kwargs = {})
#   %exp_4 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_125,), kwargs = {})
#   %sum_16 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_4, [2]), kwargs = {})
#   %log_2 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_16,), kwargs = {})
#   %add_625 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%log_2, %squeeze_1), kwargs = {})
triton_red_fused_logsumexp_26 = async_compile.triton('triton_red_fused_logsumexp_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_ptr0': '*fp64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_logsumexp_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_logsumexp_26(in_out_ptr0, in_ptr0, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (2 + 4*r1 + 4*x0*((127 + ks0) // 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (2 + 4*r1 + 4*x0*((127 + ks0) // 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl_math.abs(tmp2)
        tmp6 = tl.full([1, 1], float("inf"), tl.float64)
        tmp7 = tmp5 == tmp6
        tmp8 = tl.full([1, 1], 0.0, tl.float64)
        tmp9 = tl.where(tmp7, tmp8, tmp2)
        tmp10 = tmp4 - tmp9
        tmp11 = libdevice.exp(tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp15 = libdevice.log(tmp13)
    tmp16 = tl_math.abs(tmp2)
    tmp17 = tl.full([1, 1], float("inf"), tl.float64)
    tmp18 = tmp16 == tmp17
    tmp19 = tl.full([1, 1], 0.0, tl.float64)
    tmp20 = tl.where(tmp18, tmp19, tmp2)
    tmp21 = tmp15 + tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/64/c64yyx6dzz6cxijlhrtt5mnvoro2ruavankvpjf3jmpyvetcqt22.py
# Topologically Sorted Source Nodes: [maximum], Original ATen: [aten.amax]
# Source node to ATen node mapping:
#   maximum => amax_6
# Graph fragment:
#   %amax_6 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%select_11, [2]), kwargs = {})
triton_red_fused_amax_27 = async_compile.triton('triton_red_fused_amax_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_amax_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_amax_27(in_ptr0, out_ptr0, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (3 + 4*r1 + 4*x0*((127 + ks0) // 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/5u/c5uge6nhmvzgnti6r6xchhxsdrhi3xipy66tg62blraqdmxetuli.py
# Topologically Sorted Source Nodes: [weights], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   weights => amax_4, clone, exp_3, sub_116, sum_14
# Graph fragment:
#   %clone : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%slice_22,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_4 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone, [2], True), kwargs = {})
#   %sub_116 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %amax_4), kwargs = {})
#   %exp_3 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_116,), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_3, [2], True), kwargs = {})
triton_red_fused__softmax_28 = async_compile.triton('triton_red_fused__softmax_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'out_ptr1': '*fp64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_28(in_ptr0, out_ptr0, out_ptr1, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 3)
    x1 = xindex // 3
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float64)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 4*r2 + 4*x1*((127 + ks0) // 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp4 = tl.load(in_ptr0 + (x0 + 4*r2 + 4*x1*((127 + ks0) // 128)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = libdevice.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/cu/ccunnpzyipasvedncqb45vgrdhdo6rvv5kiszzjht42bnjoys4cm.py
# Topologically Sorted Source Nodes: [mul_14, merged], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   merged => sum_15
#   mul_14 => mul_476
# Graph fragment:
#   %mul_476 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_31, %getitem_6), kwargs = {})
#   %sum_15 : [num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_476, [2]), kwargs = {})
triton_red_fused_mul_sum_29 = async_compile.triton('triton_red_fused_mul_sum_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'in_ptr3': '*fp64', 'out_ptr0': '*fp64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mul_sum_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = ((xindex // 16) % 3)
    x2 = xindex // 48
    x4 = xindex // 16
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    x5 = (xindex % 48)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float64)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + 4*r3 + 4*x2*((127 + ks0) // 128)), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x5 + 48*r3 + 48*x2*((127 + ks0) // 128)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = libdevice.exp(tmp2)
        tmp5 = tmp3 / tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x6), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ol/col73cdf6hmqy4qc34v7y5e2nio5ys4twpdqe74aoygru5lfxpni.py
# Topologically Sorted Source Nodes: [er], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   er => add_645
# Graph fragment:
#   %add_645 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_12, %select_13), kwargs = {})
triton_poi_fused_add_30 = async_compile.triton('triton_poi_fused_add_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (34*x0 + 1632*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 34*x0 + 1632*x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/q2/cq2wlhbkdjxezqlriozvbnezlmug5wccf7eaikk4djljdcnyjbyp.py
# Topologically Sorted Source Nodes: [E, scores], Original ATen: [aten.sub, streaming_sparse.action_reduce]
# Source node to ATen node mapping:
#   E => sub_120
#   scores => action_reduce
# Graph fragment:
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_8, %select_7), kwargs = {})
#   %action_reduce : [num_users=3] = call_function[target=torch.ops.streaming_sparse.action_reduce.default](args = (%where_23, %select_7, %sub_120, %select_12, %add_645, %select_13, %slice_50, %slice_54, %select_14, %add_646, %select_15, %slice_58, %slice_62, %add_625, %select_9, %add_648, %view_17, %add_649, %view_20, %amax_6), kwargs = {})
triton_poi_fused_action_reduce_sub_31 = async_compile.triton('triton_poi_fused_action_reduce_sub_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_action_reduce_sub_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_action_reduce_sub_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + x0 + 48*x1), None)
    tmp1 = tl.load(in_ptr0 + (x0 + 48*x1), None)
    tmp2 = tmp0 - tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/fc/cfct2gsfyxuayw2zibp5tnlvtbd5fdr52pfprb5miympzguubjll.py
# Topologically Sorted Source Nodes: [ea], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   ea => add_646
# Graph fragment:
#   %add_646 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_14, %select_15), kwargs = {})
triton_poi_fused_add_32 = async_compile.triton('triton_poi_fused_add_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (544 + 34*x0 + 1632*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (545 + 34*x0 + 1632*x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ci/cciz2pxsu5ux7gsbkn62iht5fhxf5q4mdn5ba3wl2yngpgs4zvvh.py
# Topologically Sorted Source Nodes: [add_6], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_6 => add_647
# Graph fragment:
#   %add_647 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_50, %slice_54), kwargs = {})
triton_poi_fused_add_33 = async_compile.triton('triton_poi_fused_add_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 16)
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + x0 + 34*x1 + 1632*x2), None)
    tmp1 = tl.load(in_ptr0 + (18 + x0 + 34*x1 + 1632*x2), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xz/cxz2pdprpu42uwbxd65v6cpvkt3nr5u2743q6ilp5lcomjblzthh.py
# Topologically Sorted Source Nodes: [eye, excluded, rx, softmax_2], Original ATen: [aten.eye, aten.masked_fill, aten.logsumexp, aten._softmax]
# Source node to ATen node mapping:
#   excluded => full_default_10, where_8
#   eye => eq_161, iota_4
#   rx => abs_5, add_648, amax_7, eq_162, exp_5, full_default_11, log_3, sub_130, sum_17, where_9
#   softmax_2 => amax_8, div_6, exp_6, sub_131, sum_18
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %eq_161 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_32, %iota_4), kwargs = {})
#   %full_default_10 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_8 : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%eq_161, %full_default_10, %expand_3), kwargs = {})
#   %amax_7 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%where_8, [-1], True), kwargs = {})
#   %abs_5 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax_7,), kwargs = {})
#   %eq_162 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_5, inf), kwargs = {})
#   %full_default_11 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_162, %full_default_11, %amax_7), kwargs = {})
#   %sub_130 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_8, %where_9), kwargs = {})
#   %exp_5 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_130,), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_5, [-1]), kwargs = {})
#   %log_3 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_17,), kwargs = {})
#   %add_648 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%log_3, %squeeze_2), kwargs = {})
#   %amax_8 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where_8, [-1], True), kwargs = {})
#   %sub_131 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_8, %amax_8), kwargs = {})
#   %exp_6 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_131,), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_6, [-1], True), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_6, %sum_18), kwargs = {})
triton_per_fused__softmax_eye_logsumexp_masked_fill_34 = async_compile.triton('triton_per_fused__softmax_eye_logsumexp_masked_fill_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_ptr0': '*fp64', 'out_ptr3': '*fp64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_eye_logsumexp_masked_fill_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_eye_logsumexp_masked_fill_34(in_out_ptr0, in_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    r2 = rindex
    x1 = xindex // 16
    x3 = xindex
    tmp3 = tl.load(in_ptr0 + (r2 + 16*x1), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = r2
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1, 1], float("-inf"), tl.float64)
    tmp5 = tl.where(tmp2, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp9 = tl_math.abs(tmp8)
    tmp10 = tl.full([1, 1], float("inf"), tl.float64)
    tmp11 = tmp9 == tmp10
    tmp12 = tl.full([1, 1], 0.0, tl.float64)
    tmp13 = tl.where(tmp11, tmp12, tmp8)
    tmp14 = tmp5 - tmp13
    tmp15 = libdevice.exp(tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tmp5 - tmp8
    tmp20 = libdevice.exp(tmp19)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tmp20 / tmp23
    tmp25 = libdevice.log(tmp18)
    tmp26 = tmp25 + tmp13
    tl.store(out_ptr3 + (r2 + 16*x3), tmp24, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/jg/cjger5ctfegx2k37mxbm6dhdy3pz2iqyltexq3tp76ekkjskbk72.py
# Topologically Sorted Source Nodes: [eye_2, excluded_1, ax, softmax_3], Original ATen: [aten.eye, aten.masked_fill, aten.logsumexp, aten._softmax]
# Source node to ATen node mapping:
#   ax => abs_6, add_649, amax_9, eq_164, exp_7, full_default_13, log_4, sub_132, sum_19, where_11
#   excluded_1 => full_default_12, where_10
#   eye_2 => eq_163, iota_6
#   softmax_3 => amax_10, div_7, exp_8, sub_133, sum_20
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %eq_163 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_34, %iota_6), kwargs = {})
#   %full_default_12 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_10 : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%eq_163, %full_default_12, %expand_6), kwargs = {})
#   %amax_9 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%where_10, [-1], True), kwargs = {})
#   %abs_6 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax_9,), kwargs = {})
#   %eq_164 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_6, inf), kwargs = {})
#   %full_default_13 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_11 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_164, %full_default_13, %amax_9), kwargs = {})
#   %sub_132 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_10, %where_11), kwargs = {})
#   %exp_7 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_132,), kwargs = {})
#   %sum_19 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_7, [-1]), kwargs = {})
#   %log_4 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_19,), kwargs = {})
#   %add_649 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%log_4, %squeeze_3), kwargs = {})
#   %amax_10 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where_10, [-1], True), kwargs = {})
#   %sub_133 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_10, %amax_10), kwargs = {})
#   %exp_8 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_133,), kwargs = {})
#   %sum_20 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_8, [-1], True), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_8, %sum_20), kwargs = {})
triton_per_fused__softmax_eye_logsumexp_masked_fill_35 = async_compile.triton('triton_per_fused__softmax_eye_logsumexp_masked_fill_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_ptr0': '*fp64', 'out_ptr3': '*fp64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_eye_logsumexp_masked_fill_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_eye_logsumexp_masked_fill_35(in_out_ptr0, in_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    r2 = rindex
    x1 = xindex // 32
    x3 = xindex
    tmp3 = tl.load(in_ptr0 + (544 + 34*r2 + 1632*x1), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = r2
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1, 1], float("-inf"), tl.float64)
    tmp5 = tl.where(tmp2, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp9 = tl_math.abs(tmp8)
    tmp10 = tl.full([1, 1], float("inf"), tl.float64)
    tmp11 = tmp9 == tmp10
    tmp12 = tl.full([1, 1], 0.0, tl.float64)
    tmp13 = tl.where(tmp11, tmp12, tmp8)
    tmp14 = tmp5 - tmp13
    tmp15 = libdevice.exp(tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tmp5 - tmp8
    tmp20 = libdevice.exp(tmp19)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tmp20 / tmp23
    tmp25 = libdevice.log(tmp18)
    tmp26 = tmp25 + tmp13
    tl.store(out_ptr3 + (r2 + 32*x3), tmp24, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/jz/cjzoppeqcc54kycebmhik2hevtyjsfdlyux23q22d5fkr5ziw3wv.py
# Topologically Sorted Source Nodes: [den_1, den_2, den_3], Original ATen: [aten.logaddexp]
# Source node to ATen node mapping:
#   den_1 => abs_8, add_651, eq_167, eq_168, exp_10, ge_2, log1p_1, logical_and_1, logical_not_1, mul_514, ne_16, sub_135, where_15, where_16, where_17
#   den_2 => abs_9, add_652, eq_169, eq_170, exp_11, ge_3, log1p_2, logical_and_2, logical_not_2, mul_515, ne_17, sub_136, where_18, where_19, where_20
#   den_3 => abs_10, add_653, eq_171, eq_172, exp_12, ge_4, log1p_3, logical_and_3, logical_not_3, mul_516, ne_18, sub_137, where_21, where_22, where_23
# Graph fragment:
#   %eq_167 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%expand_10, %expand_10), kwargs = {})
#   %abs_8 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%expand_10,), kwargs = {})
#   %ne_16 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_8, inf), kwargs = {})
#   %mul_514 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_167, %ne_16), kwargs = {})
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%mul_514,), kwargs = {})
#   %eq_168 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%expand_10, %expand_11), kwargs = {})
#   %logical_and_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%logical_not_1, %eq_168), kwargs = {})
#   %ge_2 : [num_users=2] = call_function[target=torch.ops.aten.ge.Tensor](args = (%expand_10, %expand_11), kwargs = {})
#   %where_16 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_2, %expand_11, %expand_10), kwargs = {})
#   %where_15 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%ge_2, %expand_10, %expand_11), kwargs = {})
#   %sub_135 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_16, %where_15), kwargs = {})
#   %exp_10 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_135,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_10,), kwargs = {})
#   %add_651 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_15, %log1p_1), kwargs = {})
#   %where_17 : [num_users=7] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_1, %expand_10, %add_651), kwargs = {})
#   %eq_169 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%where_17, %where_17), kwargs = {})
#   %abs_9 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%where_17,), kwargs = {})
#   %ne_17 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_9, inf), kwargs = {})
#   %mul_515 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_169, %ne_17), kwargs = {})
#   %logical_not_2 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%mul_515,), kwargs = {})
#   %eq_170 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%where_17, %expand_12), kwargs = {})
#   %logical_and_2 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%logical_not_2, %eq_170), kwargs = {})
#   %ge_3 : [num_users=2] = call_function[target=torch.ops.aten.ge.Tensor](args = (%where_17, %expand_12), kwargs = {})
#   %where_19 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_3, %expand_12, %where_17), kwargs = {})
#   %where_18 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%ge_3, %where_17, %expand_12), kwargs = {})
#   %sub_136 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_19, %where_18), kwargs = {})
#   %exp_11 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_136,), kwargs = {})
#   %log1p_2 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_11,), kwargs = {})
#   %add_652 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_18, %log1p_2), kwargs = {})
#   %where_20 : [num_users=7] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_2, %where_17, %add_652), kwargs = {})
#   %eq_171 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%where_20, %where_20), kwargs = {})
#   %abs_10 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%where_20,), kwargs = {})
#   %ne_18 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_10, inf), kwargs = {})
#   %mul_516 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_171, %ne_18), kwargs = {})
#   %logical_not_3 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%mul_516,), kwargs = {})
#   %eq_172 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%where_20, %expand_13), kwargs = {})
#   %logical_and_3 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%logical_not_3, %eq_172), kwargs = {})
#   %ge_4 : [num_users=2] = call_function[target=torch.ops.aten.ge.Tensor](args = (%where_20, %expand_13), kwargs = {})
#   %where_22 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_4, %expand_13, %where_20), kwargs = {})
#   %where_21 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%ge_4, %where_20, %expand_13), kwargs = {})
#   %sub_137 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_22, %where_21), kwargs = {})
#   %exp_12 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_137,), kwargs = {})
#   %log1p_3 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_12,), kwargs = {})
#   %add_653 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_21, %log1p_3), kwargs = {})
#   %where_23 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_3, %where_20, %add_653), kwargs = {})
triton_poi_fused_logaddexp_36 = async_compile.triton('triton_poi_fused_logaddexp_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'in_ptr3': '*fp64', 'in_ptr4': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_logaddexp_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_logaddexp_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 512
    x3 = xindex // 32
    x0 = (xindex % 32)
    x4 = xindex
    x1 = ((xindex // 32) % 16)
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (x0 + 32*x2), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr3 + (34*x1 + 1632*x2), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr4 + (x0 + 32*x2), None, eviction_policy='evict_last')
    tmp1 = tmp0 == tmp0
    tmp2 = tl_math.abs(tmp0)
    tmp3 = tl.full([1], float("inf"), tl.float64)
    tmp4 = tmp2 != tmp3
    tmp5 = tmp1 & tmp4
    tmp6 = tmp5 == 0
    tmp8 = tmp0 == tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tmp0 >= tmp7
    tmp11 = tl.where(tmp10, tmp0, tmp7)
    tmp12 = tl.where(tmp10, tmp7, tmp0)
    tmp13 = tmp12 - tmp11
    tmp14 = libdevice.exp(tmp13)
    tmp15 = libdevice.log1p(tmp14)
    tmp16 = tmp11 + tmp15
    tmp17 = tl.where(tmp9, tmp0, tmp16)
    tmp18 = tmp17 == tmp17
    tmp19 = tl_math.abs(tmp17)
    tmp20 = tmp19 != tmp3
    tmp21 = tmp18 & tmp20
    tmp22 = tmp21 == 0
    tmp24 = tmp17 == tmp23
    tmp25 = tmp22 & tmp24
    tmp26 = tmp17 >= tmp23
    tmp27 = tl.where(tmp26, tmp17, tmp23)
    tmp28 = tl.where(tmp26, tmp23, tmp17)
    tmp29 = tmp28 - tmp27
    tmp30 = libdevice.exp(tmp29)
    tmp31 = libdevice.log1p(tmp30)
    tmp32 = tmp27 + tmp31
    tmp33 = tl.where(tmp25, tmp17, tmp32)
    tmp34 = tmp33 == tmp33
    tmp35 = tl_math.abs(tmp33)
    tmp36 = tmp35 != tmp3
    tmp37 = tmp34 & tmp36
    tmp38 = tmp37 == 0
    tmp40 = tmp33 == tmp39
    tmp41 = tmp38 & tmp40
    tmp42 = tmp33 >= tmp39
    tmp43 = tl.where(tmp42, tmp33, tmp39)
    tmp44 = tl.where(tmp42, tmp39, tmp33)
    tmp45 = tmp44 - tmp43
    tmp46 = libdevice.exp(tmp45)
    tmp47 = libdevice.log1p(tmp46)
    tmp48 = tmp43 + tmp47
    tmp49 = tl.where(tmp41, tmp33, tmp48)
    tmp50 = tmp49 == tmp49
    tmp51 = tl_math.abs(tmp49)
    tmp52 = tmp51 != tmp3
    tmp53 = tmp50 & tmp52
    tmp54 = tmp53 == 0
    tmp56 = tmp49 == tmp55
    tmp57 = tmp54 & tmp56
    tmp58 = tmp49 >= tmp55
    tmp59 = tl.where(tmp58, tmp49, tmp55)
    tmp60 = tl.where(tmp58, tmp55, tmp49)
    tmp61 = tmp60 - tmp59
    tmp62 = libdevice.exp(tmp61)
    tmp63 = libdevice.log1p(tmp62)
    tmp64 = tmp59 + tmp63
    tmp65 = tl.where(tmp57, tmp49, tmp64)
    tl.store(in_out_ptr0 + (x4), tmp65, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/nk/cnko6qli6c5qpuqz3wthd2glngshpjvoq7nxgafzlipazynltw3d.py
# Topologically Sorted Source Nodes: [max_1, rr, gt, isfinite, all_1, execute, invert_4, aa], Original ATen: [aten.max, aten.gather, aten.gt, aten.eq, aten.abs, aten.ne, aten.mul, aten.all, aten.bitwise_and, aten.bitwise_not]
# Source node to ATen node mapping:
#   aa => gather_1
#   all_1 => any_1, logical_not_4, logical_not_5
#   execute => bitwise_and
#   gt => gt_2
#   invert_4 => bitwise_not_4
#   isfinite => abs_11, eq_174, mul_524, ne_23
#   max_1 => max_1
#   rr => gather
# Graph fragment:
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%view_24, -1), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view_22, 1, %unsqueeze_42), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%getitem_9, 0), kwargs = {})
#   %eq_174 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%view_24, %view_24), kwargs = {})
#   %abs_11 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view_24,), kwargs = {})
#   %ne_23 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_11, inf), kwargs = {})
#   %mul_524 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_174, %ne_23), kwargs = {})
#   %logical_not_4 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%mul_524,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_4, -1), kwargs = {})
#   %logical_not_5 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %bitwise_and : [num_users=3] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%gt_2, %logical_not_5), kwargs = {})
#   %bitwise_not_4 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze_44,), kwargs = {})
#   %gather_1 : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view_23, 1, %unsqueeze_43), kwargs = {})
triton_per_fused_abs_all_bitwise_and_bitwise_not_eq_gather_gt_max_mul_ne_37 = async_compile.triton('triton_per_fused_abs_all_bitwise_and_bitwise_not_eq_gather_gt_max_mul_ne_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'out_ptr3': '*i64', 'out_ptr4': '*i64', 'out_ptr5': '*i1', 'out_ptr6': '*i1', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_all_bitwise_and_bitwise_not_eq_gather_gt_max_mul_ne_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_all_bitwise_and_bitwise_not_eq_gather_gt_max_mul_ne_37(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, ks0, xnumel, rnumel):
    xnumel = 288
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 512*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp1, 0))
    tmp5 = tl.broadcast_to(rindex, tmp1.shape)
    tmp4_val, tmp4_idx = triton_helpers.max_with_index(tmp1, tmp5, 0)
    tmp4 = triton_helpers.promote_to_tensor(tmp4_idx)
    tmp6 = tmp0 == tmp0
    tmp7 = tl_math.abs(tmp0)
    tmp8 = tl.full([1], float("inf"), tl.float64)
    tmp9 = tmp7 != tmp8
    tmp10 = tmp6 & tmp9
    tmp11 = tmp10 == 0
    tmp12 = tmp11.to(tl.int64)
    tmp13 = (tmp12 != 0)
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp14, 0))
    tmp17 = tl.full([1], 32, tl.int64)
    tmp18 = tl.where((tmp4 < 0) != (tmp17 < 0), tl.where(tmp4 % tmp17 != 0, tmp4 // tmp17 - 1, tmp4 // tmp17), tmp4 // tmp17)
    tmp19 = tl.full([1], 16, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 16), "index out of bounds: 0 <= tmp22 < 16")
    tmp24 = tl.load(in_ptr1 + (tmp22 + ks0*x0), None, eviction_policy='evict_last')
    tmp25 = tmp4 % tmp17
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = tmp25 != tmp26
    tmp28 = (libdevice.signbit(tmp25) != 0) if (tmp25).dtype is tl.float32 else tmp25 < 0
    tmp29 = (libdevice.signbit(tmp17) != 0) if (tmp17).dtype is tl.float32 else tmp17 < 0
    tmp30 = tmp28 != tmp29
    tmp31 = tmp27 & tmp30
    tmp32 = tmp25 + tmp17
    tmp33 = tl.where(tmp31, tmp32, tmp25)
    tmp34 = tl.full([1], 32, tl.int32)
    tmp35 = tmp33 + tmp34
    tmp36 = tmp33 < 0
    tmp37 = tl.where(tmp36, tmp35, tmp33)
    tl.device_assert((0 <= tmp37) & (tmp37 < 32), "index out of bounds: 0 <= tmp37 < 32")
    tmp39 = tl.load(in_ptr2 + (tmp37 + ks0*x0), None, eviction_policy='evict_last')
    tmp40 = tl.full([1], 0.0, tl.float64)
    tmp41 = tmp3 > tmp40
    tmp42 = tmp16 == 0
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43 == 0
    tl.store(out_ptr3 + (x0), tmp24, None)
    tl.store(out_ptr4 + (x0), tmp39, None)
    tl.store(out_ptr5 + (x0), tmp44, None)
    tl.store(out_ptr6 + (x0), tmp43, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/pn/cpnyj66czpsz6t25pnep6ob67gpnjo4zcm6ygvpwmoel6r2vx3hj.py
# Topologically Sorted Source Nodes: [isfinite_1, invert_5, sum_12], Original ATen: [aten.eq, aten.abs, aten.ne, aten.mul, aten.bitwise_not, aten.sum]
# Source node to ATen node mapping:
#   invert_5 => bitwise_not_5
#   isfinite_1 => abs_12, eq_180, mul_531, ne_25
#   sum_12 => sum_22
# Graph fragment:
#   %eq_180 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%action_reduce, %action_reduce), kwargs = {})
#   %abs_12 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%action_reduce,), kwargs = {})
#   %ne_25 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_12, inf), kwargs = {})
#   %mul_531 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_180, %ne_25), kwargs = {})
#   %bitwise_not_5 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%mul_531,), kwargs = {})
#   %sum_22 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%bitwise_not_5,), kwargs = {})
triton_red_fused_abs_bitwise_not_eq_mul_ne_sum_38 = async_compile.triton('triton_red_fused_abs_bitwise_not_eq_mul_ne_sum_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_bitwise_not_eq_mul_ne_sum_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_bitwise_not_eq_mul_ne_sum_38(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0 == tmp0
        tmp2 = tl_math.abs(tmp0)
        tmp3 = tl.full([1, 1], float("inf"), tl.float64)
        tmp4 = tmp2 != tmp3
        tmp5 = tmp1 & tmp4
        tmp6 = tmp5 == 0
        tmp7 = tmp6.to(tl.int64)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tu/ctuysfidtufqx7hon7zbgeqxrelr32cosdly6uxaqq6psmjef2vo.py
# Topologically Sorted Source Nodes: [isfinite_1, invert_5, sum_12, stats], Original ATen: [aten.eq, aten.abs, aten.ne, aten.mul, aten.bitwise_not, aten.sum, aten.stack]
# Source node to ATen node mapping:
#   invert_5 => bitwise_not_5
#   isfinite_1 => abs_12, eq_180, mul_531, ne_25
#   stats => cat_1
#   sum_12 => sum_22
# Graph fragment:
#   %eq_180 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%action_reduce, %action_reduce), kwargs = {})
#   %abs_12 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%action_reduce,), kwargs = {})
#   %ne_25 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_12, inf), kwargs = {})
#   %mul_531 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_180, %ne_25), kwargs = {})
#   %bitwise_not_5 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%mul_531,), kwargs = {})
#   %sum_22 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%bitwise_not_5,), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_46, %full_default_14, %unsqueeze_48],), kwargs = {})
triton_per_fused_abs_bitwise_not_eq_mul_ne_stack_sum_39 = async_compile.triton('triton_per_fused_abs_bitwise_not_eq_mul_ne_stack_sum_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr1': '*i64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_bitwise_not_eq_mul_ne_stack_sum_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_bitwise_not_eq_mul_ne_stack_sum_39(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 18
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/zg/czg4hl5dxqsnowddph2iuedslpqzx2rfbbixlplmwtv3pjrxson7.py
# Topologically Sorted Source Nodes: [sum_11, stats], Original ATen: [aten.sum, aten.stack]
# Source node to ATen node mapping:
#   stats => cat_1
#   sum_11 => sum_21
# Graph fragment:
#   %sum_21 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%bitwise_and,), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_46, %full_default_14, %unsqueeze_48],), kwargs = {})
triton_per_fused_stack_sum_40 = async_compile.triton('triton_per_fused_stack_sum_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr1': '*i64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_stack_sum_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_stack_sum_40(in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0).to(tl.int1)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/dc/cdcsjoymxyss5zd74hp7owunibyaljqwhji6f3emep5zhmzp3npe.py
# Topologically Sorted Source Nodes: [invert_6, masked_fill_7, scatter_7], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.scatter]
# Source node to ATen node mapping:
#   invert_6 => bitwise_not_6
#   masked_fill_7 => full_default_15, where_24
#   scatter_7 => scatter_7
# Graph fragment:
#   %bitwise_not_6 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%view_25,), kwargs = {})
#   %full_default_15 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_24 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not_6, %full_default_15, %mean), kwargs = {})
#   %scatter_7 : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%where_24, -1, %unsqueeze_49, inf), kwargs = {})
triton_poi_fused_bitwise_not_masked_fill_scatter_41 = async_compile.triton('triton_poi_fused_bitwise_not_masked_fill_scatter_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_not_masked_fill_scatter_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_not_masked_fill_scatter_41(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.int1)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tmp0 == 0
    tmp3 = float("-inf")
    tmp4 = tl.where(tmp1, tmp3, tmp2)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/kb/ckb3iiv2xm3urczfmouutfpmhqkzrlyjvhnykkp76g3pul3g74di.py
# Topologically Sorted Source Nodes: [ids_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   ids_1 => convert_element_type_7
# Graph fragment:
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_12, torch.int32), kwargs = {})
triton_poi_fused__to_copy_42 = async_compile.triton('triton_poi_fused__to_copy_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1 = args
    args.clear()
    s0 = arg1_1
    s3 = arg9_1
    s4 = arg11_1
    s5 = arg13_1
    s6 = arg15_1
    assert_size_stride(arg0_1, (12, 24, 64, 128), (128, 98304, 1536, 1))
    assert_size_stride(arg2_1, (12, 24), (24, 1))
    assert_size_stride(arg3_1, (12, 4), (4, 1))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (12, s0, 4, 128), (512*s0, 512, 128, 1))
    assert_size_stride(arg7_1, (12, s0, 4, 12), (48*s0, 48, 12, 1))
    assert_size_stride(arg8_1, (8, 128), (128, 1))
    assert_size_stride(arg10_1, (12, s3, 3, 16), (48*s3, 48, 16, 1))
    assert_size_stride(arg12_1, (12, s4, 16), (16*s4, 16, 1))
    assert_size_stride(arg14_1, (12, s5, 8, 4), (32*s5, 32, 4, 1))
    assert_size_stride(arg16_1, (12, s6, 5, 17), (85*s6, 85, 17, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf108 = empty_strided_cuda((3, ), (1, ), torch.int64)
        buf106 = reinterpret_tensor(buf108, (1, ), (1, ), 1)  # alias
        # Topologically Sorted Source Nodes: [stats], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_0.run(buf106, 1, grid=grid(1), stream=stream0)
        buf27 = empty_strided_cuda((12, 24), (24, 1), torch.int64)
        # Topologically Sorted Source Nodes: [valid_1, sum_8], Original ATen: [aten.expand, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_expand_sum_1.run(buf27, 288, s0, grid=grid(288), stream=stream0)
        buf29 = empty_strided_cuda((12, 24, 128), (3072, 128, 1), torch.bool)
        # Topologically Sorted Source Nodes: [arange, active], Original ATen: [aten.arange, aten.lt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_lt_2.run(buf27, buf29, s0, 36864, grid=grid(36864), stream=stream0)
        buf28 = empty_strided_cuda((12, 24, s0), (24*s0, s0, 1), torch.bool)
        # Topologically Sorted Source Nodes: [zeros_like_1, arange, active, mask], Original ATen: [aten.zeros_like, aten.arange, aten.lt, aten.scatter]
        triton_poi_fused_arange_lt_scatter_zeros_like_3_xnumel = 288*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_lt_scatter_zeros_like_3.run(buf28, triton_poi_fused_arange_lt_scatter_zeros_like_3_xnumel, grid=grid(triton_poi_fused_arange_lt_scatter_zeros_like_3_xnumel), stream=stream0)
        buf9 = empty_strided_cuda((12, 24, 16, 8), (128, 1536, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_4.run(arg4_1, arg0_1, buf9, 36864, 16, grid=grid(36864), stream=stream0)
        buf10 = empty_strided_cuda((12, 24, 16, 8), (3072, 128, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q2], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(buf9, buf10, 36864, grid=grid(36864), stream=stream0)
        buf48 = empty_strided_cuda((288, s0), (s0, 1), torch.int32)
        # Topologically Sorted Source Nodes: [mapping, mapping_1], Original ATen: [aten.full_like, aten.scatter]
        triton_poi_fused_full_like_scatter_6_xnumel = 288*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_like_scatter_6.run(buf48, triton_poi_fused_full_like_scatter_6_xnumel, grid=grid(triton_poi_fused_full_like_scatter_6_xnumel), stream=stream0)
        buf11 = empty_strided_cuda((12, s0, 4, 12), (48*s0, 48, 12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [st], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7_xnumel = 576*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(arg7_1, buf11, triton_poi_fused__to_copy_7_xnumel, grid=grid(triton_poi_fused__to_copy_7_xnumel), stream=stream0)
        buf14 = empty_strided_cuda((12, 1, 3), (3, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [amax], Original ATen: [aten.amax]
        triton_red_fused_amax_8_rnumel = (2 + s0) // 3
        stream0 = get_raw_stream(0)
        triton_red_fused_amax_8.run(buf11, buf14, s0, 36, triton_red_fused_amax_8_rnumel, grid=grid(36), stream=stream0)
        buf15 = empty_strided_cuda((12, 1), (1, 12), torch.float32)
        # Topologically Sorted Source Nodes: [amax], Original ATen: [aten.amax]
        stream0 = get_raw_stream(0)
        triton_per_fused_amax_9.run(buf14, buf15, 12, 3, grid=grid(12), stream=stream0)
        del buf14
        buf0 = empty_strided_cuda((12, 384, 128), (49152, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tensor_bmm_default], Original ATen: [streaming_sparse.tensor_bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_tensor_bmm_10.run(arg4_1, arg0_1, buf0, 589824, grid=grid(589824), stream=stream0)
        buf16 = empty_strided_cuda((12, 24, 16), (16, 192, 1), torch.float32)
        buf41 = empty_strided_cuda((12, 24, 16, 128), (49152, 2048, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [probes, norm], Original ATen: [aten.index_select, aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_select_linalg_vector_norm_11.run(arg4_1, arg0_1, buf16, buf41, 4608, 128, grid=grid(4608), stream=stream0)
        del arg0_1
        del arg4_1
        buf42 = reinterpret_tensor(buf9, (4608, 8), (8, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (4608, 128), (128, 1), 0), reinterpret_tensor(arg8_1, (128, 8), (1, 128), 0), out=buf42)
        del arg8_1
        del buf41
        buf1 = empty_strided_cuda((12, 128, 4*s0), (512*s0, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [tensor_bmm_default], Original ATen: [streaming_sparse.tensor_bmm]
        triton_poi_fused_tensor_bmm_12_xnumel = 6144*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_tensor_bmm_12.run(arg6_1, buf1, triton_poi_fused_tensor_bmm_12_xnumel, grid=grid(triton_poi_fused_tensor_bmm_12_xnumel), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [tensor_bmm_default], Original ATen: [streaming_sparse.tensor_bmm]
        buf2 = torch.ops.streaming_sparse.tensor_bmm.default(buf0, buf1)
        del buf0
        del buf1
        buf3 = buf2
        del buf2
        buf12 = empty_strided_cuda((12, 384, 4*s0), (1536*s0, 4*s0, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (12, 384, 8), (3072, 8, 1), 0), reinterpret_tensor(buf11, (12, 8, 4*s0), (48*s0, 1, 12), 0), out=buf12)
        del buf10
        buf4 = reinterpret_tensor(buf3, (12, 24, 16, s0, 4), (1536*s0, 64*s0, 4*s0, 4, 1), 0); del buf3  # reuse
        buf5 = empty_strided_cuda((12, 24, 16, 1), (384, 16, 1, 4608), torch.float32)
        buf6 = empty_strided_cuda((12, 24, 16, 1), (384, 16, 1, 4608), torch.float32)
        # Topologically Sorted Source Nodes: [z, softmax], Original ATen: [aten.div, aten._softmax]
        triton_red_fused__softmax_div_13_rnumel = 4*s0
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_div_13.run(buf4, buf5, buf6, s0, 4608, triton_red_fused__softmax_div_13_rnumel, grid=grid(4608), stream=stream0)
        buf43 = empty_strided_cuda((12, 24, 16, 1, 3), (1152, 48, 3, 13824, 1), torch.float64)
        # Topologically Sorted Source Nodes: [zz, lm, logsumexp_1], Original ATen: [aten._to_copy, aten.logsumexp]
        triton_red_fused__to_copy_logsumexp_14_rnumel = (2 + s0) // 3
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_logsumexp_14.run(buf4, buf43, s0, 13824, triton_red_fused__to_copy_logsumexp_14_rnumel, grid=grid(13824), stream=stream0)
        buf44 = empty_strided_cuda((12, 24, 16, 1), (384, 16, 1, 4608), torch.float64)
        # Topologically Sorted Source Nodes: [zz, lm, logsumexp_1], Original ATen: [aten._to_copy, aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_logsumexp_15.run(buf43, buf44, 4608, 3, grid=grid(4608), stream=stream0)
        buf7 = empty_strided_cuda((12, 24, 16, s0), (384*s0, 16*s0, s0, 1), torch.float32)
        buf8 = empty_strided_cuda((12, 24, 16, s0), (384*s0, 16*s0, s0, 1), torch.float32)
        buf17 = empty_strided_cuda((12, 24, 16, s0), (384*s0, 16*s0, s0, 1), torch.float32)
        buf13 = empty_strided_cuda((12, 24, 16, s0), (384*s0, 16*s0, s0, 1), torch.float32)
        buf18 = empty_strided_cuda((12, 24, 16, s0), (384*s0, 16*s0, s0, 1), torch.float32)
        buf47 = empty_strided_cuda((12, 24, 16, s0), (384*s0, 16*s0, s0, 1), torch.float64)
        # Topologically Sorted Source Nodes: [alpha, truediv_3, sub, abs_1, mul_2, sum_3, u, sqrt, mul_3, mul_4, x1, beta, mul_5, sum_5, square_1, add_2, mul_7, sum_6, zz, lm, logsumexp_1, lm_1], Original ATen: [aten.sum, aten.div, aten.sub, aten.abs, aten.mul, aten.clamp_min, aten.sqrt, aten.pow, aten.add, aten._to_copy, aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_abs_add_clamp_min_div_logsumexp_mul_pow_sqrt_sub_sum_16.run(buf4, buf5, buf6, arg7_1, buf16, buf11, buf12, buf44, buf7, buf8, buf17, buf13, buf18, buf47, s0, 4608, s0, grid=grid(4608), stream=stream0)
        del buf16
        del buf5
        del buf6
        ps0 = 1536*s0
        buf19 = reinterpret_tensor(buf12, (12, 24, 16, s0, 4), (1536*s0, 64*s0, 4*s0, 4, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [f], Original ATen: [aten.stack]
        triton_poi_fused_stack_17_xnumel = 18432*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_17.run(buf7, arg7_1, buf8, buf13, buf11, buf15, buf17, buf18, buf19, s0, ps0, triton_poi_fused_stack_17_xnumel, grid=grid(triton_poi_fused_stack_17_xnumel), stream=stream0)
        del arg7_1
        del buf11
        del buf13
        del buf15
        del buf17
        del buf18
        del buf7
        del buf8
        ps1 = 24*s0
        buf20 = empty_strided_cuda((12, 24, s0), (24*s0, s0, 1), torch.float32)
        buf21 = buf20; del buf20  # reuse
        buf22 = empty_strided_cuda((12, 24, s0), (24*s0, s0, 1), torch.float32)
        # Topologically Sorted Source Nodes: [invert, masked_fill, mul_9, sum_7, risk, ranked], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.mul, aten.sum, aten.mean, aten.scatter]
        triton_per_fused_bitwise_not_masked_fill_mean_mul_scatter_sum_18_xnumel = 288*s0
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_not_masked_fill_mean_mul_scatter_sum_18.run(buf21, buf19, arg3_1, buf22, s0, ps1, triton_per_fused_bitwise_not_masked_fill_mean_mul_scatter_sum_18_xnumel, 16, grid=grid(triton_per_fused_bitwise_not_masked_fill_mean_mul_scatter_sum_18_xnumel), stream=stream0)
        del arg3_1
        del buf19
        # Topologically Sorted Source Nodes: [invert, masked_fill, ranked], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.scatter]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_not_masked_fill_scatter_19.run(arg2_1, buf22, s0, 288, grid=grid(288), stream=stream0)
        # Topologically Sorted Source Nodes: [topk], Original ATen: [aten.topk]
        buf24 = torch.ops.aten.topk.default(buf22, 128)
        buf26 = buf24[1]
        del buf24
        aten.scatter_.src(buf28,-1,buf26,buf29)
        del buf26
        del buf29
        buf31 = empty_strided_cuda((12, 24, s0), (24*s0, s0, 1), torch.bool)
        # Topologically Sorted Source Nodes: [removable], Original ATen: [aten.scatter]
        triton_poi_fused_scatter_20_xnumel = 288*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_scatter_20.run(buf28, buf31, triton_poi_fused_scatter_20_xnumel, grid=grid(triton_poi_fused_scatter_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [removable], Original ATen: [aten.scatter]
        stream0 = get_raw_stream(0)
        triton_poi_fused_scatter_21.run(arg2_1, buf31, s0, 288, grid=grid(288), stream=stream0)
        buf33 = buf22; del buf22  # reuse
        buf37 = empty_strided_cuda((12, 24, s0), (24*s0, s0, 1), torch.float32)
        buf88 = empty_strided_cuda((288, s0), (s0, 1), torch.bool)
        # Topologically Sorted Source Nodes: [invert_3, masked_fill_3, masked_fill_4, rr, invert_4, result], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.gather, aten.scatter]
        triton_poi_fused_bitwise_not_gather_masked_fill_scatter_22_xnumel = 288*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_not_gather_masked_fill_scatter_22.run(buf31, buf21, buf28, buf33, buf37, buf88, triton_poi_fused_bitwise_not_gather_masked_fill_scatter_22_xnumel, grid=grid(triton_poi_fused_bitwise_not_gather_masked_fill_scatter_22_xnumel), stream=stream0)
        del buf31
        # Topologically Sorted Source Nodes: [invert_3, masked_fill_3, argsort], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.sort]
        buf34 = torch.ops.aten.sort.stable(buf33, stable=True, dim=2, descending=False)
        del buf33
        buf36 = buf34[1]
        del buf34
        # Topologically Sorted Source Nodes: [mapping, mapping_1], Original ATen: [aten.full_like, aten.scatter]
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_like_scatter_23.run(buf36, buf48, s0, 4608, grid=grid(4608), stream=stream0)
        buf50 = empty_strided_cuda((288, s0), (s0, 1), torch.int32)
        # Topologically Sorted Source Nodes: [mapping_2], Original ATen: [aten.scatter]
        triton_poi_fused_scatter_24_xnumel = 288*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_scatter_24.run(buf48, buf50, triton_poi_fused_scatter_24_xnumel, grid=grid(triton_poi_fused_scatter_24_xnumel), stream=stream0)
        del buf48
        # Topologically Sorted Source Nodes: [masked_fill_4, argsort_1], Original ATen: [aten.masked_fill, aten.sort]
        buf38 = torch.ops.aten.sort.stable(buf37, stable=True, dim=2, descending=True)
        buf40 = buf38[1]
        del buf38
        # Topologically Sorted Source Nodes: [mapping_2], Original ATen: [aten.scatter]
        stream0 = get_raw_stream(0)
        triton_poi_fused_scatter_25.run(buf40, buf50, s0, 9216, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [reader_reduce_default], Original ATen: [streaming_sparse.reader_reduce]
        buf52 = torch.ops.streaming_sparse.reader_reduce.default(reinterpret_tensor(buf42, (12, 24, 16, 8), (3072, 128, 8, 1), 0), arg16_1, arg14_1, arg12_1, arg10_1, buf4, buf47, reinterpret_tensor(buf28, (288, s0), (s0, 1), 0), buf50, 48, 128, 4, True)
        del arg10_1
        del arg12_1
        del arg14_1
        del arg16_1
        del buf42
        del buf47
        del buf50
        buf53 = buf52[0]
        buf54 = buf52[1]
        buf55 = buf52[2]
        del buf52
        buf57 = reinterpret_tensor(buf44, (288, 16), (16, 1), 0); del buf44  # reuse
        buf58 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [lb], Original ATen: [aten.logsumexp]
        triton_red_fused_logsumexp_26_rnumel = (127 + s0) // 128
        stream0 = get_raw_stream(0)
        triton_red_fused_logsumexp_26.run(buf58, buf54, s0, 4608, triton_red_fused_logsumexp_26_rnumel, grid=grid(4608), stream=stream0)
        buf81 = empty_strided_cuda((288, 16), (16, 1), torch.float64)
        # Topologically Sorted Source Nodes: [maximum], Original ATen: [aten.amax]
        triton_red_fused_amax_27_rnumel = (127 + s0) // 128
        stream0 = get_raw_stream(0)
        triton_red_fused_amax_27.run(buf54, buf81, s0, 4608, triton_red_fused_amax_27_rnumel, grid=grid(4608), stream=stream0)
        buf69 = reinterpret_tensor(buf43, (288, 16, 1, 3), (48, 3, 13824, 1), 0); del buf43  # reuse
        buf70 = empty_strided_cuda((288, 16, 1, 3), (48, 3, 13824, 1), torch.float64)
        # Topologically Sorted Source Nodes: [weights], Original ATen: [aten._softmax]
        triton_red_fused__softmax_28_rnumel = (127 + s0) // 128
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_28.run(buf54, buf69, buf70, s0, 13824, triton_red_fused__softmax_28_rnumel, grid=grid(13824), stream=stream0)
        buf71 = empty_strided_cuda((288, 16, 3, 16), (768, 48, 16, 1), torch.float64)
        # Topologically Sorted Source Nodes: [mul_14, merged], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29_rnumel = (127 + s0) // 128
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_29.run(buf54, buf69, buf70, buf53, buf71, s0, 221184, triton_red_fused_mul_sum_29_rnumel, grid=grid(221184), stream=stream0)
        del buf53
        del buf54
        del buf69
        del buf70
        buf59 = empty_strided_cuda((288, 16, 16), (256, 16, 1), torch.float64)
        # Topologically Sorted Source Nodes: [er], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_30.run(buf55, buf59, 73728, grid=grid(73728), stream=stream0)
        buf82 = empty_strided_cuda((288, 16, 16), (256, 16, 1), torch.float64)
        # Topologically Sorted Source Nodes: [E, scores], Original ATen: [aten.sub, streaming_sparse.action_reduce]
        stream0 = get_raw_stream(0)
        triton_poi_fused_action_reduce_sub_31.run(buf71, buf82, 73728, grid=grid(73728), stream=stream0)
        buf67 = empty_strided_cuda((288, 16, 32), (512, 32, 1), torch.float64)
        # Topologically Sorted Source Nodes: [ea], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_32.run(buf55, buf67, 147456, grid=grid(147456), stream=stream0)
        buf75 = empty_strided_cuda((288, 16, 16, 16), (4096, 256, 16, 1), torch.float64)
        # Topologically Sorted Source Nodes: [add_6], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_33.run(buf55, buf75, 1179648, grid=grid(1179648), stream=stream0)
        buf61 = empty_strided_cuda((288, 16, 16), (256, 16, 1), torch.float64)
        buf74 = empty_strided_cuda((288, 16, 16, 16), (4096, 256, 16, 1), torch.float64)
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [eye, excluded, rx, softmax_2], Original ATen: [aten.eye, aten.masked_fill, aten.logsumexp, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_eye_logsumexp_masked_fill_34.run(buf62, buf59, buf74, 73728, 16, grid=grid(73728), stream=stream0)
        buf76 = empty_strided_cuda((4608, 16, 16), (256, 16, 1), torch.float64)
        # Topologically Sorted Source Nodes: [ro], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf74, (4608, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf75, (4608, 16, 16), (256, 16, 1), 0), out=buf76)
        del buf74
        del buf75
        buf64 = empty_strided_cuda((288, 16, 32), (512, 32, 1), torch.float64)
        buf79 = empty_strided_cuda((288, 16, 32, 32), (16384, 1024, 32, 1), torch.float64)
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [eye_2, excluded_1, ax, softmax_3], Original ATen: [aten.eye, aten.masked_fill, aten.logsumexp, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_eye_logsumexp_masked_fill_35.run(buf65, buf55, buf79, 147456, 32, grid=grid(147456), stream=stream0)
        buf80 = empty_strided_cuda((4608, 32, 16), (512, 16, 1), torch.float64)
        # Topologically Sorted Source Nodes: [ao], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (4608, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf55, (4608, 32, 16), (1632, 34, 1), 546), out=buf80)
        del buf79
        buf66 = empty_strided_cuda((288, 16, 16, 32), (8192, 512, 32, 1), torch.float64)
        buf68 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [den_1, den_2, den_3], Original ATen: [aten.logaddexp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_logaddexp_36.run(buf68, buf58, buf62, buf65, buf55, buf67, 2359296, grid=grid(2359296), stream=stream0)
        # Topologically Sorted Source Nodes: [E, scores], Original ATen: [aten.sub, streaming_sparse.action_reduce]
        buf83 = torch.ops.streaming_sparse.action_reduce.default(buf68, reinterpret_tensor(buf71, (288, 16, 16), (768, 48, 1), 0), buf82, reinterpret_tensor(buf55, (288, 16, 16), (26112, 1632, 34), 0), buf59, reinterpret_tensor(buf55, (288, 16, 16), (26112, 1632, 34), 1), reinterpret_tensor(buf55, (288, 16, 16, 16), (26112, 1632, 34, 1), 2), reinterpret_tensor(buf55, (288, 16, 16, 16), (26112, 1632, 34, 1), 18), reinterpret_tensor(buf55, (288, 16, 32), (26112, 1632, 34), 544), buf67, reinterpret_tensor(buf55, (288, 16, 32), (26112, 1632, 34), 545), reinterpret_tensor(buf55, (288, 16, 32, 16), (26112, 1632, 34, 1), 546), reinterpret_tensor(buf55, (288, 16, 32, 16), (26112, 1632, 34, 1), 562), buf58, reinterpret_tensor(buf71, (288, 16, 16), (768, 48, 1), 32), buf62, reinterpret_tensor(buf76, (288, 16, 16, 16), (4096, 256, 16, 1), 0), buf65, reinterpret_tensor(buf80, (288, 16, 32, 16), (8192, 512, 16, 1), 0), buf81)
        del buf55
        del buf58
        del buf59
        del buf62
        del buf65
        del buf67
        del buf68
        del buf71
        del buf76
        del buf80
        del buf81
        del buf82
        buf84 = buf83
        del buf83
        buf89 = reinterpret_tensor(buf27, (288, 1), (1, 288), 0); del buf27  # reuse
        buf93 = empty_strided_cuda((288, 1), (1, 288), torch.int64)
        buf90 = empty_strided_cuda((288, 1), (1, 288), torch.bool)
        buf94 = empty_strided_cuda((288, ), (1, ), torch.bool)
        # Topologically Sorted Source Nodes: [max_1, rr, gt, isfinite, all_1, execute, invert_4, aa], Original ATen: [aten.max, aten.gather, aten.gt, aten.eq, aten.abs, aten.ne, aten.mul, aten.all, aten.bitwise_and, aten.bitwise_not]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_all_bitwise_and_bitwise_not_eq_gather_gt_max_mul_ne_37.run(buf84, buf36, buf40, buf89, buf93, buf90, buf94, s0, 288, 512, grid=grid(288), stream=stream0)
        del buf36
        del buf40
        buf103 = empty_strided_cuda((18, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [isfinite_1, invert_5, sum_12], Original ATen: [aten.eq, aten.abs, aten.ne, aten.mul, aten.bitwise_not, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_bitwise_not_eq_mul_ne_sum_38.run(buf84, buf103, 18, 8192, grid=grid(18), stream=stream0)
        del buf84
        aten.scatter_.src(buf88,1,buf89,buf90)
        del buf89
        del buf90
        buf107 = reinterpret_tensor(buf108, (1, ), (1, ), 2)  # alias
        # Topologically Sorted Source Nodes: [isfinite_1, invert_5, sum_12, stats], Original ATen: [aten.eq, aten.abs, aten.ne, aten.mul, aten.bitwise_not, aten.sum, aten.stack]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_bitwise_not_eq_mul_ne_stack_sum_39.run(buf103, buf107, 1, 18, grid=grid(1), stream=stream0)
        del buf103
        buf105 = reinterpret_tensor(buf108, (1, ), (1, ), 0)  # alias
        # Topologically Sorted Source Nodes: [sum_11, stats], Original ATen: [aten.sum, aten.stack]
        stream0 = get_raw_stream(0)
        triton_per_fused_stack_sum_40.run(buf94, buf105, 1, 288, grid=grid(1), stream=stream0)
        buf92 = reinterpret_tensor(buf28, (288, s0), (s0, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [aa, result_1], Original ATen: [aten.gather, aten.scatter]
        triton_poi_fused_scatter_20_xnumel = 288*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_scatter_20.run(buf88, buf92, triton_poi_fused_scatter_20_xnumel, grid=grid(triton_poi_fused_scatter_20_xnumel), stream=stream0)
        del buf88
        aten.scatter_.src(buf92,1,buf93,reinterpret_tensor(buf94, (288, 1), (1, 0), 0))
        del buf93
        del buf94
        buf96 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [invert_6, masked_fill_7, scatter_7], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.scatter]
        triton_poi_fused_bitwise_not_masked_fill_scatter_41_xnumel = 288*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_not_masked_fill_scatter_41.run(buf92, buf21, buf96, triton_poi_fused_bitwise_not_masked_fill_scatter_41_xnumel, grid=grid(triton_poi_fused_bitwise_not_masked_fill_scatter_41_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [invert_6, masked_fill_7, scatter_7], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.scatter]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_not_masked_fill_scatter_19.run(arg2_1, buf96, s0, 288, grid=grid(288), stream=stream0)
        del arg2_1
        # Topologically Sorted Source Nodes: [topk_1], Original ATen: [aten.topk]
        buf98 = torch.ops.aten.topk.default(buf96, math.trunc(0.1*float(s0)))
        del buf96
        buf100 = buf98[1]
        del buf98
        buf101 = empty_strided_cuda((12, 24, math.trunc(0.1*float(s0))), (24*math.trunc(0.1*float(s0)), math.trunc(0.1*float(s0)), 1), torch.int32)
        # Topologically Sorted Source Nodes: [ids_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_42_xnumel = 288*math.trunc(0.1*float(s0))
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_42.run(buf100, buf101, triton_poi_fused__to_copy_42_xnumel, grid=grid(triton_poi_fused__to_copy_42_xnumel), stream=stream0)
        del buf100
    return (buf101, buf4, reinterpret_tensor(buf92, (12, 24, s0), (24*s0, s0, 1), 0), buf21, buf108, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((12, 24, 64, 128), (128, 98304, 1536, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = 288
    arg2_1 = rand_strided((12, 24), (24, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((12, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg5_1 = 288
    arg6_1 = rand_strided((12, 288, 4, 128), (147456, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((12, 288, 4, 12), (13824, 48, 12, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = 288
    arg10_1 = rand_strided((12, 288, 3, 16), (13824, 48, 16, 1), device='cuda:0', dtype=torch.float64)
    arg11_1 = 288
    arg12_1 = rand_strided((12, 288, 16), (4608, 16, 1), device='cuda:0', dtype=torch.float64)
    arg13_1 = 288
    arg14_1 = rand_strided((12, 288, 8, 4), (9216, 32, 4, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = 288
    arg16_1 = rand_strided((12, 288, 5, 17), (24480, 85, 17, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

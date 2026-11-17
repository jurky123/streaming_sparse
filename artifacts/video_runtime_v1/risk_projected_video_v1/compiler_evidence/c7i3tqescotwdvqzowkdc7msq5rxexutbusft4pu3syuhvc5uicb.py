
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr3': '*fp64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_logsumexp_sub_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 2, 'backend_hash': '47F64A9F6A85E0BCC1AF063FAD76D919A3624F50FDDB2C9704DA0603639DC738', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_logsumexp_sub_19(in_ptr0, out_ptr3, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp30 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr0 + (1 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (2 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr0 + (3 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float64)
        tmp3 = tmp2.to(tl.float64)
        tmp4 = triton_helpers.maximum(tmp1, tmp3)
        tmp6 = tmp5.to(tl.float64)
        tmp7 = triton_helpers.maximum(tmp4, tmp6)
        tmp9 = tmp8.to(tl.float64)
        tmp10 = triton_helpers.maximum(tmp7, tmp9)
        tmp11 = tl_math.abs(tmp10)
        tmp12 = tl.full([1, 1], float("inf"), tl.float64)
        tmp13 = tmp11 == tmp12
        tmp14 = tl.full([1, 1], 0.0, tl.float64)
        tmp15 = tl.where(tmp13, tmp14, tmp10)
        tmp16 = tmp1 - tmp15
        tmp17 = libdevice.exp(tmp16)
        tmp18 = tmp3 - tmp15
        tmp19 = libdevice.exp(tmp18)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp6 - tmp15
        tmp22 = libdevice.exp(tmp21)
        tmp23 = tmp20 + tmp22
        tmp24 = tmp9 - tmp15
        tmp25 = libdevice.exp(tmp24)
        tmp26 = tmp23 + tmp25
        tmp27 = libdevice.log(tmp26)
        tmp28 = tmp27 + tmp15
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = triton_helpers.maximum(_tmp30, tmp29)
        _tmp30 = tl.where(rmask & xmask, tmp31, _tmp30)
    tmp30 = triton_helpers.max2(_tmp30, 1)[:, None]
    _tmp67 = tl.full([XBLOCK, RBLOCK], 0, tl.float64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp32 = tl.load(in_ptr0 + (4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr0 + (1 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr0 + (2 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp40 = tl.load(in_ptr0 + (3 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tmp32.to(tl.float64)
        tmp35 = tmp34.to(tl.float64)
        tmp36 = triton_helpers.maximum(tmp33, tmp35)
        tmp38 = tmp37.to(tl.float64)
        tmp39 = triton_helpers.maximum(tmp36, tmp38)
        tmp41 = tmp40.to(tl.float64)
        tmp42 = triton_helpers.maximum(tmp39, tmp41)
        tmp43 = tl_math.abs(tmp42)
        tmp44 = tl.full([1, 1], float("inf"), tl.float64)
        tmp45 = tmp43 == tmp44
        tmp46 = tl.full([1, 1], 0.0, tl.float64)
        tmp47 = tl.where(tmp45, tmp46, tmp42)
        tmp48 = tmp33 - tmp47
        tmp49 = libdevice.exp(tmp48)
        tmp50 = tmp35 - tmp47
        tmp51 = libdevice.exp(tmp50)
        tmp52 = tmp49 + tmp51
        tmp53 = tmp38 - tmp47
        tmp54 = libdevice.exp(tmp53)
        tmp55 = tmp52 + tmp54
        tmp56 = tmp41 - tmp47
        tmp57 = libdevice.exp(tmp56)
        tmp58 = tmp55 + tmp57
        tmp59 = libdevice.log(tmp58)
        tmp60 = tmp59 + tmp47
        tmp61 = tl_math.abs(tmp30)
        tmp62 = tmp61 == tmp44
        tmp63 = tl.where(tmp62, tmp46, tmp30)
        tmp64 = tmp60 - tmp63
        tmp65 = libdevice.exp(tmp64)
        tmp66 = tl.broadcast_to(tmp65, [XBLOCK, RBLOCK])
        tmp68 = _tmp67 + tmp66
        _tmp67 = tl.where(rmask & xmask, tmp68, _tmp67)
    tmp67 = tl.sum(_tmp67, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp69 = tl.load(in_ptr0 + (4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp71 = tl.load(in_ptr0 + (1 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp74 = tl.load(in_ptr0 + (2 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp77 = tl.load(in_ptr0 + (3 + 4*r1 + 4*ks0*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp70 = tmp69.to(tl.float64)
        tmp72 = tmp71.to(tl.float64)
        tmp73 = triton_helpers.maximum(tmp70, tmp72)
        tmp75 = tmp74.to(tl.float64)
        tmp76 = triton_helpers.maximum(tmp73, tmp75)
        tmp78 = tmp77.to(tl.float64)
        tmp79 = triton_helpers.maximum(tmp76, tmp78)
        tmp80 = tl_math.abs(tmp79)
        tmp81 = tl.full([1, 1], float("inf"), tl.float64)
        tmp82 = tmp80 == tmp81
        tmp83 = tl.full([1, 1], 0.0, tl.float64)
        tmp84 = tl.where(tmp82, tmp83, tmp79)
        tmp85 = tmp70 - tmp84
        tmp86 = libdevice.exp(tmp85)
        tmp87 = tmp72 - tmp84
        tmp88 = libdevice.exp(tmp87)
        tmp89 = tmp86 + tmp88
        tmp90 = tmp75 - tmp84
        tmp91 = libdevice.exp(tmp90)
        tmp92 = tmp89 + tmp91
        tmp93 = tmp78 - tmp84
        tmp94 = libdevice.exp(tmp93)
        tmp95 = tmp92 + tmp94
        tmp96 = libdevice.log(tmp95)
        tmp97 = tmp96 + tmp84
        tmp98 = libdevice.log(tmp67)
        tmp99 = tl_math.abs(tmp30)
        tmp100 = tmp99 == tmp81
        tmp101 = tl.where(tmp100, tmp83, tmp30)
        tmp102 = tmp98 + tmp101
        tmp103 = tmp97 - tmp102
        tl.store(out_ptr3 + (r1 + ks0*x0), tmp103, rmask & xmask)

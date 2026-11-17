"""Refine the module distance table with arithmetic and logical payload bounds."""
import json,re
from .weights import ROOT

def main():
    root=ROOT/'artifacts/video_runtime_v1/exclude_scan_video_v1';roof=json.loads((ROOT/'artifacts/video_runtime_v1/lazy_video_v1/device_roof.json').read_text());bw=roof['copy']['gbps'];fp64=roof['fp64']['tflops'];bf16=roof['bf16']['tflops']
    cells=[]
    for line in (root/'full_route_attributed/after_profile.txt').read_text().splitlines():
        c=[s.strip() for s in re.split(r'\s{2,}',line.strip())]
        if len(c)==11 and c[6].endswith(('us','ms')):cells.append(c)
    def time(prefix,total=False):
        vals=[]
        for c in cells:
            if c[0].startswith(prefix):
                v=c[8 if total else 6];vals.append(float(v[:-2])/(1000 if v.endswith('us') else 1))
        return max(vals)
    b=12*72*16;weights=b*(16**2+32**2)*8;values=b*(16+32)*16*8;logs=b*(16+32)*8
    old_bytes=weights+2*values;new_bytes=2*(values+logs)
    rows=[dict(module='旧快精算＋补偿',ms=1.2954400181770325,floor_ms=(1.5*77.913391104+24.461180928+14.523826176)/bf16,work='155.855 GFLOPs主要BF16发射量；沿用相同实现隔离值',next='已切换快实现，优先级低于路由'),
          dict(module='Reader标量＋向量',ms=time('streaming_sparse::reader_reduce',True),floor_ms=2*283115520/bw/1e6,work='工作区读写566.23MB，不含系数/非线性',next='系数重复读取、归约组织'),
          dict(module='Risk含softmax/U投影',ms=time('triton_red_fused__to_copy_abs_add_clamp_min_div')+time('triton_red_fused__softmax_div')+time('ampere_sgemm'),floor_ms=3*b*432*4*4/bw/1e6,work='Z读、prob写/读最低286.65MB，不含U/非线性',next='共享概率/特征，避免改变路由边界'),
          dict(module='512动作评分',ms=time('scalar_block'),floor_ms=2*b*16*32*8/bw/1e6,work='den/dot最低113.25MB输入',next='本轮以前融合未产生明显收益'),
          dict(module='排除候选扫描',ms=time('scan_kernel'),floor_ms=new_bytes/bw/1e6,work=f'唯一逻辑输入/输出{new_bytes/1e6:.2f}MB；未计重复加载/扫描算术',next='本轮已省掉权重矩阵；进一步优化优先级下降'),
          dict(module='原排除候选BMM（历史对照）',ms=.371787,floor_ms=max(.56623104/fp64,old_bytes/bw/1e6),work=f'0.566GFLOPs；逻辑payload{old_bytes/1e6:.2f}MB',next='已由扫描替代；原9倍算术距离修正约1.51倍')]
    for r in rows:r['distance']=r['ms']/r['floor_ms']
    (root/'module_distance.json').write_text(json.dumps(rows,indent=2))
    lines=['# 正质量扫描后的模块距离表','','形状12heads×72Q64 tiles×M16、432页。当前基线旧快精算/补偿＋A-only/split Reader＋scan，PAGE32/lazy关闭。表内路由模块使用本轮同输入profile，Attention沿用已验证相同实现隔离值。完整路由稳态4.997ms；模块不含所有零碎算子，不能直接求和替代完整路由。','','|模块|实测/ms|参考下限/ms|距离|优化空间|','|---|---:|---:|---:|---|']
    for r in rows:lines.append(f'|{r["module"]}|{r["ms"]:.3f}|{r["floor_ms"]:.3f}|{r["distance"]:.2f}×|{r["next"]}|')
    lines+=['',f'参考设备吞吐：复制{bw:.1f}GB/s，FP64 {fp64:.2f}TFLOP/s，BF16 {bf16:.2f}TFLOP/s。采用主要算术或唯一逻辑payload下限，未覆盖全部非线性、扫描通信和缓存行为；逻辑字节不是HBM计数器测量。距离不可直接当作可兑现加速。','']+[f'- {r["module"]}：{r["work"]}。' for r in rows]
    lines+=['','这次重点修正原BMM的判断：只看算术是0.041ms，但输入/输出payload约311.43MB对应0.246ms，实测约0.372ms。原报告的“9倍”仅是部分算术下限，不能表示还有9倍kernel优化空间。更有效的改动是移除矩阵形式本身，而非把SIMT点积替换到原小矩阵上。','', '短视频累计阶段：路由4.047→3.756秒，Attention约1.851秒，摘要/编码约0.446秒，clean commit约0.877秒。其余阶段未变。完整误差与时间见REPORT.zh.md。']
    (root/'MODULE_DISTANCE.zh.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:12]))
if __name__=='__main__':main()

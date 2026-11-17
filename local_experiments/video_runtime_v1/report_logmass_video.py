"""Finish cached log-mass acceptance and the module-distance accounting."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 root=ROOT/'artifacts/video_runtime_v1/logmass_video_v1';micro=ROOT/'artifacts/video_runtime_v1/logmass_cached_v1'
 rows={}
 for name in ('baseline','cached'):
  r=json.loads((root/name/'result.json').read_text());d=json.loads((root/name/'decode.json').read_text())
  rows[name]=dict(generation=r['median_seconds'],trials=r['seconds'],total=d['generation_plus_decode_seconds'],vae=d['median_seconds'],stages=r['stages'])
 errors=json.loads((root/'errors.json').read_text());quality=json.loads((root/'quality.json').read_text());errors.update({k:quality[k] for k in ['psnr','ssim','lpips','window']})
 assert errors['baseline_matches_current'] and errors['latent_equal'] and errors['rgb_equal']
 route=json.loads((micro/'full_route/result.json').read_text());assert all(route['rows']['output_equal'])
 screen={p.parent.name:json.loads(p.read_text()) for p in sorted(micro.glob('p*/result.json'))}
 rb=route['rows']['before']['gpu_ms_median'];ra=route['rows']['after']['gpu_ms_median'];a=rows['baseline'];b=rows['cached']
 result=dict(rows=rows,errors=errors,route_before_ms=rb,route_after_ms=ra,route_speedup=rb/ra,generation_speedup=a['generation']/b['generation'],total_speedup=a['total']/b['total'],canonical_flag='--cached-logmass',screening=screen)
 (root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
 (root/'status.json').write_text(json.dumps(dict(status='complete_short_video_bitwise',canonical_flag='--cached-logmass',next_action='pause goal at user request; no next-round jobs launched'),indent=2)+'\n')
 lines=['# 缓存 R4 log质量：本轮验收与模块性能距离','','## 配置与测量口径','','当前基线保留用户指定旧快精算/补偿：Page64/Q64/R4、完整128维V均值＋rank8、M16/B16、10%精算、16×32交换＋no-op；全覆盖C_s17＋Full_Pset_s17。旧RoPE、A-only/split Reader、scan-exclude、projected-risk开启；PAGE32、lazy关闭。本轮只增加 --cached-logmass，权重和候选规则均未改变。','','模块输入为真实轨迹捕获的12 heads ×72 Q64 tiles ×M16 ×432页（360历史页＋72当前页）。路由数据来自本轮同一输入完整profile；Attention使用相同未变实现此前隔离测量。每个模块只计一次，custom-op外层与内部kernel不重复加；Reader子阶段不再叠加到合计。完整路由CUDA Graph交替预热、各50次稳态计时。','','设备A100-SXM4-40GB。采用本机实测参考：复制1266.396 GB/s、BF16大GEMM 202.135 TFLOP/s、FP32 13.4865 TFLOP/s、FP64 13.6845 TFLOP/s，来源 ../lazy_video_v1/device_roof.json。以下下限由主要FLOPs或最低明确逻辑读写量推导；未包含的exp/log、归约通信、访问布局会降低实际可达速度。距离不是可承诺加速比，差额也不是实际可节省时间。逻辑payload不等于HBM计数器流量。','','## 模块耗时、理论距离与优化空间','','|模块|实测/ms|参考下限/ms|距离|参考差额/ms|优化空间与优先级|','|---|---:|---:|---:|---:|---|']
 modules=[('旧快精算＋完整均值/rank8补偿',1.29544,155.855/202.135,'中低：已换快实现，先处理Reader'),('Reader标量＋向量',1.938444,566.23104/1266.396,'高：系数复用与归约组织；向量1.448ms、标量0.490ms'),('Reader log质量（本轮两阶段）',.146266+.076797,238.87872/1266.396,'低：已消除R4重复计算；继续优化受非线性及归约限制'),('Risk主要路径，含概率/查询预处理和U投影',.173689+.230263+.066270,477.75744/1266.396,'低：保留FP32 SGEMM，主路径接近逻辑payload参考'),('512动作标量评分',.336850,113.246208/1266.396,'中：exp和中间量读写；前轮融合已筛查，收益很小'),('动作log分母生成',.324467,56.623104/1266.396,'中：距离大但下限只算输出，不能推断7倍可达收益'),('排除候选正质量扫描',.291732,180.486144/1266.396,'中低：减少重复访问/扫描通信，保留主导页稳定处理')]
 for name,ms,floor,space in modules:
  lines.append(f'|{name}|{ms:.3f}|{floor:.3f}|{ms/floor:.2f}×|{ms-floor:.3f}|{space}|')
 lines+=['','其他路由开销：probe QK约0.155ms、FP64交叉矩BMM约0.159ms、矩/增量组合约0.210ms、候选池topk/sort等约0.2ms；部分操作在融合kernel内。它们未做完整逐算子工作量审计，故不填虚构理论倍数。动作回退0.033ms，不是主要热点。表内行不是完整路由的无遗漏分解。', '', '### 下限计算依据与历史修正','','1. Attention：主要BF16发射工作量155.855 GFLOPs /202.135 TFLOP/s =0.771ms。包含当前旧快精算和补偿，两者合计实测1.295ms；不是早期较慢stream_attention的239.176 GFLOPs。','2. Reader两阶段：WORK约283.116MB，写入＋读取566.231MB /1266.396GB/s =0.447ms。未包含系数和exp/tanh，4.34倍只是部分工作量距离。向量阶段仍是最值得深入的模块；此前PAGE32、向量宽度/寄存器筛查结果应保留，避免重复无效实验。','3. 本轮log质量：输入Z为12×72×16×432×4×4字节=95.551MB，FP64局部质量47.776MB。局部写、归一化读、最终写各47.776MB，总238.879MB，对应0.189ms。profile局部生成0.146ms＋归一化0.077ms=0.223ms，距离1.18倍。若只按唯一输入＋最终输出143.327MB算单核参考，则0.113ms、距离1.97倍；这忽略了本轮实际使用的中间缓存。单核融合已试，未比两阶段更快。','4. Risk：概率相关Z读/prob写/prob读286.654MB＋保留U投影的写/读191.103MB =477.757MB，对应0.377ms。主要kernel合计0.470ms。历史表把与Reader共享的log质量计算归进纯Risk，且漏计U，不能沿用原3.86倍结论。','5. 动作评分：主要den/dot输入113.246MB，下限0.089ms；log分母输出56.623MB，下限0.045ms，未计输入及多次logaddexp。编译器已经共享第一步质量组合；前轮20组融合筛查无明确整体收益，因此不以表中大距离直接判断值得重写。','6. 扫描：主要唯一输入/输出180.486MB，下限0.143ms；GPU并行扫描存在logK通信步骤。历史排除BMM的0.566GFLOPs对应0.041ms仅为算术下限；计入311.427MB逻辑payload后约0.246ms，原0.372ms实测距离约1.51倍，而不是可兑现9倍。扫描替代是改变组织方式带来的收益。','','## 本轮针对性优化及隔离验证','','原部署Reader log质量kernel三遍扫描页面，每遍重算R4局部logsumexp。本轮用独立FP64 kernel只计算一次局部log质量，后续页面归一化继续由原torch表达式编译。没有低精度近似；libdevice exp/log和稳定最大值缩放保留。两阶段/单阶段共8组布局筛查，选择两阶段BLOCK256、4warps；2warps的配对倍数略高但其绝对时间更慢且不同GPU，未据此替换。', '', '|页数|独立原表达式/ms|两阶段/ms|加速|最大绝对差|','|---:|---:|---:|---:|---:|']
 for p in [216,288,360,432]:
  d=screen[f'p{p}_b256_w4_f0'];lines.append(f'|{p}|{d["rows"]["baseline"]["gpu_ms_median"]:.3f}|{d["rows"]["cached"]["gpu_ms_median"]:.3f}|{d["speedup"]:.3f}×|{d["max_abs"]:.3g}|')
 lines += ['',f'完整路由 {rb:.3f}→{ra:.3f}ms，{rb/ra:.3f}×；捕获事件5项路由输出全部逐位一致。独立表达式编译和部署编译生成不同归约策略，所以独立模块约0.249→0.229ms不能直接替代部署profile约0.560→0.223ms。完整路由的0.335ms收益与部署profile相符。','','## 短视频验收','','GPU3同卡顺序运行，city_walk_s701，9chunks/105帧，768×512、16FPS，5个历史chunks/360历史页；只跑短视频。完整预热后两次生成计时，VAE独立预热测量，合计不含模型加载、编译、文件读写、视频编码。阶段累计来自独立插桩运行，不与墙钟简单求和。','','|阶段|本轮前/s|本轮后/s|加速比|','|---|---:|---:|---:|']
 for label,key in [('生成','generation'),('VAE','vae'),('生成＋VAE','total')]:lines.append(f'|{label}|{a[key]:.3f}|{b[key]:.3f}|{a[key]/b[key]:.3f}×|')
 for key in ['route_and_correction','attention','summary_and_encoding','clean_commit']:
  x=a['stages'][key]['milliseconds']/1000;y=b['stages'][key]['milliseconds']/1000;lines.append(f'|{key}|{x:.3f}|{y:.3f}|{x/y:.3f}×|')
 lines += ['', '摘要/编码、clean commit、VAE包含多个不同网络/数据移动阶段，尚无完整工作量普查，不给统一理论下限。它们本轮未改动，小幅波动不归为优化收益。', '', '|优化误差指标|结果|','|---|---:|',f'|latent NMSE|{errors["latent_nmse"]}|',f'|RGB MAE /255|{errors["rgb_mae_255"]}|',f'|RGB最大绝对差 /255|{errors["rgb_max_abs_255"]}|',f'|PSNR/dB|{errors["psnr"]}（零差封顶）|',f'|SSIM|{errors["ssim"]:.9f}|',f'|LPIPS|{errors["lpips"]}|','','latent和全部RGB帧逐位一致；SSIM接近1的小偏差来自指标实现。LPIPS/SSIM/PSNR沿用[21,101)评价窗口。基线latent与上一轮已接受projected-risk输出逐位一致。隔离FP64 log质量约1e-14–2e-13差异来自计算/归约次序，未降低精度；本次完整短视频无新增误差。这里比较优化前后，不讨论算法相对Dense的近似误差。','','## 采用结论、复现及暂停点','','本轮采用 --cached-logmass，通用实验开关默认仍关闭，当前完整命令更新到 local_setup/CURRENT_VIDEO_BASELINE.txt。精算和补偿继续使用旧快实现。下一优先级仍是Reader向量系数复用，其次动作分母/评分；本轮结束后按用户要求暂停目标，不启动这些工作。','','证据：acceptance.json / errors.json / quality.json；../logmass_cached_v1/各配置result.json、full_route/before_profile.txt、after_profile.txt、result.json；原编译重复计算证据在../risk_projected_video_v1/compiler_evidence/。源码快照与哈希在source_snapshot/、source_hashes.json。脚本为check_logmass_cached.py、profile_logmass_route.py、run_logmass_video.py、quality_logmass_video.py、report_logmass_video.py。']
 lines += ['', (ROOT/'local_experiments/video_runtime_v1/LOGMASS_REVIEW.zh.md').read_text()]
 text='\n'.join(lines)+'\n';(root/'MODULE_DISTANCE.zh.md').write_text(text)
 (root/'REPORT.zh.md').write_text('# 缓存log质量验收\n\n完整报告与模块理论距离见 [MODULE_DISTANCE.zh.md](MODULE_DISTANCE.zh.md)。\n\n'+f'路由{rb:.3f}→{ra:.3f}ms；短视频生成{a["generation"]:.3f}→{b["generation"]:.3f}秒，含VAE{a["total"]:.3f}→{b["total"]:.3f}秒。latent/RGB逐位一致。\n')
 files=['logmass_cached.py','pipeline_graph.py','runtime.py','benchmark_hotspots.py','check_logmass_cached.py','profile_logmass_route.py','run_logmass_video.py','quality_logmass_video.py','report_logmass_video.py']
 snap=root/'source_snapshot';snap.mkdir(exist_ok=True);hashes={}
 for name in files:
  p=ROOT/'local_experiments/video_runtime_v1'/name;shutil.copy2(p,snap/name);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 (root/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print({k:result[k] for k in ['route_speedup','generation_speedup','total_speedup','errors']})
if __name__=='__main__':main()

import triton as tr
import triton.language as tl
from local_experiments.review_v1.tuned_attention import _tail
@tr.jit
def retry(FLAGS,Q,K,V,IDS,KEY,MEAN,DETAIL,BASIS,OUT,P:tl.constexpr,HP:tl.constexpr,C:tl.constexpr):
 if tl.load(FLAGS+tl.program_id(1)*72+tl.program_id(0)):
  _tail(Q,KEY,MEAN,DETAIL,None,None,None,None,P,4608,12,128,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=BASIS,OUT=OUT,HEAD_PAGES=HP,K=K,V=V,IDS=IDS,ANCHOR_SCALE=True,COLUMN_MASS=True,EXACT_COUNT=C,QK_SCALE=128**-.5)

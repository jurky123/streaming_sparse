from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt32_actions_vaware_20261026'
PARENT=ROOT.parent/'opt30_behavior_grouping_20261026'
PRIORITY=ROOT.parent/'opt31_priority_long_video_20261026'
ANCHOR='q64_qk_local16'
PILOTS=[1,14,24,29]
VARIANTS={
 'Base':dict(content='K32',learned=False,mode='none'),
 'K32_analytic':dict(content='K32',learned=False,mode='swap'),
 'V32_analytic':dict(content='V32',learned=False,mode='swap'),
 'K32_action':dict(content='K32',learned=True,mode='swap'),
 'V32_action':dict(content='V32',learned=True,mode='swap'),
 'V8_action':dict(content='V8',learned=True,mode='swap'),
 'K32_dynamic':dict(content='K32',learned=True,mode='dynamic'),
 'V32_dynamic':dict(content='V32',learned=True,mode='dynamic'),
}

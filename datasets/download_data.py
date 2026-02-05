from datasets import load_dataset
import pandas as pd
from pathlib import Path
from huggingface_hub import login
import os

os.environ['NO_PROXY'] = 'localhost,127.0.0.1,192.168.50.186'

# env_dir = Path(__file__).parent
env_dir = '/research/d1/gds/ytyang/kwchen/store/baselines/atomizedDesign/envs/GPQA'
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
df = pd.DataFrame(ds['train'])

# 添加数据预处理步骤
# 1. 过滤空值行（取消注释）
# df = df[df['image'].isna()]
# 2. 重置索引避免序列化问题
# df = df.reset_index(drop=True)
# 3. 转换所有非基础类型为字符串
# df = df.map(lambda x: str(x) if not isinstance(x, (int, float, str, bool)) else x)

# df = df[df['image'].isna()]  # 使用isna()判断空值，保留空值行


df.to_json(env_dir+'/dataset/output.jsonl', 
          orient='records', 
          lines=True, 
          force_ascii=False,
          default_handler=str)  # 添加异常处理

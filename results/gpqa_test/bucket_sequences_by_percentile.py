#!/usr/bin/env python3
"""
将 JSON profile 结果文件中的序列按 actual_total_tokens 的百分位数分成多个桶，
并将每个桶保存到独立的 JSON 文件中。
"""

import argparse
import json
import numpy as np
from pathlib import Path


def load_profile(filepath: str) -> dict:
    """加载 JSON profile 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_percentiles(tokens: list[float], percentiles: list[float]) -> dict[float, float]:
    """计算百分位数值"""
    arr = np.array(tokens)
    values = np.percentile(arr, percentiles)
    return dict(zip(percentiles, values))


def bucket_sequences(
    sequences: list[dict],
    p25: float,
    p55: float,
    p85: float,
) -> dict[str, list[dict]]:
    """
    根据 actual_total_tokens 的百分位数将序列分类到不同桶中。
    
    - Short: actual_total_tokens <= 25th percentile
    - Medium: 25th < actual_total_tokens <= 55th percentile
    - Long: 55th < actual_total_tokens <= 85th percentile
    - Extra_Long: actual_total_tokens > 85th percentile
    """
    buckets = {
        "short": [],
        "medium": [],
        "long": [],
        "extra_long": [],
    }
    
    for seq in sequences:
        tokens = seq.get("actual_total_tokens", 0)
        if tokens <= p25:
            buckets["short"].append(seq)
        elif tokens <= p55:
            buckets["medium"].append(seq)
        elif tokens <= p85:
            buckets["long"].append(seq)
        else:
            buckets["extra_long"].append(seq)
    
    return buckets


def save_bucket(
    profile: dict,
    sequences: list[dict],
    output_path: Path,
) -> None:
    """将桶中的序列保存到 JSON 文件"""
    output_profile = {k: v for k, v in profile.items() if k != "sequences"}
    output_profile["sequences"] = sequences
    output_profile["bucket_info"] = {
        "sequence_count": len(sequences),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_profile, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="按 actual_total_tokens 百分位数将序列分桶并保存"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="输入的 JSON profile 文件路径",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认为输入文件所在目录",
    )
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取不带扩展名的文件名用于输出命名
    stem = input_path.stem
    
    # 加载 profile
    profile = load_profile(input_path)
    sequences = profile.get("sequences", [])
    
    if not sequences:
        print("警告: 未找到 sequences 数组或数组为空")
        return
    
    # 提取 actual_total_tokens
    tokens = [s.get("actual_total_tokens", 0) for s in sequences]
    
    # 计算 25th、55th、85th 百分位数
    percentiles = [25, 55, 85]
    perc_values = compute_percentiles(tokens, percentiles)
    p25, p55, p85 = perc_values[25], perc_values[55], perc_values[85]
    
    print(f"百分位数:")
    print(f"  25th: {p25:.2f}")
    print(f"  55th: {p55:.2f}")
    print(f"  85th: {p85:.2f}")
    
    # 分桶
    buckets = bucket_sequences(sequences, p25, p55, p85)
    
    # 保存每个桶
    bucket_names = ["short", "medium", "long", "extra_long"]
    for name in bucket_names:
        bucket_seqs = buckets[name]
        output_path = output_dir / f"{stem}_{name}.json"
        save_bucket(profile, bucket_seqs, output_path)
        print(f"  {name}: {len(bucket_seqs)} 条序列 -> {output_path}")
    
    print("完成!")


if __name__ == "__main__":
    main()

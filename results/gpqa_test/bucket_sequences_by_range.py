#!/usr/bin/env python3
"""
将 JSON profile 结果文件中的序列按 actual_total_tokens 平均分成 4 个长度桶，
桶的边界由最大长度/4 确定，并将每个桶保存到独立的 JSON 文件中。
"""

import argparse
import json
from pathlib import Path


def load_profile(filepath: str) -> dict:
    """加载 JSON profile 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def bucket_sequences_by_range(
    sequences: list[dict],
    max_tokens: float,
) -> dict[str, list[dict]]:
    """
    根据 actual_total_tokens 和 max/4 边界将序列分成 4 个桶。
    
    边界: step = max / 4
    - bucket_0 (short):     actual_total_tokens <= max/4
    - bucket_1 (medium):    max/4 < actual_total_tokens <= max/2
    - bucket_2 (long):      max/2 < actual_total_tokens <= 3*max/4
    - bucket_3 (extra_long): actual_total_tokens > 3*max/4
    """
    step = max_tokens / 4
    b1, b2, b3 = step, 2 * step, 3 * step
    
    buckets = {
        "bucket_0": [],
        "bucket_1": [],
        "bucket_2": [],
        "bucket_3": [],
    }
    
    for seq in sequences:
        tokens = seq.get("actual_total_tokens", 0)
        if tokens <= b1:
            buckets["bucket_0"].append(seq)
        elif tokens <= b2:
            buckets["bucket_1"].append(seq)
        elif tokens <= b3:
            buckets["bucket_2"].append(seq)
        else:
            buckets["bucket_3"].append(seq)
    
    return buckets


def save_bucket(
    profile: dict,
    sequences: list[dict],
    output_path: Path,
    bucket_info: dict,
) -> None:
    """将桶中的序列保存到 JSON 文件"""
    output_profile = {k: v for k, v in profile.items() if k != "sequences"}
    output_profile["sequences"] = sequences
    output_profile["bucket_info"] = bucket_info
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_profile, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="按 actual_total_tokens 最大长度/4 边界将序列均分为 4 桶并保存"
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
    
    stem = input_path.stem
    
    profile = load_profile(input_path)
    sequences = profile.get("sequences", [])
    
    if not sequences:
        print("警告: 未找到 sequences 数组或数组为空")
        return
    
    tokens = [s.get("actual_total_tokens", 0) for s in sequences]
    max_tokens = max(tokens)
    min_tokens = min(tokens)
    step = max_tokens / 4
    
    print(f"actual_total_tokens 范围: [{min_tokens}, {max_tokens}]")
    print(f"桶边界 (max/4 = {step:.2f}):")
    print(f"  bucket_0: tokens <= {step:.2f}")
    print(f"  bucket_1: {step:.2f} < tokens <= {2*step:.2f}")
    print(f"  bucket_2: {2*step:.2f} < tokens <= {3*step:.2f}")
    print(f"  bucket_3: tokens > {3*step:.2f}")
    
    buckets = bucket_sequences_by_range(sequences, max_tokens)
    
    bucket_names = ["bucket_0", "bucket_1", "bucket_2", "bucket_3"]
    for name in bucket_names:
        bucket_seqs = buckets[name]
        output_path = output_dir / f"{stem}_{name}.json"
        save_bucket(
            profile,
            bucket_seqs,
            output_path,
            {
                "sequence_count": len(bucket_seqs),
                "max_tokens": max_tokens,
                "step": step,
            },
        )
        print(f"  {name}: {len(bucket_seqs)} 条序列 -> {output_path}")
    
    print("完成!")


if __name__ == "__main__":
    main()

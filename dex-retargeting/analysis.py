#!/usr/bin/env python3
import argparse
import pickle
import numpy as np

def inspect_pickle(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print("=== 1) Top-level data type ===")
    print(type(data))

    # 如果是列表，打印长度以及前几个元素结构
    if isinstance(data, list):
        print("List length:", len(data))
        if len(data) > 0:
            print("First element type:", type(data[0]))

            # 如果元素是 ndarray
            if isinstance(data[0], np.ndarray):
                print("First element shape:", data[0].shape)
                # 可选：打印前几帧数据的内容或形状
                for i, arr in enumerate(data[:2]):
                    print(f"\nElement [{i}] shape: {arr.shape}")
                    print(arr)  # 如果想看具体值可以打开注释
                    # print(arr[:5])  # 只看前5行等
            else:
                print("First element value:", data[0])

    elif isinstance(data, dict):
        print("Dict keys:", data.keys())
        for k, v in data.items():
            print(f"Key: {k} -> type: {type(v)}")
            if isinstance(v, np.ndarray):
                print(f"  shape: {v.shape}")
            elif isinstance(v, list):
                print(f"  list length: {len(v)}")

    else:
        print("Data is neither list nor dict. Directly printing info:")
        print(repr(data))


def main():
    parser = argparse.ArgumentParser(description="Inspect a pickle file.")
    parser.add_argument("--pkl", type=str, required=True,
                        help="Path to the pickle file, e.g. /path/to/output.pkl")
    args = parser.parse_args()

    inspect_pickle(args.pkl)


if __name__ == "__main__":
    main()

import argparse
import json
import time

import moe_matmul
import torch

LlamaConfig = {
    "7B": {
        "hidden_size": 4096,
        "intermediate_size": 11008,
    },
    "13B": {"hidden_size": 5120, "intermediate_size": 13824},
}

BLOCK_DIMS = [
    # MAX_THREADS_PER_BLOCK 1024
    [16, 64],
    [32, 32],
    [64, 16],
    [128, 8],
    [256, 4],
    [512, 2],
    # MAX_THREADS_PER_BLOCK 768
    [16, 48],
    [32, 24],
    [64, 12],
    [128, 8],
    [768, 1],
    [384, 2],
    [192, 4],
    [96, 8],
    [48, 16],
    [24, 32],
    [12, 64],
    # MAX_THREADS_PER_BLOCK 512
    [8, 64],
    [16, 32],
    [32, 16],
    [64, 8],
    [128, 4],
    [256, 2],
    [512, 1],
    # MAX_THREADS_PER_BLOCK 256
    [4, 64],
    [8, 32],
    [16, 16],
    [32, 8],
    [64, 4],
    [128, 2],
    [256, 1],
]


def get_layer_config(model_size, tp, vocab_size=32000):
    hidden_size = LlamaConfig[model_size]["hidden_size"]
    intermediate_size = LlamaConfig[model_size]["intermediate_size"]

    return {
        "QKV_PROJ": {"x": [1, hidden_size], "w": [hidden_size * 3 // tp, hidden_size]},
        "O_PROJ": {"x": [1, hidden_size // tp], "w": [hidden_size, hidden_size // tp]},
        "FFN1": {
            "x": [1, hidden_size],
            "w": [intermediate_size * 2 // tp, hidden_size],
        },
        "FFN2": {
            "x": [1, intermediate_size // tp],
            "w": [hidden_size, intermediate_size // tp],
        },
        # "LM_HEAD": {"x": [1, hidden_size], "w": [vocab_size // tp, hidden_size]},
    }


def runtime(mat1, mat2, block_x, block_y, loops):
    torch.cuda.synchronize()
    s_time = time.time()
    for _ in range(loops):
        out = moe_matmul.moe_gemm(mat2, mat1.view(-1, 1), block_x, block_y).view(1, -1)
    torch.cuda.synchronize()
    cost_time = time.time() - s_time

    return (cost_time / loops) * 1e6, out


def main(args):
    model_sizes, tp_sizes, vocab_sizes, dtype = (
        args.size,
        args.tp,
        args.vocab_size,
        args.dtype,
    )

    # 设置模型及层尺寸的配置
    configs = [
        get_layer_config(model_size, int(tp), int(vocab_size))
        for model_size in model_sizes.split(",")
        for tp in tp_sizes.split(",")
        for vocab_size in vocab_sizes.split(",")
    ]
    shape_configs = {}
    for cfg in configs:
        for v in cfg.values():
            x_shape = v["x"]
            w_shape = v["w"]

            shape_configs[tuple(x_shape + w_shape)] = {}

    # 寻优
    torch.set_default_dtype(torch.__dict__[dtype])
    for shape in shape_configs:
        x_shape = shape[:2]
        w_shape = shape[2:]
        mat1 = torch.rand(x_shape).cuda() / 100
        mat2 = torch.rand(w_shape).cuda() / 100
        truth = torch.matmul(mat1, mat2.T)
        for block_x, block_y in BLOCK_DIMS:
            if block_y > 64:
                continue
            if w_shape[1] // block_x < 8:
                continue
            # warm up
            _, out = runtime(mat1, mat2, block_x, block_y, 10)

            is_same = torch.allclose(truth, out, atol=1e-3)
            if not is_same:
                continue

            # run block cuda
            cost_time, _ = runtime(mat1, mat2, block_x, block_y, 1000)
            print(f"{x_shape=} {w_shape=} {block_x=} {block_y=} {cost_time=}")

            shape_configs[tuple(x_shape + w_shape)][(block_x, block_y)] = cost_time

    # 保存配置
    save_config = {}
    for k, v in shape_configs.items():
        best_block = sorted(v.items(), key=lambda o: o[1])
        if best_block:
            save_config[",".join(map(lambda x: str(x), k))] = best_block[0][0]

    json.dump(save_config, open(args.save, "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="7B,13B", help="Specify model size")
    parser.add_argument("--tp", type=str, default="1", help="Specify model tp size")
    parser.add_argument(
        "--vocab_size", type=str, default="32000", help="Specify model vocab size"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", help="Specify model weight dtype"
    )
    parser.add_argument(
        "--save", type=str, default="block_config.json", help="Specify config output"
    )
    args = parser.parse_args()
    main(args)

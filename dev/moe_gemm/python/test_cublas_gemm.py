import time

import base_matmul
import torch

matmul_kernel_lib = base_matmul.MyMatmulKernel.get_instance()


class Timer:
    def __init__(self, op_name, loops):
        self.loops = loops
        self.begin_time = 0
        self.end_time = 0
        self.op_name = op_name

    def __enter__(self):
        torch.cuda.synchronize()
        self.begin_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end_time = time.time()
        print(
            f"Average time cost of {self.op_name} is {(self.end_time - self.begin_time) / self.loops * 1000:.4f} ms"
        )


hidden_size = 4096
input1 = torch.randn(1, hidden_size).cuda().half() / 100
input2 = torch.randn(hidden_size, hidden_size).cuda().half() / 100
input3 = torch.randn(hidden_size, hidden_size * 3).cuda().half() / 100

xs = [input1] * 3
ws = [
    torch.randn(hidden_size, hidden_size).cuda().half() / 100,
    torch.randn(hidden_size, hidden_size).cuda().half() / 100,
    torch.randn(hidden_size, hidden_size).cuda().half() / 100,
]
w = torch.cat(ws, dim=0)
w2 = w.T.contiguous()


def test_matmul():
    loops = 1000
    with Timer("torch matmul", loops):
        for _ in range(loops):
            out1 = torch.matmul(input1, input3)
    with Timer("cublas matmul", loops):
        for _ in range(loops):
            out2 = matmul_kernel_lib.matmul(input1, input3)
    print(torch.allclose(out1, out2, atol=1e-3))
    print((out1 - out2).abs().max())
    print(out1)
    print(out2)


def test_batched_matmul():
    loops = 1000
    with Timer("torch batched matmul", loops):
        for _ in range(loops):
            out1 = torch.nn.functional.linear(input1, w)
            query, kv = out1.split(
                [
                    4096,
                    2 * 4096,
                ],
                dim=1,
            )
            query = query.view(-1, 4096)
            kv = kv.view(-1, 2, 4096)
            # out1 = torch.matmul(input1, w)
    with Timer("cublas batched matmul", loops):
        for _ in range(loops):
            out2 = matmul_kernel_lib.batched_matmul(xs, ws)
    # print(torch.allclose(out1, out2, atol=1e-3))
    print(out1)
    print(out2)
    # print(input1.shape, w2.shape)


if __name__ == "__main__":
    # a = [torch.tensor([[1,2,3],[4,5,6]]).cuda().half()]
    # b = [torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]]).cuda().half()]
    # print(matmul_kernel_lib.matmul(a[0], b[0]))
    test_matmul()
    # test_batched_matmul()

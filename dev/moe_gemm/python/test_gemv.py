import time

import moe_matmul
import torch


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


a_mat = torch.rand([12288, 4096]).cuda().half() / 100
b_mat = torch.rand([4096, 1]).cuda().half() / 100

c_mat = torch.rand([1, 4096]).cuda().half() / 100
d_mat = torch.rand([4096, 12288]).cuda().half() / 100


def test_gemv():
    loops = 10000
    with Timer("torch gemm", loops):
        for _ in range(loops):
            out1 = torch.matmul(c_mat, d_mat)
    with Timer("cublas gemm", loops):
        for _ in range(loops):
            out2 = moe_matmul.cublas_gemm(a_mat, b_mat)
    with Timer("fast gemv", loops):
        for _ in range(loops):
            out3 = moe_matmul.moe_gemm(a_mat, b_mat, 512, 1)
    # with Timer("transpose1 fast gemv", loops):
    #     for _ in range(loops):
    #         out4 = moe_matmul.moe_gemm(d_mat.t().contiguous(), c_mat.view(4096,1))
    with Timer("transpose2 fast gemv", loops):
        for _ in range(loops):
            out5 = moe_matmul.moe_gemm(a_mat, c_mat.view(-1, 1), 512, 1)
    # print(torch.allclose(out1, out2, atol=1e-4))
    # print(torch.allclose(out1, out3, atol=1e-4))
    # print(torch.matmul(c_mat, d_mat), "\n", out4)


if __name__ == "__main__":
    # test_gemv()
    a_mat = torch.rand([5120, 13824]).cuda().half() / 100
    b_mat = torch.rand([13824, 1]).cuda().half() / 100
    out1 = torch.matmul(a_mat, b_mat)
    out2 = moe_matmul.moe_gemm(a_mat, b_mat, 128, 1)
    print(out1)
    print(out2)

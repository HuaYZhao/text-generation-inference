import unittest

import torch
from mpi4py import MPI

import comm_lib

tp_comm = None


def init_envs():
    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    world_size = COMM.Get_size()
    tp_comm, pp_comm = comm_lib.init_nccl(world_size, 1)
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(1.0, device)
    return tp_comm, pp_comm


class TestCommLib(unittest.TestCase):
    def test_allreduce(self):
        global tp_comm
        t = torch.tensor([1, 1.1, 2, 2.01], dtype=torch.float16).to("cuda")
        comm_lib.allreduce(t, tp_comm)

        assert torch.allclose(
            t, torch.tensor([2, 2.2, 4, 4.02]).cuda().half(), atol=1e-3
        )

    def test_allgather_into_tensor(self):
        global tp_comm
        t = torch.tensor([1, 1.1, 2, 2.01], dtype=torch.float16).to("cuda")
        world_out = t.new_empty(1, 8)
        comm_lib.allgather_into_tensor(world_out, t, tp_comm)
        assert torch.allclose(
            world_out,
            torch.tensor([1, 1.1, 2, 2.01, 1, 1.1, 2, 2.01]).cuda().half(),
            atol=1e-3,  # noqa: E501
        )


if __name__ == "__main__":
    tp_comm, pp_comm = init_envs()
    unittest.main()
    comm_lib.finalize_nccl(tp_comm, pp_comm)

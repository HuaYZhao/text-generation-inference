import os
import torch
from loguru import logger
from mpi4py import MPI

import comm_lib

# Tensor Parallelism settings
RANK = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
WORLD_SIZE = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))

# CUDA memory fraction
MEMORY_FRACTION = float(os.getenv("CUDA_MEMORY_FRACTION", "1.0"))

USE_CUSTOM_NCCL = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1")) > 1 and int(os.getenv("USE_CUSTOM_NCCL", "1")) == 1


class CommGroup:
    def __init__(self, rank, size, tp_comm, pp_comm):
        self._rank = rank
        self._size = size
        self.tp_comm = tp_comm
        self.pp_comm = pp_comm

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def initialize_mpi_distributed():
    assert torch.cuda.is_available()
    # mpi initialize
    COMM = MPI.COMM_WORLD
    assert COMM.Get_size() == WORLD_SIZE, f"{COMM.Get_size()},{WORLD_SIZE}"

    # Set the device id.
    assert WORLD_SIZE <= torch.cuda.device_count(), "Each process is one gpu"
    device = RANK % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)

    # nccl initialize
    tp_comm, pp_comm = comm_lib.init_nccl(WORLD_SIZE, 1)
    process_group = CommGroup(RANK, WORLD_SIZE, tp_comm, pp_comm)

    logger.info("custom mpi and nccl is already initialized.")
    return process_group, RANK, WORLD_SIZE, COMM


def initialize_distributed():
    if USE_CUSTOM_NCCL:
        return initialize_mpi_distributed()
    else:
        from text_generation_server.utils.dist import initialize_torch_distributed
        process_group, rank, world_size = initialize_torch_distributed()
        return process_group, rank, world_size, None


def barrier(comm=None, process_group=None):
    if USE_CUSTOM_NCCL:
        comm.barrier()
    else:
        torch.distributed.barrier(group=process_group)


def allreduce(tensor, process_group):
    if USE_CUSTOM_NCCL:
        comm_lib.allreduce(tensor, process_group.tp_comm)
    else:
        torch.distributed.all_reduce(tensor, group=process_group)


def allgather_into_tensor(world_out, gather_input, process_group):
    # 在Tensor层级进行gather，避免allgather返回列表的手动concat开销
    if USE_CUSTOM_NCCL:
        comm_lib.allgather_into_tensor(world_out, gather_input, process_group.tp_comm)
    else:
        torch.distributed.all_gather_into_tensor(
            world_out, gather_input, group=process_group
        )

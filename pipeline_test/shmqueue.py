import time
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Union

import numpy as np
import torch


class QueueSpec:
    """
    Header: 8 bytes
        4 bytes: head
        4 bytes: tail
    """
    HEADER_SPECS = {
        "head": (int, 4),
        "tail": (int, 4),
    }
    TOTAL_LEN = sum([nbytes for _, (_, nbytes) in HEADER_SPECS.items()])

    def __init__(self, shm: SharedMemory) -> None:
        self.shm = shm
        self.loadMem()

    def loadMem(self):
        offset = 0
        for name, (T, nbytes) in self.HEADER_SPECS.items():
            setattr(self, name, T.from_bytes(
                self.shm.buf[offset:offset + nbytes], "little"))
            offset += nbytes

    def writeMem(self, names):
        offset = 0
        for name, (T, nbytes) in self.HEADER_SPECS.items():
            if name in names:
                self.shm.buf[offset:offset + nbytes] = \
                    getattr(self, name).to_bytes(nbytes, "little")
            offset += nbytes

    def full(self, queue_size):
        self.loadMem()
        return (self.head + 1) % queue_size == self.tail

    def empty(self, queue_size):
        self.loadMem()
        return self.head == self.tail

    def __repr__(self):
        return "QueueHeader(" + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]) + ")"


class ShmQueue():
    """
    This is a circular buffer queue wrapped around a shared memory segment.
    HACK: This is a very simple implementation, and it is only for 1 producer + 1 consumer.

    Memory layout:
        [header] [obj1] [obj2] ...

    Description:
        obj_specs: Dict[str, np.ndarray]
            The key is the name of the object, and the value is the numpy array
            that describes the shape and dtype of the object.
        device_specs: Dict[str, torch.device]
            The key is the name of the object, and the value is the torch device
            that the object should be copied to.

    Example:
        obj_specs = {
            "x": np.zeros((10, 10), dtype=np.int8),
            "y": np.zeros((10, 10), dtype=np.float32),
        }
        device_specs = {
            "x": torch.device("cuda:0"),
        }
        shm = shared_memory.SharedMemory(create=True, size=8 + ...)
        queue = ShmQueue(shm, size, obj_specs, device_specs)

        # Case 1: directly write to the shared memory
        queue.put(data)

        # Case 2: get a copy of the shared memory
        data = queue.get()

        # Case 3: manually copy to the shared memory
        writable_mem = queue.alloc()
        # copy data to writable_mem
        del writable_mem
        queue.commit()

        # Case 4: read from the shared memory
        data = queue.read()
        # do something with data...
        del data
        queue.commit()
    """

    def __init__(self,
                 queue_size: int,
                 shm: SharedMemory,
                 obj_specs: Dict[str, np.ndarray],
                 device_specs: Dict[str, torch.device] = {}
                 ) -> None:
        self.shm = shm
        self.header = QueueSpec(shm)
        self.queue_size = queue_size
        self.obj_specs = obj_specs

        self.bytes_per_item = \
            sum([array.nbytes for array in obj_specs.values()])
        total_bytes = self.header.TOTAL_LEN + queue_size * self.bytes_per_item
        if self.shm.buf.nbytes < total_bytes:
            print(
                "[ERROR]: Shared memory size is not enough for the queue, "
                "{} < {}".format(self.shm.buf.nbytes, total_bytes)
            )
            raise RuntimeError

        self.commit_options = None

        # HACK: self.obj_views[k][self.header.head, ...] = v is faster
        # self.obj_views: Dict[str, np.ndarray] = {}
        # offset = self.header.TOTAL_LEN
        # for k, v in self.obj_specs.items():
        #     self.obj_views[k] = np.frombuffer(
        #         self.shm.buf, dtype=v.dtype,
        #         count=v.size * self.queue_size, offset=offset)\
        #         .reshape((self.queue_size, *v.shape))
        #     offset += v.nbytes * self.queue_size

        # async memcpy for cuda tensor
        # TODO: this is not implemented yet
        assert not device_specs, "Not implemented for device_specs"
        self.device_specs = device_specs
        self.stream = torch.cuda.Stream() if len(self.device_specs) > 0 else None
        self.next_data: Dict[str, Union[np.ndarray, torch.Tensor]] = {}

        self.waiting_time = 0  # (sec)

    def full(self):
        return self.header.full(self.queue_size)

    def empty(self):
        return self.header.empty(self.queue_size)

    def put(self, data: Dict[str, np.ndarray]):
        waiting_start = time.time()
        while self.header.full(self.queue_size):
            pass  # HACK: busy waiting when queue is full
        self.waiting_time += time.time() - waiting_start

        offset = QueueSpec.TOTAL_LEN + self.header.head * self.bytes_per_item
        for k, v in self.obj_specs.items():
            if v.shape != data[k].shape:
                print(f"[ERROR]: Shape mismatch: {v.shape} != {data[k].shape}")
                raise RuntimeError
            if v.dtype != data[k].dtype:
                print(f"[ERROR]: Dtype mismatch: {v.dtype} != {data[k].dtype}")
                raise RuntimeError

            self.shm.buf[offset:offset + v.nbytes] = data[k].tobytes()
            offset += v.nbytes

        self.header.head = (self.header.head + 1) % self.queue_size
        self.header.writeMem(names=["head"])

    def get(self) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        waiting_start = time.time()
        while self.header.empty(self.queue_size):
            pass  # HACK: busy waiting when queue is empty
        self.waiting_time += time.time() - waiting_start

        data = {}
        offset = QueueSpec.TOTAL_LEN + self.header.tail * self.bytes_per_item
        for k, v in self.obj_specs.items():
            data[k] = np.frombuffer(
                self.shm.buf, dtype=v.dtype,
                count=v.size, offset=offset)\
                .reshape(v.shape).copy()
            offset += v.nbytes

            # cuda memcpy
            if k in self.device_specs:
                data[k] = torch.as_tensor(
                    data[k], device=self.device_specs[k])

        self.header.tail = (self.header.tail + 1) % self.queue_size
        self.header.writeMem(names=["tail"])

        return data

    def alloc(self) -> Dict[str, np.ndarray]:
        """Get data to write to the queue."""
        while self.header.full(self.queue_size):
            pass  # HACK: busy waiting when queue is full

        data = {}
        offset = QueueSpec.TOTAL_LEN + self.header.head * self.bytes_per_item
        for k, v in self.obj_specs.items():
            data[k] = np.frombuffer(
                self.shm.buf, dtype=v.dtype,
                count=v.size, offset=offset)\
                .reshape(v.shape)
            offset += v.nbytes

        self.header.head = (self.header.head + 1) % self.queue_size
        self.commit_options = ["head"]
        return data

    def read(self) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Get data (not copy) from the queue."""
        while self.header.empty(self.queue_size):
            pass  # HACK: busy waiting when queue is empty

        if self.stream is not None \
                and len(self.next_data) > 0:
            torch.cuda.current_stream().wait_stream(self.stream)
        data, self.next_data = self.next_data, {}

        offset = QueueSpec.TOTAL_LEN + self.header.tail * self.bytes_per_item
        for k, v in self.obj_specs.items():
            if k not in data:
                data[k] = np.frombuffer(
                    self.shm.buf, dtype=v.dtype,
                    count=v.size, offset=offset)\
                    .reshape(v.shape)
                offset += v.nbytes

                if k in self.device_specs:
                    data[k] = torch.as_tensor(
                        data[k], device=self.device_specs[k])

        self.header.tail = (self.header.tail + 1) % self.queue_size
        self.commit_options = ["tail"]

        # if next data is ready, prefetch device data
        if self.stream is not None and \
                not self.header.empty(self.queue_size):
            # TODO: FIXME: not async
            with torch.cuda.stream(self.stream):
                offset = QueueSpec.TOTAL_LEN + self.header.tail * self.bytes_per_item
                for k, v in self.obj_specs.items():
                    if k in self.device_specs:
                        self.next_data[k] = torch.as_tensor(
                            np.frombuffer(
                                self.shm.buf, dtype=v.dtype,
                                count=v.size, offset=offset)
                            .reshape(v.shape)
                        ).pin_memory(self.device_specs[k])

        return data

    def commit(self):
        if self.commit_options is None:
            print("[ERROR]: No data to commit")
            raise RuntimeError

        self.header.writeMem(names=self.commit_options)
        self.commit_options = None
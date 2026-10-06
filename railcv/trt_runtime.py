"""Minimal TensorRT 10 runner (JetPack 7 on Jetson Thor, or TensorRT 10 on the L4 box).

Uses torch CUDA tensors as I/O buffers, so pycuda is not needed.
Build the engine first:  python scripts/export_onnx.py <model> --build
"""

import tensorrt as trt
import torch

_DTYPES = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
}


class TRTEngine:
    def __init__(self, engine_path: str, device=None):
        self.device = device or torch.device("cuda")
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load {engine_path} (built with a different TensorRT version?)")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        assert len(self.inputs) == 1 and len(self.outputs) == 1, names
        self.out_buf = torch.empty(
            tuple(self.engine.get_tensor_shape(self.outputs[0])),
            dtype=_DTYPES[self.engine.get_tensor_dtype(self.outputs[0])],
            device=self.device,
        )
        self.in_dtype = _DTYPES[self.engine.get_tensor_dtype(self.inputs[0])]
        self.stream = torch.cuda.Stream(device=self.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, self.in_dtype).contiguous()
        self.ctx.set_tensor_address(self.inputs[0], x.data_ptr())
        self.ctx.set_tensor_address(self.outputs[0], self.out_buf.data_ptr())
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.out_buf

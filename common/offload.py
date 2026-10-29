import contextlib
import math

import torch


def _is_block_list(module):
    # the attention output projection is a ModuleList too, so only keep lists of real blocks
    return isinstance(module, torch.nn.ModuleList) and len(module) > 0 and all(
        sum(p.numel() for p in child.parameters()) > 1_000_000 for child in module)


# the norm weights and the like: streaming them saves nothing, and an unquantized one is saved by the autograd
# graph for the backward, which finds it back on the cpu when the hook of the block has already offloaded it
MIN_OFFLOADED_NUMEL = 1_000_000


def _frozen_tensors(block):
    for tensor in list(block.parameters()) + list(block.buffers()):
        if not tensor.requires_grad and tensor.numel() >= MIN_OFFLOADED_NUMEL:
            yield tensor


class BlockOffload:
    """Keeps the frozen weights of a share of the transformer blocks in pinned CPU memory.

    An offloaded block is copied to `device` right before its forward and released right after.
    Gradient checkpointing re-runs the forward during the backward, which fires the hooks again,
    so the memory is only freed if checkpointing is enabled. Trainable tensors (the adapter) never
    move, otherwise their grads would end up on the cpu next to the optimizer states. The blocks
    that are not offloaded and everything outside the blocks (embedders, norms, ...) sit on the
    device for good.
    """

    def __init__(self, model, device, percentage=100.0):
        self.device = device
        self.active = True
        self.offloaded = []

        block_params = set()
        blocks = []
        for module in model.modules():
            if _is_block_list(module):
                for block in module:
                    blocks.append(block)
                    block_params.update(id(p) for p in block.parameters())

        if len(blocks) == 0:
            raise ValueError('offload_to_cpu: no transformer block list found in the model')

        # spread the offloaded blocks evenly over the depth, so the resident ones are not all at one end
        fraction = max(0.0, min(100.0, float(percentage))) / 100.0
        offloaded = [math.floor((i + 1) * fraction + 1e-9) > math.floor(i * fraction + 1e-9)
                     for i in range(len(blocks))]

        # trainable tensors and everything outside the offloaded blocks live on the device
        offloaded_params = set()
        for block, is_offloaded in zip(blocks, offloaded):
            if is_offloaded:
                offloaded_params.update(id(t) for t in _frozen_tensors(block))
        for _, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if id(tensor) not in offloaded_params:
                tensor.data = tensor.data.to(device)

        # resident is a set shared by the hooks. Non reentrant checkpointing stops the recomputation
        # early, before the forward hook of the block runs, so a block can stay on the device after
        # its backward. Only one block is allowed to be resident: onloading the next one releases
        # the previous one
        self.resident = set()
        for block, is_offloaded in zip(blocks, offloaded):
            if not is_offloaded:
                continue
            for tensor in _frozen_tensors(block):
                tensor.data = tensor.data.to('cpu').pin_memory()
            # keep the pinned copy around: moving back is then free instead of a device to host copy
            block._cpu_copies = {id(t): t.data for t in _frozen_tensors(block)}
            block.register_forward_pre_hook(self._onload)
            block.register_forward_hook(self._offload)
            self.offloaded.append(block)

        torch.cuda.empty_cache()
        print(f'offload_to_cpu: {len(self.offloaded)} of {len(blocks)} blocks stream their frozen '
              f'weights from the cpu')

    def _release(self, block):
        for tensor in _frozen_tensors(block):
            tensor.data = block._cpu_copies[id(tensor)]
        self.resident.discard(block)

    def _onload(self, block, args, kwargs=None):
        if not self.active:
            return
        for other in list(self.resident):
            if other is not block:
                self._release(other)
        self.resident.add(block)
        for tensor in _frozen_tensors(block):
            tensor.data = tensor.data.to(self.device, non_blocking=True)

    def _offload(self, block, args, output, kwargs=None):
        if self.active:
            self._release(block)

    @contextlib.contextmanager
    def on_device(self):
        """Put every offloaded block on the device for good, e.g. for the validation images.

        The pinned copies are kept, so leaving the context just drops the device tensors.
        """
        self.active = False
        for block in self.offloaded:
            for tensor in _frozen_tensors(block):
                tensor.data = tensor.data.to(self.device)
        try:
            yield
        finally:
            for block in self.offloaded:
                self._release(block)
            self.active = True
            torch.cuda.empty_cache()


def offload_frozen_blocks(model, device, percentage=100.0):
    return BlockOffload(model, device, percentage)

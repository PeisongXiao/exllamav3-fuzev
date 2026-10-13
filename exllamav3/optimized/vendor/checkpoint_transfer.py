"""Coalesced checkpoint copies through reusable, model-sized pinned storage."""
# Derived from ExLlamaV3 by turboderp (MIT). Modifications Copyright 2026 Peisong Xiao.

import functools
import threading

import torch
from exllamav3.modules.gated_delta_net import GDNLayerState, GDNState
from exllamav3.modules.ple import PLELayerState

ARENA = None
LOCK = threading.RLock()


def layout(state):
    if state.cache.model.loaded_tp:
        return None
    groups = []
    offset = 0
    devices = set()
    for key, layer in state.cache.get_all_recurrent_layers().items():
        if type(layer) is GDNLayerState:
            tensors = (layer.recurrent_state[state.slot, :1],
                       layer.conv_state[state.slot, :, :layer.module.conv_kernel_size])
        elif type(layer) is PLELayerState:
            tensors = (layer.conv_state[state.slot, :, :layer.win],
                       layer.id_state[state.slot, :layer.ctx])
        else:
            return None
        items = []
        for tensor in tensors:
            devices.add(tensor.device)
            if tensor.device.type != "cuda" or offset % tensor.element_size():
                return None
            size = tensor.numel() * tensor.element_size()
            items.append((tensor, offset, size))
            offset += size
        groups.append((key, items))
    if len(devices) != 1 or not offset:
        return None
    return groups, offset, devices.pop()


def arena(size):
    global ARENA
    if ARENA is None or ARENA.numel() < size:
        ARENA = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
    return ARENA[:size]


def view(buffer, source, offset, size):
    return buffer[offset:offset + size].view(source.dtype).view(source.shape)


def snapshot(state, plan):
    groups, size, device = plan
    with LOCK:
        staging = arena(size)
        for _, items in groups:
            for source, offset, length in items:
                view(staging, source, offset, length).copy_(source, non_blocking=True)
        torch.cuda.current_stream(device).synchronize()
        stored = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=False)
        stored.copy_(staging)
    result = {"position": state.position, "checkpoint_size": size,
              "_coalesced_slab": stored}
    for key, items in groups:
        result[key] = tuple(view(stored, source, offset, length) for source, offset, length in items)
    return result


def restore(state, saved, plan):
    groups, size, device = plan
    with LOCK:
        staging = arena(size)
        staging.copy_(saved["_coalesced_slab"])
        for _, items in groups:
            for target, offset, length in items:
                target.copy_(view(staging, target, offset, length), non_blocking=True)
        torch.cuda.current_stream(device).synchronize()


def patches():
    original_stash, original_restore = GDNState.stash, GDNState.unstash

    @functools.wraps(original_stash)
    def stash(self):
        plan = layout(self)
        return snapshot(self, plan) if plan is not None else original_stash(self)

    @functools.wraps(original_restore)
    def unstash(self, saved):
        plan = layout(self) if "_coalesced_slab" in saved else None
        if (plan is not None and self.position == saved["position"]
                and plan[1] == saved["checkpoint_size"]):
            return restore(self, saved, plan)
        return original_restore(self, saved)

    return [(GDNState, "stash", stash), (GDNState, "unstash", unstash)]

"""ExLlamaV3 prefill optimizations, enabled by default."""

__all__ = ["enable_prefill_optimizations"]


def enable_prefill_optimizations(options: dict | None = None):
    """Enable optimized kernels, applying options before constructing a model."""
    from . import hooks
    from .install import install
    if options is not None:
        unknown = options.keys() - hooks.CONFIG.keys()
        if unknown:
            raise ValueError(f"Unknown prefill optimization options: {', '.join(sorted(unknown))}")
        hooks.CONFIG.update(options)
    install()

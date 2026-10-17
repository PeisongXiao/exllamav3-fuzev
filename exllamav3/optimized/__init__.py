"""Options for ExLlamaV3 prefill optimizations."""

__all__ = ["enable_prefill_optimizations"]


def enable_prefill_optimizations(options: dict | None = None):
    """Set prefill optimization options before Model.from_config. A call without options changes nothing."""
    from . import settings
    if options is not None:
        unknown = options.keys() - settings.CONFIG.keys()
        if unknown:
            raise ValueError(f"Unknown prefill optimization options: {', '.join(sorted(unknown))}")
        settings.CONFIG.update(options)

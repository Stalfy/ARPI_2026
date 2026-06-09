# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Utility helpers used by `module_api.common`.

This module contains small, shared helpers used across the `common`
package (for example, formatting durations for human-readable output).
"""


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    The representation uses seconds for durations under 60s, minutes and
    seconds for durations under an hour, and hours + minutes for longer
    durations.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted duration string. Examples:
            - 30.5 -> "30.5s"
            - 150 -> "2m 30s"
            - 5430 -> "1h 30m"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

"""Shared helper utilities for GTFS-RT protobuf parsing and parquet writing."""

from google.protobuf.message import Message
from google.transit import gtfs_realtime_pb2

type ProtoRow = dict[str, int | float | str | bool | None]


def hf(msg: Message, field: str) -> bool:
    """Check whether a protobuf message has a specific field set.

    Args:
        msg: The protobuf message to inspect.
        field: Name of the field to check.

    Returns:
        True if the field is set, False otherwise — including when the field is
        repeated or not part of a oneof, which raises ValueError internally.
    """
    try:
        return msg.HasField(field)
    except ValueError:
        return False


def attr(obj: Message, field: str, default: int | float | str | None = None) -> int | float | str | None:
    """Return a proto message attribute by name, falling back to a default.

    Wraps getattr to safely access optional extension fields that may not be
    defined on every feed implementation's generated class.

    Args:
        obj: The protobuf message to read from.
        field: Name of the attribute to access.
        default: Value to return when the attribute is absent. Defaults to None.

    Returns:
        The attribute value, or default if the attribute does not exist.
    """
    return getattr(obj, field, default)


def ts(msg: gtfs_realtime_pb2.TranslatedString) -> str | None:
    """Extract the first translation text from a GTFS-RT TranslatedString message.

    Args:
        msg: A TranslatedString protobuf message.

    Returns:
        The text of the first translation entry, or None if the list is empty
        or an error occurs during access.
    """
    try:
        translations = list(msg.translation)
        return translations[0].text if translations else None
    except Exception:
        return None


def str_or_ts(obj: Message, field: str) -> str | None:
    """Read a proto field that may be either a plain string or a TranslatedString.

    Args:
        obj: The protobuf message to read from.
        field: Name of the field to access.

    Returns:
        The string value, the first translation text if the field holds a
        TranslatedString, or None if the field is absent, empty, or any other type.
    """
    raw = getattr(obj, field, None)
    if raw is None:
        return None
    if isinstance(raw, gtfs_realtime_pb2.TranslatedString):
        return ts(raw)
    if isinstance(raw, str):
        return raw or None
    return None


def parse_header(hdr: gtfs_realtime_pb2.FeedHeader) -> ProtoRow:
    """Map FeedHeader fields into a flat ProtoRow dictionary.

    Empty strings are normalized to None so downstream columns remain nullable
    rather than holding empty string sentinels.

    Args:
        hdr: A FeedHeader protobuf message.

    Returns:
        A ProtoRow with the header.gtfs_realtime_version, header.incrementality,
        header.timestamp, and header.feed_version columns.
    """
    return {
        "header.gtfs_realtime_version": hdr.gtfs_realtime_version or None,
        "header.incrementality": hdr.incrementality,
        "header.timestamp": hdr.timestamp,
        "header.feed_version": attr(hdr, "feed_version") or None,
    }

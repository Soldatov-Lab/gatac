"""
In-place edit of the key-value metadata of a Parquet file.

A Parquet file ends with its footer (a Thrift ``FileMetaData`` in the compact
protocol), the footer's length and ``PAR1``. The data pages are located by
absolute offsets that all point before the footer, so the footer can be
replaced without touching the data: only the key-value list (field 5) is
re-encoded, and the rest of the footer is copied byte for byte.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Union

_MAGIC = b"PAR1"

# Thrift compact protocol types
_BOOL_TRUE, _BOOL_FALSE, _BYTE, _I16, _I32, _I64, _DOUBLE = 1, 2, 3, 4, 5, 6, 7
_BINARY, _LIST, _SET, _MAP, _STRUCT = 8, 9, 10, 11, 12

_KEY_VALUE_FIELD = 5  # FileMetaData.key_value_metadata


def _varint(buf: bytes, pos: int) -> Tuple[int, int]:
    value = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, pos
        shift += 7


def _write_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _field_header(buf: bytes, pos: int, last_id: int) -> Tuple[int, int, int]:
    """(field id, type, position after the header); type 0 is the struct's end."""
    b = buf[pos]
    pos += 1
    ftype = b & 0x0F
    if ftype == 0:
        return 0, 0, pos
    delta = b >> 4
    if delta:
        return last_id + delta, ftype, pos
    zigzag, pos = _varint(buf, pos)
    return (zigzag >> 1) ^ -(zigzag & 1), ftype, pos


def _list_header(buf: bytes, pos: int) -> Tuple[int, int, int]:
    """(size, element type, position after the header) of a list or set."""
    b = buf[pos]
    pos += 1
    size = b >> 4
    if size == 15:
        size, pos = _varint(buf, pos)
    return size, b & 0x0F, pos


def _skip(buf: bytes, pos: int, ftype: int, in_collection: bool = False) -> int:
    """Position after a value of type ``ftype``."""
    if ftype in (_BOOL_TRUE, _BOOL_FALSE):
        # A struct field's bool is in its type; a collection's takes a byte
        return pos + 1 if in_collection else pos
    if ftype == _BYTE:
        return pos + 1
    if ftype in (_I16, _I32, _I64):
        return _varint(buf, pos)[1]
    if ftype == _DOUBLE:
        return pos + 8
    if ftype == _BINARY:
        n, pos = _varint(buf, pos)
        return pos + n
    if ftype in (_LIST, _SET):
        size, etype, pos = _list_header(buf, pos)
        for _ in range(size):
            pos = _skip(buf, pos, etype, True)
        return pos
    if ftype == _MAP:
        size, pos = _varint(buf, pos)
        if size:
            types = buf[pos]
            pos += 1
            for _ in range(size):
                pos = _skip(buf, pos, types >> 4, True)
                pos = _skip(buf, pos, types & 0x0F, True)
        return pos
    if ftype == _STRUCT:
        last_id = 0
        while True:
            last_id, t, pos = _field_header(buf, pos, last_id)
            if t == 0:
                return pos
            pos = _skip(buf, pos, t)
    raise ValueError(f"Unknown Thrift type {ftype} in the Parquet footer")


def _key_value_span(footer: bytes) -> Tuple[int, int]:
    """Start and end of the key-value list's value in the footer."""
    pos, last_id = 0, 0
    while True:
        last_id, ftype, pos = _field_header(footer, pos, last_id)
        if ftype == 0:
            raise ValueError("The Parquet footer has no key-value metadata")
        end = _skip(footer, pos, ftype)
        if last_id == _KEY_VALUE_FIELD:
            return pos, end
        pos = end


def _decode_key_values(buf: bytes) -> List[Tuple[str, str]]:
    size, _, pos = _list_header(buf, 0)
    items = []
    for _ in range(size):
        key = value = ""
        last_id = 0
        while True:
            last_id, ftype, pos = _field_header(buf, pos, last_id)
            if ftype == 0:
                break
            if ftype == _BINARY and last_id in (1, 2):
                n, pos = _varint(buf, pos)
                text = buf[pos:pos + n].decode()
                pos += n
                key, value = (text, value) if last_id == 1 else (key, text)
            else:
                pos = _skip(buf, pos, ftype)
        items.append((key, value))
    return items


def _encode_key_values(items: List[Tuple[str, str]]) -> bytes:
    n = len(items)
    out = bytearray([n << 4 | _STRUCT] if n < 15 else [0xF0 | _STRUCT])
    if n >= 15:
        out += _write_varint(n)
    for key, value in items:
        for text in (key, value):
            data = text.encode()
            out += bytes([1 << 4 | _BINARY]) + _write_varint(len(data)) + data
        out.append(0)
    return bytes(out)


def update_key_value_metadata(path: Union[str, Path], updates: Dict[str, str]) -> None:
    """
    Set keys of the key-value metadata of a Parquet file, in place.

    Only the footer is rewritten: the file keeps its data and is not copied.
    On an error the old footer is put back.
    """
    with open(path, "r+b") as f:
        f.seek(-8, os.SEEK_END)
        n = struct.unpack("<I", f.read(4))[0]
        if f.read(4) != _MAGIC:
            raise ValueError(f"{path} is not a Parquet file")
        start = f.tell() - 8 - n
        f.seek(start)
        footer = f.read(n)
        lo, hi = _key_value_span(footer)
        items = dict(_decode_key_values(footer[lo:hi]))
        items.update(updates)
        new = footer[:lo] + _encode_key_values(list(items.items())) + footer[hi:]
        try:
            f.seek(start)
            f.write(new + struct.pack("<I", len(new)) + _MAGIC)
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            f.seek(start)
            f.write(footer + struct.pack("<I", n) + _MAGIC)
            f.truncate()
            raise

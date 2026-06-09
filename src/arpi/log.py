# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Stub ApplicationLogger for use outside the full module_api runtime."""


class ApplicationLogger:
    def __init__(self, name: str) -> None:
        self._name = name

    def dbg(self, msg: str, **_) -> None:
        pass

    def inf(self, msg: str, **_) -> None:
        print(f"[{self._name}] {msg}")

    def wrn(self, msg: str, **_) -> None:
        print(f"[{self._name}] WARN: {msg}")

    def err(self, msg: str, **_) -> None:
        print(f"[{self._name}] ERROR: {msg}")

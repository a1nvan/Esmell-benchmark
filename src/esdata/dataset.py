"""Archive-backed dataset source for the layer-based API."""

from __future__ import annotations

import json
import re
import tarfile
from datetime import datetime
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

from esdata.csv_repair import CsvRepairDiagnostic, CsvRepairError, UserLabelCsvRepair
from esdata.layers import SampleId, Source


class EsmellDataset(Source):
    """ID-based source for raw experiment CSV files in DVC archives."""

    _pattern = re.compile(r"exp(\d+)_(.+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.csv")
    _card_pattern = re.compile(r"experiment_card_(\d+)\.json")
    _archive_suffixes = (".tar.gz", ".tgz")

    def __init__(
        self,
        root_path: Path,
        subfolders: Union[str, Iterable[str]],
        csv_repair: Optional[Union[str, UserLabelCsvRepair]] = None,
        archive_cache: Optional[str] = None,
        archive_cache_dir: Optional[Path] = None,
        preload: bool = False,
    ) -> None:
        """Create an archive-backed source.

        ``ids`` are qualified by the archive path relative to ``root_path``
        and the archive member path.  This keeps names stable within a source
        and avoids collisions when multiple archives contain similarly named
        CSV files.
        """
        super().__init__()
        self.root_path = Path(root_path)
        if isinstance(subfolders, str):
            self.subfolders = [subfolders]
        else:
            self.subfolders = list(subfolders)

        if csv_repair is None:
            self.csv_repair = UserLabelCsvRepair(mode="off")
        elif isinstance(csv_repair, str):
            self.csv_repair = UserLabelCsvRepair(mode=csv_repair)
        else:
            self.csv_repair = csv_repair

        self._csv_repair_diagnostics: dict[SampleId, CsvRepairDiagnostic] = {}
        self._archive_cache_mode = self._normalize_archive_cache(archive_cache, preload)
        self._archive_cache_dir = self._resolve_archive_cache_dir(archive_cache_dir)
        self._memory_archive_cache: dict[tuple[Path, str], bytes] = {}
        self.experiment_cards = {}
        self.exp_cards = self.experiment_cards

        self._files = self._build_file_index()
        self._files_by_id = {meta["id"]: meta for meta in self._files}
        self._set_ids(meta["id"] for meta in self._files)

        if preload:
            self.populate_archive_cache()

    # -----------------------------
    # Layer 1: file registry
    # -----------------------------

    def _build_file_index(self) -> list[dict]:
        files = []

        for folder in self.subfolders:
            for archive_path in self._iter_archives(folder):
                domain = self._archive_domain(archive_path)
                with tarfile.open(archive_path, mode="r:gz") as archive:
                    for member in archive.getmembers():
                        if not member.isfile():
                            continue

                        member_path = Path(member.name)
                        card_match = self._card_pattern.fullmatch(member_path.name)
                        if card_match:
                            self._store_experiment_card(archive, member, domain, card_match)
                            continue

                        match = self._pattern.fullmatch(member_path.name)
                        if not match:
                            continue

                        exp_card_id_str, classname, date_str, time_str = match.groups()
                        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H-%M-%S")

                        files.append({
                            "path": member.name,
                            "filename": member_path.name,
                            "archive_path": archive_path,
                            "domain": domain,
                            "classname": classname.lower(),
                            "exp_card_id": int(exp_card_id_str),
                            "date": date_str,
                            "time": time_str,
                            "datetime": dt,
                        })

        files.sort(key=lambda item: item["datetime"])
        for meta in files:
            meta["id"] = self._make_sample_id(meta)
        return files

    def _make_sample_id(self, meta: dict) -> str:
        return meta['filename'][:-len('.csv')]

    def _store_experiment_card(self, archive, member: tarfile.TarInfo, domain: str, card_match) -> None:
        extracted = archive.extractfile(member)
        if extracted is None:
            return

        card_id = int(card_match.group(1))
        domain_cards = self.experiment_cards.setdefault(domain, {})
        domain_cards[card_id] = json.loads(extracted.read().decode("utf-8"))

    def _iter_archives(self, folder: str):
        folder_path = self.root_path / folder

        if self._is_supported_archive(folder_path):
            yield folder_path
            return

        for suffix in self._archive_suffixes:
            archive_path = self.root_path / f"{folder}{suffix}"
            if archive_path.is_file():
                yield archive_path
                return

        if not folder_path.is_dir():
            return

        archive_paths = []
        for suffix in self._archive_suffixes:
            archive_paths.extend(folder_path.glob(f"*{suffix}"))

        for archive_path in sorted(set(archive_paths)):
            if archive_path.is_file():
                yield archive_path

    @classmethod
    def _is_supported_archive(cls, path: Path) -> bool:
        path_str = str(path)
        return path.is_file() and any(path_str.endswith(suffix) for suffix in cls._archive_suffixes)

    @classmethod
    def _archive_domain(cls, archive_path: Path) -> str:
        name = archive_path.name
        for suffix in cls._archive_suffixes:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return archive_path.stem

    @staticmethod
    def _normalize_archive_cache(archive_cache: Optional[str], preload: bool) -> Optional[str]:
        if archive_cache is None:
            return "disk" if preload else None

        mode = archive_cache.lower()
        aliases = {
            "off": None,
            "none": None,
            "false": None,
            "memory": "memory",
            "ram": "memory",
            "disk": "disk",
            "dir": "disk",
        }
        if mode not in aliases:
            raise ValueError("archive_cache must be one of None, 'memory', or 'disk'")
        return aliases[mode]

    def _resolve_archive_cache_dir(self, archive_cache_dir: Optional[Path]) -> Optional[Path]:
        if self._archive_cache_mode != "disk":
            return None
        if archive_cache_dir is not None:
            return Path(archive_cache_dir)
        cache_parent = self.root_path if self.root_path.is_dir() else self.root_path.parent
        return cache_parent / ".esmell_archive_cache"

    def _archive_fingerprint(self, archive_path: Path) -> str:
        stat = archive_path.stat()
        key = f"{archive_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        return sha256(key.encode("utf-8")).hexdigest()[:20]

    def _member_cache_path(self, archive_path: Path, member_path: str) -> Path:
        if self._archive_cache_dir is None:
            raise RuntimeError("archive cache directory is not configured")

        archive_key = self._archive_fingerprint(archive_path)
        member_key = sha256(member_path.encode("utf-8")).hexdigest()
        suffix = Path(member_path).suffix or ".csv"
        return self._archive_cache_dir / archive_key / f"{member_key}{suffix}"

    @staticmethod
    def _cache_key(archive_path: Path, member_path: str) -> tuple[Path, str]:
        return Path(archive_path).resolve(), member_path

    def _iter_indexed_members_by_archive(self):
        members_by_archive: dict[Path, set[str]] = {}
        for meta in self._files:
            archive_path = Path(meta["archive_path"])
            members_by_archive.setdefault(archive_path, set()).add(meta["path"])

        for archive_path, member_paths in members_by_archive.items():
            yield archive_path, member_paths

    def _populate_archive_cache(self, archive_path: Path, member_paths: set[str]) -> int:
        cached_count = 0
        remaining = set(member_paths)

        if self._archive_cache_mode == "memory":
            remaining = {
                member_path
                for member_path in remaining
                if self._cache_key(archive_path, member_path)
                not in self._memory_archive_cache
            }
            cached_count = len(member_paths) - len(remaining)
        elif self._archive_cache_mode == "disk":
            assert self._archive_cache_dir is not None
            self._archive_cache_dir.mkdir(parents=True, exist_ok=True)
            remaining = {
                member_path
                for member_path in remaining
                if not self._member_cache_path(archive_path, member_path).is_file()
            }
            cached_count = len(member_paths) - len(remaining)

        if not remaining:
            return cached_count

        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                if not member.isfile() or member.name not in remaining:
                    continue

                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                data = extracted.read()

                if self._archive_cache_mode == "memory":
                    self._memory_archive_cache[
                        self._cache_key(archive_path, member.name)
                    ] = data
                elif self._archive_cache_mode == "disk":
                    cache_path = self._member_cache_path(archive_path, member.name)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(data)

                cached_count += 1
                remaining.remove(member.name)
                if not remaining:
                    break

        return cached_count

    def populate_archive_cache(self) -> int:
        """Preload indexed CSV members into the configured archive cache."""
        if self._archive_cache_mode is None:
            return 0

        cached_count = 0
        for archive_path, member_paths in self._iter_indexed_members_by_archive():
            cached_count += self._populate_archive_cache(archive_path, member_paths)
        return cached_count

    def clear_memory_archive_cache(self) -> int:
        """Release extracted CSV members held by the in-memory archive cache."""
        released_count = len(self._memory_archive_cache)
        self._memory_archive_cache.clear()
        return released_count

    def _read_member_bytes(self, meta: dict) -> bytes:
        archive_path = Path(meta["archive_path"])
        member_path = meta["path"]

        if self._archive_cache_mode == "memory":
            cache_key = self._cache_key(archive_path, member_path)
            cached = self._memory_archive_cache.get(cache_key)
            if cached is not None:
                return cached

            with tarfile.open(archive_path, mode="r:gz") as archive:
                extracted = archive.extractfile(member_path)
                if extracted is None:
                    raise FileNotFoundError(member_path)
                data = extracted.read()
            self._memory_archive_cache[cache_key] = data
            return data

        if self._archive_cache_mode == "disk":
            cache_path = self._member_cache_path(archive_path, member_path)
            if cache_path.is_file():
                return cache_path.read_bytes()

            with tarfile.open(archive_path, mode="r:gz") as archive:
                extracted = archive.extractfile(member_path)
                if extracted is None:
                    raise FileNotFoundError(member_path)
                data = extracted.read()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)
            return data

        with tarfile.open(archive_path, mode="r:gz") as archive:
            extracted = archive.extractfile(member_path)
            if extracted is None:
                raise FileNotFoundError(member_path)
            return extracted.read()

    # -----------------------------
    # Source protocol
    # -----------------------------

    def metadata(self, sample_id: SampleId) -> dict:
        try:
            return self._files_by_id[sample_id]
        except KeyError as exc:
            raise KeyError(f"unknown sample id: {sample_id!r}") from exc

    def label(self, sample_id: SampleId):
        return self.metadata(sample_id)["classname"]

    def get_csv_repair_diagnostics(self, sample_id: Optional[SampleId] = None):
        if sample_id is not None:
            diagnostic = self._csv_repair_diagnostics.get(sample_id)
            return None if diagnostic is None else diagnostic.as_dict()

        return {
            sample_id: diagnostic.as_dict()
            for sample_id, diagnostic in self._csv_repair_diagnostics.items()
        }

    def load_csv(self, sample_id: SampleId) -> pd.DataFrame:
        """Lazily load one raw dataframe by stable sample ID."""
        meta = self.metadata(sample_id)
        data = self._read_member_bytes(meta)

        if self.csv_repair.mode == "off":
            df = pd.read_csv(BytesIO(data), sep=",", decimal=".", encoding="utf-8")
        else:
            text = data.decode("utf-8")
            try:
                text, diagnostic = self.csv_repair.prepare(text, source=meta["filename"])
            except CsvRepairError as exc:
                self._csv_repair_diagnostics[sample_id] = exc.diagnostic
                raise
            self._csv_repair_diagnostics[sample_id] = diagnostic
            df = pd.read_csv(StringIO(text), sep=",", decimal=".")

        return df


__all__ = ["EsmellDataset"]

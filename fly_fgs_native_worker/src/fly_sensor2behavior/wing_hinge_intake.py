"""Fail-closed, non-downloading intake for the Melis wing-hinge dataset.

This module only inspects a local file supplied by the caller.  It never
downloads the CaltechDATA artifact.  A successfully hashed and structurally
valid file is still only a ``candidate_unreviewed`` artifact and can never, by
itself, pass or unblock ``hinge.heldout_wing_prediction``.

The HDF5 dependency is deliberately optional and imported only by metadata
inspection functions.  Hashing, topology validation, split construction, and
overlap auditing remain pure-Python so their scientific contracts can be
tested with manufactured metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


# Primary-source lock.  The content endpoint redirects to an expiring object
# URL; callers must not persist or treat that redirect as source identity.
CALTECH_RECORD_URI = "https://data.caltech.edu/records/aypcy-ck464"
CALTECH_RECORD_API_URI = "https://data.caltech.edu/api/records/aypcy-ck464"
CALTECH_FILE_API_URI = (
    "https://data.caltech.edu/api/records/aypcy-ck464/files/"
    "main_muscle_and_wing_data.h5"
)
CALTECH_CONTENT_URI = CALTECH_FILE_API_URI + "/content"
CALTECH_DOI = "10.22002/aypcy-ck464"
SOURCE_FILE_NAME = "main_muscle_and_wing_data.h5"
SOURCE_SIZE_BYTES = 2_642_932_080
SOURCE_MD5 = "8aa8629237b5cc7f845c8fd23872c815"
SOURCE_LICENSE = "CC0-1.0"
SOURCE_LICENSE_URI = "https://creativecommons.org/publicdomain/zero/1.0/legalcode"
UPSTREAM_CODE_URI = "https://github.com/FlyRanch/wing-hinge-cnn"
UPSTREAM_CODE_COMMIT = "cd5081aac460754b8ff3d6426835920e43542f95"
N_WBS_DTYPE = "int64"

CANDIDATE_STATUS = "candidate_unreviewed"
HINGE_GATE_ID = "hinge.heldout_wing_prediction"
DATE_GROUPED_SPLIT_STRATEGY = "whole_acquisition_date_sha256_rank_v1"
DEFAULT_SPLIT_SEED = "melis-wing-hinge-heldout-v2"
LEAKY_FIRST_30_SPLIT_STRATEGY = "first_30_wingbeats_per_movie"
CAUSAL_WINDOW_ALIGNMENT = "completed_9_wingbeats_before_target_v2"
PUBLISHED_ACAUSAL_WINDOW_ALIGNMENT = "leading_9_rows_starting_at_target"

EXPECTED_SESSION_COUNT = 74
EXPECTED_ACQUISITION_DATE_COUNT = 33
EXPECTED_MOVIE_COUNT = 377

INPUT_WINDOW_WINGBEATS = 9
CANONICAL_INPUT_OFFSETS = tuple(range(-INPUT_WINDOW_WINGBEATS, 0))
CANONICAL_TARGET_OFFSET = 0
CANONICAL_FUTURE_INPUT_COUNT = 0
CANONICAL_MUSCLE_FEATURES = (
    "b1",
    "b2",
    "b3",
    "i1",
    "i2",
    "iii1",
    "iii24",
    "iii3",
    "hg1",
    "hg2",
    "hg3",
    "hg4",
    "freq",
)
CANONICAL_INPUT_SHAPE = (INPUT_WINDOW_WINGBEATS, len(CANONICAL_MUSCLE_FEATURES))
CANONICAL_TARGET_COMPONENTS = (
    ("a_theta_L", 20),
    ("a_eta_L", 24),
    ("a_phi_L", 16),
    ("a_xi_L", 20),
)
CANONICAL_TARGET_WIDTH = sum(width for _name, width in CANONICAL_TARGET_COMPONENTS)
CANONICAL_TARGET_SHAPE = (CANONICAL_TARGET_WIDTH,)

SESSION_ID_PATTERN = re.compile(
    r"^Session_(?P<day>[0-9]{2})_(?P<month>[0-9]{2})_"
    r"(?P<year>[0-9]{4})_(?P<hour>[0-9]{2})_(?P<minute>[0-9]{2})$"
)
MOVIE_ID_PATTERN = re.compile(r"^mov_(?P<index>[1-9][0-9]*)$")
_MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

ROOT_DATASET_NAMES = frozenset(
    {
        "N_mov",
        "N_triggers",
        "b1_raw",
        "b2_raw",
        "b3_raw",
        "ca_scaling",
        "hg1_raw",
        "hg2_raw",
        "hg3_raw",
        "hg4_raw",
        "i1_raw",
        "i2_raw",
        "iii1_raw",
        "iii24_raw",
        "iii3_raw",
        "muscle_file_loc",
        "muscle_file_name",
        "nm_raw",
        "pr_raw",
        "t_vec",
        "tpd_raw",
        "trigger_t",
        "wingkin_file_loc",
        "wingkin_file_name",
    }
)

PER_WINGBEAT_ARRAYS = frozenset(
    {
        "T",
        "time_wb",
        "freq",
        "t_wbs",
        *("%s_wbs" % name for name in CANONICAL_MUSCLE_FEATURES[:-1]),
    }
)
SPATIAL_COEFFICIENT_ARRAYS = frozenset(
    "a_%s_%s" % (axis, side)
    for axis in ("x", "y", "z")
    for side in ("L", "R")
)
ANGULAR_COEFFICIENT_WIDTHS = {
    "a_theta_L": 20,
    "a_theta_R": 20,
    "a_eta_L": 24,
    "a_eta_R": 24,
    "a_phi_L": 16,
    "a_phi_R": 16,
    "a_xi_L": 20,
    "a_xi_R": 20,
}
REQUIRED_MOVIE_ARRAYS = frozenset(
    set(PER_WINGBEAT_ARRAYS)
    | {"s1_s2"}
    | set(SPATIAL_COEFFICIENT_ARRAYS)
    | set(ANGULAR_COEFFICIENT_WIDTHS)
)

RAW_SEQUENCE_ARRAYS = frozenset(
    {
        *("%s_%s" % (angle, part) for angle in ("theta", "eta", "phi") for part in (
            "head",
            "thorax",
            "abdomen",
            "wing_L",
            "wing_R",
        )),
        "xi_wing_L",
        "xi_wing_R",
        *("%s_%s" % (axis, part) for axis in ("x", "y", "z") for part in (
            "head",
            "thorax",
            "abdomen",
            "wing_L",
            "wing_R",
        )),
    }
)
_DECONVOLUTION_PATTERN = re.compile(
    r"^(?:b1|b2|b3|i1|i2|iii1|iii24|iii3|hg1|hg2|hg3|hg4)_"
    r"wbs_deconv_0\.005_(?:0\.01|0\.025|0\.05|0\.075|0\.1|0\.125|"
    r"0\.15|0\.175|0\.2)$"
)

DEFAULT_SPLIT_FRACTIONS = (
    ("calibration", 0.70),
    ("validation", 0.15),
    ("held_out", 0.15),
)
REQUIRED_SPLIT_NAMES = tuple(name for name, _fraction in DEFAULT_SPLIT_FRACTIONS)


class WingHingeIntakeError(ValueError):
    """Raised when local evidence violates the locked intake contract."""


@dataclass(frozen=True)
class FileDigest:
    """Observed digests for one immutable regular-file read."""

    path: str
    size_bytes: int
    md5: str
    sha256: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class CandidateReceipt:
    """Unreviewed local observation of the primary-source artifact."""

    path: str
    size_bytes: int
    expected_md5: str
    observed_md5: str
    observed_sha256: str
    observed_device: int
    observed_inode: int
    observed_mtime_ns: int
    observed_ctime_ns: int
    status: str = CANDIDATE_STATUS
    hinge_gate_id: str = HINGE_GATE_ID
    hinge_gate_unblocked: bool = False
    source_doi: str = CALTECH_DOI
    source_file_name: str = SOURCE_FILE_NAME
    source_record_uri: str = CALTECH_RECORD_URI
    source_record_api_uri: str = CALTECH_RECORD_API_URI
    source_file_api_uri: str = CALTECH_FILE_API_URI
    source_content_uri: str = CALTECH_CONTENT_URI
    source_license: str = SOURCE_LICENSE
    source_license_uri: str = SOURCE_LICENSE_URI
    upstream_code_uri: str = UPSTREAM_CODE_URI
    upstream_code_commit: str = UPSTREAM_CODE_COMMIT

    def __post_init__(self) -> None:
        if self.status != CANDIDATE_STATUS:
            raise WingHingeIntakeError("candidate status must remain unreviewed")
        if (
            not isinstance(self.path, str)
            or not self.path
            or os.path.abspath(self.path) != self.path
        ):
            raise WingHingeIntakeError("candidate receipt path must be absolute")
        if self.hinge_gate_id != HINGE_GATE_ID or self.hinge_gate_unblocked:
            raise WingHingeIntakeError(
                "%s cannot be unblocked by dataset intake" % HINGE_GATE_ID
            )
        if self.size_bytes != SOURCE_SIZE_BYTES:
            raise WingHingeIntakeError("candidate receipt has an unexpected byte count")
        if self.expected_md5 != SOURCE_MD5 or self.observed_md5 != SOURCE_MD5:
            raise WingHingeIntakeError("candidate receipt does not match the source MD5")
        if not _SHA256_PATTERN.fullmatch(self.observed_sha256):
            raise WingHingeIntakeError("observed SHA-256 must be lowercase hexadecimal")
        identity_values = (
            self.observed_device,
            self.observed_inode,
            self.observed_mtime_ns,
            self.observed_ctime_ns,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in identity_values
        ):
            raise WingHingeIntakeError(
                "candidate receipt file identity fields must be non-negative integers"
            )
        locked = (
            (self.source_doi, CALTECH_DOI),
            (self.source_file_name, SOURCE_FILE_NAME),
            (self.source_record_uri, CALTECH_RECORD_URI),
            (self.source_record_api_uri, CALTECH_RECORD_API_URI),
            (self.source_file_api_uri, CALTECH_FILE_API_URI),
            (self.source_content_uri, CALTECH_CONTENT_URI),
            (self.source_license, SOURCE_LICENSE),
            (self.source_license_uri, SOURCE_LICENSE_URI),
            (self.upstream_code_uri, UPSTREAM_CODE_URI),
            (self.upstream_code_commit, UPSTREAM_CODE_COMMIT),
        )
        if any(observed != expected for observed, expected in locked):
            raise WingHingeIntakeError("candidate receipt source metadata is not locked")


@dataclass(frozen=True)
class ArrayMetadata:
    name: str
    shape: Tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class MovieMetadata:
    movie_id: str
    n_wbs: int
    n_wbs_shape: Tuple[int, ...]
    n_wbs_dtype: str
    arrays: Tuple[ArrayMetadata, ...]


@dataclass(frozen=True)
class SessionMetadata:
    session_id: str
    root_dataset_names: Tuple[str, ...]
    movies: Tuple[MovieMetadata, ...]


@dataclass(frozen=True)
class HingeTopology:
    sessions: Tuple[SessionMetadata, ...]


@dataclass(frozen=True)
class TopologyExpectations:
    session_count: int = EXPECTED_SESSION_COUNT
    acquisition_date_count: int = EXPECTED_ACQUISITION_DATE_COUNT
    movie_count: int = EXPECTED_MOVIE_COUNT

    def __post_init__(self) -> None:
        values = (self.session_count, self.acquisition_date_count, self.movie_count)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        ):
            raise WingHingeIntakeError("topology expectations must be positive integers")


OFFICIAL_TOPOLOGY_EXPECTATIONS = TopologyExpectations()


@dataclass(frozen=True)
class TopologyReport:
    session_count: int
    acquisition_date_count: int
    movie_count: int
    input_shape: Tuple[int, int]
    target_shape: Tuple[int, ...]
    input_features: Tuple[str, ...]
    target_components: Tuple[Tuple[str, int], ...]
    metadata_only: bool = True


@dataclass(frozen=True)
class WindowSemantics:
    """Input offsets relative to the wingbeat whose kinematics are predicted."""

    alignment: str
    input_offsets: Tuple[int, ...]
    target_offset: int
    future_input_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.alignment, str) or not self.alignment:
            raise WingHingeIntakeError("window alignment must be a non-empty string")
        if len(self.input_offsets) != INPUT_WINDOW_WINGBEATS:
            raise WingHingeIntakeError("window must contain exactly nine input rows")
        if any(isinstance(offset, bool) or not isinstance(offset, int) for offset in self.input_offsets):
            raise WingHingeIntakeError("window offsets must be integers")
        if tuple(sorted(set(self.input_offsets))) != self.input_offsets:
            raise WingHingeIntakeError("window offsets must be unique and increasing")
        if isinstance(self.target_offset, bool) or not isinstance(self.target_offset, int):
            raise WingHingeIntakeError("target offset must be an integer")
        # The v2 protocol issues a prediction at onset of the target beat.
        # Offset zero is therefore same-beat leakage and is counted together
        # with strictly future rows.
        observed_future = sum(
            offset >= self.target_offset for offset in self.input_offsets
        )
        if self.future_input_count != observed_future:
            raise WingHingeIntakeError(
                "future-input count does not match the declared input offsets"
            )


CANONICAL_WINDOW_SEMANTICS = WindowSemantics(
    alignment=CAUSAL_WINDOW_ALIGNMENT,
    input_offsets=CANONICAL_INPUT_OFFSETS,
    target_offset=CANONICAL_TARGET_OFFSET,
    future_input_count=CANONICAL_FUTURE_INPUT_COUNT,
)


@dataclass(frozen=True)
class DateAssignment:
    acquisition_date: str
    split: str


@dataclass(frozen=True)
class DateGroupedSplit:
    seed: str
    strategy: str
    fractions: Tuple[Tuple[str, float], ...]
    assignments: Tuple[DateAssignment, ...]
    window_semantics: WindowSemantics = CANONICAL_WINDOW_SEMANTICS
    rejects_published_first_30_split: bool = True


@dataclass(frozen=True)
class SplitAudit:
    zero_overlap: bool
    all_dates_assigned_once: bool
    all_sessions_assigned_once: bool
    all_movies_assigned_once: bool
    all_windows_assigned_once: bool
    window_alignment: str
    input_offsets: Tuple[int, ...]
    target_offset: int
    future_input_count: int
    published_acausal_window_rejected: bool
    date_counts: Tuple[Tuple[str, int], ...]
    session_counts: Tuple[Tuple[str, int], ...]
    movie_counts: Tuple[Tuple[str, int], ...]
    window_counts: Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class CanonicalJsonReceipt:
    """Canonical JSON plus its content digest for immutable persistence."""

    canonical_json: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_json, str) or not self.canonical_json:
            raise WingHingeIntakeError("canonical JSON receipt must not be empty")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise WingHingeIntakeError("canonical JSON receipt SHA-256 is malformed")
        observed = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        if observed != self.sha256:
            raise WingHingeIntakeError("canonical JSON receipt digest mismatch")


@dataclass(frozen=True)
class CandidateIntake:
    receipt: CandidateReceipt
    topology: HingeTopology
    topology_report: TopologyReport
    split: DateGroupedSplit
    split_audit: SplitAudit
    status: str = CANDIDATE_STATUS
    hinge_gate_id: str = HINGE_GATE_ID
    hinge_gate_unblocked: bool = False

    def __post_init__(self) -> None:
        if self.status != CANDIDATE_STATUS or self.hinge_gate_unblocked:
            raise WingHingeIntakeError(
                "a structurally valid candidate remains unreviewed and cannot unblock the hinge gate"
            )
        if self.hinge_gate_id != HINGE_GATE_ID:
            raise WingHingeIntakeError("unexpected hinge gate identifier")
        if not self.split_audit.zero_overlap:
            raise WingHingeIntakeError("candidate split must have zero overlap")
        if self.split.window_semantics != CANONICAL_WINDOW_SEMANTICS:
            raise WingHingeIntakeError("candidate split must use the causal window contract")
        if self.split_audit.future_input_count != 0:
            raise WingHingeIntakeError("candidate split must contain zero future inputs")
        observed_report = validate_topology(self.topology)
        if observed_report != self.topology_report:
            raise WingHingeIntakeError("candidate topology report is inconsistent")
        observed_audit = audit_date_grouped_split(self.topology, self.split)
        if observed_audit != self.split_audit:
            raise WingHingeIntakeError("candidate split audit is inconsistent")

    @property
    def candidate_receipt_document(self) -> CanonicalJsonReceipt:
        return canonical_candidate_receipt(self.receipt)

    @property
    def split_receipt_document(self) -> CanonicalJsonReceipt:
        return canonical_split_receipt(
            self.receipt,
            self.topology_report,
            self.split,
            self.split_audit,
        )


def _validate_md5(value: str, label: str) -> str:
    if not isinstance(value, str) or not _MD5_PATTERN.fullmatch(value):
        raise WingHingeIntakeError("%s must be a lowercase MD5 digest" % label)
    return value


def _stat_identity(value: os.stat_result) -> Tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _receipt_identity(receipt: CandidateReceipt) -> Tuple[int, int, int, int, int]:
    return (
        receipt.observed_device,
        receipt.observed_inode,
        receipt.size_bytes,
        receipt.observed_mtime_ns,
        receipt.observed_ctime_ns,
    )


def _assert_receipt_file_identity(
    value: os.stat_result, receipt: CandidateReceipt, *, phase: str
) -> None:
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise WingHingeIntakeError(
            "candidate must remain a non-symlink regular file during %s" % phase
        )
    if _stat_identity(value) != _receipt_identity(receipt):
        raise WingHingeIntakeError(
            "candidate file identity or timestamps changed during %s" % phase
        )


def hash_regular_file(
    path: os.PathLike,
    *,
    expected_size: int,
    expected_md5: str,
    chunk_size: int = 4 * 1024 * 1024,
) -> FileDigest:
    """Hash exactly one regular file without following symlinks.

    ``expected_size`` and ``expected_md5`` are caller-supplied so manufactured
    fixtures can exercise this pure core.  Official intake always supplies the
    locked constants above.
    """

    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise WingHingeIntakeError("expected_size must be an integer")
    if expected_size < 0:
        raise WingHingeIntakeError("expected_size must be non-negative")
    _validate_md5(expected_md5, "expected_md5")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise WingHingeIntakeError("chunk_size must be a positive integer")

    candidate = Path(path)
    try:
        before = os.lstat(str(candidate))
    except OSError as exc:
        raise WingHingeIntakeError("candidate file is not accessible: %s" % exc) from exc
    if stat.S_ISLNK(before.st_mode):
        raise WingHingeIntakeError("candidate file must not be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise WingHingeIntakeError("candidate path must be a regular file")
    if before.st_size != expected_size:
        raise WingHingeIntakeError(
            "candidate byte count mismatch: expected %d, observed %d"
            % (expected_size, before.st_size)
        )

    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(candidate), flags)
    except OSError as exc:
        raise WingHingeIntakeError("candidate file could not be opened safely: %s" % exc) from exc

    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    observed_bytes = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise WingHingeIntakeError("opened candidate is not a regular file")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise WingHingeIntakeError("candidate file changed while it was opened")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                observed_bytes += len(block)
                md5.update(block)
                sha256.update(block)
            after = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if observed_bytes != expected_size or after.st_size != expected_size:
        raise WingHingeIntakeError("candidate size changed during hashing")
    if _stat_identity(after) != _stat_identity(before):
        raise WingHingeIntakeError("candidate file changed during hashing")
    observed_md5 = md5.hexdigest()
    if observed_md5 != expected_md5:
        raise WingHingeIntakeError(
            "candidate MD5 mismatch: expected %s, observed %s"
            % (expected_md5, observed_md5)
        )
    return FileDigest(
        path=os.path.abspath(str(candidate)),
        size_bytes=observed_bytes,
        md5=observed_md5,
        sha256=sha256.hexdigest(),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        mtime_ns=int(after.st_mtime_ns),
        ctime_ns=int(after.st_ctime_ns),
    )


def verify_official_candidate_file(path: os.PathLike) -> CandidateReceipt:
    """Hash a local candidate against the exact CaltechDATA artifact."""

    digest = hash_regular_file(
        path,
        expected_size=SOURCE_SIZE_BYTES,
        expected_md5=SOURCE_MD5,
    )
    return CandidateReceipt(
        path=digest.path,
        size_bytes=digest.size_bytes,
        expected_md5=SOURCE_MD5,
        observed_md5=digest.md5,
        observed_sha256=digest.sha256,
        observed_device=digest.device,
        observed_inode=digest.inode,
        observed_mtime_ns=digest.mtime_ns,
        observed_ctime_ns=digest.ctime_ns,
    )


def acquisition_date_for_session(session_id: str) -> str:
    """Return an ISO acquisition date from a locked session identifier."""

    if not isinstance(session_id, str):
        raise WingHingeIntakeError("session identifier must be a string")
    match = SESSION_ID_PATTERN.fullmatch(session_id)
    if match is None:
        raise WingHingeIntakeError("malformed session identifier: %r" % session_id)
    fields = match.groupdict()
    value = "%s_%s_%s_%s_%s" % (
        fields["day"],
        fields["month"],
        fields["year"],
        fields["hour"],
        fields["minute"],
    )
    try:
        timestamp = datetime.strptime(value, "%d_%m_%Y_%H_%M")
    except ValueError as exc:
        raise WingHingeIntakeError(
            "session identifier contains an invalid acquisition timestamp"
        ) from exc
    return timestamp.date().isoformat()


def _expected_array_shapes(n_wbs: int) -> Dict[str, Tuple[int, ...]]:
    shapes: Dict[str, Tuple[int, ...]] = {
        name: (n_wbs,) for name in PER_WINGBEAT_ARRAYS
    }
    shapes["s1_s2"] = (n_wbs, 2)
    shapes.update({name: (20, n_wbs) for name in SPATIAL_COEFFICIENT_ARRAYS})
    shapes.update(
        {name: (width, n_wbs) for name, width in ANGULAR_COEFFICIENT_WIDTHS.items()}
    )
    return shapes


def canonical_movie_metadata(
    movie_id: str,
    n_wbs: int,
    *,
    include_raw_sequences: bool = False,
    raw_sequence_length: Optional[int] = None,
) -> MovieMetadata:
    """Construct canonical manufactured metadata without importing h5py."""

    if isinstance(n_wbs, bool) or not isinstance(n_wbs, int) or n_wbs <= 0:
        raise WingHingeIntakeError("n_wbs must be a positive integer")
    arrays = [
        ArrayMetadata(name=name, shape=shape, dtype="float64")
        for name, shape in sorted(_expected_array_shapes(n_wbs).items())
    ]
    if include_raw_sequences:
        length = raw_sequence_length if raw_sequence_length is not None else n_wbs * 75
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise WingHingeIntakeError("raw_sequence_length must be positive")
        arrays.extend(
            ArrayMetadata(name=name, shape=(length,), dtype="float64")
            for name in sorted(RAW_SEQUENCE_ARRAYS)
        )
    return MovieMetadata(
        movie_id=movie_id,
        n_wbs=n_wbs,
        n_wbs_shape=(),
        n_wbs_dtype=N_WBS_DTYPE,
        arrays=tuple(arrays),
    )


def canonical_session_metadata(
    session_id: str, movies: Iterable[MovieMetadata]
) -> SessionMetadata:
    """Construct a canonical manufactured session for pure contract tests."""

    return SessionMetadata(
        session_id=session_id,
        root_dataset_names=tuple(sorted(ROOT_DATASET_NAMES)),
        movies=tuple(movies),
    )


def _validate_movie(movie: MovieMetadata, session_id: str) -> None:
    match = MOVIE_ID_PATTERN.fullmatch(movie.movie_id) if isinstance(movie.movie_id, str) else None
    if match is None:
        raise WingHingeIntakeError(
            "malformed movie identifier in %s: %r" % (session_id, movie.movie_id)
        )
    if isinstance(movie.n_wbs, bool) or not isinstance(movie.n_wbs, int) or movie.n_wbs <= 0:
        raise WingHingeIntakeError("%s/%s N_wbs must be positive" % (session_id, movie.movie_id))
    if movie.n_wbs_shape != ():
        raise WingHingeIntakeError("%s/%s N_wbs must be scalar" % (session_id, movie.movie_id))
    if movie.n_wbs_dtype != N_WBS_DTYPE:
        raise WingHingeIntakeError(
            "%s/%s N_wbs must be %s"
            % (session_id, movie.movie_id, N_WBS_DTYPE)
        )

    arrays: Dict[str, ArrayMetadata] = {}
    for array in movie.arrays:
        if array.name in arrays:
            raise WingHingeIntakeError(
                "%s/%s has duplicate array %s" % (session_id, movie.movie_id, array.name)
            )
        arrays[array.name] = array
    missing = sorted(REQUIRED_MOVIE_ARRAYS - set(arrays))
    if missing:
        raise WingHingeIntakeError(
            "%s/%s missing canonical arrays: %s"
            % (session_id, movie.movie_id, ", ".join(missing))
        )
    unknown = sorted(
        name
        for name in arrays
        if name not in REQUIRED_MOVIE_ARRAYS
        and name not in RAW_SEQUENCE_ARRAYS
        and _DECONVOLUTION_PATTERN.fullmatch(name) is None
    )
    if unknown:
        raise WingHingeIntakeError(
            "%s/%s has unknown arrays: %s"
            % (session_id, movie.movie_id, ", ".join(unknown))
        )

    expected_shapes = _expected_array_shapes(movie.n_wbs)
    for name, expected_shape in expected_shapes.items():
        array = arrays[name]
        if array.dtype != "float64":
            raise WingHingeIntakeError(
                "%s/%s/%s must be float64" % (session_id, movie.movie_id, name)
            )
        if array.shape != expected_shape:
            raise WingHingeIntakeError(
                "%s/%s/%s shape mismatch: expected %r, observed %r"
                % (session_id, movie.movie_id, name, expected_shape, array.shape)
            )
    for name, array in arrays.items():
        if name in expected_shapes:
            continue
        if array.dtype != "float64":
            raise WingHingeIntakeError(
                "%s/%s/%s must be float64" % (session_id, movie.movie_id, name)
            )
        if name in RAW_SEQUENCE_ARRAYS:
            valid_shape = len(array.shape) == 1 and array.shape[0] > 0
        else:
            valid_shape = array.shape == (movie.n_wbs,)
        if not valid_shape:
            raise WingHingeIntakeError(
                "%s/%s/%s has an invalid auxiliary shape"
                % (session_id, movie.movie_id, name)
            )


def validate_topology(
    topology: HingeTopology,
    *,
    expectations: TopologyExpectations = OFFICIAL_TOPOLOGY_EXPECTATIONS,
) -> TopologyReport:
    """Validate a pure metadata snapshot against the locked HDF5 schema."""

    if not isinstance(topology, HingeTopology):
        raise WingHingeIntakeError("topology must be HingeTopology")
    sessions: Dict[str, SessionMetadata] = {}
    movie_count = 0
    acquisition_dates = set()
    for session in topology.sessions:
        if session.session_id in sessions:
            raise WingHingeIntakeError("duplicate session identifier %s" % session.session_id)
        acquisition_dates.add(acquisition_date_for_session(session.session_id))
        sessions[session.session_id] = session
        root_names = tuple(session.root_dataset_names)
        if len(root_names) != len(set(root_names)):
            raise WingHingeIntakeError("%s has duplicate root datasets" % session.session_id)
        missing_root = sorted(ROOT_DATASET_NAMES - set(root_names))
        unknown_root = sorted(set(root_names) - ROOT_DATASET_NAMES)
        if missing_root or unknown_root:
            raise WingHingeIntakeError(
                "%s root schema mismatch; missing=%s unknown=%s"
                % (session.session_id, missing_root, unknown_root)
            )
        movie_ids = set()
        for movie in session.movies:
            if movie.movie_id in movie_ids:
                raise WingHingeIntakeError(
                    "%s has duplicate movie identifier %s"
                    % (session.session_id, movie.movie_id)
                )
            movie_ids.add(movie.movie_id)
            _validate_movie(movie, session.session_id)
            movie_count += 1

    observed = (len(sessions), len(acquisition_dates), movie_count)
    expected = (
        expectations.session_count,
        expectations.acquisition_date_count,
        expectations.movie_count,
    )
    if observed != expected:
        raise WingHingeIntakeError(
            "topology count mismatch: expected sessions/dates/movies %r, observed %r"
            % (expected, observed)
        )
    if CANONICAL_INPUT_SHAPE != (9, 13) or CANONICAL_TARGET_SHAPE != (80,):
        raise WingHingeIntakeError("canonical model schema constants are inconsistent")
    return TopologyReport(
        session_count=observed[0],
        acquisition_date_count=observed[1],
        movie_count=observed[2],
        input_shape=CANONICAL_INPUT_SHAPE,
        target_shape=CANONICAL_TARGET_SHAPE,
        input_features=CANONICAL_MUSCLE_FEATURES,
        target_components=CANONICAL_TARGET_COMPONENTS,
    )


def validate_split_strategy(strategy: str) -> str:
    """Reject the published within-movie first-30 validation split."""

    if strategy == LEAKY_FIRST_30_SPLIT_STRATEGY:
        raise WingHingeIntakeError(
            "the published first-30 split is rejected: adjacent 9-wingbeat windows overlap "
            "and every movie appears in both train and validation"
        )
    if strategy != DATE_GROUPED_SPLIT_STRATEGY:
        raise WingHingeIntakeError("unknown or unsafe split strategy: %r" % strategy)
    return strategy


def validate_window_semantics(semantics: WindowSemantics) -> WindowSemantics:
    """Require nine completed inputs strictly before the predicted wingbeat.

    This check is independent of the split-strategy name.  Renaming the
    published ``x[j:j+9] -> y[j]`` policy therefore cannot conceal its one
    same-beat and eight future input rows.
    """

    if not isinstance(semantics, WindowSemantics):
        raise WingHingeIntakeError("split must include WindowSemantics")
    if semantics.alignment == PUBLISHED_ACAUSAL_WINDOW_ALIGNMENT:
        raise WingHingeIntakeError(
            "published x[j:j+9] -> y[j] alignment has nine same-or-future inputs"
        )
    if semantics != CANONICAL_WINDOW_SEMANTICS:
        raise WingHingeIntakeError(
            "window must be nine completed rows strictly before the target wingbeat"
        )
    if semantics.future_input_count != 0:
        raise WingHingeIntakeError("causal window must have exactly zero future inputs")
    return semantics


def _validated_fractions(
    fractions: Sequence[Tuple[str, float]], number_of_dates: int
) -> Tuple[Tuple[str, float], ...]:
    values = tuple(fractions)
    if not values:
        raise WingHingeIntakeError("split fractions must not be empty")
    names = [item[0] for item in values]
    if len(names) != len(set(names)):
        raise WingHingeIntakeError("split names must be unique")
    if any(not isinstance(name, str) or not name for name in names):
        raise WingHingeIntakeError("split names must be non-empty strings")
    if set(names) != set(REQUIRED_SPLIT_NAMES):
        raise WingHingeIntakeError(
            "splits must be calibration, validation, and held_out"
        )
    parsed = []
    for name, fraction in values:
        if isinstance(fraction, bool):
            raise WingHingeIntakeError("split fractions must be numeric")
        try:
            number = float(fraction)
        except (TypeError, ValueError) as exc:
            raise WingHingeIntakeError("split fractions must be numeric") from exc
        if not math.isfinite(number) or number <= 0.0:
            raise WingHingeIntakeError("split fractions must be finite and positive")
        parsed.append((name, number))
    if not math.isclose(sum(value for _name, value in parsed), 1.0, abs_tol=1e-12):
        raise WingHingeIntakeError("split fractions must sum to one")
    if number_of_dates < len(parsed):
        raise WingHingeIntakeError("not enough acquisition dates for non-empty splits")
    return tuple(parsed)


def _allocation_counts(
    number_of_dates: int, fractions: Tuple[Tuple[str, float], ...]
) -> Tuple[int, ...]:
    raw = [number_of_dates * fraction for _name, fraction in fractions]
    counts = [int(math.floor(value)) for value in raw]
    remaining = number_of_dates - sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    for empty_index, count in enumerate(tuple(counts)):
        if count != 0:
            continue
        donor = max(range(len(counts)), key=lambda index: counts[index])
        if counts[donor] <= 1:
            raise WingHingeIntakeError("could not allocate a non-empty date split")
        counts[donor] -= 1
        counts[empty_index] += 1
    return tuple(counts)


def build_date_grouped_split(
    topology: HingeTopology,
    *,
    seed: str = DEFAULT_SPLIT_SEED,
    fractions: Sequence[Tuple[str, float]] = DEFAULT_SPLIT_FRACTIONS,
    strategy: str = DATE_GROUPED_SPLIT_STRATEGY,
    window_semantics: WindowSemantics = CANONICAL_WINDOW_SEMANTICS,
) -> DateGroupedSplit:
    """Assign complete acquisition dates by stable SHA-256 rank."""

    validate_split_strategy(strategy)
    validate_window_semantics(window_semantics)
    if not isinstance(seed, str) or not seed:
        raise WingHingeIntakeError("split seed must be a non-empty string")
    dates = sorted(
        {acquisition_date_for_session(session.session_id) for session in topology.sessions}
    )
    parsed_fractions = _validated_fractions(fractions, len(dates))
    counts = _allocation_counts(len(dates), parsed_fractions)
    ranked_dates = sorted(
        dates,
        key=lambda value: (
            hashlib.sha256((seed + "\0" + value).encode("utf-8")).digest(),
            value,
        ),
    )
    assignments = []
    cursor = 0
    for (split_name, _fraction), count in zip(parsed_fractions, counts):
        for acquisition_date in ranked_dates[cursor : cursor + count]:
            assignments.append(DateAssignment(acquisition_date, split_name))
        cursor += count
    if cursor != len(ranked_dates):
        raise WingHingeIntakeError("internal date allocation error")
    return DateGroupedSplit(
        seed=seed,
        strategy=strategy,
        fractions=parsed_fractions,
        assignments=tuple(sorted(assignments, key=lambda item: item.acquisition_date)),
        window_semantics=window_semantics,
    )


def _ensure_pairwise_disjoint(groups: Mapping[str, set], label: str) -> None:
    names = sorted(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise WingHingeIntakeError(
                    "%s overlap between %s and %s" % (label, left, right)
                )


def audit_date_grouped_split(
    topology: HingeTopology, split: DateGroupedSplit
) -> SplitAudit:
    """Prove that dates, sessions, movies, and 9-wingbeat windows do not overlap."""

    validate_split_strategy(split.strategy)
    validate_window_semantics(split.window_semantics)
    if not split.rejects_published_first_30_split:
        raise WingHingeIntakeError("split must explicitly reject the published first-30 policy")
    parsed_fractions = _validated_fractions(
        split.fractions,
        len({acquisition_date_for_session(item.session_id) for item in topology.sessions}),
    )
    split_names = tuple(name for name, _fraction in parsed_fractions)
    assignment_map: Dict[str, str] = {}
    for assignment in split.assignments:
        if assignment.acquisition_date in assignment_map:
            raise WingHingeIntakeError(
                "duplicate acquisition-date assignment %s" % assignment.acquisition_date
            )
        if assignment.split not in split_names:
            raise WingHingeIntakeError("assignment uses an unknown split")
        try:
            datetime.strptime(assignment.acquisition_date, "%Y-%m-%d")
        except ValueError as exc:
            raise WingHingeIntakeError("assignment has an invalid ISO date") from exc
        assignment_map[assignment.acquisition_date] = assignment.split

    expected_dates = {
        acquisition_date_for_session(session.session_id) for session in topology.sessions
    }
    if set(assignment_map) != expected_dates:
        raise WingHingeIntakeError("split assignments do not exactly cover acquisition dates")

    dates_by_split = {name: set() for name in split_names}
    sessions_by_split = {name: set() for name in split_names}
    movies_by_split = {name: set() for name in split_names}
    windows_by_split = {name: set() for name in split_names}
    for acquisition_date, split_name in assignment_map.items():
        dates_by_split[split_name].add(acquisition_date)
    for session in topology.sessions:
        acquisition_date = acquisition_date_for_session(session.session_id)
        split_name = assignment_map[acquisition_date]
        sessions_by_split[split_name].add(session.session_id)
        for movie in session.movies:
            movie_key = (session.session_id, movie.movie_id)
            movies_by_split[split_name].add(movie_key)
            # The key records both the target and every input bound.  Inputs
            # are [target-9, ..., target-1], never same-beat or future rows.
            windows_by_split[split_name].update(
                (
                    session.session_id,
                    movie.movie_id,
                    target_index,
                    target_index + split.window_semantics.input_offsets[0],
                    target_index + split.window_semantics.input_offsets[-1],
                )
                for target_index in range(INPUT_WINDOW_WINGBEATS, movie.n_wbs)
            )

    for label, groups in (
        ("date", dates_by_split),
        ("session", sessions_by_split),
        ("movie", movies_by_split),
        ("window", windows_by_split),
    ):
        _ensure_pairwise_disjoint(groups, label)
    observed_sessions = set().union(*sessions_by_split.values())
    observed_movies = set().union(*movies_by_split.values())
    expected_sessions = {session.session_id for session in topology.sessions}
    expected_movies = {
        (session.session_id, movie.movie_id)
        for session in topology.sessions
        for movie in session.movies
    }
    if observed_sessions != expected_sessions or observed_movies != expected_movies:
        raise WingHingeIntakeError("split audit did not assign every session and movie")

    def counts(groups: Mapping[str, set]) -> Tuple[Tuple[str, int], ...]:
        return tuple((name, len(groups[name])) for name in split_names)

    return SplitAudit(
        zero_overlap=True,
        all_dates_assigned_once=True,
        all_sessions_assigned_once=True,
        all_movies_assigned_once=True,
        all_windows_assigned_once=True,
        window_alignment=split.window_semantics.alignment,
        input_offsets=split.window_semantics.input_offsets,
        target_offset=split.window_semantics.target_offset,
        future_input_count=split.window_semantics.future_input_count,
        published_acausal_window_rejected=True,
        date_counts=counts(dates_by_split),
        session_counts=counts(sessions_by_split),
        movie_counts=counts(movies_by_split),
        window_counts=counts(windows_by_split),
    )


def _load_h5py():
    try:
        import h5py  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise WingHingeIntakeError(
            "HDF5 metadata inspection requires optional dependency 'h5py'"
        ) from exc
    return h5py


def read_hdf5_topology(path: os.PathLike) -> HingeTopology:
    """Read only HDF5 names, shapes, dtypes, and scalar ``N_wbs`` values.

    This function deliberately does not read any bulk dataset values.  It is a
    structural primitive; official intake additionally requires the exact
    size/MD5 receipt produced by :func:`verify_official_candidate_file`.
    """

    h5py = _load_h5py()
    candidate = Path(path)
    try:
        local_stat = os.lstat(str(candidate))
    except OSError as exc:
        raise WingHingeIntakeError("HDF5 candidate is not accessible: %s" % exc) from exc
    if stat.S_ISLNK(local_stat.st_mode) or not stat.S_ISREG(local_stat.st_mode):
        raise WingHingeIntakeError("HDF5 candidate must be a non-symlink regular file")

    sessions = []
    try:
        with h5py.File(str(candidate), "r") as handle:
            for session_id in sorted(handle.keys()):
                node = handle[session_id]
                if not isinstance(node, h5py.Group):
                    raise WingHingeIntakeError(
                        "top-level child %s must be an HDF5 group" % session_id
                    )
                root_dataset_names = []
                movie_groups = []
                for child_name in sorted(node.keys()):
                    child = node[child_name]
                    if MOVIE_ID_PATTERN.fullmatch(child_name):
                        if not isinstance(child, h5py.Group):
                            raise WingHingeIntakeError(
                                "%s/%s must be an HDF5 group" % (session_id, child_name)
                            )
                        movie_groups.append((child_name, child))
                    else:
                        if child_name not in ROOT_DATASET_NAMES:
                            raise WingHingeIntakeError(
                                "%s has unknown child %s" % (session_id, child_name)
                            )
                        if not isinstance(child, h5py.Dataset):
                            raise WingHingeIntakeError(
                                "%s/%s must be an HDF5 dataset" % (session_id, child_name)
                            )
                        root_dataset_names.append(child_name)
                movies = []
                for movie_id, movie_group in movie_groups:
                    child_names = set(movie_group.keys())
                    if "N_wbs" not in child_names:
                        raise WingHingeIntakeError(
                            "%s/%s is missing scalar N_wbs" % (session_id, movie_id)
                        )
                    n_wbs_node = movie_group["N_wbs"]
                    if not isinstance(n_wbs_node, h5py.Dataset):
                        raise WingHingeIntakeError(
                            "%s/%s/N_wbs must be a dataset" % (session_id, movie_id)
                        )
                    if tuple(n_wbs_node.shape) != ():
                        raise WingHingeIntakeError(
                            "%s/%s/N_wbs must be scalar" % (session_id, movie_id)
                        )
                    n_wbs = int(n_wbs_node[()])
                    arrays = []
                    for array_name in sorted(child_names - {"N_wbs"}):
                        array_node = movie_group[array_name]
                        if not isinstance(array_node, h5py.Dataset):
                            raise WingHingeIntakeError(
                                "%s/%s/%s must be a dataset"
                                % (session_id, movie_id, array_name)
                            )
                        arrays.append(
                            ArrayMetadata(
                                name=array_name,
                                shape=tuple(int(value) for value in array_node.shape),
                                dtype=str(array_node.dtype),
                            )
                        )
                    movies.append(
                        MovieMetadata(
                            movie_id=movie_id,
                            n_wbs=n_wbs,
                            n_wbs_shape=tuple(n_wbs_node.shape),
                            n_wbs_dtype=str(n_wbs_node.dtype),
                            arrays=tuple(arrays),
                        )
                    )
                sessions.append(
                    SessionMetadata(
                        session_id=session_id,
                        root_dataset_names=tuple(root_dataset_names),
                        movies=tuple(movies),
                    )
                )
    except WingHingeIntakeError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise WingHingeIntakeError("could not inspect HDF5 metadata: %s" % exc) from exc
    return HingeTopology(sessions=tuple(sessions))


def inspect_verified_candidate(
    path: os.PathLike,
    receipt: CandidateReceipt,
    *,
    expectations: TopologyExpectations = OFFICIAL_TOPOLOGY_EXPECTATIONS,
) -> Tuple[HingeTopology, TopologyReport]:
    """Inspect metadata only after an exact official-file receipt exists."""

    absolute_path = os.path.abspath(str(path))
    if receipt.path != absolute_path:
        raise WingHingeIntakeError("receipt path does not match the inspected candidate")
    if receipt.status != CANDIDATE_STATUS or receipt.hinge_gate_unblocked:
        raise WingHingeIntakeError("only an unreviewed candidate receipt is accepted")
    try:
        before = os.lstat(absolute_path)
    except OSError as exc:
        raise WingHingeIntakeError(
            "candidate is inaccessible before HDF5 inspection: %s" % exc
        ) from exc
    _assert_receipt_file_identity(before, receipt, phase="pre-inspection check")
    topology = read_hdf5_topology(absolute_path)
    try:
        after = os.lstat(absolute_path)
    except OSError as exc:
        raise WingHingeIntakeError(
            "candidate is inaccessible after HDF5 inspection: %s" % exc
        ) from exc
    _assert_receipt_file_identity(after, receipt, phase="post-inspection check")
    if _stat_identity(after) != _stat_identity(before):
        raise WingHingeIntakeError("candidate changed during HDF5 inspection")
    report = validate_topology(topology, expectations=expectations)
    return topology, report


def _canonical_json_receipt(kind: str, payload: Mapping[str, object]) -> CanonicalJsonReceipt:
    document = {
        "kind": kind,
        "schema_version": "1.0.0",
        **payload,
    }
    try:
        encoded = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise WingHingeIntakeError("receipt is not canonically serializable") from exc
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return CanonicalJsonReceipt(canonical_json=encoded, sha256=digest)


def canonical_candidate_receipt(receipt: CandidateReceipt) -> CanonicalJsonReceipt:
    """Serialize an unreviewed source receipt deterministically."""

    if not isinstance(receipt, CandidateReceipt):
        raise WingHingeIntakeError("candidate receipt has the wrong type")
    return _canonical_json_receipt(
        "melis_wing_hinge_candidate_receipt",
        {"receipt": asdict(receipt)},
    )


def canonical_split_receipt(
    receipt: CandidateReceipt,
    topology_report: TopologyReport,
    split: DateGroupedSplit,
    audit: SplitAudit,
) -> CanonicalJsonReceipt:
    """Bind a permanent split and leakage audit to the exact source receipt."""

    if not isinstance(topology_report, TopologyReport):
        raise WingHingeIntakeError("topology report has the wrong type")
    if not isinstance(split, DateGroupedSplit) or not isinstance(audit, SplitAudit):
        raise WingHingeIntakeError("split receipt inputs have the wrong type")
    validate_split_strategy(split.strategy)
    validate_window_semantics(split.window_semantics)
    if not audit.zero_overlap or audit.future_input_count != 0:
        raise WingHingeIntakeError("only a zero-overlap causal split can be serialized")
    expected_semantics = (
        split.window_semantics.alignment,
        split.window_semantics.input_offsets,
        split.window_semantics.target_offset,
        split.window_semantics.future_input_count,
    )
    observed_semantics = (
        audit.window_alignment,
        audit.input_offsets,
        audit.target_offset,
        audit.future_input_count,
    )
    if observed_semantics != expected_semantics:
        raise WingHingeIntakeError("split audit window semantics are inconsistent")
    if not all(
        (
            audit.all_dates_assigned_once,
            audit.all_sessions_assigned_once,
            audit.all_movies_assigned_once,
            audit.all_windows_assigned_once,
            audit.published_acausal_window_rejected,
        )
    ):
        raise WingHingeIntakeError("split audit is incomplete")
    source_document = canonical_candidate_receipt(receipt)
    return _canonical_json_receipt(
        "melis_wing_hinge_permanent_split_receipt",
        {
            "candidate_receipt_sha256": source_document.sha256,
            "split": asdict(split),
            "split_audit": asdict(audit),
            "topology_report": asdict(topology_report),
        },
    )


def intake_official_candidate(
    path: os.PathLike,
    *,
    split_seed: str = DEFAULT_SPLIT_SEED,
    fractions: Sequence[Tuple[str, float]] = DEFAULT_SPLIT_FRACTIONS,
) -> CandidateIntake:
    """Hash, inspect, and split one local official-file candidate.

    The return value is intentionally non-promotional: even complete success
    remains ``candidate_unreviewed`` and leaves the hinge gate blocked.
    """

    _load_h5py()  # Fail before hashing a multi-gigabyte file when unavailable.
    receipt = verify_official_candidate_file(path)
    topology, report = inspect_verified_candidate(path, receipt)
    split = build_date_grouped_split(
        topology,
        seed=split_seed,
        fractions=fractions,
        strategy=DATE_GROUPED_SPLIT_STRATEGY,
    )
    audit = audit_date_grouped_split(topology, split)
    return CandidateIntake(
        receipt=receipt,
        topology=topology,
        topology_report=report,
        split=split,
        split_audit=audit,
    )

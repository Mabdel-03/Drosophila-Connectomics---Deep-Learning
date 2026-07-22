"""Fail-closed contract tests for the Melis wing-hinge dataset intake.

The official artifact is intentionally not downloaded in CI.  The scientific
topology and split rules are exercised with manufactured metadata, while the
byte-level receipt path is exercised with a small local file.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
import types
from dataclasses import replace
from datetime import date, timedelta

import pytest

from fly_sensor2behavior import wing_hinge_intake as hinge


MOVIE_IDS = ("mov_1", "mov_3", "mov_6", "mov_10", "mov_15", "mov_21")
N_WBS = 12


@pytest.fixture(scope="module")
def official_topology() -> hinge.HingeTopology:
    """Manufacture the observed 74-session/33-date/377-movie topology."""

    acquisition_dates = [date(2021, 1, 1) + timedelta(days=index) for index in range(33)]
    sessions_per_date = (3,) * 8 + (2,) * 25
    sessions = []
    session_index = 0
    for acquisition_date, session_count in zip(acquisition_dates, sessions_per_date):
        for within_date in range(session_count):
            session_id = "Session_%s_%02d_%02d" % (
                acquisition_date.strftime("%d_%m_%Y"),
                8 + within_date,
                (17 * within_date) % 60,
            )
            movie_count = 6 if session_index < 7 else 5
            movies = tuple(
                hinge.canonical_movie_metadata(movie_id, N_WBS)
                for movie_id in MOVIE_IDS[:movie_count]
            )
            sessions.append(hinge.canonical_session_metadata(session_id, movies))
            session_index += 1

    topology = hinge.HingeTopology(tuple(sessions))
    assert len(topology.sessions) == 74
    assert sum(len(session.movies) for session in topology.sessions) == 377
    return topology


def _replace_first_session(
    topology: hinge.HingeTopology, session: hinge.SessionMetadata
) -> hinge.HingeTopology:
    return hinge.HingeTopology((session,) + topology.sessions[1:])


def _replace_first_movie(
    topology: hinge.HingeTopology, movie: hinge.MovieMetadata
) -> hinge.HingeTopology:
    first = topology.sessions[0]
    changed = replace(first, movies=(movie,) + first.movies[1:])
    return _replace_first_session(topology, changed)


def _official_receipt(
    path: str = "/tmp/main_muscle_and_wing_data.h5",
    *,
    file_stat=None,
) -> hinge.CandidateReceipt:
    identity = file_stat or type(
        "ManufacturedStat",
        (),
        {"st_dev": 0, "st_ino": 0, "st_mtime_ns": 0, "st_ctime_ns": 0},
    )()
    return hinge.CandidateReceipt(
        path=path,
        size_bytes=hinge.SOURCE_SIZE_BYTES,
        expected_md5=hinge.SOURCE_MD5,
        observed_md5=hinge.SOURCE_MD5,
        observed_sha256="0" * 64,
        observed_device=int(identity.st_dev),
        observed_inode=int(identity.st_ino),
        observed_mtime_ns=int(identity.st_mtime_ns),
        observed_ctime_ns=int(identity.st_ctime_ns),
    )


def test_primary_source_lock_and_model_shapes_are_exact() -> None:
    assert hinge.CALTECH_DOI == "10.22002/aypcy-ck464"
    assert hinge.CALTECH_RECORD_URI == "https://data.caltech.edu/records/aypcy-ck464"
    assert hinge.CALTECH_FILE_API_URI.endswith("/files/main_muscle_and_wing_data.h5")
    assert hinge.CALTECH_CONTENT_URI == hinge.CALTECH_FILE_API_URI + "/content"
    assert hinge.SOURCE_FILE_NAME == "main_muscle_and_wing_data.h5"
    assert hinge.SOURCE_SIZE_BYTES == 2_642_932_080
    assert hinge.SOURCE_MD5 == "8aa8629237b5cc7f845c8fd23872c815"
    assert hinge.SOURCE_LICENSE == "CC0-1.0"
    assert hinge.UPSTREAM_CODE_COMMIT == "cd5081aac460754b8ff3d6426835920e43542f95"
    assert hinge.CANONICAL_INPUT_SHAPE == (9, 13)
    assert hinge.CANONICAL_TARGET_SHAPE == (80,)
    assert sum(width for _name, width in hinge.CANONICAL_TARGET_COMPONENTS) == 80
    assert hinge.CANONICAL_INPUT_OFFSETS == (-9, -8, -7, -6, -5, -4, -3, -2, -1)
    assert hinge.CANONICAL_TARGET_OFFSET == 0
    assert hinge.CANONICAL_FUTURE_INPUT_COUNT == 0
    assert hinge.DEFAULT_SPLIT_SEED == "melis-wing-hinge-heldout-v2"
    assert tuple(name for name, _value in hinge.DEFAULT_SPLIT_FRACTIONS) == (
        "calibration",
        "validation",
        "held_out",
    )


def test_candidate_receipt_is_permanently_unreviewed() -> None:
    receipt = _official_receipt()
    assert receipt.status == "candidate_unreviewed"
    assert receipt.hinge_gate_id == "hinge.heldout_wing_prediction"
    assert receipt.hinge_gate_unblocked is False

    with pytest.raises(hinge.WingHingeIntakeError, match="unreviewed"):
        replace(receipt, status="validated")
    with pytest.raises(hinge.WingHingeIntakeError, match="cannot be unblocked"):
        replace(receipt, hinge_gate_unblocked=True)
    with pytest.raises(hinge.WingHingeIntakeError, match="not locked"):
        replace(receipt, source_record_uri="https://example.invalid")
    with pytest.raises(hinge.WingHingeIntakeError, match="SHA-256"):
        replace(receipt, observed_sha256="A" * 64)


def test_hash_regular_file_observes_md5_sha256_and_absolute_path(tmp_path) -> None:
    payload = (b"manufactured wing-hinge fixture\x00" * 19) + b"end"
    candidate = tmp_path / "candidate.h5"
    candidate.write_bytes(payload)

    receipt = hinge.hash_regular_file(
        candidate,
        expected_size=len(payload),
        expected_md5=hashlib.md5(payload).hexdigest(),
        chunk_size=7,
    )

    assert receipt.path == os.path.abspath(str(candidate))
    assert receipt.size_bytes == len(payload)
    assert receipt.md5 == hashlib.md5(payload).hexdigest()
    assert receipt.sha256 == hashlib.sha256(payload).hexdigest()
    observed_stat = candidate.stat()
    assert receipt.device == observed_stat.st_dev
    assert receipt.inode == observed_stat.st_ino
    assert receipt.mtime_ns == observed_stat.st_mtime_ns
    assert receipt.ctime_ns == observed_stat.st_ctime_ns


def test_hash_regular_file_rejects_wrong_bytes_and_non_regular_paths(tmp_path) -> None:
    payload = b"not the official dataset"
    candidate = tmp_path / "candidate.h5"
    candidate.write_bytes(payload)
    expected_md5 = hashlib.md5(payload).hexdigest()

    with pytest.raises(hinge.WingHingeIntakeError, match="byte count mismatch"):
        hinge.hash_regular_file(
            candidate, expected_size=len(payload) + 1, expected_md5=expected_md5
        )
    with pytest.raises(hinge.WingHingeIntakeError, match="MD5 mismatch"):
        hinge.hash_regular_file(
            candidate, expected_size=len(payload), expected_md5="0" * 32
        )
    with pytest.raises(hinge.WingHingeIntakeError, match="regular file"):
        hinge.hash_regular_file(
            tmp_path, expected_size=0, expected_md5=hashlib.md5(b"").hexdigest()
        )

    alias = tmp_path / "candidate-link.h5"
    alias.symlink_to(candidate)
    with pytest.raises(hinge.WingHingeIntakeError, match="symlink"):
        hinge.hash_regular_file(
            alias, expected_size=len(payload), expected_md5=expected_md5
        )


def test_topology_accepts_non_contiguous_movies_and_reports_exact_counts(
    official_topology: hinge.HingeTopology,
) -> None:
    report = hinge.validate_topology(official_topology)

    assert report.session_count == 74
    assert report.acquisition_date_count == 33
    assert report.movie_count == 377
    assert report.input_shape == (9, 13)
    assert report.target_shape == (80,)
    assert report.metadata_only is True
    assert [movie.movie_id for movie in official_topology.sessions[0].movies] == list(
        MOVIE_IDS
    )


@pytest.mark.parametrize(
    ("session_id", "message"),
    (
        ("Session_1_01_2021_08_00", "malformed session"),
        ("Session_31_02_2021_08_00", "invalid acquisition timestamp"),
        ("session_01_01_2021_08_00", "malformed session"),
    ),
)
def test_topology_rejects_malformed_session_ids(
    official_topology: hinge.HingeTopology, session_id: str, message: str
) -> None:
    malformed = replace(official_topology.sessions[0], session_id=session_id)
    with pytest.raises(hinge.WingHingeIntakeError, match=message):
        hinge.validate_topology(_replace_first_session(official_topology, malformed))


def test_topology_rejects_duplicate_sessions_and_wrong_counts(
    official_topology: hinge.HingeTopology,
) -> None:
    duplicated = hinge.HingeTopology(
        official_topology.sessions + (official_topology.sessions[0],)
    )
    with pytest.raises(hinge.WingHingeIntakeError, match="duplicate session"):
        hinge.validate_topology(duplicated)

    with pytest.raises(hinge.WingHingeIntakeError, match="topology count mismatch"):
        hinge.validate_topology(hinge.HingeTopology(official_topology.sessions[:-1]))


def test_topology_rejects_unknown_missing_and_duplicate_root_datasets(
    official_topology: hinge.HingeTopology,
) -> None:
    first = official_topology.sessions[0]
    mutations = (
        replace(first, root_dataset_names=first.root_dataset_names + ("mystery",)),
        replace(first, root_dataset_names=first.root_dataset_names[:-1]),
        replace(first, root_dataset_names=first.root_dataset_names + (first.root_dataset_names[0],)),
    )
    messages = ("root schema mismatch", "root schema mismatch", "duplicate root")
    for mutation, message in zip(mutations, messages):
        with pytest.raises(hinge.WingHingeIntakeError, match=message):
            hinge.validate_topology(_replace_first_session(official_topology, mutation))


def test_topology_rejects_malformed_and_duplicate_movie_ids(
    official_topology: hinge.HingeTopology,
) -> None:
    first_session = official_topology.sessions[0]
    malformed = replace(first_session.movies[0], movie_id="mov_0")
    with pytest.raises(hinge.WingHingeIntakeError, match="malformed movie"):
        hinge.validate_topology(_replace_first_movie(official_topology, malformed))

    duplicated = replace(
        first_session,
        movies=(first_session.movies[0], first_session.movies[0]) + first_session.movies[2:],
    )
    with pytest.raises(hinge.WingHingeIntakeError, match="duplicate movie"):
        hinge.validate_topology(_replace_first_session(official_topology, duplicated))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"n_wbs_shape": (1,)}, "must be scalar"),
        ({"n_wbs_dtype": "uint64"}, "must be int64"),
        ({"n_wbs": 0}, "must be positive"),
    ),
)
def test_topology_rejects_invalid_n_wbs_metadata(
    official_topology: hinge.HingeTopology, changes: dict, message: str
) -> None:
    movie = replace(official_topology.sessions[0].movies[0], **changes)
    with pytest.raises(hinge.WingHingeIntakeError, match=message):
        hinge.validate_topology(_replace_first_movie(official_topology, movie))


def test_topology_rejects_missing_unknown_duplicate_and_malformed_arrays(
    official_topology: hinge.HingeTopology,
) -> None:
    movie = official_topology.sessions[0].movies[0]
    arrays = movie.arrays

    missing = replace(movie, arrays=arrays[1:])
    with pytest.raises(hinge.WingHingeIntakeError, match="missing canonical arrays"):
        hinge.validate_topology(_replace_first_movie(official_topology, missing))

    unknown = replace(
        movie,
        arrays=arrays + (hinge.ArrayMetadata("mystery", (N_WBS,), "float64"),),
    )
    with pytest.raises(hinge.WingHingeIntakeError, match="unknown arrays"):
        hinge.validate_topology(_replace_first_movie(official_topology, unknown))

    duplicate = replace(movie, arrays=arrays + (arrays[0],))
    with pytest.raises(hinge.WingHingeIntakeError, match="duplicate array"):
        hinge.validate_topology(_replace_first_movie(official_topology, duplicate))

    wrong_dtype = replace(
        movie,
        arrays=(replace(arrays[0], dtype="float32"),) + arrays[1:],
    )
    with pytest.raises(hinge.WingHingeIntakeError, match="must be float64"):
        hinge.validate_topology(_replace_first_movie(official_topology, wrong_dtype))

    wrong_shape = replace(
        movie,
        arrays=(replace(arrays[0], shape=(N_WBS, 1)),) + arrays[1:],
    )
    with pytest.raises(hinge.WingHingeIntakeError, match="shape mismatch"):
        hinge.validate_topology(_replace_first_movie(official_topology, wrong_shape))


def test_published_first_30_and_unknown_split_strategies_are_rejected() -> None:
    with pytest.raises(hinge.WingHingeIntakeError, match="first-30"):
        hinge.validate_split_strategy(hinge.LEAKY_FIRST_30_SPLIT_STRATEGY)
    with pytest.raises(hinge.WingHingeIntakeError, match="unknown or unsafe"):
        hinge.validate_split_strategy("random_wingbeat_split")


def test_future_input_alignment_is_rejected_even_under_a_new_name(
    official_topology: hinge.HingeTopology,
) -> None:
    published = hinge.WindowSemantics(
        alignment=hinge.PUBLISHED_ACAUSAL_WINDOW_ALIGNMENT,
        input_offsets=tuple(range(9)),
        target_offset=0,
        future_input_count=9,
    )
    renamed = replace(published, alignment="apparently_safe_new_strategy_name")
    same_beat = hinge.WindowSemantics(
        alignment="trailing_window_that_still_reads_target_beat",
        input_offsets=(-8, -7, -6, -5, -4, -3, -2, -1, 0),
        target_offset=0,
        future_input_count=1,
    )

    with pytest.raises(hinge.WingHingeIntakeError, match="nine same-or-future inputs"):
        hinge.build_date_grouped_split(
            official_topology, window_semantics=published
        )
    with pytest.raises(hinge.WingHingeIntakeError, match="nine completed rows"):
        hinge.build_date_grouped_split(official_topology, window_semantics=renamed)
    with pytest.raises(hinge.WingHingeIntakeError, match="nine completed rows"):
        hinge.build_date_grouped_split(official_topology, window_semantics=same_beat)

    safe = hinge.build_date_grouped_split(official_topology)
    relabelled = replace(safe, window_semantics=renamed)
    with pytest.raises(hinge.WingHingeIntakeError, match="nine completed rows"):
        hinge.audit_date_grouped_split(official_topology, relabelled)


def test_date_grouped_split_is_deterministic_complete_and_zero_overlap(
    official_topology: hinge.HingeTopology,
) -> None:
    first = hinge.build_date_grouped_split(official_topology, seed="fixed-science-seed")
    repeated = hinge.build_date_grouped_split(official_topology, seed="fixed-science-seed")
    other = hinge.build_date_grouped_split(official_topology, seed="another-seed")

    assert first == repeated
    assert first.assignments != other.assignments
    assert first.rejects_published_first_30_split is True
    assert first.window_semantics == hinge.CANONICAL_WINDOW_SEMANTICS
    assert tuple(name for name, _fraction in first.fractions) == (
        "calibration",
        "validation",
        "held_out",
    )

    audit = hinge.audit_date_grouped_split(official_topology, first)
    assert audit.zero_overlap is True
    assert audit.all_dates_assigned_once is True
    assert audit.all_sessions_assigned_once is True
    assert audit.all_movies_assigned_once is True
    assert audit.all_windows_assigned_once is True
    assert audit.window_alignment == hinge.CAUSAL_WINDOW_ALIGNMENT
    assert audit.input_offsets == (-9, -8, -7, -6, -5, -4, -3, -2, -1)
    assert audit.target_offset == 0
    assert audit.future_input_count == 0
    assert audit.published_acausal_window_rejected is True
    assert sum(value for _split, value in audit.date_counts) == 33
    assert sum(value for _split, value in audit.session_counts) == 74
    assert sum(value for _split, value in audit.movie_counts) == 377
    assert sum(value for _split, value in audit.window_counts) == 377 * (
        N_WBS - hinge.INPUT_WINDOW_WINGBEATS
    )

    date_assignment = {
        assignment.acquisition_date: assignment.split for assignment in first.assignments
    }
    for session in official_topology.sessions:
        assert hinge.acquisition_date_for_session(session.session_id) in date_assignment


def test_split_audit_rejects_duplicate_missing_and_unsafe_assignments(
    official_topology: hinge.HingeTopology,
) -> None:
    split = hinge.build_date_grouped_split(official_topology)

    duplicate = replace(split, assignments=split.assignments + (split.assignments[0],))
    with pytest.raises(hinge.WingHingeIntakeError, match="duplicate acquisition-date"):
        hinge.audit_date_grouped_split(official_topology, duplicate)

    missing = replace(split, assignments=split.assignments[:-1])
    with pytest.raises(hinge.WingHingeIntakeError, match="exactly cover"):
        hinge.audit_date_grouped_split(official_topology, missing)

    unsafe = replace(split, rejects_published_first_30_split=False)
    with pytest.raises(hinge.WingHingeIntakeError, match="explicitly reject"):
        hinge.audit_date_grouped_split(official_topology, unsafe)

    with pytest.raises(
        hinge.WingHingeIntakeError,
        match="calibration, validation, and held_out",
    ):
        hinge.build_date_grouped_split(
            official_topology,
            fractions=(("train", 0.7), ("validation", 0.15), ("test", 0.15)),
        )


def test_candidate_intake_cannot_be_promoted_by_structural_success(
    official_topology: hinge.HingeTopology,
) -> None:
    report = hinge.validate_topology(official_topology)
    split = hinge.build_date_grouped_split(official_topology)
    audit = hinge.audit_date_grouped_split(official_topology, split)
    candidate = hinge.CandidateIntake(
        receipt=_official_receipt(),
        topology=official_topology,
        topology_report=report,
        split=split,
        split_audit=audit,
    )

    assert candidate.status == "candidate_unreviewed"
    assert candidate.hinge_gate_unblocked is False
    candidate_document = candidate.candidate_receipt_document
    split_document = candidate.split_receipt_document
    assert candidate_document == candidate.candidate_receipt_document
    assert split_document == candidate.split_receipt_document
    assert candidate_document.sha256 == hashlib.sha256(
        candidate_document.canonical_json.encode("utf-8")
    ).hexdigest()
    assert split_document.sha256 == hashlib.sha256(
        split_document.canonical_json.encode("utf-8")
    ).hexdigest()
    candidate_json = json.loads(candidate_document.canonical_json)
    split_json = json.loads(split_document.canonical_json)
    assert candidate_json["receipt"]["status"] == "candidate_unreviewed"
    assert candidate_json["receipt"]["hinge_gate_unblocked"] is False
    assert split_json["candidate_receipt_sha256"] == candidate_document.sha256
    assert split_json["split"]["window_semantics"]["future_input_count"] == 0
    assert split_json["split_audit"]["zero_overlap"] is True
    with pytest.raises(hinge.WingHingeIntakeError, match="cannot unblock"):
        replace(candidate, hinge_gate_unblocked=True)
    with pytest.raises(hinge.WingHingeIntakeError, match="remains unreviewed"):
        replace(candidate, status="validated")


def test_inspection_rejects_same_size_file_replacement(
    monkeypatch, tmp_path, official_topology: hinge.HingeTopology
) -> None:
    candidate = tmp_path / hinge.SOURCE_FILE_NAME
    candidate.touch()
    os.truncate(candidate, hinge.SOURCE_SIZE_BYTES)
    before = candidate.stat()
    receipt = _official_receipt(os.path.abspath(str(candidate)), file_stat=before)

    def replace_during_inspection(_path):
        replacement = tmp_path / "same-size-replacement.h5"
        replacement.touch()
        os.truncate(replacement, hinge.SOURCE_SIZE_BYTES)
        os.replace(replacement, candidate)
        return official_topology

    monkeypatch.setattr(hinge, "read_hdf5_topology", replace_during_inspection)
    with pytest.raises(
        hinge.WingHingeIntakeError, match="identity or timestamps changed"
    ):
        hinge.inspect_verified_candidate(candidate, receipt)


def test_h5py_is_lazy_and_optional(monkeypatch, tmp_path) -> None:
    candidate = tmp_path / "metadata-only.h5"
    candidate.write_bytes(b"not read because h5py is unavailable")
    original_import = builtins.__import__

    def import_without_h5py(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "h5py":
            raise ModuleNotFoundError("manufactured missing optional dependency")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_h5py)
    with pytest.raises(hinge.WingHingeIntakeError, match="optional dependency 'h5py'"):
        hinge.read_hdf5_topology(candidate)


def test_hdf5_reader_reads_only_scalar_n_wbs_and_never_bulk_arrays(
    monkeypatch, tmp_path
) -> None:
    class FakeDataset:
        def __init__(self, shape, dtype, scalar=None):
            self.shape = shape
            self.dtype = dtype
            self.scalar = scalar
            self.read_count = 0

        def __getitem__(self, key):
            self.read_count += 1
            if self.shape != () or key != ():
                raise AssertionError("bulk dataset value was read")
            return self.scalar

    class FakeGroup(dict):
        pass

    class FakeFile(FakeGroup):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    scalar = FakeDataset((), "int64", 12)
    bulk = FakeDataset((20, 12), "float64")
    movie = FakeGroup({"N_wbs": scalar, "a_theta_L": bulk})
    session = FakeGroup({"N_mov": FakeDataset((), "int64", 1), "mov_3": movie})
    fake_file = FakeFile({"Session_01_01_2021_08_00": session})
    fake_h5py = types.SimpleNamespace(
        Dataset=FakeDataset,
        Group=FakeGroup,
        File=lambda _path, _mode: fake_file,
    )
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)
    candidate = tmp_path / "metadata-only.h5"
    candidate.write_bytes(b"manufactured file object")

    topology = hinge.read_hdf5_topology(candidate)

    assert topology.sessions[0].movies[0].movie_id == "mov_3"
    assert topology.sessions[0].movies[0].n_wbs == 12
    assert scalar.read_count == 1
    assert bulk.read_count == 0

from itertools import permutations
from pathlib import Path
from typing import Tuple
import pandas as pd
import re
from pandera.errors import SchemaError
from glass_onion.engine import SyncableContent
from glass_onion.player import (
    PlayerSyncEngine,
    PlayerSyncLayer,
    PlayerSyncSimilarityMethod,
    PlayerSyncableContent,
    birth_dates_are_exact_match,
    birth_dates_conflict,
)
import pytest


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-21",
        "01-21-2026",
        "21-01-2026",
        "Jan 21, 2026",
        "January 21, 2026",
        "2026-01-26T00:00Z",
        "2026-01-26T00:00:00Z",
        "2026-01-26T00:00:00.000Z",
    ],
)
def test_init_syncable_content_birth_date_is_valid_format(value: str):
    content = PlayerSyncableContent(
        "provider_a",
        pd.DataFrame(
            [
                {
                    "provider_a_player_id": "1",
                    "player_name": "test",
                    "birth_date": value,
                    "team_id": "1",
                }
            ]
        ),
    )

    assert content.validate_data_schema()


def test_init_syncable_content_birth_date_is_not_valid_format():
    with pytest.raises(
        SchemaError,
        match=re.escape("Unknown datetime string format, unable to parse: test"),
    ):
        PlayerSyncableContent(
            "provider_a",
            pd.DataFrame(
                [
                    {
                        "provider_a_player_id": "1",
                        "player_name": "test",
                        "birth_date": "test",
                        "team_id": "1",
                    }
                ]
            ),
        )


@pytest.mark.parametrize(
    "column",
    [
        "provider_a_player_id",
        "player_name",
        "team_id",
    ],
)
def test_init_syncable_content_prevent_mixed_values(column: str):
    base = {
        "provider_a_player_id": "1",
        "birth_date": "2026-01-01",
        "team_id": "1",
        "player_name": "test",
        "player_nickname": "test1",
        "jersey_number": "1",
    }
    dataset = []

    for i in range(0, 10):
        c = base.copy()
        c["provider_a_player_id"] = str(i)

        if i % 2 == 1:
            c[column] = pd.NA

        dataset.append(c)

    df = pd.DataFrame(dataset)

    with pytest.raises(
        SchemaError,
        match=re.escape(f"non-nullable series '{column}' contains null values"),
    ):
        PlayerSyncableContent(
            "provider_a",
            df,
        )


@pytest.mark.parametrize(
    "column",
    [
        "player_nickname",
        "birth_date",
        "jersey_number",
    ],
)
def test_init_syncable_content_allow_mixed_values(column: str):
    base = {
        "provider_a_player_id": "1",
        "birth_date": "2026-01-01",
        "team_id": "1",
        "player_name": "test",
        "player_nickname": "test1",
        "jersey_number": "1",
    }
    dataset = []

    for i in range(0, 10):
        c = base.copy()
        c["provider_a_player_id"] = str(i)

        if i % 2 == 1:
            c[column] = pd.NA

        dataset.append(c)

    df = pd.DataFrame(dataset)

    c = PlayerSyncableContent(
        "provider_a",
        df,
    )
    assert c.validate_data_schema()


@pytest.mark.parametrize(
    "column",
    [
        "provider_a_player_id",
        "team_id",
        "jersey_number",
    ],
)
def test_init_syncable_content_allow_mixed_types(column: str):
    base = {
        "provider_a_player_id": "1",
        "birth_date": "2026-01-01",
        "team_id": "1",
        "player_name": "test",
        "player_nickname": "test1",
        "jersey_number": "1",
    }
    dataset = []

    for i in range(0, 10):
        c = base.copy()
        c["provider_a_player_id"] = str(i)

        if i % 2 == 1:
            c[column] = i

        dataset.append(c)

    df = pd.DataFrame(dataset)

    c = PlayerSyncableContent(
        "provider_a",
        df,
    )
    assert c.validate_data_schema()


@pytest.mark.parametrize(
    "column",
    ["player_nickname", "player_name", "birth_date"],
)
def test_init_syncable_content_prevent_mixed_types(column: str):
    base = {
        "provider_a_player_id": "1",
        "birth_date": "2026-01-01",
        "team_id": "1",
        "player_name": "test",
        "player_nickname": "test1",
        "jersey_number": "1",
    }
    dataset = []

    for i in range(0, 10):
        c = base.copy()
        c["provider_a_player_id"] = str(i)

        if i % 2 == 1:
            c[column] = i

        dataset.append(c)

    df = pd.DataFrame(dataset)

    with pytest.raises(
        SchemaError,
        match=re.escape(f"expected series '{column}' to have type str"),
    ):
        PlayerSyncableContent(
            "provider_a",
            df,
        )


def test_init_missing_columns():
    with pytest.raises(
        SchemaError,
        match=re.escape(
            "column 'team_id' not in dataframe. Columns in dataframe: ['provider_a_player_id', 'player_name']"
        ),
    ):
        PlayerSyncableContent(
            "provider_a",
            data=pd.DataFrame([{"provider_a_player_id": "1", "player_name": "A"}]),
        )


def test_init_unreliable_columns():
    left = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_player_id": "1",
                    "player_name": "A",
                    "team_id": "A",
                    "jersey_number": pd.NA,
                }
            ]
        ),
    )

    right = PlayerSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_b_player_id": "1",
                    "player_name": "A",
                    "team_id": "A",
                    "jersey_number": "1",
                }
            ]
        ),
    )

    engine = PlayerSyncEngine([left, right], verbose=True)
    assert set(["player_name", "team_id"]) == set(engine.join_columns)


@pytest.mark.parametrize(
    "layer, method, expected_matches",
    [
        # base case: ensure string similarity hits
        (
            PlayerSyncLayer(
                title="test layer",
                match_methodology=PlayerSyncSimilarityMethod.NAIVE,
                threshold=0,
            ),
            "synchronize_with_naive_match",
            1,
        ),
        # what if someone passes NULL to string similarity?
        (
            PlayerSyncLayer(
                title="test layer",
                match_methodology=None,
            ),
            "synchronize_with_cosine_similarity",
            1,
        ),
        # what if we use different string fields? This won't match in a small sample
        (
            PlayerSyncLayer(
                title="test layer", input_fields=("player_name", "player_nickname")
            ),
            "synchronize_with_cosine_similarity",
            0,
        ),
        # what if we use different string fields AND drop the threshold? this will now match
        (
            PlayerSyncLayer(
                title="test layer",
                input_fields=("player_name", "player_nickname"),
                threshold=0,
            ),
            "synchronize_with_cosine_similarity",
            1,
        ),
        # what if we require certain fields to be equal? This shouldn't match because jersey numbers are different
        (
            PlayerSyncLayer(
                title="test layer", other_equal_fields=["team_id", "jersey_number"]
            ),
            "synchronize_with_cosine_similarity",
            0,
        ),
        # what if we adjust the birth date one day forward? This shouldn't find any matches because the default other_equal_fields includes birth_date
        (
            PlayerSyncLayer(
                title="test layer",
                date_adjustment=pd.Timedelta(days=1),
            ),
            "synchronize_with_cosine_similarity",
            0,
        ),
        # what if we adjust the birth date one day forward AND ignore birth_date? This should find matches
        (
            PlayerSyncLayer(
                title="test layer",
                date_adjustment=pd.Timedelta(days=1),
                other_equal_fields=["team_id"],
            ),
            "synchronize_with_cosine_similarity",
            1,
        ),
        # what if we adjust the birth date one day forward AND swap the month/day? This shouldn't find any matches because the default other_equal_fields includes birth_date
        (
            PlayerSyncLayer(title="test layer", swap_birth_month_day=True),
            "synchronize_with_cosine_similarity",
            0,
        ),
    ],
)
def test_synchronize_using_layer(
    layer: PlayerSyncLayer, method: str, expected_matches: int, mocker
):
    left = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_player_id": "1",
                    "player_name": "ABCD",
                    "player_nickname": "AB",
                    "team_id": "A",
                    "jersey_number": "1",
                    "birth_date": "1970-01-02",
                }
            ]
        ),
    )

    right = PlayerSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_b_player_id": "1",
                    "player_name": "ABCD",
                    "player_nickname": "AB",
                    "team_id": "A",
                    "jersey_number": "0",
                    "birth_date": "1970-01-02",
                }
            ]
        ),
    )

    engine = PlayerSyncEngine([left, right], verbose=True)
    spy = mocker.spy(engine, method)

    result = engine.synchronize_using_layer(left, right, layer)
    if method == "synchronize_with_naive_match":
        spy.assert_called_once_with(left, right, layer.input_fields)
    else:
        spy.assert_called_once_with(
            left, right, layer.input_fields, layer.similarity_threshold
        )
    assert set(["provider_a_player_id", "provider_b_player_id"]) == set(result.columns)

    assert len(result) == expected_matches


@pytest.mark.parametrize(
    "left_df, right_df, remove_columns, expected_layers",
    [
        (
            pd.DataFrame(
                [
                    {
                        "provider_a_player_id": "1",
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": "1",
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "provider_b_player_id": "1",
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": "0",
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            [],
            3,
        ),
        (
            pd.DataFrame(
                [
                    {
                        "provider_a_player_id": "1",
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": "1",
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "provider_b_player_id": "1",
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": "0",
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            ["jersey_number"],
            1,
        ),
        (
            pd.DataFrame(
                [
                    {
                        "provider_a_player_id": "1",
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": "1",
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "provider_b_player_id": "1",
                        "player_name": "ABCD",
                        "player_nickname": "AB",
                        "team_id": "A",
                        "jersey_number": "0",
                        "birth_date": "1970-01-02",
                    }
                ]
            ),
            ["birth_date"],
            2,
        ),
    ],
)
def test_synchronize_pair(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    remove_columns: list[str],
    expected_layers: int,
    mocker,
):
    left = PlayerSyncableContent(
        "provider_a",
        data=left_df,
    )

    right = PlayerSyncableContent(
        "provider_b",
        data=right_df,
    )

    if len(remove_columns) > 0:
        left.data.drop(remove_columns, axis=1, inplace=True)
        right.data.drop(remove_columns, axis=1, inplace=True)

    engine = PlayerSyncEngine([left, right], verbose=True)
    spy = mocker.spy(engine, "synchronize_using_layer")

    result = engine.synchronize_pair(left, right)
    assert isinstance(result, PlayerSyncableContent)
    assert len(result.data) == 1
    assert spy.call_count == expected_layers


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # identical
        ("1996-11-15", "1996-11-15", False),
        # within 1 day (timezone shift)
        ("1996-11-15", "1996-11-16", False),
        ("1996-11-16", "1996-11-15", False),
        # 2 days apart -> conflict
        ("1996-11-15", "1996-11-17", True),
        # different years -> conflict
        ("1996-11-15", "2004-02-01", True),
        # month/day swap (exact and within 1 day of swapped)
        ("1996-11-05", "1996-05-11", False),
        ("1996-05-11", "1996-11-05", False),
        ("1996-11-05", "1996-05-12", False),
        ("1996-11-05", "1996-05-13", True),
        # swapping BOTH sides is a no-op and does not rescue a conflict
        ("1996-03-04", "1996-03-06", True),
        # Jan 1 in the same year (year-only placeholder)
        ("1996-01-01", "1996-11-15", False),
        ("1996-11-15", "1996-01-01", False),
        # Jan 1 in a different year -> conflict (unless Dec 31 of the year before)
        ("1997-01-01", "1996-11-15", True),
        ("1997-01-01", "1996-12-31", False),
        # timestamps with time of day / timezone: compare calendar dates
        ("1996-11-15T23:00:00Z", "1996-11-16T01:00:00Z", False),
        ("1996-11-15T23:00:00Z", "1996-11-17T00:00:00Z", True),
        # missing, blank, unparseable, or epoch-zero sentinel (1970-01-01) -> no conflict
        (pd.NA, "1996-11-15", False),
        ("1996-11-15", None, False),
        (pd.NA, pd.NA, False),
        ("", "1996-11-15", False),
        ("not-a-date", "1996-11-15", False),
        ("1970-01-01", "1996-11-15", False),
        ("1970-01-01T00:00:00Z", "1996-11-15", False),
    ],
)
def test_birth_dates_conflict(a, b, expected: bool):
    assert birth_dates_conflict(a, b) is expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("1996-11-15", "1996-11-15", True),
        ("1996-11-15T00:00:00Z", "1996-11-15", True),
        ("1996-11-15", "1996-11-16", False),
        ("1996-11-05", "1996-05-11", False),
        ("1996-01-01", "1996-01-01", False),
        ("1996-01-01", "1996-11-15", False),
        (pd.NA, "1996-11-15", False),
        ("1970-01-01", "1970-01-01", False),
    ],
)
def test_birth_dates_are_exact_match(a, b, expected: bool):
    assert birth_dates_are_exact_match(a, b) is expected


@pytest.mark.parametrize("na_value", [pd.NA, None])
def test_init_syncable_content_all_null_optional_string_columns(na_value):
    content = PlayerSyncableContent(
        "provider_a",
        pd.DataFrame(
            [
                {
                    "provider_a_player_id": "1",
                    "player_name": "Kim Min-jae",
                    "player_nickname": na_value,
                    "team_id": "KOR",
                    "birth_date": na_value,
                    "jersey_number": na_value,
                }
            ]
        ),
    )
    assert content.validate_data_schema()


def test_synchronize_using_layer_does_not_mutate_inputs_and_normalizes_dates():
    left = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_player_id": "1",
                    "player_name": "Kim Min-jae",
                    "player_nickname": "Min-jae",
                    "team_id": "KOR",
                    "jersey_number": "3",
                    "birth_date": "1996-11-15T00:00:00Z",
                }
            ]
        ),
    )
    right = PlayerSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_b_player_id": "2",
                    "player_name": "Kim Min-jae",
                    "player_nickname": "Min-jae",
                    "team_id": "KOR",
                    "jersey_number": "4",
                    "birth_date": "Nov 16, 1996",
                }
            ]
        ),
    )

    engine = PlayerSyncEngine([left, right])
    layer = PlayerSyncLayer(
        title="Layer 2: +1d shift",
        date_adjustment=pd.Timedelta(days=1),
        swap_birth_month_day=False,
    )
    res = engine.synchronize_using_layer(left, right, layer)
    assert len(res) == 1
    # Neither input's birth_date column was mutated in-place
    assert left.data["birth_date"].iloc[0] == "1996-11-15T00:00:00Z"
    assert right.data["birth_date"].iloc[0] == "Nov 16, 1996"


@pytest.mark.parametrize(
    "left_row, right_rows, expected_matches",
    [
        # Layer 1 (same jersey + team + name) vetoed by conflicting birth dates
        (
            {
                "player_name": "Kim Min-jae",
                "player_nickname": "Kim Min-jae",
                "jersey_number": "3",
                "birth_date": "1996-11-15",
            },
            [
                {
                    "player_name": "Kim Min-jae",
                    "player_nickname": "Kim Min-jae",
                    "jersey_number": "3",
                    "birth_date": "2004-02-01",
                }
            ],
            0,
        ),
        # Layer 3/4/5 (different jerseys) vetoed by conflicting birth dates
        (
            {
                "player_name": "Kim Min-jae",
                "player_nickname": "Min-jae",
                "jersey_number": "3",
                "birth_date": "1996-11-15",
            },
            [
                {
                    "player_name": "Kim Min-jae",
                    "player_nickname": "Min-jae",
                    "jersey_number": "15",
                    "birth_date": "2004-02-01",
                }
            ],
            0,
        ),
        # Layer 5 (threshold=0 fallback on last unmatched players) vetoed by conflicting birth dates
        (
            {
                "player_name": "Alice Smith",
                "player_nickname": "Alice",
                "jersey_number": pd.NA,
                "birth_date": "1995-03-10",
            },
            [
                {
                    "player_name": "Bob Jones",
                    "player_nickname": "Bob",
                    "jersey_number": pd.NA,
                    "birth_date": "2001-08-22",
                }
            ],
            0,
        ),
        # Veto still applies when another row has a null birth_date (which disables Layer 2)
        (
            {
                "player_name": "Kim Min-jae",
                "player_nickname": "Min-jae",
                "jersey_number": "3",
                "birth_date": "1996-11-15",
            },
            [
                {
                    "player_name": "Kim Min-jae",
                    "player_nickname": "Min-jae",
                    "jersey_number": "15",
                    "birth_date": "2004-02-01",
                },
                {
                    "player_name": "Unrelated Player",
                    "player_nickname": "Unrelated",
                    "jersey_number": "99",
                    "birth_date": pd.NA,
                },
            ],
            0,
        ),
        # Vetoing a conflicting homonym before 1-to-1 uniqueness check lets the true player match
        (
            {
                "player_name": "Kim Min-jae",
                "player_nickname": "Min-jae",
                "jersey_number": pd.NA,
                "birth_date": "1996-11-15",
            },
            [
                {
                    "player_name": "Kim Min-jae",
                    "player_nickname": "Min-jae",
                    "jersey_number": pd.NA,
                    "birth_date": "1996-11-15",
                },
                {
                    "player_name": "Kim Min-jae",
                    "player_nickname": "Min-jae",
                    "jersey_number": pd.NA,
                    "birth_date": "2004-02-01",
                },
            ],
            1,
        ),
        # Layer 2 +1 day adjustment (input1 + 1 day == input2)
        (
            {
                "player_name": "Kim Min-jae",
                "player_nickname": "Min-jae",
                "jersey_number": "3",
                "birth_date": "1996-11-15",
            },
            [
                {
                    "player_name": "K. Min-jae",
                    "player_nickname": "Min-jae",
                    "jersey_number": "4",
                    "birth_date": "1996-11-16",
                }
            ],
            1,
        ),
        # Layer 2 -1 day adjustment (input1 - 1 day == input2)
        (
            {
                "player_name": "Kim Min-jae",
                "player_nickname": "Min-jae",
                "jersey_number": "3",
                "birth_date": "1996-11-16",
            },
            [
                {
                    "player_name": "K. Min-jae",
                    "player_nickname": "Min-jae",
                    "jersey_number": "4",
                    "birth_date": "1996-11-15",
                }
            ],
            1,
        ),
        # Layer 2 month/day swap
        (
            {
                "player_name": "Kim Min-jae",
                "player_nickname": "Min-jae",
                "jersey_number": "3",
                "birth_date": "1996-11-05",
            },
            [
                {
                    "player_name": "K. Min-jae",
                    "player_nickname": "Min-jae",
                    "jersey_number": "4",
                    "birth_date": "1996-05-11",
                }
            ],
            1,
        ),
    ],
)
def test_synchronize_pair_birth_date_veto(
    left_row: dict, right_rows: list[dict], expected_matches: int
):
    left = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame([{"provider_a_player_id": "1", "team_id": "A", **left_row}]),
    )
    right = PlayerSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [
                {"provider_b_player_id": str(i + 1), "team_id": "A", **r}
                for i, r in enumerate(right_rows)
            ]
        ),
    )

    engine = PlayerSyncEngine([left, right])
    result = engine.synchronize_pair(left, right)
    assert len(result.data) == expected_matches
    assert left.data["birth_date"].iloc[0] == left_row["birth_date"]


def test_synchronize_end_to_end_preserves_vetoed_homonyms_and_same_name_teammates():
    # Two providers list different "Kim Min-jae" players (1996 vs 2004) without jersey numbers;
    # synchronize() must keep them as two separate rows rather than collapsing them on join_columns.
    left = PlayerSyncableContent(
        "provider_a",
        data=pd.DataFrame(
            [
                {
                    "provider_a_player_id": "a_1996",
                    "player_name": "Kim Min-jae",
                    "player_nickname": "Kim Min-jae",
                    "team_id": "KOR",
                    "jersey_number": pd.NA,
                    "birth_date": "1996-11-15",
                }
            ]
        ),
    )
    right = PlayerSyncableContent(
        "provider_b",
        data=pd.DataFrame(
            [
                {
                    "provider_b_player_id": "b_2004",
                    "player_name": "Kim Min-jae",
                    "player_nickname": "Kim Min-jae",
                    "team_id": "KOR",
                    "jersey_number": pd.NA,
                    "birth_date": "2004-02-01",
                }
            ]
        ),
    )

    synced = PlayerSyncEngine([left, right]).synchronize().data
    assert len(synced) == 2
    assert (
        len(synced.dropna(subset=["provider_a_player_id", "provider_b_player_id"])) == 0
    )


@pytest.mark.parametrize(
    "order",
    list(permutations(["provider_a", "provider_b", "provider_c"])),
)
def test_synchronize_three_provider_birth_date_drift_chain_rejected(
    order: Tuple[str, ...],
):
    # A (15 Nov) <-> B (16 Nov) and B (16 Nov) <-> C (17 Nov) are each within 1 day,
    # but A and C are 2 days apart and conflict; the chain must not merge A and C in any provider order.
    dob_by_provider = {
        "provider_a": "1996-11-15",
        "provider_b": "1996-11-16",
        "provider_c": "1996-11-17",
    }
    content = [
        PlayerSyncableContent(
            p,
            data=pd.DataFrame(
                [
                    {
                        f"{p}_player_id": f"{p}_1",
                        "player_name": "Kim Min-jae",
                        "player_nickname": "Kim Min-jae",
                        "team_id": "KOR",
                        "jersey_number": "3",
                        "birth_date": dob_by_provider[p],
                    }
                ]
            ),
        )
        for p in order
    ]

    synced = PlayerSyncEngine(content).synchronize().data
    assert len(synced) == 3
    for p in order:
        assert synced[f"{p}_player_id"].notna().sum() == 1


@pytest.mark.parametrize(
    "order",
    list(permutations(["provider_a", "provider_b", "provider_c"])),
)
@pytest.mark.parametrize(
    "bridge_dob",
    [pd.NA, "1996-01-01"],
)
def test_synchronize_three_provider_ambiguous_bridge_between_conflicting_players_rejected(
    order: Tuple[str, ...],
    bridge_dob,
):
    # Provider B has a missing or Jan-1 placeholder birth date and could pairwise-match both
    # Provider A (1996-11-15) and Provider C (1996-05-20 / 2004-02-01), which conflict with each other.
    dob_by_provider = {
        "provider_a": "1996-11-15",
        "provider_b": bridge_dob,
        "provider_c": "1996-05-20" if pd.notna(bridge_dob) else "2004-02-01",
    }
    content = [
        PlayerSyncableContent(
            p,
            data=pd.DataFrame(
                [
                    {
                        f"{p}_player_id": f"{p}_1",
                        "player_name": "Kim Min-jae",
                        "player_nickname": "Kim Min-jae",
                        "team_id": "KOR",
                        "jersey_number": "3",
                        "birth_date": dob_by_provider[p],
                    }
                ]
            ),
        )
        for p in order
    ]

    synced = PlayerSyncEngine(content).synchronize().data
    assert len(synced) == 3


@pytest.mark.parametrize(
    "order",
    list(permutations(["provider_a", "provider_b", "provider_c", "provider_d"])),
)
def test_synchronize_four_provider_preserves_exact_cohort_when_bridge_conflicts(
    order: Tuple[str, ...],
):
    # Providers A and D both have 1996-11-15 (exact match), Provider B has NA birth date,
    # and Provider C has 2004-02-01 (conflict). A and D must unify while B and C stay separate.
    dob_by_provider = {
        "provider_a": "1996-11-15",
        "provider_b": pd.NA,
        "provider_c": "2004-02-01",
        "provider_d": "1996-11-15",
    }
    content = [
        PlayerSyncableContent(
            p,
            data=pd.DataFrame(
                [
                    {
                        f"{p}_player_id": f"{p}_1",
                        "player_name": "Kim Min-jae",
                        "player_nickname": "Kim Min-jae",
                        "team_id": "KOR",
                        "jersey_number": "3",
                        "birth_date": dob_by_provider[p],
                    }
                ]
            ),
        )
        for p in order
    ]

    synced = PlayerSyncEngine(content).synchronize().data
    assert len(synced) == 3
    ad_row = synced[
        (synced["provider_a_player_id"] == "provider_a_1")
        & (synced["provider_d_player_id"] == "provider_d_1")
    ]
    assert len(ad_row) == 1
    assert pd.isna(ad_row["provider_b_player_id"].iloc[0])
    assert pd.isna(ad_row["provider_c_player_id"].iloc[0])


@pytest.mark.parametrize(
    "order",
    list(permutations(["provider_a", "provider_b", "provider_c"])),
)
def test_synchronize_three_provider_compatible_missing_dob_bridge_unifies(
    order: Tuple[str, ...],
):
    # Provider B has NA birth date and bridges Provider A (1996-11-15) and Provider C (1996-11-15);
    # because A and C do not conflict, all three unify into a single row.
    rows_by_provider = {
        "provider_a": {
            "player_name": "Timothy Weah",
            "player_nickname": "Timothy Weah",
            "jersey_number": "21",
            "birth_date": "2000-02-22",
        },
        "provider_b": {
            "player_name": "Timothy Weah",
            "player_nickname": "T. Weah",
            "jersey_number": "23",
            "birth_date": pd.NA,
        },
        "provider_c": {
            "player_name": "T. Weah",
            "player_nickname": "T. Weah",
            "jersey_number": "22",
            "birth_date": "2000-02-22",
        },
    }
    content = [
        PlayerSyncableContent(
            p,
            data=pd.DataFrame(
                [
                    {
                        f"{p}_player_id": f"{p}_1",
                        "team_id": "USA",
                        **rows_by_provider[p],
                    }
                ]
            ),
        )
        for p in order
    ]

    synced = PlayerSyncEngine(content).synchronize().data
    assert len(synced) == 1
    for p in order:
        assert synced[f"{p}_player_id"].iloc[0] == f"{p}_1"


def test_synchronize_same_name_teammates_preserved_without_jersey_numbers():
    # Two distinct teammates share player_name ("Tae-hyeon Kim") and team_id ("KOR"),
    # and an unrelated player has jersey_number=NA (so join_columns drops jersey_number).
    # Both Tae-hyeon Kim rows must survive final deduplication.
    providers = ["provider_a", "provider_b", "provider_c"]
    content = [
        PlayerSyncableContent(
            p,
            data=pd.DataFrame(
                [
                    {
                        f"{p}_player_id": f"{p}_1996",
                        "player_name": "Tae-hyeon Kim",
                        "player_nickname": "Tae-hyeon Kim",
                        "team_id": "KOR",
                        "jersey_number": "4",
                        "birth_date": "1996-12-19",
                    },
                    {
                        f"{p}_player_id": f"{p}_2000",
                        "player_name": "Tae-hyeon Kim",
                        "player_nickname": "Tae-hyeon Kim",
                        "team_id": "KOR",
                        "jersey_number": "15",
                        "birth_date": "2000-09-17",
                    },
                    {
                        f"{p}_player_id": f"{p}_other",
                        "player_name": "Son Heung-min",
                        "player_nickname": "Son",
                        "team_id": "KOR",
                        "jersey_number": pd.NA,
                        "birth_date": "1992-07-08",
                    },
                ]
            ),
        )
        for p in providers
    ]

    synced = PlayerSyncEngine(content).synchronize().data
    kim_rows = synced[synced["player_name"] == "Tae-hyeon Kim"]
    assert len(kim_rows) == 2
    assert set(kim_rows["provider_a_player_id"]) == {
        "provider_a_1996",
        "provider_a_2000",
    }

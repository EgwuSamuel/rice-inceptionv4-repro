import numpy as np
import pytest

from ricedx import data as D

CLASSES = ["Bacterial leaf blight", "Brown spot", "Health", "Leaf smut"]


def test_source_parsed_from_augmentor_filename():
    name = "Bacterial_leaf_blight_original_DSC_0365.JPG_01646849-182b-4932-8f59-bf48ac20a425.JPG"
    assert D.source_of(name) == "DSC_0365.JPG"
    name = "Health_original_IMG_20190419_095646.jpg_015d5748-92a7-45e0-81b7-8670d0aac106.jpg"
    assert D.source_of(name) == "IMG_20190419_095646.jpg"
    assert D.source_of("100001.jpg") is None


def test_index_counts_groups(synthetic_root):
    df = D.build_index(synthetic_root / "riceleaf", CLASSES)
    assert len(df) == 4 * 6 * 4
    assert df["group"].nunique() == 4 * 6
    assert df.groupby("group")["label"].nunique().max() == 1


def test_find_dataset_root_auto(synthetic_root):
    found = D.find_dataset_root("auto", CLASSES, search_roots=(str(synthetic_root),))
    assert found == synthetic_root / "riceleaf"


def test_group_kfold_has_no_leakage(synthetic_root):
    df = D.build_index(synthetic_root / "riceleaf", CLASSES)
    folds = D.group_kfold(df, 3, seed=0)
    tested = np.concatenate([te for _, te in folds])
    assert sorted(tested) == list(range(len(df)))  # every image tested exactly once
    for tr, te in folds:
        D.assert_disjoint_groups(df, tr, te)
        assert D.group_overlap_fraction(df, tr, te) == 0.0
        assert set(df.iloc[te]["label"]) == set(range(4))  # stratified


def test_random_split_leaks_and_group_split_does_not(synthetic_root):
    df = D.build_index(synthetic_root / "riceleaf", CLASSES)
    tr, te = D.random_split(df, 0.25, seed=0)
    assert D.group_overlap_fraction(df, tr, te) > 0.5
    tr, te = D.group_split(df, 0.25, seed=0)
    assert D.group_overlap_fraction(df, tr, te) == 0.0
    a, b = D.split_train_val(df, tr, 0.25, seed=0, grouped=True)
    D.assert_disjoint_groups(df, a, b)
    with pytest.raises(AssertionError):
        D.assert_disjoint_groups(df, *D.random_split(df, 0.5, seed=1))


def test_external_index(synthetic_root):
    m = {"bacterial_leaf_blight": "Bacterial leaf blight", "brown_spot": "Brown spot", "normal": "Health"}
    ext = D.build_external_index(synthetic_root / "paddy", m, CLASSES, max_per_class=2)
    assert len(ext) == 6
    assert set(ext["label"]) == {0, 1, 2}


def test_phash_groups_identical_images(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    b = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    for i, im in enumerate([a, a, b]):
        im.save(tmp_path / f"{i}.png")
    g = D._phash_groups([tmp_path / f"{i}.png" for i in range(3)])
    assert g[0] == g[1] != g[2]

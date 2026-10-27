"""Reproducible outer holdout and inner validation partitions."""
from sklearn.model_selection import train_test_split


def split_holdout(frame, target, *, test_size=0.2, random_state=42, stratified=True):
    return train_test_split(
        frame, test_size=test_size, random_state=random_state,
        stratify=frame[target] if stratified else None,
    )


def split_train_validation_holdout(frame, target, *, test_size=0.2,
                                   validation_size=0.2, random_state=42,
                                   stratified=True):
    train, holdout = split_holdout(
        frame, target, test_size=test_size, random_state=random_state,
        stratified=stratified,
    )
    train, validation = split_holdout(
        train, target, test_size=validation_size, random_state=random_state,
        stratified=stratified,
    )
    return train, validation, holdout

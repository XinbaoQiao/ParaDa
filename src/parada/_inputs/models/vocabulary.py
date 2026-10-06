from __future__ import annotations

from dataclasses import dataclass

from timm.data.imagenet_info import ImageNetInfo

from parada._inputs.hashing import canonical_sha256


@dataclass(frozen=True, slots=True)
class SourceVocabulary:
    id: str
    class_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    row_order_hash: str

    @classmethod
    def from_names(
        cls,
        vocabulary_id: str,
        class_names: list[str] | tuple[str, ...],
        *,
        class_ids: list[str] | tuple[str, ...] | None = None,
    ) -> SourceVocabulary:
        names = tuple(str(name) for name in class_names)
        ids = (
            tuple(str(item) for item in class_ids)
            if class_ids is not None
            else tuple(str(index) for index in range(len(names)))
        )
        if not names or len(names) != len(ids):
            raise ValueError("source vocabulary IDs and names must be nonempty and aligned")
        if len(set(ids)) != len(ids):
            raise ValueError("source vocabulary class IDs must be unique")
        return cls(
            id=vocabulary_id,
            class_ids=ids,
            class_names=names,
            row_order_hash=canonical_sha256({"ids": ids, "names": names}),
        )

    def validate_head_rows(self, row_count: int) -> None:
        if row_count != len(self.class_names):
            raise ValueError(
                f"classifier head has {row_count} rows but vocabulary has {len(self.class_names)}"
            )


def imagenet_vocabulary(subset: str = "imagenet-22k") -> SourceVocabulary:
    info = ImageNetInfo(subset)
    descriptions = info.label_descriptions(detailed=False)
    if not isinstance(descriptions, list):
        raise TypeError("timm ImageNet descriptions unexpectedly returned a mapping")
    return SourceVocabulary.from_names(
        vocabulary_id=subset, class_ids=info.label_names(), class_names=descriptions
    )

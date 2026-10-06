from __future__ import annotations

from types import MappingProxyType
from typing import Literal

from parada._inputs.hashing import canonical_sha256

OfficialDatasetId = Literal["cifar10", "cifar100", "dtd", "oxford_pets"]
OFFICIAL_CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer/elk",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
OFFICIAL_CIFAR100_CLASS_NAMES = (
    "apple",
    "aquarium fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak tree",
    "orange",
    "orchid",
    "otter",
    "palm tree",
    "pear",
    "pickup truck",
    "pine tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow tree",
    "wolf",
    "woman",
    "worm",
)
_DTD_CANONICAL_CLASS_NAMES = (
    "banded",
    "blotchy",
    "braided",
    "bubbly",
    "bumpy",
    "chequered",
    "cobwebbed",
    "cracked",
    "crosshatched",
    "crystalline",
    "dotted",
    "fibrous",
    "flecked",
    "freckled",
    "frilly",
    "gauzy",
    "grid",
    "grooved",
    "honeycombed",
    "interlaced",
    "knitted",
    "lacelike",
    "lined",
    "marbled",
    "matted",
    "meshed",
    "paisley",
    "perforated",
    "pitted",
    "pleated",
    "polka-dotted",
    "porous",
    "potholed",
    "scaly",
    "smeared",
    "spiralled",
    "sprinkled",
    "stained",
    "stratified",
    "striped",
    "studded",
    "swirly",
    "veined",
    "waffled",
    "woven",
    "wrinkled",
    "zigzagged",
)
OFFICIAL_DTD_CLASS_NAMES = tuple(f"{name} texture" for name in _DTD_CANONICAL_CLASS_NAMES)
_OXFORD_PETS_CANONICAL_CLASS_NAMES = (
    "Abyssinian",
    "American Bulldog",
    "American Pit Bull Terrier",
    "Basset Hound",
    "Beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "Boxer",
    "British Shorthair",
    "Chihuahua",
    "Egyptian Mau",
    "English Cocker Spaniel",
    "English Setter",
    "German Shorthaired",
    "Great Pyrenees",
    "Havanese",
    "Japanese Chin",
    "Keeshond",
    "Leonberger",
    "Maine Coon",
    "Miniature Pinscher",
    "Newfoundland",
    "Persian",
    "Pomeranian",
    "Pug",
    "Ragdoll",
    "Russian Blue",
    "Saint Bernard",
    "Samoyed",
    "Scottish Terrier",
    "Shiba Inu",
    "Siamese",
    "Sphynx",
    "Staffordshire Bull Terrier",
    "Wheaten Terrier",
    "Yorkshire Terrier",
)
OFFICIAL_OXFORD_PETS_CLASS_NAMES = tuple(
    f"{name}, a type of pet" for name in _OXFORD_PETS_CANONICAL_CLASS_NAMES
)
_OFFICIAL_CLASS_NAMES = MappingProxyType(
    {
        "cifar10": OFFICIAL_CIFAR10_CLASS_NAMES,
        "cifar100": OFFICIAL_CIFAR100_CLASS_NAMES,
        "dtd": OFFICIAL_DTD_CLASS_NAMES,
        "oxford_pets": OFFICIAL_OXFORD_PETS_CLASS_NAMES,
    }
)
OFFICIAL_CLASS_ORDER_SHA256 = MappingProxyType(
    {
        dataset_id: canonical_sha256(class_names)
        for dataset_id, class_names in _OFFICIAL_CLASS_NAMES.items()
    }
)


def resolve_official_dataset_id(dataset_id: str) -> OfficialDatasetId:
    """Resolve an exact formal dataset ID without heuristic prefix matching."""
    if dataset_id == "cifar10":
        return "cifar10"
    if dataset_id == "cifar100":
        return "cifar100"
    if dataset_id == "dtd":
        return "dtd"
    if dataset_id == "oxford_pets":
        return "oxford_pets"
    raise ValueError(
        f"official_code_13a2240 class names support only "
        f"cifar10/cifar100/dtd/oxford_pets, found {dataset_id!r}"
    )


def resolve_official_class_names(dataset_id: str) -> tuple[str, ...]:
    """Return the immutable prompt class-name tuple for a formal dataset."""
    return _OFFICIAL_CLASS_NAMES[resolve_official_dataset_id(dataset_id)]

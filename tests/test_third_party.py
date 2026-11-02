"""The vendored Limite model code must stay byte-identical to the pinned upstream revision."""

import hashlib
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "src/limite_experiments/third_party/limite"
PINNED = {
    "modeling_limite.py": "8d44f9e8f7e0c7d811ecab68e9e93abda197cba01f48514935d917c30953d0ea",
    "configuration_limite.py": "69f101ad280b14b3ab9e9b0076685d022191cc5abcf4981666867c0249522ddb",
    "registration.py": "2c9bbd09280a57b68064b5853ce6208c37d0342a299abfd149764aa481cad8ca",
    "contract.py": "253d7511b4c89f74eb7f4919bf466a841844a4699079d148deb80f8e4ed61fe6",
    "violetto_config.json": "d080b1f7a04a44c44cb0deb1d3fcb7ce99212fce5573f5e15b74b7f84b74463d",
}


def test_vendored_sources_are_pinned():
    for name, digest in PINNED.items():
        assert hashlib.sha256((DIR / name).read_bytes()).hexdigest() == digest, name

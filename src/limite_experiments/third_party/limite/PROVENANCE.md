# Third-party code: Limite model implementation

These files are **unmodified copies** of the Hugging Face remote code of
[`paradigma-inc/limite-1b-violetto`](https://huggingface.co/paradigma-inc/limite-1b-violetto) at revision
`b1f3d572ccacb6919f4d64c321b70ba034ddaef2`, the checkpoint pinned by
[`limite-violetto`](https://github.com/paradigma-inc/limite-violetto) @ `7ddad34`.

- Copyright: Paradigma (paradigma-inc).
- Licence: Apache License 2.0 (see the model card and the upstream repository). A copy of the licence is in
  `LICENSE` at the root of this repository.
- Source: `https://huggingface.co/paradigma-inc/limite-1b-violetto/resolve/<revision>/<file>`.

| File | sha256 |
|---|---|
| `modeling_limite.py` | `8d44f9e8f7e0c7d811ecab68e9e93abda197cba01f48514935d917c30953d0ea` |
| `configuration_limite.py` | `69f101ad280b14b3ab9e9b0076685d022191cc5abcf4981666867c0249522ddb` |
| `registration.py` | `2c9bbd09280a57b68064b5853ce6208c37d0342a299abfd149764aa481cad8ca` |
| `contract.py` | `253d7511b4c89f74eb7f4919bf466a841844a4699079d148deb80f8e4ed61fe6` |
| `violetto_config.json` (the checkpoint's `config.json`) | `d080b1f7a04a44c44cb0deb1d3fcb7ce99212fce5573f5e15b74b7f84b74463d` |

They are used only to run Limite's forward pass with a custom attention function (`scripts/value_study.py judge`).
`tests/test_third_party.py` checks the hashes, so any edit is caught.

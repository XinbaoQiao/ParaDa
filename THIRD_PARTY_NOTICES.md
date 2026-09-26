# Dependencies and attribution

The numerical source is extracted from the project's own implementation. No external
repository source, datasets, model weights, or dependency binaries are bundled.

Runtime dependencies are obtained separately by the installer:

| Dependency | Upstream license | Upstream |
| --- | --- | --- |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch |
| Safetensors | Apache-2.0 | https://github.com/huggingface/safetensors |

Development/build tools include pytest (MIT) and Hatchling (MIT). Their installed
distributions supply their own license notices. Their code is not incorporated here.

The text-to-visual mapping follows the behavior of the reference alignment work
`recycling4vlalignment` (revision `13a22403018292b5877a77a25b1f91a112b043c4`, MIT),
implemented independently. The labeled-support stage trains client-local additive residuals with
cross-entropy and aggregates them by sample count over five rounds. These acknowledgments do not imply endorsement or numerical
equivalence with the original projects.

Pretrained models, text-generation services, and datasets are external inputs subject
to their respective terms. This review snapshot grants no final project-wide
redistribution license and does not redistribute those assets.

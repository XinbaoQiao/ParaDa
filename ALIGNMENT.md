# Implementation and manuscript alignment

This snapshot is an extraction of existing numerical operators, with a new portable
tensor-input interface. It is not a fresh implementation of the equations or a claim
that every historical experiment followed the supplied manuscript.

| Item | Current implementation | Supplied manuscript |
| --- | --- | --- |
| Source mapper | 3072-wide MLP, source-only fitting, 500 epochs | Same architecture and optimizer profile |
| Target text | Three supplied embedding views, mapped independently | Three GPT-4o descriptions encoded with CLIP-B/32 |
| K=0 mixture | rho=0.4, selected by a train-only validation workflow | rho=0.5, described as fixed |
| K>0 mixture | rho=1.0, obtained by that selection workflow | rho=1.0, described as fixed |
| Query access in adaptation | No query features or labels | No query labels |

The `current` profile uses 0.4/1.0. The optional `manuscript` profile uses 0.5/1.0.
These names describe coefficients, not evidence tiers. Selection history is not
reclassified as a fixed-before-validation design. Neither profile implies that
the manuscript tables have been reproduced by this packaged interface.

The source MLP constructor in the existing shared-checkpoint implementation runs
before the function resets the random generators to the requested seed. Therefore
the seed alone does not determine fresh initial weights. This export preserves that
ordering; a reused hash-verified source checkpoint is the reliable way to continue
the same trajectory. Correcting initialization would change fresh-run results and
requires a separately validated experiment. CPU/GPU bitwise equivalence is not claimed.

Historical runners used other coefficients and description-generator identities.
They are not interchangeable with this profile. The supplied tensors must carry
their own model, description, class-order, and preprocessing provenance. The package
does not contain the assets needed to certify the historical score tables.

Unrelated analytic adaptation methods, alternative classifier ensembles, abandoned
experiments, orchestration state, and runtime outputs are outside this source release.

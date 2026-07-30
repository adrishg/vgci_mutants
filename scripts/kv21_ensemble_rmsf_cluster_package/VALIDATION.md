# Validation record

The package was validated before release with the uploaded files:

- one F412L masked AlphaFold2 tetramer recycle structure;
- 8SD3;
- 8SDA.

## Automated tests

```text
5 tests passed
```

The tests cover:

- correct conversion of three-letter amino-acid names;
- Kabsch rigid-body recovery;
- cyclic mapping generation;
- geometric pore-ring ordering;
- inclusive mask-range construction.

## End-to-end test

The complete workflow was executed on the uploaded AlphaFold model:

```text
input inspection
reference mapping and alignment
model alignment
shard merge
chain-resolved RMSF profile generation
symmetry-averaged profile generation
experimental-reference metric generation
```

The model aligned successfully with complete raw 1–600 Cα coverage in all four chains.

The geometric ring-order check found:

```text
8SD3 ring order: A-B-D-C
8SDA ring order: A-B-C-D
uploaded AF2 model ring order: A-B-C-D
```

This demonstrates why alphabetical chain order cannot be assumed for experimental tetramers.

The 8SDA-to-8SD3 unmasked S1–S3 core fit was:

```text
1.068 Å over 232 matched Cα atoms
```

The uploaded F412L masked model-to-8SD3 core fit was:

```text
4.843 Å over 232 matched Cα atoms
```

The raw-to-experimental residue mapping included:

```text
raw 405 -> experimental 403
raw 414 -> experimental 412
```

No uploaded PDB files are redistributed inside the package.

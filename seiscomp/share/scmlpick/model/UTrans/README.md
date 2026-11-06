# UTrans (Foundation-EQCCT) model directory

This directory is where the fine-tuned **Foundation-EQCCT** checkpoints must be
placed to run scmlpick with `eqcct.architecture = utrans_eqcct`.

## What goes here

Two `.h5` files, analogous to the plain-EQCCT `test_trainer_XXX.h5` files in
`../ModelPS/`:

- A P-picker checkpoint (`eqcct.Pmodel`)
- An S-picker checkpoint (`eqcct.Smodel`)

Each file must be a **combined** checkpoint containing both the fine-tuned
U-Trans encoder branch and its picking head for that phase, saved as
`model.save_weights(...)` from the exact graph built by
`predictor.utrans_eqcct.build_utrans_eqcct_graph()` (one branch at a time,
i.e. `Model(inputs, logitp)` for P and `Model(inputs, logits)` for S).

## These files do NOT exist yet

Unlike plain EQCCT, no pretrained Foundation-EQCCT picking checkpoint ships
with the U-Trans repository — only the Stage 1 self-supervised encoder
(`UTrans_Foundation.h5`) is released. Producing the P/S checkpoints requires
fine-tuning, offline, using the U-Trans repository's own training notebooks:

```
examples/downstream/pwave_eqcct/01_train.ipynb
examples/downstream/swave_eqcct/01_train.ipynb
```

against a labeled seismic dataset (STEAD/TXED/INSTANCE-formatted HDF5 with
`p_arrival_sample`/`s_arrival_sample`/`snr_db` attributes), starting from
`weights/UTrans_Foundation.h5`. Until that fine-tuning is done and the
resulting checkpoints are copied here, selecting `utrans_eqcct` will fail at
`model.load_weights(...)` with a missing-file error.

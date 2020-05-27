# Direction decoding

This repository contains code used in the publication "Effects of aging on the
encoding of walking direction in the human brain" (Koch, Li, Polk &
Schuck, 2020).

For details please contact the corresponding author (Christoph Koch,
  koch@mpib-berlin.mpg.de).

---

## Repository structure

Below you will find a short guide on folder contents and script purposes.

### Overall structure

```
.
├── analysis
│   ├── backwards_walking
│   │   └── analysis_backwards_walking.rmd
│   └── curve_fitting
│       └── create_confusion_matrices.Rmd
├── behavioral
│   ├── logfile_1_create.R
│   └── logfile_2_add_buffer.R
├── decoding
│   ├── backwards_decoding
│   │   └── train_beta_test_raw
│   │       ├── results
│   │       │   └── _combine_results.py
│   │       └── train_beta_test_raw.py
│   ├── train_beta_test_beta
│   │   ├── results
│   │   │   └── _combine_results.py
│   │   └── train_beta_test_beta.py
│   └── train_beta_test_raw
│       ├── results
│       │   └── _combine_results.py
│       └── train_beta_test_raw.py
└── README.md
```

### Detailed structure

```
├── analysis
```

Folder containing scripts using the decoding data (classification accuracies)
and transforming them into other formats to ask more specific questions (e.g.
differences in tuning width).

```
│   ├── backwards_walking
```

Folder containing scripts to prepare further analysis of predictions made by a
classifier that was trained on beta maps of forward walking events but tested
on backwards walking events.

```
│   │   └── analysis_backwards_walking.rmd
```

Script creating confusion functions for backwards walking events as well as
visual influence scores for each participant and specified mask.

```
│   └── curve_fitting
```

Folder containing scripts to prepare analysis of predictions made by a
classifier that was trained on forward walking events and tested on either beta
maps of forward walking events or single TRs of forward walking events.

```
│       └── create_confusion_matrices.Rmd
```

Script calculating confusion functions and fitting both models
(Gaussian & uniform) to confusion function. This results in parameters of
fitted curves (e.g. tuning width in case of Gaussian model) as well as the
fitted curves for both models for each participant and each specified mask.

```
├── behavioral

```

Folder contianing scripts working on behavioral data (logfiles of movement in
  the virtual environment during scanning).

```
│   ├── logfile_1_create.R
```

Script to create movement events for each TR scanned.

```
│   └── logfile_2_add_buffer.R
```

Script to add buffering (even vs. odd events) to movement data.

```
├── decoding
```

Folder containing scripts for all kinds of direction decoding done. In short,
a classifier tries to predict walking directions in a testing set using the
information of walking direction and associated brain activation patterns in a
training set.

```
│   ├── backwards_decoding
```

Contains scripts to obtain predictions of a classifier trained on forward
walking beta maps and tested on single TRs of backwards walking events.

```
│   │   └── train_beta_test_raw
```

Folder to clarify that training was done on beta maps and testing was done on
single TRs.

```
│   │       ├── results
```

Folder containing results of the above classification

```
│   │       │   └── _combine_results.py
```

Script to combine results of each participant into a larger data frame since
decoding scripts are run for each subject individually (to enable parallel
  processing).

```
│   │       └── train_beta_test_raw.py
```

Script training a classifier on beta maps of forward walking events and
testing it on single TRs of backwards walking events to predict walking
direction.

```
│   ├── train_beta_test_beta
```

Contains scripts to obtain predictions of a classifier trained on forward
walking beta maps and tested on forward walking beta maps.

```
│   │   ├── results
│   │   │   └── _combine_results.py
```

See above.

```
│   │   └── train_beta_test_beta.py
```
Script training a classifier on beta maps of forward walking events and
testing it on beta maps of forward walking events to predict walking
direction.


```
│   └── train_beta_test_raw
```

Folder containing decoding analysis of training on beta maps of forward walking
events and testing on single TRs of forward walkinf events.

```
│       ├── results
│       │   └── _combine_results.py
│       └── train_beta_test_raw.py
```

See above.

---

# Beyond visual plausibility: quantitative assessment of attention-based interpretability in dermoscopic lesion classification

Code, splits and reproduction commands for the paper. Every number, table and figure in the
paper can be produced from this repository.

The work examines a two-stage hierarchical classifier for HAM10000 with two lesion-focused
modules, TEME (topology-enhanced multi-scale encoding) and LTV (localize-then-verify attention),
and asks two questions: does the design improve classification accuracy under an evaluation
protocol strict enough that the answer does not depend on a single training run, and do its
attention maps localize the lesion as claimed.

---

## 1. Setup

```bash
conda create -n skin python=3.10 && conda activate skin
pip install torch torchvision timm pandas numpy scikit-learn scipy matplotlib pillow tqdm
```

### Data

Both the images and the lesion masks are in the same Harvard Dataverse record,
<https://doi.org/10.7910/DVN/DBW86T>:

| File | Contents |
|---|---|
| `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip` | the 10,015 dermatoscopic images |
| `HAM10000_metadata.csv` | diagnosis, lesion identifier, age, sex, localization |
| `HAM10000_segmentations_lesion_tschandl.zip` | binary lesion masks for all 10,015 images |

If you use this data, cite the dataset and, for the masks, the study they were produced for:

* Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of
  multi-source dermatoscopic images of common pigmented skin lesions. *Sci Data* 5, 180161 (2018).
* Tschandl, P. *et al.* Human-computer collaboration for skin cancer recognition.
  *Nat Med* 26, 1229-1234 (2020).

The dataset is released under CC BY-NC 4.0.

```bash
git clone https://github.com/shinhosoo/beyond-visual-plausibility.git
cd beyond-visual-plausibility
# unzip the images under one directory and the masks under another, then
ln -s /path/to/HAM10000_segmentations_lesion_tschandl masks_ham10000
cp env.sh.example env.sh        # edit the paths inside
source env.sh                   # sets PYTHONPATH and the HAM_* variables
```

`env.sh` must be sourced before every run. It sets the data root, the split root, the output
root and the GPU index. The defaults inside the code already match the paper
(`HAM_WEIGHT_DECAY=5e-5`, `HAM_AUGMENT=paper`), so sourcing it is about paths rather than
hyper-parameters.

### Splits

The split used in the paper is in `splits_paper_seed42/`, so nothing has to be generated before
running anything. It can be rebuilt and checked with:

```bash
python make_splits.py --metadata $HAM_DATA_ROOT/HAM10000_metadata.csv \
                      --out splits_paper_seed42 --seed 42
python check_split.py
```

---

## 2. Evaluation protocol

The protocol affects the numbers more than any design choice, so it is stated explicitly.

* **Lesion-level splitting.** Images of the same lesion never appear in two subsets
  (7,010 / 981 / 2,024 images), generated once with seed 42. The seeds reported in the paper vary
  the training run rather than the partition. Accuracies above 0.99 reported elsewhere on
  HAM10000 come from image-level splits.
* **One split for both stages.** The Stage 2 branches are trained only on subsets of the Stage 1
  training split, so no evaluation image is seen at any stage of the pipeline. `make_splits.py`
  enforces this and `check_split.py` verifies it.
* **Cross-validation for design comparisons.** Folds are built from the pooled training and
  validation images with `GroupKFold(groups=lesion_id)`. The test split is never part of a fold.
  The validation split alone is too small to separate variants whose accuracies differ by less
  than a percentage point, so it is not used to choose among them.
* **Seeds include the router.** `SEED` in `SkinCancer/configs/training_config.py` is a constant,
  so running `SkinCancer.main.run_stage1` with a different seed would reuse the same Stage 1
  weights. `experiments/seed_run.py` replaces `SEED` and `STAGE1_MODEL_PATH` at run time and
  `scripts/run_seed.sh` goes through it. Multi-seed results must be produced this way, otherwise the
  reported spread omits routing variability.
* **Noise floor.** Retraining an identical configuration changes fold accuracy by up to 1.0 pp
  and test accuracy by +/- 0.0114 over five seeds. Differences below this are not interpreted.
* **Masks.** Lesion masks are used for the alignment loss during training and as ground truth
  for the overlap metrics. No trained model needs a mask at inference.

---

## 3. Reproducing the paper

All commands assume `source env.sh`. Long runs are best started with `nohup ... &`.

### 3.1 Base models

```bash
# single-stage classifier, five cross-validation folds (backbone of the interpretability study)
for K in 0 1 2 3 4; do python -m experiments.residual --model flat --fold $K --seed 42; done

# single-stage classifier on the standard split, seeds 42-46
bash scripts/run_seeds.sh
```

### 3.2 Ablation of TEME and LTV over five seeds

```bash
SEEDS="43 44" HAM_GPU=0 nohup bash scripts/run_seed.sh > logs/s4344.log 2>&1 &
SEEDS="45 46" HAM_GPU=1 nohup bash scripts/run_seed.sh > logs/s4546.log 2>&1 &
bash scripts/run_all_ablations.sh          # seed 42
python -m experiments.acc_table
```

### 3.3 Class-wise metrics, calibration and MEL--NV confusion

```bash
python -m experiments.paper_tables --out figs
```

### 3.4 Explanation quality

```bash
BATCH=A bash scripts/run_loc_batch.sh           # unsupervised LTV and design decomposition
BATCH=B bash scripts/run_loc_batch.sh
LAMS="0.2 1.0 5.0" bash scripts/run_next.sh     # mask supervision
for K in 0 1 2 3 4; do
  python -m experiments.faithful --fold $K --names orig sup0.2_bias_local
done
python -m experiments.loc_table             # overlap
python -m experiments.faith_table           # faithfulness
```

### 3.5 How many masks the supervision needs

```bash
FRACS="0.1 0.25 0.5" bash scripts/run_frac.sh
python -m experiments.loc_table | grep frac
```

### 3.6 Backbone baselines and the second-backbone replication

```bash
for BB in convnext_tiny resnet50 efficientnet_b3 vit_small_patch16_384; do
  BB=$BB STEP=flat bash scripts/run_backbone.sh
done
BB=convnext_tiny STEP=unsup bash scripts/run_backbone.sh
BB=convnext_tiny STEP=sup   bash scripts/run_backbone.sh
HAM_OUTPUT_ROOT=$PWD/outputs_convnext_tiny python -m experiments.loc_table
HAM_OUTPUT_ROOT=$PWD/outputs_convnext_tiny python -m experiments.faith_table
```

### 3.7 Design variants and statistics

```bash
bash scripts/run_residual.sh ; bash scripts/run_residual_lr.sh    # correction configurations
bash scripts/run_explore.sh  ; bash scripts/run_explore_fix.sh    # LTV as the classification head
MODES="A B" bash scripts/run_mid.sh                       # mid-level injection
bash scripts/run_hier_sup.sh                              # mask-supervised hierarchy
STEPS="lcrop bgdim" bash scripts/run_bg.sh                # lesion-constrained training
bash scripts/run_zoom.sh                                  # re-input of the magnified lesion
python -m experiments.residual_cv --models flat d1 d1lr d3lr midA midB lcrop bgdim h1sup
python -m experiments.stats_extra                     # equivalence tests, Holm correction
```

### 3.8 Checking the reported numbers

Once the predictions above exist, every number reported in the paper can be recomputed from them
and compared with the manuscript:

```bash
python -m experiments.verify_paper
```

The script reads only saved predictions, so it needs no GPU. Each line prints the recomputed
value next to the one in the paper and marks it as `ok` or `MISMATCH`.

### 3.9 Figures

```bash
python -m experiments.fig_style --which 2 4 6 7 --out figs
python -m experiments.fig_qual --fold 0 --n 300 --out figs/fig3
python -m experiments.fig_vis --which A B --fold 0 --n 150 --out figs
```

Progress of the running batches:

```bash
bash scripts/batch_status.sh
```

---

## 4. Layout

```
SkinCancer/              backbone, data, training and evaluation
  configs/               paths and hyper-parameters, all overridable by HAM_* variables
  models/                stage1, stage2, TEME, LTV, routing
  data/                  split readers, loaders, transforms
  train/, eval/          training loops, losses, metrics, bootstrap CIs
  main/                  entry points for Stage 1, Stage 2 and soft routing
experiments/             the experiments of this paper
  residual.py            cross-validated training; correction configurations
  residual_lr.py         the same with a larger head learning rate
  explore*.py            LTV used as the classification head
  midres.py              injection of refined features into the backbone
  hier*.py               hierarchical design variants
  localize.py            overlap of a saliency map with the lesion mask
  localize_sup.py        mask supervision of the attention
  localize_sup2.py       mask supervision keeping the global path
  sup_frac.py            how many masks the supervision needs
  faithful.py            deletion and insertion test
  ltv_fix.py             LTV with the masking order corrected
  zoom.py, zoom2.py      magnified re-input; context-margin sweep
  aug_cv.py, aug2.py     class-aware and lesion-aware augmentation
  seed_run.py            runs a module with a different seed, retraining Stage 1
  bb_run.py              runs a module with a different backbone
  *_table.py             print the result tables
  stats_extra.py         equivalence tests and multiple-comparison correction
  verify_paper.py        recompute the reported numbers and compare them with the paper
  fig_*.py               figures
ablation/                seed-42 runs of the ablation settings
splits_paper_seed42/     lesion-level splits used for every reported number
scripts/                 batch runners (run_*.sh) and batch_status.sh
make_splits.py           build the splits
check_split.py           verify that the splits contain no leakage
```

## 5. Notes

* `experiments/bb_run.py` and `experiments/seed_run.py` change the backbone, the seed and the output
  directory at run time, so the sources under `SkinCancer/` stay unmodified, and each configuration
  writes to its own `outputs*/` directory.
* `ablation/` and `scripts/run_all_ablations.sh` produce the seed-42 column of the multi-seed
  tables. They read the same splits as everything else, so their Stage 2 branches are trained
  only on subsets of the Stage 1 training split.
* Checkpoints, predictions and figures are not tracked here. The commands above regenerate them
  under `outputs/`, one subdirectory per configuration.

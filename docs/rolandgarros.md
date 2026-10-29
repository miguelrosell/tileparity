# First real run on rolandgarros (UZH)

A step-by-step guide for the first run on real data: two samples at one scale (0.5 µm/px, 112 µm tiles), with the mask comparison and the 3-D UMAP QC switched on:

* **ZEN36** — HEST-1k, standard Visium, colorectal cancer (cohort B);
* **A_P1_CRC** — Visium HD, colorectal cancer, patient P1 (cohort A).

Everything — software, downloads and runs — happens on the server; nothing is needed on your laptop except a terminal. rolandgarros has 128 CPUs, 512 GB of RAM, 58 TB of disk and an NVIDIA L4 GPU (24 GB); wimbledon is the login node. rolandgarros is shared with the rest of the group, so the jobs below ask for a quarter of it (32 CPUs, 128 GB).

**Two software routes.** The login node has no conda, no module system and no containers, so this guide follows **route A (conda)**, which needs no administrator: conda is installed in your own folder, Nextflow comes from bioconda, and Nextflow creates the pipeline's three software environments from `containers/*/environment.yml`. The whole run is **one SLURM job on rolandgarros**; inside it Nextflow runs every task on that node (its *local* executor), so no partition names or GPU requests per task are needed. Route B (containers) is at the end, for when Apptainer is available.

## 0. One folder for everything

Choose a folder on a large disk and call it `$TP` in what follows. On rolandgarros the home directories are on the server's 58 TB disk, so `$HOME/tileparity` (i.e. `/home/<user>/tileparity`) is the natural choice (`df -h $HOME` shows the free space). Everything goes under it:

```
$TP/
├── miniforge3/      conda itself (~0.5 GB)
├── env.sh           the environment variables, read at the start of every session and job
├── tileparity/      the code (git clone)
├── cache/           Nextflow's files, the pipeline's conda environments (~10 GB), Hugging Face's cache
├── models/uni2h/    the UNI2-h weights (2.7 GB)
├── data/            HEST and Visium HD files (~10 GB for these two samples)
└── runs/<name>/     one folder per run: its job script, job log, Nextflow's log, work/ (tens of GB;
                     can be deleted after a successful run) and results/
```

## 1. Log in and check the machine (5 minutes)

```bash
ssh <user>@wimbledon.mls.uzh.ch         # the login node
srun --pty -c 4 --mem 16G -t 4:00:00 bash               # a shell on rolandgarros, through SLURM
hostname && nproc && free -g && df -h $HOME             # rolandgarros; CPUs, memory, disk
curl -sI https://conda.anaconda.org | head -1           # internet access: "HTTP/2 200" (needed to install and download)
sinfo -o "%P %G %l %c %m"               # SLURM partitions, GPUs (gres), time limits, CPUs and memory per node
srun --gres=gpu:1 -c 1 --mem 1G -t 5 nvidia-smi         # the L4 and its driver (the PyTorch build used needs >= 525)
```

SLURM only shows the GPU to jobs that ask for it: without `--gres=gpu:1`, `nvidia-smi` answers "No devices were found". rolandgarros has a single partition, `main`, which is the default (`main*` in `sinfo`), and its GPU is `gpu:1`, so the job scripts below need no `-p` and ask for the GPU with `--gres=gpu:1`. Type `exit` to leave the `srun` shell when you are done.

## 2. Install conda and Nextflow (once, 10 minutes)

```bash
export TP=$HOME/tileparity               # your folder
mkdir -p $TP && cd $TP
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $TP/miniforge3     # -b: no questions; -p: install here
source $TP/miniforge3/etc/profile.d/conda.sh
conda create -y -n nf -c conda-forge -c bioconda nextflow=26.10.3 openjdk=21 huggingface_hub git
```

`nf` is a small environment with Nextflow, Java 21, git and `huggingface_hub` (for the downloads). Then write `$TP/env.sh`:

```bash
cat > $TP/env.sh <<'END'
export TP=$HOME/tileparity
source $TP/miniforge3/etc/profile.d/conda.sh
conda activate nf
export NXF_HOME=$TP/cache/nextflow              # Nextflow's own files
export NXF_CONDA_CACHEDIR=$TP/cache/conda       # the pipeline's environments: created once, reused by every run
export HF_HOME=$TP/cache/huggingface            # Hugging Face downloads
export PIP_CACHE_DIR=$TP/cache/pip              # pip's downloads while the environments are created
export NXF_OPTS='-Xms1g -Xmx4g'                 # memory of Nextflow itself
mkdir -p $NXF_HOME $NXF_CONDA_CACHEDIR $HF_HOME $PIP_CACHE_DIR   # Nextflow stops if its folder does not exist
END
source $TP/env.sh
nextflow -version && java -version              # 26.10.3 and 21
```

From now on, every session starts with `source $HOME/tileparity/env.sh`.

## 3. Get the code (1 minute)

```bash
cd $TP
git clone https://github.com/miguelrosell/tileparity.git
git -C tileparity log -1 --oneline       # the version you run; later updates: git -C tileparity pull
```

The pipeline is run from this folder (`nextflow run $TP/tileparity`), so the code that runs is exactly the code you see.

## 4. Test the set-up on the synthetic data (20–40 minutes the first time)

The test profile runs the whole pipeline on a tiny synthetic slide (no data and no token needed; UNI2-h runs as a random 1-block network in test mode). Run on the GPU, it is also the first test of the CUDA path. Every run gets its own folder under `$TP/runs/`, with its job script inside; the job starts in the folder it was submitted from (`cd $SLURM_SUBMIT_DIR`), so its log, Nextflow's files, `work/` and `results/` all land in that folder:

```bash
mkdir -p $TP/runs/test && cd $TP/runs/test
cat > test.sbatch <<'END'
#!/bin/bash
#SBATCH --job-name=tp_test
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
source $HOME/tileparity/env.sh
cd $SLURM_SUBMIT_DIR
nvidia-smi
nextflow run $TP/tileparity -profile test,conda,gpu --outdir results -w work
END
sbatch test.sbatch                       # from this folder; prints the job id
squeue -u $USER                          # is it running?
tail -f tp_test_<jobid>.log              # follow it (Ctrl-C stops following, not the job)
```

The first run creates the three environments in `$TP/cache/conda` (Python tools; PyTorch with CUDA, about 3 GB; R with Bioconductor), which takes most of the time; later runs reuse them. It should end with `tileparity finished: results in results`. If it does not, see section 9.

## 5. Download the data (30–60 minutes, mostly waiting)

You need a Hugging Face token (`HF_TOKEN`) with access to `MahmoodLab/hest` and `MahmoodLab/UNI2-h` (request access on both pages first). Downloads can run in `tmux` (so they survive a lost connection) or as a job (`sbatch --wrap "source $HOME/tileparity/env.sh && ..."`).

```bash
source $HOME/tileparity/env.sh
export HF_TOKEN=hf_xxx
```

**HEST-1k, ZEN36** (standard Visium, CRC). The script writes a samplesheet row with HEST's own pixel size:

```bash
python $TP/tileparity/scripts/hest_download.py --ids ZEN36 --outdir $TP/data/hest --samplesheet $TP/data/zen36.csv --cohort B
```

**Visium HD, P1 CRC.** On the 10x Genomics page [Visium HD CytAssist Gene Expression Libraries of Human CRC](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc), choose the P1 CRC sample and copy the links of two files: the **microscope image** (`.btf`, the H&E image Space Ranger was given, `Visium_HD_Human_Colon_Cancer_tissue_image.btf`; not the CytAssist image) and the **binned outputs** (`.tar.gz`). Put them in a tab-separated table and download:

```bash
printf 'sample\tkind\turl\nA_P1_CRC\timage\t<image link>\nA_P1_CRC\tbinned_outputs\t<binned outputs link>\n' > $TP/data/p1_urls.tsv
bash $TP/tileparity/scripts/download_urls.sh $TP/data/p1_urls.tsv $TP/data/raw
ls $TP/data/raw/A_P1_CRC        # the .btf image and binned_outputs/ (square_002um, square_008um, square_016um)
```

**UNI2-h weights** (2.7 GB, once):

```bash
python $TP/tileparity/bin/tp_download_uni2h.py --outdir $TP/models/uni2h
ls -lh $TP/models/uni2h          # pytorch_model.bin and its checksum
```

## 6. Write the samplesheet

The samplesheet of this run has two rows: the ZEN36 row that `hest_download.py` wrote to `$TP/data/zen36.csv` (with HEST's pixel size and metadata), and a row for the Visium HD sample with the same columns. `tiling` stays empty (grid tiles for both; ZEN36's spots are split between tiles by area), and so does `mpp` for Visium HD (Space Ranger's files give it). This writes it, with the paths in full:

```bash
mkdir -p $TP/runs/two_samples && cd $TP/runs/two_samples
python - <<'END'
import csv, glob, os
tp = os.environ["TP"]
zen = list(csv.DictReader(open(f"{tp}/data/zen36.csv")))
hd = {"sample": "A_P1_CRC", "image": glob.glob(f"{tp}/data/raw/A_P1_CRC/*.btf")[0],
      "spatial": f"{tp}/data/raw/A_P1_CRC/binned_outputs", "spatial_format": "spaceranger_hd",
      "patient": "P1", "condition": "CRC", "cohort": "A"}
columns = list(dict.fromkeys(list(zen[0]) + list(hd)))
with open("samplesheet.csv", "w", newline="") as fh:
    out = csv.DictWriter(fh, fieldnames=columns, restval="")
    out.writeheader()
    out.writerows([hd] + zen)
END
cat samplesheet.csv
```

## 7. Run the two samples

In `$TP/runs/two_samples` (where the samplesheet is), `job.config` tells Nextflow how much of the machine it may use, and `run.sbatch` asks SLURM for a quarter of rolandgarros (32 CPUs, 128 GB) and the GPU; Nextflow then runs all tasks inside that allocation:

```bash
cd $TP/runs/two_samples
cat > job.config <<'END'
// Nextflow runs the tasks inside this SLURM job: never more CPUs and memory at once than the job has
executor {
    name   = 'local'
    cpus   = 32
    memory = '120 GB'
}
END
cat > run.sbatch <<'END'
#!/bin/bash
#SBATCH --job-name=tp_two
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.log
source $HOME/tileparity/env.sh
cd $SLURM_SUBMIT_DIR
nvidia-smi
nextflow run $TP/tileparity -profile conda,gpu -c job.config \
    --input samplesheet.csv --outdir results -w work \
    --max_cpus 32 --max_memory '120.GB' --max_time '24.h' \
    --uni2h_weights $TP/models/uni2h/pytorch_model.bin --uni2h_batch_size 64 \
    --scales_mpp 0.5 --mask_compare --min_tissue_extract 0.3 --qc_umap \
    -resume
END
sbatch run.sbatch && tail -f tp_two_<jobid>.log
```

* `-profile conda,gpu`: software from the conda environments, UNI2-h on the GPU; no `uzh_rolandgarros` profile here, because that one sends every task to SLURM as a separate job.
* `-c job.config`: inside a job, Nextflow would otherwise count every CPU and all the memory of the machine as its own and could start more tasks at once than the job has room for. 120 of the 128 GB, so that Nextflow itself has room too.
* `--max_cpus 32 --max_memory '120.GB' --max_time '24.h'`: no single task asks for more than the job has (LBP asks for 32 CPUs, so the LBP tasks run one after the other).
* `--uni2h_batch_size 64`: 64 tiles at a time fit the L4's 24 GB.
* `--scales_mpp 0.5`: one scale for this first run. `--mask_compare`: the mask comparison figure. `--min_tissue_extract 0.3`: features from 30 % tissue on, so the 30/50/80 % thresholds can be compared afterwards without rerunning (`keep` still means ≥ 50 %). `--qc_umap`: the 3-D UMAP page.
* By default both extractors read both input sets: the slide (`rgb`) and the slide rebuilt from H and E only (`rgb_he`), with each slide's own H and E colours.
* `-resume`: if the job stops (time limit, error, a fix), submit the same script again: finished tasks are reused.

## 8. What to look at, in this order

Paths are relative to `$TP/runs/two_samples/results/`.

1. `pipeline_info/execution_report_*.html`: did every task finish, how long did each process take and how much memory did it use (this sets the resources of the full cohort).
2. `<sample>/mpp0.5000/prepare/<id>.overlay.png`: the blue units must lie exactly on the tissue (a shift means a wrong image or pixel size).
3. `prepare/<id>.mask_compare.png` and `prepare/<id>.tissue_thresholds.png`: is the saturation mask good enough, and which threshold looks right?
4. `channels/<id>.rgb_he_qc.png`: does the H&E-only version look like the slide, and where did it change? `channels/<id>.channels.json` has the slide's H and E colours (`stain_vectors`; a `note` appears if the standard ones had to be used).
5. `lbp/<id>.lbp_qc.png`: the share of irregular LBP patterns per radius on real tissue (which radii carry information).
6. `qc/mpp0.5000/combined_mpp0.5000.qc_umap.html` (copy it to your laptop, e.g. with `scp`, and open it in a browser): do the two samples separate in every view (batch effect) or only in some, and does `rgb_he` change that?
7. `st/<id>.aggregate.json` and `counts_summary.tsv`: share of molecules in the tiles, counts per tile.
8. `export/mpp0.5000/spe/combined_mpp0.5000.rds` loads in R (`reducedDimNames(spe)` lists `UNI2H_RGB`, `UNI2H_RGB_HE`, `LBP_RGB`, `LBP_RGB_HE`, `INTENSITY_HE`), and `metadata(spe)$tileparity$lineage` names the original images.

## 9. If something fails

* The log names the failing process and its work directory; `.command.err` and `.command.log` there hold the error, `.command.sh` the exact command. `.nextflow.log` in the run's folder has Nextflow's side.
* *`nextflow` says it cannot create a path under `cache/`*: the folders of `env.sh` do not exist yet; `source $HOME/tileparity/env.sh` creates them (its last line).
* *`sbatch` cannot find the script, or the results land somewhere else*: submit from the run's folder (`cd $TP/runs/<name>` first); the job works in the folder it was submitted from.
* *Creating a conda environment fails or is slow*: check the internet access (section 1) and the free space in `$TP/cache/conda`.
* *UNI2-h runs on the CPU or fails with CUDA errors*: the job log starts with `nvidia-smi`; if it shows no GPU, the `--gres` line is wrong for this cluster (see `sinfo` in section 1). A driver older than 525 is too old for the PyTorch build of the pipeline.
* *Job pending forever*: `squeue -u $USER` shows the reason, often that other jobs use the CPUs, memory or the GPU (`Resources`); `squeue` shows whose.
* *Out of memory or time*: a task killed for memory or time is retried once with twice the memory and time (up to `--max_memory` and `--max_time`); if it fails again, the error says which process. The whole job has 72 h; resubmitting it continues where it stopped.
* *Pixel-size or alignment errors* from Process 1 are deliberate stops: the message says which sources disagree.

## 10. Clustering the views (`--cluster`, tileparity 0.4.0 or later)

`--cluster` clusters every feature view of every slide on its own and measures how much the views agree ([design.md](design.md#optional-clustering-of-every-view-per-slide---cluster) explains the method). To add it to the run of section 7, update the code and add `--cluster` to the `nextflow run` command of `run.sbatch` (for example after `--qc_umap`), then submit the same script again:

```bash
source $HOME/tileparity/env.sh
git -C $TP/tileparity pull               # tileparity 0.4.0 or later
cd $TP/runs/two_samples
nano run.sbatch                          # add --cluster to the nextflow command
sbatch run.sbatch
```

The Python environment gains one package (igraph, for Leiden), so Nextflow creates it again and runs the whole pipeline again once (about as long as the first run, plus a few minutes for the environment); later runs reuse everything. The clustering itself takes about a minute per sample and scale. Then, in `results/<sample>/mpp0.5000/cluster/`:

1. `<id>.cluster_agreement_keep.png`: how much every pair of views agrees at k = 8, and every view with EXPRESSION for k = 2 to 15, against the spatial null (the chance agreement of two patchy maps).
2. `<id>.cluster_maps_keep.png`: the clusters of every view on the slide, coloured after the matching expression clusters.
3. `<id>.cluster_contingency_keep.png`: which clusters of each view correspond to which expression clusters.
4. `<id>.cluster_agreement_extended.png` and `<id>.cluster_maps_extended.png`: the same for the tiles from 30 % tissue on that carry RNA; similar results mean the conclusions do not depend on the 50 % threshold.

For the whole cohort, use a new folder (`$TP/runs/cohort`) with the full samplesheet and the same job script without `--scales_mpp 0.5`, so that the default scales are used (0.2857, 0.5 and 1.0 µm/px; slides scanned more coarsely than 0.2857 µm/px skip that scale with a warning).

## Route B: containers (Apptainer or Singularity)

If Apptainer or Singularity becomes available on the login node and the compute nodes, Nextflow can run on the login node and send every task to SLURM as its own job, each in the pipeline's container image (no environments to create):

```bash
cd $TP/runs/two_samples
nextflow run $TP/tileparity -profile uzh_rolandgarros,apptainer \
    --input samplesheet.csv --outdir results -w work \
    --uni2h_weights $TP/models/uni2h/pytorch_model.bin \
    --scales_mpp 0.5 --mask_compare --min_tissue_extract 0.3 --qc_umap
```

Run it in `tmux` on the login node (Nextflow must keep running while the jobs run). The jobs go to the default partition (`main`) and the UNI2-h job asks for `--gres=gpu:1`; `--slurm_queue`, `--slurm_gpu_queue` and `--slurm_gpu_options` change that. `conf/uzh_rolandgarros.config` sets the SLURM executor, the resource ceilings (64 CPUs, 400 GB, 72 h per task) and the GPU request; the images (`ghcr.io/miguelrosell/tileparity-{python,uni2h,r}:0.4.0`) are public and pulled once into `~/.cache/tileparity/apptainer`. The same profile also works with conda (`-profile uzh_rolandgarros,conda`) when the login node has conda and internet access.

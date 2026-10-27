# First real run on rolandgarros (UZH)

A step-by-step guide for the first run on real data: two samples at one scale (0.5 µm/px, 112 µm tiles), with the mask comparison and the 3-D UMAP QC switched on:

* **ZEN36** — HEST-1k, standard Visium, colorectal cancer (cohort B);
* **A_P1_CRC** — Visium HD, colorectal cancer, patient P1 (cohort A).

Everything — software, downloads and runs — happens on the server; nothing is needed on your laptop except a terminal. rolandgarros has 128 CPUs, 512 GB of RAM, 58 TB of disk and an NVIDIA L4 GPU (24 GB); wimbledon is the login node.

**Two software routes.** The login node has no conda, no module system and no containers, so this guide follows **route A (conda)**, which needs no administrator: conda is installed in your own folder, Nextflow comes from bioconda, and Nextflow creates the pipeline's three software environments from `containers/*/environment.yml`. The whole run is **one SLURM job on rolandgarros**; inside it Nextflow runs every task on that node (its *local* executor), so no partition names or GPU requests per task are needed. Route B (containers) is at the end, for when Apptainer is available.

## 0. One folder for everything

Choose a folder on a large disk of rolandgarros (`df -h` lists the disks) and call it `$TP` in what follows, e.g. `/data/<user>/tileparity`. Everything goes under it, so your home directory (often small) is not used:

```
$TP/
├── miniforge3/      conda itself (~0.5 GB)
├── env.sh           the environment variables, read at the start of every session and job
├── tileparity/      the code (git clone)
├── cache/           Nextflow's files, the pipeline's conda environments (~10 GB), Hugging Face's cache
├── models/uni2h/    the UNI2-h weights (2.7 GB)
├── data/            HEST and Visium HD files (~10 GB for these two samples)
├── work/            Nextflow's work directory (tens of GB; can be deleted after a successful run)
└── results/         the outputs
```

## 1. Log in and check the machine (5 minutes)

```bash
ssh <user>@wimbledon...                 # the login node
ssh rolandgarros                        # if it can be reached directly; otherwise get a shell on it through SLURM:
#   srun --pty -c 4 --mem 16G -t 4:00:00 bash          (add -p <partition> or -w rolandgarros if needed)
nproc && free -g && df -h               # CPUs, memory, disks
nvidia-smi                              # the L4 and the driver version (the PyTorch build used needs >= 525)
curl -sI https://conda.anaconda.org | head -1         # internet access: "HTTP/2 200" (needed to install and download)
sinfo -o "%P %G %l %c %m"               # SLURM partitions, GPUs (gres), time limits, CPUs and memory per node
```

Note the partition that contains rolandgarros and how its GPU is named in the `%G` column (e.g. `gpu:l4:1`): the job scripts below may need `#SBATCH -p <partition>` and that name in `--gres`.

## 2. Install conda and Nextflow (once, 10 minutes)

```bash
export TP=/data/<user>/tileparity        # your folder
mkdir -p $TP && cd $TP
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $TP/miniforge3     # -b: no questions; -p: install here
source $TP/miniforge3/etc/profile.d/conda.sh
conda create -y -n nf -c conda-forge -c bioconda nextflow=26.10.3 openjdk=21 huggingface_hub git
```

`nf` is a small environment with Nextflow, Java 21, git and `huggingface_hub` (for the downloads). Then write `$TP/env.sh` (replace `<user>`):

```bash
cat > $TP/env.sh <<'END'
export TP=/data/<user>/tileparity
source $TP/miniforge3/etc/profile.d/conda.sh
conda activate nf
export NXF_HOME=$TP/cache/nextflow              # Nextflow's own files, not in the home directory
export NXF_CONDA_CACHEDIR=$TP/cache/conda       # the pipeline's environments: created once, reused by every run
export HF_HOME=$TP/cache/huggingface            # Hugging Face downloads
export NXF_OPTS='-Xms1g -Xmx4g'                 # memory of Nextflow itself
END
source $TP/env.sh
nextflow -version && java -version              # 26.10.3 and 21
```

From now on, every session starts with `source $TP/env.sh` (with `TP` set, or with the full path).

## 3. Get the code (1 minute)

```bash
cd $TP
git clone https://github.com/miguelrosell/tileparity.git
git -C tileparity log -1 --oneline       # the version you run; later updates: git -C tileparity pull
```

The pipeline is run from this folder (`nextflow run $TP/tileparity`), so the code that runs is exactly the code you see.

## 4. Test the set-up on the synthetic data (20–40 minutes the first time)

The test profile runs the whole pipeline on a tiny synthetic slide (no data and no token needed; UNI2-h runs as a random 1-block network in test mode). Run on the GPU, it is also the first test of the CUDA path. Write `$TP/test.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=tp_test
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
# add "#SBATCH -p <partition>" (and/or "#SBATCH -w rolandgarros") if the cluster needs them
source /data/<user>/tileparity/env.sh
cd $TP
nvidia-smi
nextflow run $TP/tileparity -profile test,conda,gpu --outdir $TP/results/test -w $TP/work/test
```

```bash
sbatch $TP/test.sbatch                   # prints the job id
squeue -u $USER                          # is it running?
tail -f $TP/tp_test_<jobid>.log          # follow it (Ctrl-C stops following, not the job)
```

The first run creates the three environments in `$TP/cache/conda` (Python tools; PyTorch with CUDA, about 3 GB; R with Bioconductor), which takes most of the time; later runs reuse them. It should end with `tileparity finished: results in .../results/test`. If it does not, see section 9.

(If rolandgarros is reached directly by `ssh` and its CPUs are not managed by SLURM, run the commands of the job script in a `tmux` session instead.)

## 5. Download the data (30–60 minutes, mostly waiting)

You need a Hugging Face token (`HF_TOKEN`) with access to `MahmoodLab/hest` and `MahmoodLab/UNI2-h` (request access on both pages first). Downloads can run in `tmux` (so they survive a lost connection) or as a job (`sbatch --wrap "source $TP/env.sh && ..."`).

```bash
source $TP/env.sh
export HF_TOKEN=hf_xxx
```

**HEST-1k, ZEN36** (standard Visium, CRC). The script writes a samplesheet row with HEST's own pixel size:

```bash
python $TP/tileparity/scripts/hest_download.py --ids ZEN36 --outdir $TP/data/hest --samplesheet $TP/data/zen36.csv --cohort B
```

**Visium HD, P1 CRC.** On the 10x Genomics page [Visium HD CytAssist Gene Expression Libraries of Human CRC](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc), choose the P1 CRC sample and copy the links of two files: the **microscope image** (`.btf`, the H&E image Space Ranger was given; not the CytAssist image) and the **binned outputs** (`.tar.gz`). Put them in a tab-separated table and download:

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

`$TP/samplesheet_two.csv`: copy the ZEN36 row of `$TP/data/zen36.csv` (it has the pixel size) and add the Visium HD row. Leave `tiling` empty (grid tiles for both; ZEN36's spots are split between tiles by area) and `mpp` empty for Visium HD (Space Ranger's files give it). Write the paths in full:

```csv
sample,image,spatial,spatial_format,tiling,mpp,mask,patient,condition,cohort
A_P1_CRC,/data/<user>/tileparity/data/raw/A_P1_CRC/<image>.btf,/data/<user>/tileparity/data/raw/A_P1_CRC/binned_outputs,spaceranger_hd,,,,P1,CRC,A
ZEN36,/data/<user>/tileparity/data/hest/wsis/ZEN36.tif,/data/<user>/tileparity/data/hest/st/ZEN36.h5ad,hest,,<mpp from zen36.csv>,,<patient>,<condition>,B
```

## 7. Run the two samples

`$TP/run_two.sbatch` asks SLURM for half of rolandgarros (64 CPUs, 400 GB) and the GPU for up to three days; Nextflow then runs all tasks inside that allocation:

```bash
#!/bin/bash
#SBATCH --job-name=tp_two
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.log
source /data/<user>/tileparity/env.sh
cd $TP
nextflow run $TP/tileparity -profile conda,gpu \
    --input $TP/samplesheet_two.csv --outdir $TP/results/two_samples -w $TP/work/two_samples \
    --max_cpus 64 --max_memory '400.GB' --max_time '72.h' \
    --uni2h_weights $TP/models/uni2h/pytorch_model.bin --uni2h_batch_size 64 \
    --scales_mpp 0.5 --mask_compare --min_tissue_extract 0.3 --qc_umap \
    -resume
```

```bash
sbatch $TP/run_two.sbatch && tail -f $TP/tp_two_<jobid>.log
```

* `-profile conda,gpu`: software from the conda environments, UNI2-h on the GPU; no `uzh_rolandgarros` profile here, because that one sends every task to SLURM as a separate job.
* `--max_cpus 64 --max_memory '400.GB'`: no task asks for more than the job has (LBP asks for 32 CPUs, so two LBP tasks run at the same time).
* `--uni2h_batch_size 64`: 64 tiles at a time fit the L4's 24 GB.
* `--scales_mpp 0.5`: one scale for this first run. `--mask_compare`: the mask comparison figure. `--min_tissue_extract 0.3`: features from 30 % tissue on, so the 30/50/80 % thresholds can be compared afterwards without rerunning (`keep` still means ≥ 50 %). `--qc_umap`: the 3-D UMAP page.
* By default both extractors read both input sets: the slide (`rgb`) and the slide rebuilt from H and E only (`rgb_he`), with each slide's own H and E colours.
* `-resume`: if the job stops (time limit, error, a fix), submit the same script again: finished tasks are reused.

## 8. What to look at, in this order

1. `results/two_samples/pipeline_info/execution_report_*.html`: did every task finish, how long did each process take and how much memory did it use (this sets the resources of the full cohort).
2. `<sample>/mpp0.5000/prepare/<id>.overlay.png`: the blue units must lie exactly on the tissue (a shift means a wrong image or pixel size).
3. `prepare/<id>.mask_compare.png` and `prepare/<id>.tissue_thresholds.png`: is the saturation mask good enough, and which threshold looks right?
4. `channels/<id>.rgb_he_qc.png`: does the H&E-only version look like the slide, and where did it change? `channels/<id>.channels.json` has the slide's H and E colours (`stain_vectors`; a `note` appears if the standard ones had to be used).
5. `lbp/<id>.lbp_qc.png`: the share of irregular LBP patterns per radius on real tissue (which radii carry information).
6. `qc/mpp0.5000/combined_mpp0.5000.qc_umap.html` (copy it to your laptop, e.g. with `scp`, and open it in a browser): do the two samples separate in every view (batch effect) or only in some, and does `rgb_he` change that?
7. `st/<id>.aggregate.json` and `counts_summary.tsv`: share of molecules in the tiles, counts per tile.
8. `export/mpp0.5000/spe/combined_mpp0.5000.rds` loads in R (`reducedDimNames(spe)` lists `UNI2H_RGB`, `UNI2H_RGB_HE`, `LBP_RGB`, `LBP_RGB_HE`, `INTENSITY_HE`), and `metadata(spe)$tileparity$lineage` names the original images.

## 9. If something fails

* The log names the failing process and its work directory; `.command.err` and `.command.log` there hold the error, `.command.sh` the exact command. `$TP/.nextflow.log` has Nextflow's side.
* *Creating a conda environment fails or is slow*: check the internet access (section 1) and the free space in `$TP/cache/conda`. `-profile mamba,gpu` instead of `conda,gpu` solves environments faster (Miniforge includes mamba).
* *UNI2-h runs on the CPU or fails with CUDA errors*: the job log starts with `nvidia-smi`; if it shows no GPU, the `--gres` line is wrong for this cluster (see `sinfo` in section 1). A driver older than 525 is too old for the PyTorch build of the pipeline.
* *Job pending forever*: `squeue -u $USER` shows the reason; a partition, account or QOS may be required (`#SBATCH -p ...`, `#SBATCH -A ...`, `#SBATCH --qos ...`).
* *Out of memory or time*: a task killed for memory or time is retried once with twice the memory and time; if it fails again, the error says which process. The whole job has 72 h; resubmitting it continues where it stopped.
* *Pixel-size or alignment errors* from Process 1 are deliberate stops: the message says which sources disagree.

## Route B: containers (Apptainer or Singularity)

If Apptainer or Singularity becomes available on the login node and the compute nodes, Nextflow can run on the login node and send every task to SLURM as its own job, each in the pipeline's container image (no environments to create):

```bash
nextflow run $TP/tileparity -profile uzh_rolandgarros,apptainer \
    --input $TP/samplesheet_two.csv --outdir $TP/results/two_samples -w $TP/work/two_samples \
    --slurm_queue <cpu partition> --slurm_gpu_queue <gpu partition> --slurm_gpu_options '--gres=gpu:1' \
    --uni2h_weights $TP/models/uni2h/pytorch_model.bin \
    --scales_mpp 0.5 --mask_compare --min_tissue_extract 0.3 --qc_umap
```

Run it in `tmux` on the login node (Nextflow must keep running while the jobs run). `conf/uzh_rolandgarros.config` sets the SLURM executor, the resource ceilings (64 CPUs, 400 GB, 72 h per task) and the GPU request; the images (`ghcr.io/miguelrosell/tileparity-{python,uni2h,r}:0.3.0`) are public and pulled once into `~/.cache/tileparity/apptainer`. The same profile also works with conda (`-profile uzh_rolandgarros,conda`) when the login node has conda and internet access.

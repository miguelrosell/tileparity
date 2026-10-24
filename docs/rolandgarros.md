# First real run on rolandgarros (UZH, SLURM)

A step-by-step guide for the first run on real data: one HEST-1k standard Visium CRC section (ZEN36, cohort B) and one Visium HD CRC slide (P1, cohort A), at one scale (0.5 µm/px), with the mask comparison and the 3-D UMAP QC switched on. Commands are run on the login node; every computation runs as a SLURM job.

Placeholders: `$DATA` is a folder with space for the data (~10 GB), `$WORK` a folder for Nextflow's work directory (tens of GB; scratch space is ideal), `$RUN` the folder you launch from.

## 0. Check the machine (5 minutes)

```bash
java -version                 # 17 or newer (otherwise: module avail java / conda install openjdk=21)
apptainer --version || singularity --version
sinfo -o "%P %G %l %c %m"     # partitions, GPUs (gres), time limits, CPUs and memory per node
```

Note the CPU partition and the GPU partition and how their GPUs are named in the `%G` column (e.g. `gpu:l4:1`); they go into `--slurm_queue`, `--slurm_gpu_queue` and `--slurm_gpu_options`. If neither Apptainer nor Singularity is available, use `-profile uzh_rolandgarros,conda` below instead of `apptainer` (it creates the environments from `containers/*/environment.yml`, which needs internet on the login node).

Install Nextflow in your home if it is not there (any version ≥ 25.04):

```bash
curl -s https://get.nextflow.io | bash && mkdir -p ~/bin && mv nextflow ~/bin/ && ~/bin/nextflow -version
```

## 1. Get the code and test the cluster set-up (15 minutes)

```bash
cd $RUN
git clone https://github.com/miguelrosell/tileparity.git
nextflow run miguelrosell/tileparity -r main -profile test,uzh_rolandgarros,apptainer \
    --outdir results_test -w $WORK/test \
    --slurm_queue <cpu partition> --slurm_gpu_queue <gpu partition> --slurm_gpu_options '--gres=gpu:1'
```

This runs the synthetic test (no data, no token; UNI2-h in test mode) through SLURM, pulls the three container images and, because the server profile sets `use_gpu`, runs the UNI2-h step on a GPU node: it is the first test of the GPU path. It should end with `tileparity finished`. If it does not, see section 6 before going on.

## 2. Download the data (30–60 minutes, mostly waiting)

You need a Hugging Face token (`HF_TOKEN`) with access to `MahmoodLab/hest` and `MahmoodLab/UNI2-h`, and `huggingface_hub` for Python:

```bash
export HF_TOKEN=hf_xxx
python3 -m pip install --user huggingface_hub
```

**HEST-1k, ZEN36** (standard Visium, CRC). The script writes a samplesheet row with HEST's own pixel size:

```bash
python3 tileparity/scripts/hest_download.py --ids ZEN36 --outdir $DATA/hest --samplesheet $DATA/zen36.csv --cohort B
```

**Visium HD, P1 CRC.** On the 10x Genomics page [Visium HD CytAssist Gene Expression Libraries of Human CRC](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc), choose the P1 CRC sample and copy the links of two files: the **microscope image** (`.btf`, the H&E image Space Ranger was given; not the CytAssist image) and the **binned outputs** (`.tar.gz`). Put them in a table (tab-separated) and download:

```bash
printf 'sample\tkind\turl\nA_P1_CRC\timage\t<image link>\nA_P1_CRC\tbinned_outputs\t<binned outputs link>\n' > $DATA/p1_urls.tsv
bash tileparity/scripts/download_urls.sh $DATA/p1_urls.tsv $DATA/raw
ls $DATA/raw/A_P1_CRC        # the .btf image and binned_outputs/ (square_002um, square_008um, square_016um)
```

**UNI2-h weights** (about 2.7 GB for 681 M parameters, once):

```bash
apptainer exec docker://ghcr.io/miguelrosell/tileparity-uni2h:0.2.0 \
    python tileparity/bin/tp_download_uni2h.py --outdir ~/models/uni2h
# or, with huggingface_hub installed:  bash tileparity/scripts/download_uni2h_weights.sh ~/models/uni2h
```

## 3. Write the samplesheet

Copy the ZEN36 row of `$DATA/zen36.csv` (it has the pixel size) and add the Visium HD row; leave `tiling` empty (grid tiles for both; ZEN36's spots are split between tiles by area) and `mpp` empty for Visium HD (Space Ranger's files give it):

```csv
sample,image,spatial,spatial_format,tiling,mpp,mask,patient,condition,cohort
A_P1_CRC,<DATA>/raw/A_P1_CRC/<image>.btf,<DATA>/raw/A_P1_CRC/binned_outputs,spaceranger_hd,,,,P1,CRC,A
ZEN36,<DATA>/hest/wsis/ZEN36.tif,<DATA>/hest/st/ZEN36.h5ad,hest,,<mpp from zen36.csv>,,<patient>,<condition>,B
```

## 4. Run

In a `tmux` (or `screen`) session, so the run survives a lost connection:

```bash
export NXF_OPTS='-Xms1g -Xmx4g'
nextflow run miguelrosell/tileparity -r main -profile uzh_rolandgarros,apptainer \
    --input samplesheet_monday.csv --outdir results_monday -w $WORK/monday \
    --scales_mpp 0.5 \
    --uni2h_weights ~/models/uni2h/pytorch_model.bin \
    --slurm_queue <cpu partition> --slurm_gpu_queue <gpu partition> --slurm_gpu_options '--gres=gpu:1' \
    --mask_compare --min_tissue_extract 0.3 --qc_umap
```

* `--scales_mpp 0.5`: one scale (112 µm tiles) for this first run.
* `--mask_compare`: the mask comparison figure; `--min_tissue_extract 0.3`: features from 30 % tissue on, so the 30/50/80 % thresholds can be compared afterwards without rerunning (`keep` still means ≥ 50 %).
* `--qc_umap`: the 3-D UMAP page.
* After any interruption or fix, run the same command again with `-resume`: finished tasks are reused.

## 5. What to look at, in this order

1. `results_monday/pipeline_info/execution_report_*.html`: did every task finish, how long did each process take, how much memory did it use (this sets the resources of the full run).
2. `<sample>/mpp0.5000/prepare/<id>.overlay.png`: the blue units must lie exactly on the tissue (a shift means a wrong image or pixel size).
3. `prepare/<id>.mask_compare.png` and `prepare/<id>.tissue_thresholds.png`: is the saturation mask good enough, and which threshold looks right?
4. `lbp/<id>.lbp_qc.png`: the share of irregular LBP patterns per radius on real tissue (which radii carry information).
5. `qc/mpp0.5000/combined_mpp0.5000.qc_umap.html` (copy it to your laptop and open it in a browser): do the two samples separate in every view (batch effect) or only in some?
6. `st/<id>.aggregate.json` and `counts_summary.tsv`: share of molecules in the tiles, counts per tile.
7. `export/mpp0.5000/spe/combined_mpp0.5000.rds` loads in R, and `metadata(spe)$tileparity$lineage` names the original images.

## 6. If something fails

* The message on screen names the failing process and its work directory; `.command.err` and `.command.log` there hold the error, `.command.sh` the exact command. `.nextflow.log` in `$RUN` has Nextflow's side.
* *Image pull fails*: the images must be public (GitHub > Packages > tileparity-* > Package settings > Change visibility), or pull them once by hand: `apptainer pull docker://ghcr.io/miguelrosell/tileparity-python:0.2.0`.
* *Job pending forever / invalid partition*: check `--slurm_queue` / `--slurm_gpu_queue` against `sinfo`; extra sbatch options (account, QOS) go into `--slurm_options '--account=... --qos=...'`.
* *UNI2-h runs on the CPU or fails with CUDA errors*: check the GPU request (`--slurm_gpu_options`) and that the job sees the GPU (`nvidia-smi` in the work directory's `.command.log` is not printed by default; run `srun -p <gpu partition> --gres=gpu:1 nvidia-smi`).
* *Out of memory or time*: a task killed by SLURM for memory or time is retried once with twice the memory and time; if it fails again, the error says which process. The server profile allows up to 64 CPUs, 400 GB and 72 h per job.
* *Pixel-size or alignment errors* from Process 1 are deliberate stops: the message says which sources disagree.

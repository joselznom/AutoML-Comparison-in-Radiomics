# AutoML-Comparison-in-Radiomics

Official code for the paper **"Benchmarking AutoML Frameworks for Radiomics: A Comparative Evaluation of Performance, Efficiency and Usability"**.

This work provides a comprehensive evaluation of general-purpose and radiomics-specific Automated Machine Learning (AutoML) frameworks on ten diverse radiomics classification tasks.

This repository contains the source code and dockerfiles to reproduce the results of the manuscript:

The public datasets used are:

* **OpenRadiomics** → https://openradiomics.org/  
  * BraTS → https://openradiomics.org/?page_id=1163  
  * TCIA  → https://openradiomics.org/?page_id=1144
* **WORC Database** → https://github.com/MStarmans91/WORCDatabase

---

* **dockerfiles/** – Each Dockerfile encapsulates all dependencies required to run the corresponding AutoML framework.  
  The images are intentionally kept independent so you can benchmark a single framework without building the others.
* **scripts/** – Training code that reproduce the experiments reported in the manuscript.

---

## Data Format & Directory Structure

Each script expects **a single CSV file named `data.csv`** inside the directory passed via the `-i / --input` flag.

```
/path/to/your_dataset/
└── data.csv
```

* Rows = individual subjects / lesions / scans (one case per row)
* Columns 1‒N−1 = clinical and radiomic features (numerical)
* Last column (N) = binary target label (0 / 1).

---

## Prerequisites

* [Docker](https://docs.docker.com/get-docker/)

---

## Building the Docker Images

All images can be built from the repository root. Below you find the exact commands (feel free to change the `-t` tag to
whatever naming convention you prefer):

```bash
# 1. AutoGluon
docker build -f dockerfiles/Dockerfile_autogluon   -t radiomics/autogluon   .

# 2. H2O AutoML
docker build -f dockerfiles/Dockerfile_h2o         -t radiomics/h2o         .

# 3. LightAutoML
docker build -f dockerfiles/Dockerfile_ligthautoml -t radiomics/lightautoml .

# 4. MLjar Supervised
docker build -f dockerfiles/Dockerfile_mljar       -t radiomics/mljar       .

# 5. PyCaret
docker build -f dockerfiles/Dockerfile_pycaret     -t radiomics/pycaret     .

# 6. TPOT
docker build -f dockerfiles/Dockerfile_tpot        -t radiomics/tpot        .
```

---

## Running the Containers

The minimal command to launch an experiment is:

```bash
  docker run -v <path_to_data>:/folder <docker_image>:<docker_tag> -i /folder
```
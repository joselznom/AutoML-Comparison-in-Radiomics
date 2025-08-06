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

* **dockerfiles_** Each Dockerfile encapsulates all dependencies required to run the corresponding AutoML framework.  
  The images are intentionally kept independent so you can benchmark a single framework without building the others.
* **scripts_** Training code that reproduce the experiments reported in the manuscript.

---

## Data Format

Each script expects **a single CSV file named `data.csv`** inside the directory passed via the `-i / --input` flag.

```
/path/to/your_dataset/
└── data.csv
```

* Rows = individual subjects
* Columns = clinical and radiomic features
* Last column = binary target label (0 / 1).

---

##  Docker Images

All images can be built from the repository root. Below you find the exact commands (feel free to change the `-t` tag to
whatever naming convention you prefer):

> Prerequisites: [Docker](https://docs.docker.com/get-docker/)

```bash
docker build -f <path_to_dockerfile> -t <docker_image_name>:<docker_tag> .

# Example
docker build -f dockerfiles/Dockerfile_autogluon  -t automl:autogluon  .
```

```bash
# Running the containers
docker run -v <path_to_data>:/folder <docker_image>:<docker_tag> -i /folder
```

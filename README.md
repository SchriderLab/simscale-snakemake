# simscale-snakemake
Snakemake pipeline for parsing simulation outputs across scaling factors.

# Dependencies
Two main things are needed to run this pipeline: `snakemake` and `conda`. Please refer to the [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and [snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html) documentation for installation instructions.

# Simulation outcome files
In order to use this pipeline, simulation outcomes must be stored such as that each scaling factor has its own directory named `Q[scaling factor]`. Inside each directory there should be three main categories of files: 
1. CSV files storing the mutation type, origin generation, and fixation generation for each mutation fixed after any burn-in. The headers for these files must be `mutation_id, origin_gen, fix_gen` and each file should be named `fixation_[replicate].csv`.
2. CSV files storing the mutation fixation probabilities for each mutation types. The headers for these files are just the mutation SLiM mutation type ids (e.g. `m1, m2, m3`). Each file should be named `fixation_prob_[replicate].csv`.
3. Samples of the population in vcf after the end of the simulation, each named `sample_[replicate].vcf`.

One thing to note here is that the replicate numbers should match up for each scaling factor. For instance `fixation_1.csv`, `fixation_prob_1.csv`, and `sample_1.csv` must correspond to the same simulation replicate.
An example of directory structure for simulation results that includes scaling factors of 1, 2, and 5 would be:

```.
.
└── simulation_results/
    ├── Q1/
    │   ├── fixation_1.csv
    │   ├── fixation_2.csv
    │   ├── ...
    │   ├── fixation_prob_1.csv
    │   ├── fixation_prob_2.csv
    │   ├── ...
    │   ├── sample_1.vcf
    │   └── sample_2.vcf
    ├── Q2/
    │   ├── fixation_1.csv
    │   ├── fixation_2.csv
    │   ├── ...
    │   ├── fixation_prob_1.csv
    │   ├── fixation_prob_2.csv
    │   ├── ...
    │   ├── sample_1.vcf
    │   └── sample_2.vcf
    └── Q5/
        ├── fixation_1.csv
        ├── fixation_2.csv
        ├── ...
        ├── fixation_prob_1.csv
        ├── fixation_prob_2.csv
        ├── ...
        ├── sample_1.vcf
        └── sample_2.vcf
```

We have included the simulations used in the paper "Population size rescaling significantly biases outcomes of forward-in-time population genetic simulations" in the `sims` directory. These files contain the eidos code used to output the simulation outcomes as specified.

# Configuration file
This file is located in `config/config.yml` and is required. The following variables must be specified:

`src_data_dir`: The location of the source directory for the recorded simulation outcomes. See file information and example directory structure in the previous section.

`output_dir`: The location of the output directory. The pipeline will create this directory and its two subdirectories (`graphs` and `summary_stats`) if they do not exist.

`chr_len`: The chromosome length as specified in SLiM for the simulation scenario. This is necessary for linkage disequilibrium calcuations.

`batch_size`: The number of replicates to be batched together during the initial parsing of the data. It is particularly important to select this appropriately when using a high-performance computing cluster as it will determine the number of replicates processed per job. 

`conda_env`: The name of the conda environment to be used when running jobs on a high-performance computing cluster. Since `snakemake=6.14.0`, it is possible to specify the name of an existing conda envrionment. Otherwise, an environment file name can be specified as long as it exists in `workflow/envs/`. We have provided a conda environment file in this repository which contains all the necessary dependencies. Users may just specify this file and allow `snakemake` to handle the environment installation or they may install the environment themselves and specify its name in the `conda_env` variable. Please see the [snakemake documentation](https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html#integrated-package-management) for more information about integrated package management. 

Other variables in this file are optional:

`mutation_labels`: Allows users to map mutation labels to the mutation type ids stored in the outcome files. This will affect how these mutations are labelled in the graphs and tables produced by the pipeline. If not specified, the pipeline will default to usig the SLiM muatation ids specified in the outcome files. 

`fixation`, `sfs`, `fixationprobs`: Allow users to specify which mutations to be used for plots and statistics (mean percent error and KL divergence) for a specific outcome through the sub-variable `muts`. Additionally, they allows users to specify the limits of x axis for the plots of these outcomes using the sub-variable `xlim`. 

`sample_sizes`: Allows users to specify the sample sizes for subsampling of replicates when calculating mean percent error, KL divergence, and average root mean squared error (for LD). Note that the pipeline will always calculate these with all replicates, but users may wish to see the impact of using a smaller sample size. The resulting plots and summary tables will have a suffix denoting the sample size (e.g. `fixation_100.svg`), with a suffix of `0` denoting the use of all replicates.

# Profile 
`snakemake` allows the specification of profiles using the `--profile` option. Each profile is located in the `profile` directory, and must contain a `config.yaml` file. This allows the specification of default command-line arguments and a cluster job submission environment. It also allows the specification of job resources for a rule. We have included an example profile for the `SLURM` utility which also specified default resources and resources for two rules. Please refer to the `snakemake` [profiles documentation](https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles) for more information.

# Running the pipeline

Once all the necessary variables in the `config.yaml` file are specified, the pipeline can be run using the `snakemake` command. Please note that when not using a profile, `snakemake` requires specification for the number of cores to be used. For example, this runs the pipeline using 1 core:

```sh
snakemake --cores 1
```

Using the included slurm profile, for example:

```sh
snakemake --profile profile/slurm
```

# Pipeline results
Once the pipeline is done, it should produce a `full_data.csv` file in the specified `output_dir`. In side the `output_dir` there should be two directores:
- `graphs`: contains all the graphs including graphs of the outcomes, graphs of classifier accuracy, graphs of the mean percent errors, and graphs of KL divergence and root mean squared error (for linkage disequilibrium).

- `summary_stats`: contains tables for the average values of the outcomes at each scaling factor, classifier accuracy, and values for mean percent errors, KL divergence, and linkage disequilibrium root mean squared error.

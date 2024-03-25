# jaxoptimizers

The training pipeline (besides optimizers and online learners) are mostly following from this repo: [jaxgptc4](https://github.com/acutkosky/jaxgptc4/tree/main). This repo aims to test the performance of different optimizers on the task of training GPT2 model on hugging face C4 dataset.

* Run `source scc_setup.sh` to set up the virtual environment and relevant dependency.
* Run `python train.py` to run the main training program.
    - For additional command line arguments, please refer to `conf/config_gpt2.yaml` for more detailed documentation.
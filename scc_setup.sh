module load python3/3.10.12

[ ! -d "env" ] && python -m venv env

source env/bin/activate
pip install -r requirements.txt

# manually download jax to match cuda version
module unload cuda
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
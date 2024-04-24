Install:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Install transformers fork:

```bash
git clone https://github.com/robertgshaw2-neuralmagic/transformers.git
cd transformers
git checkout fp8-hack
pip install -e .
cd ..
```

Quantize:

```bah
python3 quantize.py
```

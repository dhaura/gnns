# gnns

```bash
python3 -m venv venv
source venv/bin/activate
```

```bash
pip install torch==2.3 
pip install scipy networkx numpy pandas matplotlib seaborn
pip install wheel
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu124.html
```

```bash
cd data
wget https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.zip
unzip cora.zip
rm cora.zip
```

```bash
wget https://linqs-data.soe.ucsc.edu/public/datasets/pubmed-diabetes/pubmed-diabetes.zip
unzip pubmed-diabetes.zip
rm pubmed-diabetes.zip
```

```bash
wget https://snap.stanford.edu/graphsage/reddit.zip
unzip reddit.zip
rm reddit.zip
```

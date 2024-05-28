
# Pytorch Lightning Demo

This is a very simple pytorch lightning demo to bootstrap a new project. It uses the MNIST dataset 
and two different models. Uses Lightning CLI. Each different model has each own config file.
Should be enough in order to bootstrap a project for a single dataset and comparison of different models.

Run using 

```
python main.py fit --model ViT --config config-vit.yaml
```

or 

```
python main.py fit --model SimpleViT --config config-simplevit.yaml
```

In order to install dependencies

```
python3 -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```

Enjoy!


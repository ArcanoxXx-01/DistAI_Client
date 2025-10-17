import os, json

CONFIG_PATH = os.path.expanduser("config/config.json")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {"error": "Archivo config.json no encontrado"}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

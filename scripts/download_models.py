from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))
SentenceTransformer(cfg["embed"]["model"])
for k in ("overall", "absa"):
    AutoTokenizer.from_pretrained(cfg[k]["model"])
    AutoModelForSequenceClassification.from_pretrained(cfg[k]["model"])
print("models cached")

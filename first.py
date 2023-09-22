from transformers import pipeline
from loguru import logger

classifier = pipeline("sentiment-analysis")
sample = "I've been waiting for a HuggingFace course my whole life."
c = classifier(sample)
logger.info(sample)
logger.info(c)

ner = pipeline("ner", grouped_entities=True)
sample = "My name is Sylvain and I work at Hugging Face in Brooklyn."
n = ner(sample)
logger.info(sample)
logger.info(n)



# Spark-NLP

## Introduction

### Annotators
Annotator does a context specific NLP tasks, like tokenization, stemming, NER, etc. There are 2 types to it:
* Those that are trainable **APPROACH** : _trains DataFrame and produces a model._
* Those that are already trained **MODEL**: _transforms one DataFrame to another through models._
* Sometime certain annotators might neither say Model/ Approach, like normalizer, tokenizer, so they might be trainable or trained.

#### Properties of Annotators
1. Inputs: Columns in a DataFrame (1 or more)
2. Outputs: single output (usually)
3. Additional Parameters: for fine tuning or cutomising behaviour.

#### Most Important Annotators (Frequently used)
1. Document
2. Token
3. Chunk

#### Structure of Annotation
Annotation is the output of every annotator.
They contain:
* Type:
* Begin: _where the annotation begins_
* End: _where the annotation ends_
* Result: _main outcome_
* Metadata: _other info_
* Embeddings: _used by word embeddings annotators_

### Components:
Spark-NLP majorly comprises of 2 components:
  1. Estimators:  Used for training annotators | `fit()`
  2. Transformers: Result of fitting process and applies changes to target dataset.

 



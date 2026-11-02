// MAUVE max length 2048 to cover the full articles (~800 tokens)
(import 'base.libsonnet').metrics(2048, 'dslim/bert-base-NER', 128)

// Fidelity metrics shared by every dataset config (sim-metrics-<dataset>.jsonnet).
// MAUVE on OLMoE features; NER-entity 1-gram KL with sentence splitting (medical datasets use Medical-NER,
// the rest bert-base-NER), NER inputs processed in overlapping 512-token windows (stride 128); length KL.
{
  metrics(mauve_max_text_length, ner_model, ner_batch_size):: {
    mauve: {
      '()': 'src.fidelity.metrics.Mauve',
      use_cuda: true,
      mauve_args: {
        max_text_length: mauve_max_text_length,
        featurize_model_name: 'allenai/OLMoE-1B-7B-0125',
      },
    },
    entity_ngrams: {
      '()': 'src.fidelity.metrics.ngramFrequency',
      ngram: [1],
      ner_only: true,
      batch_size: ner_batch_size,
      topk: 1000,
      ner_model: ner_model,
      vectorizer_args: {
        stop_words: 'english',
        max_df: 0.95,
        min_df: 2,
      },
      split_by_sentence: true,
      max_length: 512,
      stride: 128,
    },
    lengths: {
      '()': 'src.fidelity.metrics.TextLengthDistribution',
      show_histogram: false,
    },
  },
}

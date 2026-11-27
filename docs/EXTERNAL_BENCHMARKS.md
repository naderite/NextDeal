# External insurance benchmarks

The benchmark script works without the BH Assurance extract, the advisor database, or Ollama. It uses source-specific adapters for two publicly available datasets and reusable scoring functions for portfolio reconstruction and one-year holdings persistence.

Place the files below in one directory.

| Local filename | Published source |
| --- | --- |
| `coil-training.tsv` | [UCI CoIL training file](https://kdd.ics.uci.edu/databases/tic/ticdata2000.txt) |
| `coil-test.tsv` | [UCI CoIL test file](https://kdd.ics.uci.edu/databases/tic/ticeval2000.txt) |
| `coil-test-target.tsv` | [UCI CoIL test labels](https://kdd.ics.uci.edu/databases/tic/tictgts2000.txt) |
| `motor-portfolio.csv` | Motor vehicle insurance data.csv from [Lledó and Pavía (2024), Mendeley Data V2](https://doi.org/10.17632/5cxyb5fp4f.2), CC BY 4.0 |

Run from the NextDeal root with its Python environment.

```sh
.venv/bin/python scripts/external_benchmarks.py /path/to/data --output reports/external_benchmarks.json
```

The ranking benchmark masks one held product at a time, fits on the publisher's training customers, and evaluates on its original test customers. It measures how well the model reconstructs a consumer portfolio. The three most common products are defined from training data.

The holdings benchmark uses counts recorded one calendar year apart for each policyholder–product key. It reports how many records match and how many reach the observed anniversary window. Correlation describes matched records and is subject to survivorship bias. This dataset combines policy lapse causes and has no individual missed-invoice label.

These runs validate the external benchmarks.  Approved add-on coverage, invoice-level non-payment, independent English labels, and genuine advisor review remain separate validation tasks.


## Using another dataset in the advisor app

Set `NEXTDEAL_DATA_DIR` to a directory prepared in the existing raw and processed file schemas:

- `raw/pm_product_portfolio.csv` and `raw/pp_contract_renewals.json` for source profiles and contracts.
- `processed/pm_companies_clean.csv`, `processed/pm_episodes.csv`, `processed/pp_renewals_clean.csv` and `processed/catalogue.json` for model inputs and product metadata.
- `raw/source_pages/manifest.json` and its referenced HTML files for retrieval.

The app defaults to a runtime specific to that dataset directory and trains its own recommendation artifact. To set `NEXTDEAL_RUNTIME` explicitly, use a fresh directory for each dataset. The app's evaluation endpoint withholds the bundled BH Assurance report in this mode. The CoIL adapter below prepares public portfolios for the app. The Spanish motor file is used by the holdings benchmark.

### Run the app with CoIL portfolios

```sh
.venv/bin/python scripts/prepare_coil.py /path/to/coil-training.tsv /path/to/new-coil-data
NEXTDEAL_DATA_DIR=/path/to/new-coil-data ./scripts/start.sh
```

The adapter imports the publisher's training split only. The app uses its existing portfolio workflow to serve the anonymized consumer portfolios. CoIL does not provide billing dates, contractual terms, or renewal cases, so those views have no CoIL contract records. Use the published test split for evaluation only.

### Run the renewal queue with the Spanish motor portfolio

```sh
.venv/bin/python scripts/prepare_motor.py /path/to/motor-portfolio.csv /path/to/new-motor-data --as-of 2017-12-31
NEXTDEAL_DATA_DIR=/path/to/new-motor-data ./scripts/start.sh
```

A fresh runtime starts at the adapter's as-of date. Existing runtimes keep their saved clock. By default the adapter loads the 500 earliest upcoming renewals with at least 30 days of lead time on that date. Use `--limit 0 --minimum-lead-days 0` for all observed active records with an upcoming renewal. The date displayed in the expiry column represents the source's scheduled next renewal. The source key identifies a policyholder–product pair, so it serves as the imported case identifier. Payment status is unknown, and the portfolio provides no verified guarantee details. The adapter does not train a recommendation ranker.

### Reproduce the advisor ranker and renewal replay

```sh
.venv/bin/python scripts/benchmark_coil_advisor.py /path/to/coil-files --output reports/external_coil_advisor_ranking.json
.venv/bin/python scripts/benchmark_renewal_replay.py /path/to/prepared-motor-data --as-of 2017-12-31 --output reports/external_renewal_replay.json
```

Both scripts use disposable application databases. The CoIL benchmark uses the advisor's own ranking function and the publisher's held-out labels. The renewal replay uses scripted approvals, so it measures scheduling and dispatch mechanics.

### Evaluate coverage additions, payment risk, and English retrieval

- **Coverage additions.** Download `Dataset of motor insurance portfolio.csv` from [Espinosa, Lledó and Atance, Mendeley Data V1](https://doi.org/10.17632/sw4jmdb2sm.1), CC BY 4.0. Run `.venv/bin/python scripts/benchmark_coverages.py /path/to/motor-coverages.csv --output reports/external_coverage_suggestions.json`. The score uses observed additions. For contract eligibility, supply terms or an approved catalogue.
- **Renewal payment proxy.** Download the [DataHack training table](https://datahack-prod.s3.amazonaws.com/train_file/train_ZoGVYWq.csv). Run `.venv/bin/python scripts/benchmark_renewal_payment.py /path/to/renewal.csv --output reports/external_renewal_payment.json`. The label is non-renewal and the table has no invoice dates.
- **Actual premium non-payment.** Run `.venv/bin/python scripts/benchmark_payment.py /path/to/invoices.csv --validation-start YYYY-MM-DD --test-start YYYY-MM-DD --observed-through YYYY-MM-DD --output /path/to/results.json`. Provide unique `invoice_id`, `policy_id`, `issued_on`, `due_on`, `paid_on`, and `premium` columns. Use ISO dates and leave `paid_on` empty for invoices not fully paid by the observation cutoff. Define the horizon with `--horizon-days`.
- **English answer retrieval.** From [InsuranceQA v2](https://github.com/shuzi/insuranceQA/tree/master/V2), download `vocabulary`, `InsuranceQA.label2answer.raw.encoded.gz`, and `InsuranceQA.question.anslabel.raw.100.pool.solr.test.encoded.gz` into a folder. Run `.venv/bin/python scripts/benchmark_insuranceqa.py /path/to/insuranceqa --output reports/external_insuranceqa.json`. The benchmark uses the questions and labels from that test file and searches all answers. The publisher provides the data for research use only. For semantic retrieval, add `--dense-model models/e5-small --cache /path/to/embedding-cache`. The model must already be installed locally. The first full encoding run on a CPU can take several minutes.

### Evaluate or configure a different English intent taxonomy

Download `data/data_full.json` from [CLINC150](https://github.com/clinc/oos-eval/tree/master/data) and run:

```sh
.venv/bin/python scripts/benchmark_clinc.py /path/to/data_full.json --output reports/external_clinc_intent.json
```

This uses the publisher's in-scope train and test splits. It measures the classifier on CLINC150's general assistant intents. To train the advisor app on a different taxonomy, put a JSON array of `[message, label]` training pairs in `processed/intent_training.json` in the external dataset directory. Each label should have representative messages and there must be at least two labels. Reserve any held-out labels and messages for evaluation. The app keeps explicit refusal handling for advisor conversations.

### Test judge agreement on an independent sample

Download `source_info.jsonl` and `response.jsonl` from the [RAGTruth dataset](https://github.com/ParticleMedia/RAGTruth/tree/main/dataset). With the configured local Ollama model running, use:

```sh
.venv/bin/python scripts/benchmark_ragtruth_judge.py /path/to/ragtruth --sample-size 24 --output /path/to/judge-results.json
```

The output checkpoints each judgment. Rerun with the same model, configuration, and output path to resume. Only a completed output has Cohen's κ. The sample is balanced by human groundedness label and contains one response per source. This tests the judge on general QA and does not establish advisor acceptance.

### Evaluate corporate duplicate matching

Export `restaurant.RData` and `restaurant_gold.RData` from the [restaurant R package](https://github.com/cran/restaurant/tree/master/data) to `restaurant.csv` and `restaurant_gold.csv`. Run `.venv/bin/python scripts/benchmark_record_linkage.py /path/to/csv-directory --output reports/external_record_linkage.json`.

The original rule found 80 of 112 matches with no false positives. The revised rule found 107 with no false positives. The 95.5% recall and 100% precision come from the dataset used to develop the revised rule. An independent sample of business records is needed to estimate generalization.

For a labeled candidate-pair CSV from [RUC-DataLab's entity resolution benchmark collection](https://huggingface.co/datasets/RUC-DataLab/ER-dataset), pass the CSV path with `--labeled-pairs`. `restaurants1.csv` was used to develop the exact-name address extension. On independent labeled pairs from `restaurants4.csv`, precision was 99.1% and recall was 89.2%. On `restaurants3.csv`, which supplies only names and street addresses, precision was 100% and recall was 47.7%. These results measure pair classification on the supplied candidates. The report is `reports/external_business_linkage.json`.

To reproduce the optional InsuranceQA cross-encoder experiment, add `--reranker /path/to/ms-marco-TinyBERT-L2-v2` to the dense benchmark command. Download the model files from the [Sentence Transformers model repository](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2). The reranker scores the union of the top 32 lexical and top 32 dense results for each question. Its recall@5 was 41.5%, compared with 45.5% for E5 alone. Shortlist recall was 72.1%, below the 85% target. This experiment is benchmark-only.

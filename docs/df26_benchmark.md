# DF26 como benchmark e treino inicial

Este documento descreve como usar o DF26 no pipeline Bronze/Silver/Gold do
projeto. O DF26 entra como dataset local versionado e não como fonte YouTube.

## Papel metodológico

O DF26 deve ser usado com separação explícita entre:

| Subconjunto | Uso no projeto |
|---|---|
| Reais | classe negativa/referência real |
| Fakes open-weight | treino experimental de detector sob protocolo LOGOS |
| Fakes comerciais | avaliação final OOD com modelo congelado |

O pipeline preserva essa regra no manifesto por meio das colunas:

| Coluna | Interpretação |
|---|---|
| `benchmark_dataset` | identifica `DF26` |
| `df26_clip_id` | identifica o clipe base |
| `df26_scenario` | cenário do vídeo |
| `df26_generator` | gerador específico |
| `df26_generator_canonical` | nome normalizado usado pelo protocolo |
| `df26_generator_family` | família do gerador para LOGOS |
| `df26_generator_availability` | `real`, `open_weight`, `commercial` ou `unknown` |
| `df26_training_role` | papel permitido no experimento |
| `df26_allowed_for_training` | indica se pode participar do treino |
| `df26_allowed_for_evaluation` | indica se pode ser avaliado |

Essas colunas são metadados de governança. Elas são preservadas na Gold, mas
não entram como sinais numéricos do modelo.

## Preparação de acesso

Antes de reproduzir em uma nova máquina:

1. Aceite os termos do dataset no Hugging Face.
2. Faça login local:

```bash
huggingface-cli login
```

3. Instale dependências:

```bash
pip install -r requirements.txt
```

## Baixar e preparar Bronze

Baixar o DF26 e gerar o manifesto:

```bash
python -m src.data_engineering.ingestion.df26 prepare \
  --repo-id DF26/DF26 \
  --dataset-dir data/external/df26 \
  --manifest-output data/bronze/manifests/bronze_manifest_df26.csv \
  --splits-output data/bronze/manifests/df26_logos_splits.csv
```

Se o dataset já estiver baixado, gere apenas o manifesto:

```bash
python -m src.data_engineering.ingestion.df26 manifest \
  --dataset-dir data/external/df26 \
  --output data/bronze/manifests/bronze_manifest_df26.csv \
  --splits-output data/bronze/manifests/df26_logos_splits.csv
```

Para uma rodada piloto balanceada:

```bash
python -m src.data_engineering.ingestion.df26 manifest \
  --dataset-dir data/external/df26 \
  --output data/bronze/manifests/bronze_manifest_df26.csv \
  --splits-output data/bronze/manifests/df26_logos_splits.csv \
  --limit 60
```

Use `--limit 0` ou omita `--limit` para o DF26 completo.

## Executar Silver e Gold

Rodada completa com o manifesto DF26:

```bash
python -m src.data_engineering.pipeline build \
  --skip-ingest \
  --manifest data/bronze/manifests/bronze_manifest_df26.csv \
  --videos-dir data/external/df26 \
  --metadata-dir data/df26/silver/face_metadata_json \
  --silver-dir data/df26/silver \
  --gold-dir data/df26/gold \
  --groups abcde \
  --max-frames 25 \
  --sample-fps 5 \
  --temporal-min-points 3 \
  --generate-missing-metadata true \
  --face-detector-model models/face_detector.task \
  --face-model experimentos/grupo_b/data/extracted/face_landmarker.task \
  --segmenter-model models/image_segmenter.task \
  --with-gx \
  --fail-on-error
```

Artefatos esperados:

| Camada | Artefato |
|---|---|
| Bronze | `data/bronze/manifests/bronze_manifest_df26.csv` |
| Bronze protocolo | `data/bronze/manifests/df26_logos_splits.csv` |
| Silver metadata | `data/df26/silver/face_metadata_json/` |
| Silver frame | `data/df26/silver/frame_features/` |
| Silver video | `data/df26/silver/video_features/video_features.parquet` |
| Silver temporal | `data/df26/silver/temporal_features/temporal_features.parquet` |
| Gold regional | `data/df26/gold/gold_video_region_dataset.parquet` |
| Gold treino | `data/df26/gold/gold_training_dataset.parquet` |

## Reprodução com DVC

O fluxo DF26 fica em `pipelines/df26/dvc.yaml` para não tornar o smoke test padrão pesado.

Executar tudo:

```bash
dvc repro pipelines/df26/dvc.yaml:df26_validate_data_contracts
```

Executar somente download e manifesto:

```bash
dvc repro pipelines/df26/dvc.yaml:df26_prepare_manifest
```

Parâmetros ficam em `params.yaml`, na seção `df26`.

## Protocolo LOGOS

O arquivo `df26_logos_splits.csv` materializa folds leave-one-generator-family-out:

| Fold | Teste |
|---|---|
| `logos_leave_HunyuanVideo_1.5` | fakes HunyuanVideo 1.5 |
| `logos_leave_LTX_2.3` | fakes LTX 2.3 |
| `logos_leave_Wan_2.2` | fakes Wan 2.2 |
| `commercial_final` | fakes comerciais para avaliação final |

Nos folds LOGOS:

- vídeos reais entram como referência de treino;
- fakes open-weight da família deixada de fora entram como teste;
- fakes open-weight das outras famílias entram como treino;
- fakes comerciais ficam como `holdout_commercial`.

No fold `commercial_final`, os fakes comerciais entram como `test` e os
open-weight ficam como `excluded_open_weight`.

## Observações

- Não use fakes comerciais para seleção de hiperparâmetros, calibração,
  threshold tuning, representação, aumento de dados ou escolha de modelo.
- A Gold preserva os campos `df26_*` para auditoria e avaliação por gerador.
- Antes da modelagem final, remova explicitamente metadados como
  `df26_generator` e `df26_generator_family` da matriz de treino.
- `max_frames=25` é um padrão inicial coerente para vídeos curtos; aumente ou
  remova esse limite se o custo computacional permitir.

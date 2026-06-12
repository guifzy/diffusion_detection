# Contratos de Dados

Este documento define os contratos oficiais da primeira versao do pipeline de engenharia de dados. Um contrato define a granularidade, camada, campos obrigatorios e valores aceitos de cada ativo produzido ou consumido pelo pipeline.

## bronze_manifest

Camada: Bronze

Granularidade: uma linha por video de origem.

Objetivo: registrar tudo que entrou ou tentou entrar no sistema antes do processamento forense.

Campos obrigatorios:

| Campo | Tipo logico | Descricao |
| --- | --- | --- |
| `video_id` | string | Identificador unico do video. Para YouTube, usa o id retornado pelo `yt-dlp`. |
| `source_url` | string | URL original do video. |
| `filename` | string | Nome do arquivo salvo localmente. |
| `storage_path` | string | Caminho local ou futuro URI no data lake. |
| `sha256` | string | Hash do arquivo bruto, quando o download foi bem-sucedido. |
| `downloaded_at` | datetime | Timestamp ISO-8601 da tentativa de ingestao. |
| `label` | string | Classe supervisionada quando conhecida: `Real`, `Fake` ou vazio. |
| `status` | string | `pending`, `downloaded`, `failed` ou `skipped`. |
| `error_message` | string | Erro registrado quando a ingestao falha. |
| `source_type` | string | `youtube`, `manual_upload` ou `dataset_local`. |

Produtor atual: `python -m src.data_engineering.ingestion`

Destino local: `data/bronze/manifests/bronze_manifest.csv`

## frame_metadata

Camada: Silver

Granularidade: uma linha por frame processado.

Objetivo: registrar regioes visuais extraidas do video e a origem da bbox usada no frame.

Campos obrigatorios:

| Campo | Tipo logico | Descricao |
| --- | --- | --- |
| `video_id` | string | Identificador do video. |
| `frame_id` | integer | Indice original do frame no video. |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | integer | Caixa facial principal. |
| `bbox_expanded_x1`, `bbox_expanded_y1`, `bbox_expanded_x2`, `bbox_expanded_y2` | integer | Caixa expandida usada para borda/contexto. |
| `source` | string | `detector`, `tracker`, `last_bbox` ou `fallback_center`. |
| `detector_score` | float | Confianca do detector quando disponivel. |
| `frame_width` | integer | Largura do frame. |
| `frame_height` | integer | Altura do frame. |
| `processed_at` | datetime | Timestamp da execucao. |
| `pipeline_version` | string | Versao logica do pipeline. |

Produtor atual: `python -m src.data_engineering.preprocessing`

Destino local estruturado: `data/silver/face_metadata/{video_id}.parquet` ou `.csv` como fallback.

Destino auxiliar JSON usado pelo extrator atual: `data/silver/face_metadata_json/{video_id}_meta.json`.

## frame_features

Camada: Silver

Granularidade: uma linha por frame processado.

Objetivo: guardar os sinais A-E extraidos em cada frame.

Campos obrigatorios de governanca:

| Campo | Tipo logico | Descricao |
| --- | --- | --- |
| `video_id` | string | Identificador do video. |
| `frame_id` | integer | Frame original processado. |
| `metadata_idx` | integer | Linha de metadata usada para esse frame. |
| `label` | string | Classe quando conhecida. |
| `feature_groups_used` | string | Grupos ativados, por exemplo `abcde`. |
| `processed_at` | datetime | Timestamp da extracao. |
| `pipeline_version` | string | Versao logica do pipeline. |

As demais colunas sao features numericas dos grupos A-E.

Produtor atual: `python -m src.api` ou `src.shared.features.extractor`

Destino local: `data/silver/frame_features/{video_id}.parquet` ou `.csv` como fallback.

## video_features

Camada: Silver

Granularidade: uma linha por video.

Objetivo: agregar features frame-level em estatisticas por video antes da curadoria para treino.

Campos obrigatorios de governanca:

| Campo | Tipo logico | Descricao |
| --- | --- | --- |
| `video_id` | string | Identificador do video. |
| `label` | string | Classe quando conhecida. |
| `n_frames` | integer | Quantidade de frames usados nas features. |
| `metadata_rows_used` | integer | Quantidade de linhas de metadata usadas. |
| `feature_groups_used` | string | Grupos ativados. |
| `aggregated_at` | datetime | Timestamp da agregacao. |
| `pipeline_version` | string | Versao logica do pipeline. |
| `missing_feature_ratio` | float | Proporcao de valores ausentes nas features agregadas. |

As demais colunas sao agregacoes numericas, hoje com sufixos `_mean`, `_std` e `_median`.

Produtor atual: `python -m src.data_engineering.datasets`

Destino local: `data/silver/video_features/video_features.parquet` ou `.csv` como fallback.

## gold_training_dataset

Camada: Gold

Granularidade: uma linha por video treinavel ou auditavel.

Objetivo: servir como dataset oficial para treinamento e avaliacao do modelo.

Campos obrigatorios adicionais:

| Campo | Tipo logico | Descricao |
| --- | --- | --- |
| `video_id` | string | Identificador do video. |
| `target_label` | string | Classe alvo para treino. |
| `dataset_split` | string | `train`, `validation`, `test` ou `unassigned`. |
| `is_trainable` | boolean | Indica se a linha pode entrar no treino. |
| `quality_flag` | string | `ok`, `review`, `insufficient_metadata`, `missing_label` ou `feature_failure`. |
| `missing_feature_ratio` | float | Proporcao de features ausentes. |
| `pipeline_version` | string | Versao logica do pipeline. |

Produtor atual: `python -m src.data_engineering.datasets`

Destino local: `data/gold/gold_training_dataset.parquet` ou `.csv` como fallback.

## prediction_payload

Camada: Serving

Granularidade: uma resposta por video analisado.

Objetivo: definir o contrato futuro entre o backend SaaS e a camada de modelo.

Campos obrigatorios:

| Campo | Tipo logico | Descricao |
| --- | --- | --- |
| `video_id` | string | Identificador do video analisado. |
| `prediction` | string | `Real`, `Fake` ou `Unknown`. |
| `score_fake` | float | Probabilidade/score para fake. |
| `score_real` | float | Probabilidade/score para real. |
| `model_version` | string | Versao do modelo usado. |
| `feature_pipeline_version` | string | Versao do pipeline de features. |
| `processed_at` | datetime | Timestamp da predicao. |
| `top_signals` | list | Principais sinais usados para explicabilidade. |
| `processing_status` | string | `success`, `failed` ou `partial`. |
| `error_message` | string | Erro registrado quando houver falha. |

Produtor futuro: API/backend de inferencia.

# Detecção de vídeos gerados por IA com sinais forenses espaciais, espectrais e temporais

## Visão Geral

Este repositório implementa uma pipeline reprodutível para investigar a
detecção de vídeos reais e sintéticos por meio de sinais forenses
interpretáveis. A proposta atual não é treinar uma rede profunda para aprender
a representação visual de ponta a ponta. O sistema calcula descritores
explícitos de textura, borda, nitidez, estrutura local, resíduo, frequência,
fotometria, regiões semânticas e temporalidade; em seguida, um classificador
supervisionado tabular pode aprender a fronteira de decisão entre vídeos reais
e falsos.

O foco científico é avaliar se inconsistências compartilhadas por diferentes
modelos gerativos de vídeo podem ser capturadas por métricas determinísticas,
auditáveis e explicáveis. A versão atual do projeto está preparada para:

- ingerir vídeos por links externos ou por datasets locais;
- processar o benchmark DF26 com protocolo LOGOS;
- detectar regiões de rosto, olhos, boca, corpo e fundo com MediaPipe;
- extrair sinais espaciais e espectrais por frame e região;
- derivar métricas temporais por série regional;
- construir datasets Silver e Gold reprodutíveis;
- versionar dados com DVC e publicar artefatos em MinIO.



Objetivo Do Estudo

O objetivo é construir uma metodologia experimental para discriminar vídeos
reais e falsos a partir de uma representação forense explícita. O estudo parte
da hipótese de que vídeos gerados por IA podem preservar realismo visual global
e ainda apresentar desvios estatísticos ou físicos em domínios locais:

- microtextura;
- bordas e gradientes;
- nitidez e altas frequências;
- estrutura local;
- resíduo visual;
- distribuição espectral;
- fotometria regional;
- estabilidade temporal;
- coerência entre rosto, olhos, boca, corpo e fundo.

Formalmente, para um vídeo \(V\), composto por frames
\(I_1, I_2, \ldots, I_T\), regiões \(r \in R\) e sinais \(k \in K\), o
pipeline calcula:

\[
x_{v,t,r,k} = \phi_k(I_{v,t}, r)
\]

em que \(\phi_k\) é um descritor determinístico. Em seguida, as observações por
frame e região são agregadas:

\[
z_V = \Psi(\Phi(I_1), \Phi(I_2), \ldots, \Phi(I_T)).
\]

O classificador supervisionado atua apenas após a representação:

\[
\hat{y} = g_\theta(z_V).
\]

Essa separação é metodologicamente importante: o projeto não aprende a
representação visual por CNN, transformer ou foundation model; ele aprende a
fronteira de decisão sobre sinais interpretáveis.

## Pergunta Científica

> Até que ponto violações texturais, fotométricas, espectrais, geométricas e
> temporais compartilhadas por diferentes geradores de vídeo podem ser
> capturadas por descritores explicitamente definidos e usadas por modelos
> tabulares para detectar vídeos gerados por IA?

## Fundamentação Científica

A metodologia se apoia em quatro linhas de literatura:

| Linha                          | Contribuição Para O Projeto                                                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Detectores profundos de vídeo | VideoMAE, TimeSformer, DeMamba e AIGVDet mostram que informação espaço-temporal é discriminativa em vídeos gerados.                                      |
| Priors físicos e temporais    | D3, NSG-VD, STALL, V-PVP, WaveRep e Grab-3D motivam segunda ordem temporal, coerência física, dinâmica local, geometria e pistas forenses de baixo nível. |
| Image forensics clássico      | LBP, Sobel, Laplaciano, SIFT, resíduos, estatísticas espectrais e textura estatística sustentam descritores explícitos e interpretáveis.                 |
| Auditoria experimental         | VidAudit e estudos de shortcut learning alertam para viés de FPS, duração, resolução, compressão, movimento e fonte do dataset.                         |

## Datasets Suportados

O repositório atualmente suporta dois fluxos principais de dados.

### 1. Dataset Por Links / YouTube

Fluxo padrão usado para smoke test, amostras próprias e vídeos catalogados por
CSV.

Entrada:

```text
data/bronze/manifests/smoke_source.csv
data/bronze/manifests/video-metadata-publish-with-links.csv
```

Contrato mínimo:

```text
link,label
```

Convenção de label:

| Valor No CSV | Label Interno |
| ------------ | ------------- |
| `true`     | `Real`      |
| `false`    | `Fake`      |

Artefatos principais:

| Camada                   | Caminho                                                     |
| ------------------------ | ----------------------------------------------------------- |
| Bronze vídeos           | `data/bronze/videos/`                                     |
| Bronze manifesto         | `data/bronze/manifests/bronze_manifest.csv`               |
| Silver metadata          | `data/silver/face_metadata_json/`                         |
| Silver frame features    | `data/silver/frame_features/`                             |
| Silver video features    | `data/silver/video_features/video_features.parquet`       |
| Silver temporal features | `data/silver/temporal_features/temporal_features.parquet` |
| Gold regional            | `data/gold/gold_video_region_dataset.parquet`             |
| Gold treino              | `data/gold/gold_training_dataset.parquet`                 |

Execução DVC:

```bash
dvc repro validate_data_contracts
```

### 2. DF26

O DF26 foi incorporado como benchmark local de experimentação inicial e treino
controlado. Ele entra como `dataset_local`, separado do fluxo padrão, para não
sobrescrever os artefatos de smoke test ou de datasets por links.

Artefatos DF26:

| Camada                   | Caminho                                                          |
| ------------------------ | ---------------------------------------------------------------- |
| Dataset bruto            | `data/external/df26/`                                          |
| Bronze manifesto         | `data/bronze/manifests/bronze_manifest_df26.csv`               |
| Folds LOGOS              | `data/bronze/manifests/df26_logos_splits.csv`                  |
| Silver metadata          | `data/df26/silver/face_metadata_json/`                         |
| Silver frame features    | `data/df26/silver/frame_features/`                             |
| Silver video features    | `data/df26/silver/video_features/video_features.parquet`       |
| Silver temporal features | `data/df26/silver/temporal_features/temporal_features.parquet` |
| Gold regional            | `data/df26/gold/gold_video_region_dataset.parquet`             |
| Gold treino              | `data/df26/gold/gold_training_dataset.parquet`                 |

O manifesto DF26 preserva metadados de governança:

| Coluna                          | Função                                                 |
| ------------------------------- | -------------------------------------------------------- |
| `benchmark_dataset`           | identifica o benchmark                                   |
| `df26_relative_path`          | caminho relativo do vídeo no dataset                    |
| `df26_clip_id`                | identificador do clipe base                              |
| `df26_scenario`               | cenário do vídeo                                       |
| `df26_generator`              | nome original do gerador                                 |
| `df26_generator_canonical`    | nome normalizado do gerador                              |
| `df26_generator_family`       | família usada para LOGOS                                |
| `df26_generator_availability` | `real`, `open_weight`, `commercial` ou `unknown` |
| `df26_training_role`          | papel experimental permitido                             |
| `df26_allowed_for_training`   | indica se pode participar de treino                      |
| `df26_allowed_for_evaluation` | indica se pode ser avaliado                              |

Essas colunas são mantidas até a Gold como metadados de auditoria. Elas não
devem ser usadas como features do modelo.

Observação: a coluna `dataset_split` da Gold continua sendo uma partição
genérica do pipeline. Para experimentos DF26, a modelagem deve unir a Gold com
`data/bronze/manifests/df26_logos_splits.csv` por `video_id` e usar os folds
LOGOS como protocolo experimental.

#### Protocolo LOGOS

O DF26 é preparado com folds leave-one-generator-family-out:

| Fold                             | Teste                                       |
| -------------------------------- | ------------------------------------------- |
| `logos_leave_HunyuanVideo_1.5` | fakes HunyuanVideo 1.5                      |
| `logos_leave_LTX_2.3`          | fakes LTX 2.3                               |
| `logos_leave_Wan_2.2`          | fakes Wan 2.2                               |
| `commercial_final`             | fakes comerciais para avaliação final OOD |

Regra experimental:

| Subconjunto       | Papel                                  |
| ----------------- | -------------------------------------- |
| Reais             | referência real                       |
| Fakes open-weight | treino/teste LOGOS                     |
| Fakes comerciais  | avaliação final com modelo congelado |

Não use fakes comerciais para treino, seleção de hiperparâmetros, calibração,
threshold tuning, augmentation, representation learning ou escolha de modelo.

Execução DVC:

```bash
dvc repro pipelines/df26/dvc.yaml:df26_validate_data_contracts
```

Após a execução, envie os artefatos versionados para o MinIO:

```bash
dvc push
```

Documentação detalhada:

```text
docs/df26_benchmark.md
```

## Pipeline Bronze/Silver/Gold

O pipeline segue uma arquitetura de engenharia de dados em camadas.

```text
Bronze
  vídeos brutos, manifestos e hashes

Silver Metadata
  regiões detectadas por frame

Silver Frame Features
  sinais A-E por frame, região e track

Silver Video Features
  agregações estáticas por vídeo e região

Silver Temporal Features
  derivadas e estatísticas temporais por vídeo, região e track

Gold Video-Region
  dataset regional para EDA, auditoria e ablação

Gold Training
  matriz final com uma linha por vídeo
```

## Padronização Analítica Do Vídeo

Os vídeos brutos não são transcodificados nem sobrescritos. A padronização
ocorre na camada analítica, preservando rastreabilidade do arquivo original.

| Dimensão               | Campo / Parâmetro                                          | Decisão Atual                   |
| ----------------------- | ----------------------------------------------------------- | -------------------------------- |
| FPS original            | `video_fps`                                               | registrado a partir do arquivo   |
| FPS analítico          | `sample_fps`                                              | padrão`5.0`                   |
| Tempo físico           | `timestamp_s`                                             | usado para derivadas por segundo |
| Frames máximos padrão | `max_frames`                                              | `4` no smoke, `25` no DF26   |
| Duração               | `duration_s`                                              | registrada por vídeo            |
| Frame count             | `frame_count`                                             | registrado por vídeo            |
| Resolução original    | `original_frame_width`, `original_frame_height`         | preservada para auditoria        |
| Resolução analítica  | `standardized_frame_width`, `standardized_frame_height` | maior lado limitado a`640`     |
| Região/identidade      | `region`, `region_type`, `track_id`                   | mantém rastreabilidade regional |

A amostragem temporal usa o FPS original \(f_o\) e o FPS analítico \(f_s\):

\[
s = \max\left(1, \operatorname{round}\left(\frac{f_o}{f_s}\right)\right)
\]

Os índices processados são:

\[
\{0, s, 2s, 3s, \ldots\}.
\]

Quando `max_frames` é definido, a seleção final é uma subamostra uniforme dos
índices já padronizados por `sample_fps`.

O redimensionamento analítico usa:

\[
\alpha = \min\left(\frac{640}{\max(H,W)}, 1\right).
\]

As regiões detectadas no frame original são reescaladas pela mesma razão.

## Detecção De Regiões

O pré-processamento usa MediaPipe em três componentes:

| Componente     | Uso                                                       |
| -------------- | --------------------------------------------------------- |
| FaceDetector   | localiza faces no frame completo e em janelas sobrepostas |
| FaceLandmarker | recupera landmarks faciais nos crops detectados           |
| ImageSegmenter | auxilia corpo/pessoa e fundo quando disponível           |

Regiões geradas:

| Região        | `region_type`    | Observação                                    |
| -------------- | ------------------ | ----------------------------------------------- |
| Rosto completo | `rosto_completo` | uma instância por face detectada               |
| Olhos          | `olhos`          | derivada dos landmarks faciais                  |
| Boca           | `boca`           | derivada dos landmarks faciais                  |
| Corpo          | `corpo`          | segmentação de pessoa ou fallback geométrico |
| Fundo          | `fundo`          | complemento operacional das regiões ocupadas   |

Quando há múltiplas faces, o pipeline cria regiões rastreáveis:

```text
rosto_completo_1, olhos_1, boca_1, corpo_1
rosto_completo_2, olhos_2, boca_2, corpo_2
fundo
```

O campo `track_id` mantém a associação entre regiões da mesma pessoa quando
possível.

## Métricas Atuais Por Família De Sinal

### Grupo A - Textura, Bordas E Nitidez

| Sinal           | Métricas Implementadas                                                                                                                                                                                             | Hipótese                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| LBP multiescala | `entropy_norm`, `uniformity`, `active_bin_ratio`, `hist_js_distance`                                                                                                                                        | microtextura sintética pode apresentar distribuição local menos natural ou instável           |
| Sobel           | `magnitude_mean`, `magnitude_std`, `magnitude_median`, `magnitude_p95`, `gradient_energy`, `magnitude_entropy_norm`, `orientation_entropy_norm`, `orientation_coherence`, `strong_gradient_ratio` | bordas geradas podem ter transições artificiais, suavização ou coerência direcional anômala |
| Laplaciano      | `signed_mean`, `signed_std`, `signed_variance`, `signed_energy`, `abs_mean`, `abs_median`, `abs_p95`, `entropy_norm`, `kurtosis`, `standardized_tail_ratio`                                     | geração, compressão e upsampling podem alterar nitidez e altas frequências                    |

LBP usa escalas:

```text
(P=8, R=1), (P=16, R=2), (P=24, R=3)
```

### Grupo B - Estrutura Local

| Sinal            | Métricas Implementadas                                                                                                                                                                                                               | Hipótese                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| SIFT             | `kp_count`, `kp_density`, `kp_coverage`, `response_mean`, `response_std`, `size_mean`, `size_std`, `orientation_entropy_norm`, `orientation_coherence`, `descriptor_entropy_norm`, `descriptor_self_similarity` | regiões geradas podem ter keypoints menos estáveis, menos distribuídos ou descritores excessivamente similares |
| Patch similarity | `sim_mean`, `sim_std`, `sim_median`, `sim_p95`, `sampled_patch_count`, `candidate_patch_count`, `sampling_coverage`                                                                                                     | síntese pode induzir redundância ou auto-similaridade local anômala                                            |

### Grupo C - Resíduo Bilateral

| Sinal              | Métricas Implementadas                                                                                                                                          | Hipótese                                                                                     |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Resíduo bilateral | `signed_mean`, `std`, `rms`, `mad`, `abs_p95`, `entropy_norm`, `kurtosis`, `horizontal_lag1_corr`, `vertical_lag1_corr`, `channel_corr_mean` | o residual high-pass pode capturar diferenças estatísticas entre textura natural e síntese |

Observação metodológica: esse residual é um baseline high-pass. Ele não deve ser
interpretado como PRNU ou ruído de sensor isolado.

### Grupo D - Frequência Espacial

| Sinal        | Métricas Implementadas                                                                                                                                                                                                          | Hipótese                                                                                        |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| FFT espacial | `mean_log_amplitude`, `std_log_amplitude`, `low_power_ratio`, `mid_power_ratio`, `high_power_ratio`, `radial_centroid`, `spectral_entropy_norm`, `spectral_flatness`, `angular_anisotropy`, `spectral_slope` | geradores podem alterar a distribuição de potência entre baixas, médias e altas frequências |

A FFT usa crops por região e janela de Hann antes do espectro:

\[
P(u,v) = |\mathcal{F}(W \cdot (I - \bar{I}))|^2.
\]

### Grupo E - Fotometria E Candidatos De Sombra

| Sinal                | Métricas Implementadas                                                                                                                                                                                                                                                                                                                                                                                                                                                | Hipótese                                                                                                           |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Fotometria regional  | `l_mean`, `l_std`, `l_contrast_p90_norm`, `l_iqr`, `l_entropy_norm`, `a_mean`, `a_std`, `b_mean`, `b_std`, `saturation_mean`, `saturation_std`, `value_mean`, `illumination_mean`, `illumination_std`, `illumination_gradient_energy`, `illumination_gradient_std`, `illumination_gradient_coherence`, `illumination_gradient_direction`, `dark_pixel_ratio`, `bright_pixel_ratio`, `black_clip_ratio`, `white_clip_ratio` | vídeos falsos podem apresentar inconsistência de luminância, crominância e campo de iluminação entre regiões |
| Assimetria facial    | `photo_face_lr_luma_asymmetry`, `photo_face_tb_luma_asymmetry`, `photo_face_quadrant_luma_imbalance`                                                                                                                                                                                                                                                                                                                                                             | iluminação facial real tende a preservar relações espaciais coerentes                                           |
| Candidatos de sombra | `candidate_ratio`, `illumination_ratio_mean`, `illumination_ratio_std`, `illumination_ratio_p10`, `candidate_depth_mean`, `candidate_to_lit_luminance_ratio`, `candidate_chromaticity_shift`, `boundary_density`, `boundary_gradient_mean`                                                                                                                                                                                                           | sombras aproximadas e regiões de baixa iluminação podem variar de modo não físico                              |

Observação metodológica: candidatos de sombra ainda são proxies fotométricos.
Eles não constituem uma reconstrução física completa de sombra 3D.

## Contrastes Regionais

Para sinais comparáveis entre regiões, o pipeline calcula contrastes entre
face/região-alvo, borda e fundo:

\[
d_s = x_A - x_B
\]

\[
d_a = |x_A - x_B|
\]

\[
d_n = \frac{x_A - x_B}{|x_A| + |x_B| + 10^{-6}}.
\]

Esses contrastes são relevantes porque um vídeo falso pode parecer plausível no
frame completo e ainda apresentar discrepâncias entre rosto, olhos, boca, corpo
e fundo.

## Temporalidade Implementada

A temporalidade é calculada sobre séries de sinais por:

```text
video_id, region, region_type, track_id
```

Para cada sinal escalar \(x_t\), ordenado por `timestamp_s`, o pipeline calcula
primeira derivada:

\[
\Delta x_t =
\frac{x_t - x_{t-1}}{t_t - t_{t-1}}.
\]

Essa normalização por tempo físico evita que o delta dependa do FPS original do
arquivo.

A segunda derivada é:

\[
\Delta^2 x_t =
\frac{\Delta x_t - \Delta x_{t-1}}{\tau_t - \tau_{t-1}},
\]

em que \(\tau_t\) é o instante médio da primeira diferença. Em amostragem
uniforme, a intuição é:

\[
\Delta^2 x_t \approx x_{t+1} - 2x_t + x_{t-1}.
\]

A motivação vem de trabalhos que exploram diferenças de segunda ordem e
inconsistência temporal em vídeos gerados. Aqui a operação é aplicada sobre
sinais handcrafted:

\[
\Delta^2\Phi(I_t)
\]

em vez de embeddings aprendidos por redes profundas.

### Métricas Temporais

Para cada série elegível, são calculados:

| Família Temporal    | Métricas                                                                      |
| -------------------- | ------------------------------------------------------------------------------ |
| Primeira ordem       | `d1_mean`, `d1_std`, `d1_median`, `d1_mad`, `d1_iqr`, `d1_p95_abs` |
| Segunda ordem        | `d2_mean`, `d2_std`, `d2_median`, `d2_mad`, `d2_iqr`, `d2_p95_abs` |
| Autocorrelação     | `lag1_autocorr`, `lag2_autocorr`                                           |
| Frequência temporal | `temporal_energy`, `temporal_high_energy_ratio`, `temporal_entropy_norm` |

Na Gold final, para controlar dimensionalidade, são mantidos principalmente:

```text
d1_std
d1_mad
d1_p95_abs
d2_std
d2_mad
d2_p95_abs
lag1_autocorr
temporal_high_energy_ratio
```

Os sinais temporalizados são escolhidos por uma lista canônica de tokens
interpretáveis. Colunas de controle de qualidade iniciadas por `qc_` não entram
na temporalização.

## Gold Final

O projeto gera dois datasets Gold:

| Artefato                              | Grão                          | Uso                                 |
| ------------------------------------- | ------------------------------ | ----------------------------------- |
| `gold_video_region_dataset.parquet` | uma linha por vídeo e região | EDA, auditoria regional e ablação |
| `gold_training_dataset.parquet`     | uma linha por vídeo           | treino e serving                    |

A Gold de treino agrega regiões por `region_type`. Exemplo de colunas:

```text
video_id
target_label
dataset_split
is_trainable
missing_feature_ratio
rosto_completo__lbp_r1_p8_face_entropy_norm_mean
rosto_completo__temporal__lbp_r1_p8_face_entropy_norm__d1_std
olhos__photo_face_l_mean_mean
boca__temporal__sobel_face_gradient_energy__d2_std
fundo__fft_background_high_power_ratio_mean
```

Quando há múltiplas faces, as regiões são consolidadas por tipo:

```text
rosto_completo_1 + rosto_completo_2 -> rosto_completo__
olhos_1 + olhos_2                 -> olhos__
boca_1 + boca_2                   -> boca__
corpo_1 + corpo_2                 -> corpo__
fundo                              -> fundo__
```

Isso evita tratar múltiplas regiões do mesmo vídeo como amostras independentes
no treinamento final.

## Qualidade E Validação

O pipeline valida contratos e qualidade dos dados com:

- contratos internos em `src/shared/contracts/schemas.py`;
- validação de tabelas Bronze/Silver/Gold;
- Great Expectations quando `--with-gx` é usado;
- relatório em `data/reports/pipeline_latest.json` ou
  `data/reports/pipeline_latest_df26.json`.

Métricas de qualidade acompanhadas:

| Métrica                           | Interpretação                                                                         |
| ---------------------------------- | --------------------------------------------------------------------------------------- |
| `missing_feature_ratio`          | proporção de features ausentes na Gold                                                |
| `temporal_missing_feature_ratio` | proporção de derivados temporais ausentes                                             |
| `n_frames`                       | frames usados por vídeo/região                                                        |
| `metadata_rows_used`             | linhas de metadados utilizadas                                                          |
| `n_temporal_frames`              | frames disponíveis para temporalidade                                                  |
| `temporal_time_span_s`           | janela temporal efetiva                                                                 |
| `is_trainable`                   | indica se a amostra pode entrar em treino                                               |
| `quality_flag`                   | `ok`, `review`, `insufficient_metadata`, `missing_label` ou `feature_failure` |

## Reprodutibilidade Com DVC E MinIO

O DVC versiona dados por hashes e outputs declarados nos pipelines. O MinIO é o
remote S3 local/privado.

Configuração atual esperada:

```text
remote = minio
url = s3://datalake/dvc
endpointurl = http://localhost:9000
```

Fluxo reprodutível:

```bash
docker compose up -d minio
python -m src.data_engineering.infra init
dvc repro validate_data_contracts
dvc push
```

Para DF26:

```bash
huggingface-cli login
dvc repro pipelines/df26/dvc.yaml:df26_validate_data_contracts
dvc push
```

Papel de cada camada:

| Camada                | Responsabilidade                                                                    |
| --------------------- | ----------------------------------------------------------------------------------- |
| Git                   | versiona código,`params.yaml`, `dvc.yaml`, `pipelines/df26/dvc.yaml` e locks |
| DVC local             | calcula hashes e mantém cache local dos artefatos                                  |
| MinIO via`dvc push` | armazena dados versionados de forma recuperável                                    |
| `publish-lake`      | publica uma visão navegável do`data/`, mas não substitui o versionamento DVC   |

Para reproduzir em outra máquina:

```bash
git clone <repo>
cd tcc
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d minio
python -m src.data_engineering.infra init
dvc pull
dvc repro validate_data_contracts
```

Para reproduzir DF26 do zero, também é necessário aceitar os termos do dataset
no Hugging Face e autenticar com `huggingface-cli login`.

## Execução Local

### Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
```

### Smoke Test Padrão

```bash
dvc repro validate_data_contracts
```

### Pipeline Manual Com Links

```bash
python -m src.data_engineering.pipeline build \
  --source-csv data/bronze/manifests/smoke_source.csv \
  --manifest data/bronze/manifests/bronze_manifest.csv \
  --videos-dir data/bronze/videos \
  --metadata-dir data/silver/face_metadata_json \
  --groups abcde \
  --max-frames 4 \
  --sample-fps 5 \
  --temporal-min-points 3 \
  --generate-missing-metadata true \
  --face-detector-model models/face_detector.task \
  --face-model experimentos/grupo_b/data/extracted/face_landmarker.task \
  --segmenter-model models/image_segmenter.task \
  --with-gx \
  --fail-on-error
```

### Pipeline DF26

```bash
python -m src.data_engineering.ingestion.df26 prepare
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

## Estrutura Do Repositório

```text
.
├── data/
│   ├── bronze/              # manifestos e vídeos do fluxo padrão
│   ├── external/df26/       # DF26 bruto baixado do Hugging Face
│   ├── silver/              # Silver do fluxo padrão
│   ├── gold/                # Gold do fluxo padrão
│   ├── df26/                # Silver/Gold específicos do DF26
│   └── reports/             # relatórios de qualidade e métricas
├── docs/
│   ├── df26_benchmark.md    # guia do DF26
│   └── contrato_sinais_v0_2.md
├── experimentos/
│   ├── grupo_a/
│   ├── grupo_b/
│   ├── grupo_c/
│   ├── grupo_d/
│   ├── grupo_e/
│   └── pre_processamento/
├── pipelines/
│   └── df26/dvc.yaml        # pipeline DVC separado do DF26
├── src/
│   ├── api/
│   ├── data_engineering/
│   ├── ml/
│   └── shared/
├── tests/
├── dvc.yaml                 # pipeline padrão
├── params.yaml              # parâmetros dos pipelines
└── requirements.txt
```

## Estado Atual

Implementado:

- ingestão por links e ingestão DF26;
- manifesto Bronze para DF26 com metadados de licença/protocolo;
- folds LOGOS para DF26;
- detecção regional com MediaPipe;
- extração dos grupos A-E;
- padronização analítica de FPS, timestamps e resolução;
- temporalidade por região e track;
- Gold regional e Gold final por vídeo;
- validação de contratos;
- DVC para pipeline padrão e DF26;
- integração com MinIO como remote DVC.

Em andamento / futuro:

- EDA estatístico dos sinais na amostra DF26;
- seleção de features e redução de dimensionalidade;
- treinamento de modelos baseline;
- avaliação por gerador e LOGOS;
- reflexos oculares;
- sombras físicas mais completas;
- rPPG, geometria 3D, optical flow e tracking denso;
- comparação com embeddings/modelos profundos.

## Limitações Atuais

- O corpo depende de segmentação de pessoa ou fallback geométrico; ainda não é
  segmentação corporal multi-instância perfeita.
- O fundo é um complemento operacional das regiões ocupadas, não uma
  reconstrução semântica completa da cena.
- A temporalidade atual é tabular e estatística; não é um transformer ou modelo
  sequencial profundo.
- O smoke test valida funcionamento, não desempenho científico.
- A Gold pode ficar muito larga; a modelagem deve incluir análise de nulos,
  redundância, correlação e seleção de features.
- O DF26 exige acesso autenticado e respeito às restrições de uso do dataset.

## Próximo Passo Científico

O próximo passo natural é executar o DF26 ou uma amostra controlada, auditar
visualmente as regiões e realizar EDA dos sinais:

- distribuição por classe;
- outliers;
- nulos;
- sinais constantes;
- correlação;
- separação por região;
- separação por família de sinal;
- contribuição dos derivados temporais;
- ablação por gerador no protocolo LOGOS.

Só depois disso a modelagem deve consolidar os classificadores baseline.

## Referências

- Bansal et al. (2024), *VideoPhy: Evaluating Physical Commonsense for Video Generation*.
- Tong et al. (2022), *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*.
- Bertasius, Wang e Torresani (2021), *Is Space-Time Attention All You Need for Video Understanding?*
- Chen et al. (2024), *DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark*.
- Bai, Lin e Cao (2024), *AI-Generated Video Detection via Spatio-Temporal Anomaly Learning*.
- Zheng et al. (2025), *Training-Free AI-Generated Video Detection Using Second-Order Features*.
- Zhang et al. (2025), *Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection*.
- Corvi, Cozzolino, Prashnani, De Mello, Nagano e Verdoliva (2025), *Seeing What Matters: Generalizable AI-generated Video Detection with Forensic-Oriented Augmentation*.
- Cui et al. (2026), *Rethinking the Readout: Unlocking Video Backbones for AI-Generated Video Detection*.
- Ben Hayun et al. (2026), *Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods*.
- Chen, Karaoglu e Gevers (2025), *Detecting AI-Generated Videos from 3D Geometric Temporal Consistency*.
- Qiu, Zhao e Qu (2026), *Beyond Semantics: Uncovering the Physics of Fakes via Universal Physical Descriptors for Cross-Modal Synthetic Detection*.
- *Handcrafted Feature Fusion for Reliable Detection of AI-Generated Images* (2026).
- Qing et al. (2026), *Moiré Video Authentication: A Physical Signature Against AI Video Generation*.
- Cakiroglu et al. (2026), *Auditing Generalization in AI-Generated Video Detection: A Six-Control Protocol and the VidAudit Toolkit*.
- Michels, Jorissen e Michiels (2026), *Dataset Biases and Shortcut Learning in Motion-Based AI-Generated Video Detection*.
- Ojala, Pietikäinen e Mäenpää (2002), *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns*.
- Haralick, Shanmugam e Dinstein (1973), *Textural Features for Image Classification*.
- Lowe (2004), *Distinctive Image Features from Scale-Invariant Keypoints*.
- Tomasi e Manduchi (1998), *Bilateral Filtering for Gray and Color Images*.
- Wang et al. (2021), *M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection*.
- Chen e Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*.
- Breiman (2001), *Random Forests*.
- Cortes e Vapnik (1995), *Support-Vector Networks*.

## Links De Datasets

Dataset principal da etapa atual:

- DF26: [https://huggingface.co/datasets/DF26/DF26](https://huggingface.co/datasets/DF26/DF26)

Datasets auxiliares/referenciais previamente considerados:

- DigiFakeAV: [https://huggingface.co/datasets/cambrain/DigiFakeAV](https://huggingface.co/datasets/cambrain/DigiFakeAV)
- Deepfake-Eval-2024: [https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024)

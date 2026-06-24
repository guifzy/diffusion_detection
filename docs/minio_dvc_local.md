# Execucao local com DVC + MinIO

Este projeto usa a Opcao A:

```text
pipeline escreve em data/
DVC versiona os outputs locais
dvc push envia o cache versionado para MinIO
pipeline de treino faz dvc pull
```

Nesse desenho, o pipeline nao escreve diretamente em `s3://...`. Ele escreve em `data/`, e o DVC envia a versao dos artefatos para o MinIO.

## 1. Secrets locais

Copie o exemplo:

```bash
cp .env.example .env
```

Valores locais padrao:

```bash
MINIO_ROOT_USER=tccadmin
MINIO_ROOT_PASSWORD=tccadmin123
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=tccadmin
MINIO_SECRET_KEY=tccadmin123
MINIO_BUCKET=tcc-datalake
MINIO_DVC_PREFIX=dvc
MINIO_LAKE_PREFIX=lake
MINIO_SECURE=false
DVC_REMOTE_NAME=minio
```

Para TCC/local, esses secrets sao suficientes. Para producao, trocar usuario/senha e nunca commitar `.env`.

## 2. Subir MinIO local

```bash
docker compose up -d minio
```

Console:

```text
http://localhost:9001
```

Login/senha padrao:

```text
tccadmin / tccadmin123
```

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

O DVC precisa de suporte S3/MinIO via `dvc-s3`.

## 4. Configurar bucket e remote DVC

Carregue o `.env` no shell:

```bash
set -a
source .env
set +a
```

Depois rode:

```bash
python -m src.data_engineering.infra init
```

Esse comando:

```text
cria o bucket tcc-datalake no MinIO
inicializa .dvc se necessario
configura remote minio em s3://tcc-datalake/dvc
salva access_key_id e secret_access_key em .dvc/config.local
define o remote minio como default
```

## 5. Rodar pipeline e enviar dados para MinIO

```bash
dvc repro
dvc push
```

Ou via helper Python:

```bash
python -m src.data_engineering.infra repro-push
```

Para publicar tambem uma copia navegavel por camadas no MinIO:

```bash
python -m src.data_engineering.infra publish-lake
python -m src.data_engineering.infra list-lake
```

Isso cria objetos em:

```text
s3://tcc-datalake/lake/bronze/...
s3://tcc-datalake/lake/silver/...
s3://tcc-datalake/lake/gold/...
s3://tcc-datalake/lake/reports/...
```

O DVC continua sendo a fonte de versionamento. A copia `lake/` existe para navegacao, auditoria e apresentacao das camadas.

## 5.1. Rodar pelo container do pipeline

O Docker roda dois papeis diferentes:

```text
minio         -> storage S3 local
data-pipeline -> ambiente Python/DVC reprodutivel
```

No host/WSL, o endpoint do MinIO e `localhost:9000`.
Dentro do container `data-pipeline`, o endpoint correto e `minio:9000`.
O `docker-compose.yml` ja faz esse override automaticamente.

Build da imagem:

```bash
docker compose --profile pipeline build data-pipeline
```

Inicializar bucket e remote DVC dentro do container:

```bash
docker compose --profile pipeline run --rm data-pipeline \
  python -m src.data_engineering.infra init
```

Rodar testes dentro do container:

```bash
docker compose --profile pipeline run --rm data-pipeline \
  python -m pytest tests -q
```

Rodar DVC dentro do container:

```bash
docker compose --profile pipeline run --rm data-pipeline dvc repro
docker compose --profile pipeline run --rm data-pipeline dvc push
```

Ou executar o helper:

```bash
docker compose --profile pipeline run --rm data-pipeline \
  python -m src.data_engineering.infra repro-push
```

## 6. Como o pipeline de treino consumira Gold

Em outro momento/ambiente local:

```bash
git pull
pip install -r requirements.txt
set -a
source .env
set +a
python -m src.data_engineering.infra init
dvc pull
```

Depois disso, o dataset Gold fica disponivel em:

```text
data/gold/gold_training_dataset.parquet
```

O pipeline de treino deve consumir apenas esse contrato Gold.

## 7. Diagnostico

```bash
python -m src.data_engineering.infra check
dvc remote list
dvc doctor
```

## 7.1. Validacao bloqueante, GX, metricas e plots

O stage `validate_data_contracts` roda:

```bash
python -m src.data_engineering.pipeline validate \
  --report data/reports/pipeline_latest.json \
  --with-gx \
  --fail-on-error
```

Ele produz:

```text
data/reports/pipeline_latest.json
data/reports/metrics.json
data/reports/plots/gold_distributions.csv
data/reports/plots/stage_counts.csv
data/reports/logs/pipeline_YYYYMMDD.jsonl
```

`--fail-on-error` faz o comando retornar erro quando houver problemas bloqueantes, por exemplo:

```text
contratos falhando
Bronze vazio
Silver sem metadata/features
Gold vazio
Gold sem linhas treinaveis
missing_feature_ratio acima do limite
fallback_center_ratio acima do limite
```

`--with-gx` executa tambem validacoes formais com Great Expectations quando o pacote esta instalado.

Depois do DVC:

```bash
dvc metrics show
dvc plots show
```

## 8. Papel do Prefect

O Prefect deve agendar:

```text
dvc repro
dvc metrics show
dvc push
```

Ou chamar:

```bash
python -m src.data_engineering.orchestration.prefect_flow
```

O Prefect nao substitui o DVC. O DVC continua sendo o executor reprodutivel e versionador dos dados.

Execucao unica:

```bash
python -m src.data_engineering.orchestration.prefect_flow run
```

Execucao agendada local, mantendo um processo vivo:

```bash
python -m src.data_engineering.orchestration.prefect_flow serve \
  --interval-seconds 86400
```

Ou com cron:

```bash
python -m src.data_engineering.orchestration.prefect_flow serve \
  --cron "0 3 * * *"
```

Deployment em um work pool existente:

```bash
python -m src.data_engineering.orchestration.prefect_flow deploy \
  --work-pool-name local-process
```

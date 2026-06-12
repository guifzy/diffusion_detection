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


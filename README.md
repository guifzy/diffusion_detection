# Ensemble híbrido de análise espacial, espectral e temporal para detecção de vídeos gerados por diffusion

## Visão geral

Este projeto investiga a detecção de vídeos sintéticos gerados por modelos de difusão por meio de um **ensemble híbrido** de sinais visuais. Em vez de depender de um único classificador, a proposta combina evidências de diferentes domínios (textura, estrutura, ruído, frequência, física e robustez), além de abordagens temporais e de transfer learning.

O foco central é a **detecção orientada a sinais**: identificar padrões estatísticos e físicos que tendem a divergir entre vídeos reais e vídeos sintéticos, especialmente quando os artefatos ficam sutis para inspeção humana.

## Objetivo

Construir uma pipeline reprodutível para:

- extrair sinais discriminativos de vídeos reais e fake;
- organizar experimentos por grupos metodológicos;
- comparar desempenho de métricas manuais e modelos aprendidos;
- evoluir para um ensemble final com melhor robustez.

## Metodologia

A metodologia está dividida em dois eixos complementares:

1. **Análise de sinais** (features explicáveis e interpretáveis);
2. **Métodos baseados em modelos** (aprendizado espacial/temporal e transfer learning).

### 1) Análise de sinais

#### Grupo A - textura

Captura microestruturas e variações locais de padrão visual.

- **LBP**: descreve textura local, variação espacial e inconsistência entre regiões.
- **Sobel**: enfatiza gradientes e bordas, útil para transições artificiais e incoerência direcional.
- **Laplacian**: destaca componentes de alta frequência e nitidez.
- **Entropia global**: mede complexidade/dispersão de informação no frame.

#### Grupo B - estrutura

Foca na coerência geométrica e na organização espacial de detalhes.

- **SIFT**: keypoints e descritores para estabilidade estrutural.
- **Patch similarity (self-similarity)**: redundância local que pode indicar síntese.

#### Grupo C - resíduo de alta passagem

Avalia assinaturas de ruído natural vs. ruído residual sintético.

- **Resíduo bilateral**: baseline de alta passagem após suavização preservadora de bordas.
- **Estatísticas robustas**: RMS, MAD, caudas e entropia do resíduo.
- **Dependência espacial e cromática**: autocorrelação lag-1 e correlação entre canais.

#### Grupo D - frequência (FFT)

Observa distribuição espectral e simetrias no domínio da frequência.

- **Razões de potência**: distribuição em baixas, médias e altas frequências.
- **Centroide e inclinação radial**: posição e decaimento da energia espectral.
- **Anisotropia angular**: direção preferencial da potência espectral.
- **Entropia e planicidade**: dispersão da potência no espectro.

#### Grupo E - fotometria regional

Procura inconsistências com o comportamento óptico esperado no mundo real.

- **Fotometria**: luminância, contraste, crominância, saturação e campo suavizado entre regiões.
- **Assimetria facial de luminância**: diferenças normalizadas entre lados e quadrantes.
- **Candidatos de sombra**: iluminação Retinex em luminância linear, profundidade, borda e cromaticidade.
- **Reflexos oculares**: planejados; não fazem parte da versão atual.

#### Grupo F - robustez (planejado)

Testa estabilidade dos sinais sob perturbações controladas.

- **Rotação**: sensibilidade dos descritores a mudanças de orientação.
- **Delta estrutural (FFT/gradiente)**: variação estrutural após transformações.

### 2) Métodos com modelos de difusão e aprendizado

#### Modelos de difusão

- **Stable Diffusion pré-treinado**: usado como referência de padrões gerativos e para análise de comportamento de síntese.

#### Temporal

- **CNN**: extração de padrões espaciais por frame.
- **Arquitetura Transformers**: modelagem de dependências temporais e inconsistências ao longo do vídeo.

## Foco em detecção de sinais

A proposta prioriza sinais interpretáveis porque:

- aumenta a explicabilidade dos resultados;
- facilita diagnóstico de erro por tipo de artefato;
- permite combinar sinais complementares em um ensemble mais robusto;
- reduz dependência exclusiva de um modelo caixa-preta.

Na prática, cada grupo de sinais captura uma faceta diferente do problema. A decisão final pode ser obtida por agregação de score, votação ponderada ou metamodelo, aproveitando o que cada grupo detecta melhor.

## Arquitetura atual do repositório

```text
.
├── data/
│   ├── bronze/              # vídeos brutos e manifestos de ingestão
│   ├── silver/              # metadados e features processadas
│   ├── gold/                # dataset final para treino
│   └── reports/             # relatórios de qualidade/execução
├── experimentos/
│   ├── grupo_a/             # notebooks e documentação do Grupo A (textura)
│   ├── grupo_b/             # notebooks e documentação do Grupo B (estrutura)
│   ├── grupo_c/             # notebooks e documentação do Grupo C
│   ├── grupo_d/             # notebooks do Grupo D (frequência/FFT)
│   ├── grupo_e/             # notebooks e documentação do Grupo E (física/iluminação)
│   └── pre_processamento/   # notebooks de pré-processamento
├── output_exemples/         # exemplos de saídas e resultados
├── src/
│   ├── shared/              # contratos, storage, paths, sinais A-E e utilitários comuns
│   ├── data_engineering/    # Bronze/Silver/Gold, DVC, Prefect e qualidade dos dados
│   ├── ml/                  # domínio futuro de treino, avaliação e registry de modelos
│   └── api/                 # serviços usados pela futura API/SaaS
├── tests/                   # testes por domínio do monorepo
├── dvc.yaml                 # estágios reprodutíveis do pipeline de dados
├── params.yaml              # parâmetros do pipeline DVC
├── requirements.txt         # dependências do projeto
└── README.md                # este documento
```

## Pré-processamento e extração de regiões

O pipeline inclui o módulo `src.data_engineering.preprocessing` para preparar os vídeos e extrair regiões de interesse por frame:

O pré-processamento utiliza MediaPipe em duas etapas: primeiro o FaceDetector
localiza faces no frame completo e em janelas sobrepostas; depois o
FaceLandmarker recupera os landmarks em crops expandidos. A partir disso sao
produzidas regioes explicitas por frame:

- **rosto completo**: malha facial completa associada a cada identidade;
- **olhos**: subconjunto ocular dos landmarks faciais;
- **boca**: subconjunto oral dos landmarks faciais;
- **corpo**: região corporal segmentada ou aproximada geometricamente;
- **fundo**: complemento das regiões ocupadas no frame.

Essas regiões são usadas nas análises espacial e espectral e servirão de base
para a futura análise temporal.

## Metadados e organização dos arquivos

Os metadados do projeto são organizados por contratos de dados:

- **CSV de entrada Bronze**: `link,label`, onde `label=true` representa vídeo real e `label=false` representa vídeo falso.
- **Manifesto Bronze**: `bronze_manifest.csv`, fonte de verdade após a ingestão, com `video_id`, `source_url`, `filename`, `storage_path`, `sha256`, `label`, `status` e rastreabilidade.
- **JSON auxiliar Silver**: metadados detalhados por vídeo, frame e região para reuso do extrator atual.
- **Parquet/CSV Silver e Gold**: ativos tabulares contratados para validação, treinamento e auditoria por região.

## Formato dos arquivos de vídeo

Atualmente, o projeto utiliza vídeos brutos em `data/bronze/videos/`, manifestos em `data/bronze/manifests/` e metadados/features derivados na camada `data/silver/`.

- `data/bronze/manifests/video-metadata-publish-with-links.csv`: CSV de entrada com apenas `link,label`.
- `data/bronze/manifests/bronze_manifest.csv`: manifesto oficial de ingestão, gerado pelo pipeline.
- `data/silver/face_metadata_json/*_meta.json`: metadados auxiliares por vídeo com informações da extração de regiões.
- `data/silver/face_metadata/`: versão tabular contratada dos metadados regionais.
- `data/silver/frame_features/`: features por frame e região.
- `data/silver/video_features/`: features agregadas por vídeo e região.
- `data/gold/gold_training_dataset.parquet`: dataset oficial para treino em grão vídeo/região.

Essa organização facilita leitura rápida dos dados nos notebooks e padroniza a extração de sinais espaciais, espectrais e temporais.

## Resultados e métricas

Os resultados são salvos em dois níveis:

- **Frame level**: métricas por frame e região armazenadas em formato de **DataFrame** para análise fina ao longo do tempo.
- **Video-region level (final)**: resumos estatísticos agregados dos frames de cada região do vídeo.

Na versão atual, o nível de vídeo contém média, desvio padrão e mediana das métricas por frame. Esses resumos são invariantes à ordem e não constituem análise temporal. Deltas e demais sinais temporais permanecem planejados para uma etapa posterior.

### Estrutura esperada de saída

Cada grupo deve gerar DataFrames com a seguinte estrutura mínima:

```python
# Frame-level (salvar em CSV ou pickle)
frame_results = pd.DataFrame({
    'video_id': [...],
    'frame': [...],
    'region': [...],      # 'face', 'contorno', 'fundo'
    'sinal_1': [...],
    'sinal_2': [...],
    ...
})

# Video-level (agregado)
video_results = pd.DataFrame({
    'video_id': [...],
    'label': [...],       # real / fake
    'sinal_1_mean': [...],
    'sinal_1_std': [...],
    'sinal_1_median': [...],
    ...
})
```

Utilização futura: Multibranch CNN com Transformers temporal para modelar sequências completas de vídeo.

## Dataset utilizado

O dataset de referência utilizado neste projeto é o **DigiFakeAV e Deepfake-Eval-2024:**

Para mais informações sobre o mesmo, acessar o seu repositório:

- Hugging Face:[ DigiFakeAV](https://huggingface.co/datasets/cambrain/DigiFakeAV)
- Hugging Face: [Deepfake-Eval-2024](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024/tree/main)

## Como navegar (primeiro acesso)

1. **Entenda o projeto**: leia este `README.md` para compreender objetivos e grupos metodológicos
2. **Explore os experimentos**: acesse `experimentos/` para consultar os notebooks por grupo
3. **Leia os contratos**: consulte `docs/contracts.md`
4. **Verifique a fonte Bronze**: inspecione `data/bronze/manifests/video-metadata-publish-with-links.csv`
5. **Execute o pipeline reprodutível**: use `dvc repro` depois de configurar MinIO/DVC

**Para começar rapidamente:**

```bash
cp .env.example .env
docker compose up -d minio
pip install -r requirements.txt
python -m src.data_engineering.infra init
dvc repro
dvc push
```

O guia completo de execução local com Docker, MinIO e DVC está em `docs/minio_dvc_local.md`.

## Ambiente e execução

### Requisitos

- Python 3.10+ (recomendado 3.11 ou superior)
- pip ou conda
- ~10GB espaço livre (para dataset + ambiente)

### Instalação

**1. Clone o repositório e entre no diretório:**

```bash
cd projetos/tcc
```

**2. Crie e ative um ambiente virtual:**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou
.venv\Scripts\activate     # Windows
```

**3. Instale as dependências:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
```

**4. (Opcional) Para exportar notebooks para PDF:**

```bash
# Linux/macOS
python -m playwright install-deps chromium

# macOS (via Homebrew)
brew install chromium
```

### Fluxo sugerido de trabalho

1. Organize vídeos de entrada em `data/bronze/videos/` ou use `python -m src.data_engineering.ingestion` para baixá-los via YouTube
2. Execute pré-processamento: `python -m src.data_engineering.preprocessing --video data/bronze/videos/exemplo.mp4`
3. Extraia features de produção: `python -m src.api data/bronze/videos/exemplo.mp4 --groups abcde`
4. Para treinamento, gere o Gold local: `python -m src.data_engineering.datasets --groups abcde`
5. Use notebooks em `experimentos/` para análise comparativa e validação metodológica
6. Integre o dataset Gold ao modelo final

## Status do projeto

Projeto em evolução incremental com base funcional para extração de sinais e análise comparativa.

### Estado atual

**Componentes implementados:**

- **Estrutura de experimentos ativa**: 5 notebooks metodológicos em `experimentos/` (grupos A, B, C, D e E) + pré-processamento
- **Pipeline de pré-processamento**: módulo `src.data_engineering.preprocessing` para detecção facial, extração de regiões (face/contorno/fundo) e geração de metadados
- **Base de dados local**:
  - `data/bronze/videos/`: vídeos brutos
  - `data/bronze/manifests/`: catálogo de origem e manifesto de ingestão
  - `data/silver/face_metadata_json/`: metadados auxiliares (`*_meta.json`)
  - `data/silver/face_metadata/`, `data/silver/frame_features/` e `data/silver/video_features/`: ativos Silver estruturados
  - `data/gold/`: dataset final de treino
- **Documentação técnica**: cada grupo (A, B, C, D e E) possui `README.md` com escopo, métricas, avaliação exploratória e limitações


## Referências

- **Textura, frequência e inconsistência física**:
  [Deepfake forensics: a survey of digital forensic methods for multimodal deepfake identification on social media](https://www.researchgate.net/publication/399736959_DeepFake_Detection_Through_Deep_Learning_A_Comprehensive_Review)
- **Textura (LBP-like), sinais espaciais e features estatísticas**:
  [Deepfake detection: Enhancing performance with spatiotemporal texture and deep learning feature fusion](https://www.sciencedirect.com/science/article/pii/S1110866524000987)
- **FFT (domínio da frequência)**:
  [M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection](https://arxiv.org/abs/2104.09770)
- **FFT (domínio de frequência) + integração multimodal**:
  [Cross-modal deepfake detection: integrating textual and frequency domains](https://www.researchgate.net/publication/403705865_Cross-modal_deepfake_detection_integrating_textual_and_frequency_domains)
- **Pesquisador forense com vídeos e conteúdos de referência**:
  [Willard S. Ribeiro, PhD](https://www.instagram.com/willardsribeiro.ia/)

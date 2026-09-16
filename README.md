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

## Contribuição metodológica v0.3.0

A versão atual formaliza o projeto como uma abordagem de **representação
forense explícita com fronteira de decisão aprendida**. O sistema não aprende a
representação visual por CNN, transformer ou foundation model; ele calcula um
vetor determinístico de sinais interpretáveis e permite que um classificador
tabular aprenda a fronteira de decisão.

Formalmente, para um vídeo \(V\) com frames \(I_1,\ldots,I_T\), regiões
\(r \in R\) e sinais \(k \in K\), o pipeline calcula:

\[
x_{v,t,r,k} = \phi_k(I_{v,t}, r)
\]

em que \(\phi_k\) é um descritor fixo, implementado por fórmula explícita. A
representação final do vídeo é obtida por um agregador determinístico:

\[
z_V = \Psi(\Phi(I_1), \Phi(I_2), \ldots, \Phi(I_T)).
\]

Somente após essa etapa entra o modelo supervisionado:

\[
\hat{y} = g_\theta(z_V).
\]

Essa distinção é central: o projeto é **sem aprendizado de representação**, mas
não é sem aprendizado. O aprendizado ocorre na fronteira de decisão
\(g_\theta\), que pode ser LogReg, SVM, Random Forest, XGBoost ou MLP tabular.

### Fundamentação científica

A hipótese do projeto é sustentada por quatro linhas de literatura:

1. detectores profundos de vídeo, como VideoMAE, TimeSformer, DeMamba e AIGVDet,
   demonstram que há informação espacial e temporal discriminativa em vídeos
   gerados;
2. métodos com priors explícitos, como D3, V-PVP, STALL, NSG-VD, WaveRep e
   Grab-3D, indicam que restrições temporais, físicas, geométricas e
   frequenciais podem melhorar generalização;
3. trabalhos de image forensics e synthetic image detection mostram que LBP,
   Sobel, Laplaciano, DCT, wavelets, GLCM e estatísticas de residual podem
   manter poder discriminativo entre datasets e arquiteturas;
4. auditorias recentes de detecção de vídeo alertam para atalhos experimentais,
   especialmente duração, FPS, resolução, compressão, fonte do dataset e
   quantidade de movimento.

Assim, a pergunta científica fica:

> Até que ponto violações físicas, fotométricas, geométricas, texturais,
> espectrais e temporais compartilhadas por diferentes arquiteturas de geração
> de vídeo podem ser capturadas por descritores explicitamente definidos e
> utilizadas por classificadores supervisionados leves?

## Estabilidade de vídeo e padronização analítica

Os arquivos brutos da camada Bronze não são sobrescritos nem transcodificados.
A padronização ocorre na camada analítica, para preservar reprodutibilidade e
rastreabilidade do dado original.

| Dimensão | Decisão v0.3.0 | Justificativa |
|---|---|---|
| FPS original | registrado como `video_fps` | preserva propriedades do arquivo fonte |
| FPS efetivo de análise | controlado por `sample_fps` | permite comparar vídeos com FPS original diferente |
| Tempo físico | registrado em `timestamp_s` | permite calcular derivadas por segundo |
| Número máximo de frames | controlado por `max_frames` | limita custo computacional em smoke/experimentos |
| Tamanho original | `original_frame_width`, `original_frame_height` | usado para auditoria e controle de viés |
| Tamanho analítico | maior lado limitado por `standardized_max_size=640` | estabiliza escala dos descritores espaciais |
| Identidade/região | `track_id`, `region`, `region_type` | preserva rastreabilidade por pessoa e região |

A amostragem é definida sobre o FPS original do vídeo. Para um vídeo com FPS
nativo \(f_o\) e FPS analítico \(f_s\), o passo aproximado é:

\[
s = \max\left(1, \operatorname{round}\left(\frac{f_o}{f_s}\right)\right).
\]

Os frames amostrados são:

\[
\{0, s, 2s, 3s, \ldots\}.
\]

Quando `max_frames` é informado, o pipeline seleciona uma subamostra uniforme
dos índices já padronizados por `sample_fps`. Isso evita que vídeos longos
dominem o custo de extração e mantém o experimento reproduzível.

O redimensionamento usa escala:

\[
\alpha = \min\left(\frac{640}{\max(H,W)}, 1\right),
\]

mantendo a proporção do frame. As regiões detectadas no frame original são
reescaladas pela mesma razão durante a extração dos sinais.

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

Na versão `0.3.0`, a temporalidade implementada no pipeline não depende de CNN
ou transformer. Ela é calculada diretamente sobre os sinais determinísticos
extraídos por frame e região.

## Temporalidade implementada

Para cada sinal escalar \(x_{t}\), ordenado por `timestamp_s` dentro de
`video_id`, `region`, `region_type` e `track_id`, o pipeline calcula derivadas
temporais discretas.

Primeira ordem:

\[
\Delta x_t =
\frac{x_t - x_{t-1}}{t_t - t_{t-1}}.
\]

Essa medida representa a velocidade de mudança do sinal por segundo. Ela é
necessária porque vídeos podem ter FPS nativo diferente; sem normalização por
\(t_t - t_{t-1}\), o delta dependeria do codec/fonte do vídeo, e não apenas do
fenômeno visual.

Segunda ordem:

\[
\Delta^2 x_t =
\frac{\Delta x_t - \Delta x_{t-1}}{\tau_t - \tau_{t-1}},
\]

em que \(\tau_t\) é o instante médio associado à primeira diferença. Em
amostragem uniforme, essa expressão é equivalente à intuição:

\[
\Delta^2 x_t \approx x_{t+1} - 2x_t + x_{t-1}.
\]

A segunda ordem mede instabilidade na própria taxa de mudança. A motivação vem
de trabalhos como D3, que exploram diferenças de segunda ordem em vídeo gerado,
e é transposta aqui para o espaço handcrafted:

\[
\Delta^2\Phi(I_t)
\]

em vez de aplicar a operação sobre embeddings aprendidos por uma rede.

Para cada série temporal, são extraídos:

| Família temporal | Métricas implementadas | Interpretação |
|---|---|---|
| Primeira ordem | média, desvio, mediana, MAD, IQR, p95 absoluto | magnitude e robustez da variação frame a frame |
| Segunda ordem | média, desvio, mediana, MAD, IQR, p95 absoluto | instabilidade da taxa de mudança |
| Autocorrelação | lag 1 e lag 2 | persistência temporal do sinal |
| Frequência temporal | energia, razão de alta energia, entropia | flicker e oscilação temporal |

Os derivados temporais são calculados apenas para sinais canônicos com
interpretação temporal plausível. Colunas de controle de qualidade iniciadas por
`qc_` não entram na temporalização nem no dataset final de treino.

## Gold final para modelagem

A versão `0.3.0` altera a organização final dos dados para separar análise
regional de treino do modelo.

| Artefato | Grão | Função |
|---|---|---|
| `data/silver/frame_features/{video_id}.parquet` | vídeo + frame + região | sinais A-E estáticos por frame |
| `data/silver/video_features/video_features.parquet` | vídeo + região | agregações estáticas por região |
| `data/silver/temporal_features/temporal_features.parquet` | vídeo + região + track | derivados temporais |
| `data/gold/gold_video_region_dataset.parquet` | vídeo + região | EDA, auditoria e ablação regional |
| `data/gold/gold_training_dataset.parquet` | vídeo | matriz final para treino e serving |

O modelo principal deve consumir `gold_training_dataset.parquet`, que possui uma
linha por vídeo. Isso evita tratar múltiplas regiões do mesmo vídeo como
amostras independentes e reduz risco de vazamento experimental.

A Gold final agrega regiões por `region_type`. Quando há mais de uma face,
`rosto_completo_1` e `rosto_completo_2`, por exemplo, são consolidados sob o
prefixo `rosto_completo__`. O mesmo vale para `olhos`, `boca`, `corpo` e
`fundo`.

Exemplo de colunas finais:

```text
video_id
target_label
dataset_split
is_trainable
rosto_completo__lbp_r1_p8_face_entropy_norm_mean
rosto_completo__temporal__lbp_r1_p8_face_entropy_norm__d1_std
olhos__photo_face_l_mean_mean
boca__temporal__sobel_face_gradient_energy__d2_std
fundo__fft_background_high_power_ratio_mean
```

## Matriz metodológica dos sinais

| Sinal | Hipótese forense | Fórmula/base matemática | Região | Transformação temporal | Status |
|---|---|---|---|---|---|
| LBP multiescala | vídeos sintéticos podem apresentar microtextura local menos natural ou instável | histograma LBP uniforme; entropia normalizada \(H(p)/\log_2(n)\), uniformidade \(\sum_i p_i^2\) e distância Jensen-Shannon entre regiões | rosto, olhos, boca, corpo, fundo e contexto operacional | \(\Delta\), \(\Delta^2\), autocorrelação e energia temporal sobre entropia/uniformidade | manter |
| Sobel | bordas e orientações podem apresentar transições artificiais, excesso de suavização ou coerência direcional anômala | \(m=\sqrt{G_x^2+G_y^2}\), \(\theta=\operatorname{atan2}(G_y,G_x)\), coerência \(\left|\sum m_i e^{i\theta_i}/(\sum m_i+\epsilon)\right|\) | todas as regiões | variação temporal de magnitude, energia, coerência e gradientes fortes | manter |
| Laplaciano | síntese, compressão e upsampling podem alterar nitidez e altas frequências | \(\nabla^2 I\), variância, energia média \(\mathbb{E}[(\nabla^2I)^2]\), estatísticas absolutas e caudas | todas as regiões | oscilação de nitidez por \(\Delta\), \(\Delta^2\) e energia temporal | manter |
| SIFT | estrutura local pode ser menos persistente ou menos distribuída em regiões geradas | densidade de keypoints, cobertura espacial, resposta média, entropia de orientação e similaridade interna de descritores | principalmente rosto, boca e regiões com área suficiente | temporalidade aplicada a densidade, cobertura, resposta e auto-similaridade | exploratório |
| Patch similarity | modelos generativos podem induzir repetição ou auto-similaridade local anômala | similaridade por cosseno entre patches normalizados; média, desvio, mediana e p95 | todas as regiões com patches válidos | estabilidade temporal da similaridade | secundário |
| Residual bilateral | smoothing preservador de bordas remove estrutura de baixa frequência; o residual funciona como proxy high-pass, não PRNU | \(R = I - B(I)\), RMS, MAD, p95 absoluto, entropia, autocorrelação espacial e correlação cromática | todas as regiões | variação temporal do residual e persistência | baseline |
| FFT espacial | imagens sintéticas podem apresentar distribuição espectral não natural | PSD com janela de Hann; \(E_{low}/E\), \(E_{mid}/E\), \(E_{high}/E\), entropia, flatness, anisotropia e slope radial | todas as regiões | oscilação temporal das bandas e energia temporal espectral | manter |
| Fotometria regional | inconsistências de luminância/cor podem indicar composição ou geração não física | estatísticas em Lab/HSV, contraste \(P95-P05\), crominância, saturação e gradiente de iluminação suavizado | rosto, olhos, boca, corpo e fundo | flicker, \(\Delta\), \(\Delta^2\) e persistência regional | manter |
| Assimetria fotométrica facial | iluminação facial real tende a preservar relações espaciais coerentes sob mesma cena | diferenças normalizadas esquerda-direita, topo-base e desequilíbrio por quadrantes | rosto completo | variação temporal das assimetrias | manter |
| Candidatos de sombra | baixa luminância estruturada pode revelar inconsistência fotométrica, mas não equivale a sombra física 3D | luminância linear, iluminação Retinex aproximada, razão de iluminação, densidade de candidatos e bordas | todas as regiões | persistência temporal e mudanças abruptas dos candidatos | exploratório |
| Corpo | corpo funciona como região contextual da pessoa, útil para coerência pessoa-cena | aplica os mesmos descritores de textura, borda, frequência, residual e fotometria | corpo associado ao `track_id` facial | estabilidade corpo-rosto e corpo-fundo por derivados temporais | manter como contexto |
| Fundo | fundo controla artefatos globais de compressão, câmera, iluminação e cena | mesmos sinais calculados no complemento das regiões ocupadas | fundo global | controle temporal global e contraste com regiões da pessoa | manter como controle |

## Contrastes regionais

Para sinais escalares comparáveis, o pipeline preserva diferenças entre região
alvo, borda e fundo:

\[
d_s = x_A - x_B
\]

\[
d_a = |x_A - x_B|
\]

\[
d_n =
\frac{x_A - x_B}{|x_A| + |x_B| + 10^{-6}}.
\]

Esses contrastes são importantes porque a hipótese do projeto não depende
apenas do valor absoluto de uma região. Um vídeo falso pode parecer plausível
globalmente e ainda apresentar discrepância local entre rosto, olhos, boca,
corpo e fundo.

## Limitações metodológicas atuais

- O corpo é obtido por segmentação de pessoa do MediaPipe ou fallback geométrico
  associado à face; ainda não é uma segmentação corporal multi-instância
  perfeita.
- O fundo é um complemento operacional das regiões ocupadas e deve ser tratado
  como controle/contexto, não como reconstrução semântica completa da cena.
- Reflexos oculares, rPPG, geometria 3D e vanishing points ainda não integram a
  versão implementada.
- A temporalidade atual é baseada em séries de sinais por região; optical flow e
  tracking geométrico denso permanecem como extensão futura.
- A Gold gerada por smoke test valida funcionamento, mas não sustenta conclusão
  estatística. EDA e modelagem exigem amostra maior e controle por fonte,
  gerador, duração, FPS, resolução, compressão e movimento.

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
para a análise temporal implementada na versão `0.3.0`.

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
- `data/silver/temporal_features/`: derivados temporais por vídeo, região e track.
- `data/gold/gold_video_region_dataset.parquet`: dataset regional para EDA e ablação.
- `data/gold/gold_training_dataset.parquet`: dataset oficial para treino em grão vídeo.

Essa organização facilita leitura rápida dos dados nos notebooks e padroniza a extração de sinais espaciais, espectrais e temporais.

## Resultados e métricas

Os resultados são salvos em quatro níveis:

- **Frame level**: métricas por frame e região armazenadas em formato de **DataFrame** para análise fina ao longo do tempo.
- **Temporal level**: derivadas e estatísticas temporais calculadas a partir das séries por região.
- **Video-region level**: resumos estáticos e temporais de cada região do vídeo.
- **Video level (final)**: consolidação por `region_type`, com uma linha por vídeo para treino e serving.

Na versão atual, o nível de vídeo combina estatísticas estáticas, derivadas
temporais e indicadores de qualidade. A temporalidade é calculada antes da
Gold final, preservando a ordem dos frames por `timestamp_s`.

### Estrutura esperada de saída

Cada grupo deve gerar DataFrames com a seguinte estrutura mínima:

```python
# Frame-level (salvar em CSV ou pickle)
frame_results = pd.DataFrame({
    'video_id': [...],
    'frame_id': [...],
    'timestamp_s': [...],
    'region': [...],      # 'rosto_completo_1', 'olhos_1', 'boca_1', 'corpo_1', 'fundo'
    'region_type': [...], # 'rosto_completo', 'olhos', 'boca', 'corpo', 'fundo'
    'track_id': [...],
    'sinal_1': [...],
    'sinal_2': [...],
    ...
})

# Video-level final
video_results = pd.DataFrame({
    'video_id': [...],
    'target_label': [...],
    'dataset_split': [...],
    'rosto_completo__sinal_1_mean': [...],
    'rosto_completo__temporal__sinal_1__d1_std': [...],
    'fundo__sinal_2_median': [...],
    ...
})
```

Utilização principal: modelos tabulares supervisionados sobre representação
forense explícita, com avaliação futura por ablação de famílias de sinais e
comparação entre classificadores.

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

### Benchmark DF26

O repositório possui um fluxo específico para usar o DF26 como benchmark local
e base inicial de experimentação. O dataset é tratado como `dataset_local`,
com manifesto Bronze próprio e metadados de licença/protocolo preservados até a
Gold.

Arquivos principais:

| Artefato | Função |
|---|---|
| `src/data_engineering/ingestion/df26.py` | download Hugging Face e conversão DF26 para Bronze |
| `data/bronze/manifests/bronze_manifest_df26.csv` | manifesto Bronze DF26 |
| `data/bronze/manifests/df26_logos_splits.csv` | folds LOGOS e avaliação comercial final |
| `pipelines/df26/dvc.yaml` | reprodução separada do fluxo DF26 |
| `docs/df26_benchmark.md` | instruções completas de uso |

Execução resumida:

```bash
huggingface-cli login
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

Para DVC:

```bash
dvc repro pipelines/df26/dvc.yaml:df26_validate_data_contracts
```

O arquivo `df26_logos_splits.csv` separa fakes open-weight para o protocolo
leave-one-generator-family-out e mantém fakes comerciais como avaliação final
OOD. Esses campos são metadados de governança e não devem ser usados como
features do modelo.

## Status do projeto

Projeto em evolução incremental com base funcional para extração de sinais e análise comparativa.

### Estado atual

**Componentes implementados:**

- **Estrutura de experimentos ativa**: 5 notebooks metodológicos em `experimentos/` (grupos A, B, C, D e E) + pré-processamento
- **Pipeline de pré-processamento**: módulo `src.data_engineering.preprocessing` para detecção MediaPipe de rosto, olhos, boca, corpo e fundo
- **Base de dados local**:
  - `data/bronze/videos/`: vídeos brutos
  - `data/bronze/manifests/`: catálogo de origem e manifesto de ingestão
  - `data/silver/face_metadata_json/`: metadados auxiliares (`*_meta.json`)
  - `data/silver/face_metadata/`, `data/silver/frame_features/`, `data/silver/video_features/` e `data/silver/temporal_features/`: ativos Silver estruturados
  - `data/gold/gold_video_region_dataset.parquet`: dataset regional para EDA
  - `data/gold/gold_training_dataset.parquet`: dataset final por vídeo para treino
- **Documentação técnica**: cada grupo (A, B, C, D e E) possui `README.md` com escopo, métricas, avaliação exploratória e limitações


## Referências

- **VideoPhy**: Bansal et al. (2024), *VideoPhy: Evaluating Physical Commonsense for Video Generation*.
  Fundamenta a diferença entre realismo visual e aderência física em vídeos gerados.
- **VideoMAE**: Tong et al. (2022), *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*.
  Referência para representação espaço-temporal aprendida em vídeo.
- **TimeSformer**: Bertasius, Wang e Torresani (2021), *Is Space-Time Attention All You Need for Video Understanding?*.
  Referência para modelagem profunda espaço-temporal.
- **DeMamba**: Chen et al. (2024), *DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark*.
  Exemplo de detector com representação aprendida para AIGV.
- **AIGVDet / STALL**: Bai, Lin e Cao (2024), *AI-Generated Video Detection via Spatio-Temporal Anomaly Learning*.
  Referência para uso de RGB e movimento/fluxo óptico em detecção de vídeo gerado.
- **D3**: Zheng et al. (2025), *Training-Free AI-Generated Video Detection Using Second-Order Features*.
  Principal motivação para derivados temporais de segunda ordem.
- **NSG-VD**: Zhang et al. (2025), *Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection*.
  Fundamenta a importância de priors físicos espaço-temporais.
- **WaveRep / forensic-oriented augmentation**: Corvi, Cozzolino, Prashnani, De Mello, Nagano e Verdoliva (2025),
  *Seeing What Matters: Generalizable AI-generated Video Detection with Forensic-Oriented Augmentation*.
  Fundamenta pistas forenses de baixo nível e generalização entre geradores.
- **V-PVP**: Cui et al. (2026), *Rethinking the Readout: Unlocking Video Backbones for AI-Generated Video Detection*.
  Motiva preservar dinâmica local em vez de pooling global prematuro.
- **Spatial-temporal likelihoods**: Ben Hayun et al. (2026), *Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods*.
  Referência para estatísticas espaço-temporais de vídeos reais.
- **Grab-3D**: Chen, Karaoglu e Gevers (2025), *Detecting AI-Generated Videos from 3D Geometric Temporal Consistency*.
  Fundamenta geometria temporal como extensão futura.
- **Universal physical descriptors**: Qiu, Zhao e Qu (2026), *Beyond Semantics: Uncovering the Physics of Fakes via Universal Physical Descriptors for Cross-Modal Synthetic Detection*.
  Referência próxima da filosofia handcrafted em imagens geradas.
- **Handcrafted Feature Fusion** (2026), *Handcrafted Feature Fusion for Reliable Detection of AI-Generated Images*.
  Sustenta fusão de LBP, DCT, GLCM, wavelets e classificadores tabulares.
- **Moiré Video Authentication**: Qing et al. (2026), *Moiré Video Authentication: A Physical Signature Against AI Video Generation*.
  Exemplo de assinatura física determinística em vídeo.
- **VidAudit**: Cakiroglu et al. (2026), *Auditing Generalization in AI-Generated Video Detection: A Six-Control Protocol and the VidAudit Toolkit*.
  Referência para controles de duração, FPS, resolução, compressão, movimento e dataset.
- **Motion shortcut learning**: Michels, Jorissen e Michiels (2026), *Dataset Biases and Shortcut Learning in Motion-Based AI-Generated Video Detection*.
  Alerta contra usar quantidade bruta de movimento como atalho.
- **LBP**: Ojala, Pietikäinen e Mäenpää (2002), *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns*.
  Base clássica dos descritores LBP.
- **GLCM/Haralick**: Haralick, Shanmugam e Dinstein (1973), *Textural Features for Image Classification*.
  Referência para textura estatística; extensão recomendada.
- **SIFT**: Lowe (2004), *Distinctive Image Features from Scale-Invariant Keypoints*.
  Base para keypoints e descritores locais.
- **Bilateral filter**: Tomasi e Manduchi (1998), *Bilateral Filtering for Gray and Color Images*.
  Base do residual bilateral usado como high-pass proxy.
- **M2TR**: Wang et al. (2021), *M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection*.
  Referência para relevância do domínio de frequência em deepfake detection.
- **XGBoost**: Chen e Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*.
  Classificador tabular recomendado para baseline forte.
- **Random Forests**: Breiman (2001), *Random Forests*.
  Baseline ensemble para dados tabulares.
- **Support-Vector Networks**: Cortes e Vapnik (1995), *Support-Vector Networks*.
  Baseline linear/não linear por margem e kernels.

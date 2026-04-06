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

- **LBP**: descreve textura local e uniformidade de padrões.
- **Sobel**: enfatiza gradientes e bordas, útil para transições artificiais.
- **Laplacian**: destaca componentes de alta frequência e nitidez.
- **Entropia global**: mede complexidade/dispersão de informação no frame.

#### Grupo B - estrutura

Foca na coerência geométrica e na organização espacial de detalhes.

- **SIFT**: keypoints e descritores para estabilidade estrutural.
- **Autocorrelação**: periodicidade e repetição de padrões espaciais.
- **Patch similarity (self-similarity)**: redundância local que pode indicar síntese.

#### Grupo C - ruído

Avalia assinaturas de ruído natural vs. ruído residual sintético.

- **Residual noise**: componente de ruído após remoção de conteúdo estrutural.
- **Energia do ruído**: intensidade total do ruído residual.
- **Variância do ruído**: dispersão temporal/espacial do ruído.

#### Grupo D - frequência (FFT)

Observa distribuição espectral e simetrias no domínio da frequência.

- **Energia central (DC)**: concentração de energia em baixas frequências.
- **Energia nos eixos (cruz)**: distribuição em componentes horizontais/verticais.
- **Simetria horizontal/vertical**: padrões especulares no espectro.
- **Anisotropia**: direção preferencial da energia espectral.
- **Entropia da FFT**: complexidade espectral global.

#### Grupo E - física

Procura inconsistências com o comportamento óptico esperado no mundo real.

- **Iluminação**: coerência de sombras e distribuição de luz facial.
- **Reflexos oculares**: consistência de highlights e reflexões nos olhos.

#### Grupo F - robustez

Testa estabilidade dos sinais sob perturbações controladas.

- **Rotação**: sensibilidade dos descritores a mudanças de orientação.
- **Delta estrutural (FFT/gradiente)**: variação estrutural após transformações.

### 2) Métodos com modelos de difusão e aprendizado

#### Modelos de difusão

- **Stable Diffusion pré-treinado**: usado como referência de padrões gerativos e para análise de comportamento de síntese.

#### Temporal

- **CNN**: extração de padrões espaciais por frame.
- **Long Short-Term Memory (LSTM)**: modelagem de dependências temporais e inconsistências ao longo do vídeo.

#### Transfer Learning

- **XceptionNet**: backbone eficiente para sinais de manipulação facial.
- **VGG16**: baseline clássico para comparação de representações visuais.

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
│   ├── raw/                 # dados brutos
│   └── extracted/           # dados processados, metadados e frames extraídos
├── experimentos/
│   ├── grupo_a/             # notebooks e documentação do Grupo A (textura)
│   └── grupo_b/             # notebooks e documentação do Grupo B (estrutura)
├── models/                  # modelos treinados, checkpoints e artefatos
├── output_exemples/         # exemplos de saídas e resultados
├── src/
│   └── create_metadata.py   # scripts utilitários para arquitetura do fluxo
├── requirements.txt         # dependências do projeto
└── README.md                # este documento
```

## Formato dos arquivos de vídeo

Atualmente, o projeto trabalha com vídeos já convertidos para uma representação de frames e metadados auxiliares.

- `*.frames.npy`: tensor com os frames do vídeo, normalmente em `uint8`, no formato `(num_frames, height, width, 3)`.
- `*.num_frames.txt`: quantidade total de frames do vídeo.
- `*.height.txt`: altura de cada frame.
- `*.width.txt`: largura de cada frame.
- `metadata.json`: índice com informações do dataset (id do vídeo, rótulo e caminhos para os arquivos acima).

Essa organização facilita leitura rápida dos dados nos notebooks e padroniza a extração de sinais espaciais, espectrais e temporais.

## Dataset utilizado

O dataset de referência utilizado neste projeto é o **DigiFakeAV e Deepfake-Eval-2024:**

Para mais informações sobre o mesmo, acessar o seu repositório:

- Hugging Face:[ DigiFakeAV](https://huggingface.co/datasets/cambrain/DigiFakeAV)
- Hugging Face: [Deepfake-Eval-2024](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024/tree/main)

## Como navegar (primeiro acesso)

1. Comece por este `README.md` para entender os objetivos e os grupos metodológicos.
2. Acesse `experimentos/`para consultar os notebooks de experimentação
3. Verifique `data/extracted/metadata.json` para mapeamento dos dados processados.
4. Use `src/create_metadata.py` para rotinas auxiliares de organização de metadados.

## Ambiente e execução

### Requisitos

- Python 3.10+ (recomendado)
- Dependências em `requirements.txt`

### Instalação

```bash
pip install -r requirements.txt
```

### Fluxo sugerido

1. Preparar dados em `data/raw/`.
2. Extrair/processar para `data/extracted/`.
3. Processar/utilizar para `data/processed/`
4. Rodar experimentos por grupo em `experimentos/`.
5. Consolidar métricas e comparar abordagens.
6. Evoluir para ensemble híbrido final.

## Status do projeto

Projeto em evolução incremental por grupos de sinais. A base atual prioriza:

- organização dos experimentos em blocos metodológicos;
- validação de sinais interpretáveis;
- preparação para integração com modelos temporais e transfer learning.

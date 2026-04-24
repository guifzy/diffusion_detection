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
- **Autocorrelação**: periodicidade e repetição de padrões espaciais.

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
│   ├── extracted/           # metadados consolidados do dataset (ex.: metadata.json)
│   ├── metadata/            # metadados por vídeo/região e CSV com links/labels
│   └── videos/              # vídeos de entrada utilizados no processamento
├── experimentos/
│   ├── grupo_a/             # notebooks e documentação do Grupo A (textura)
│   ├── grupo_b/             # notebooks e documentação do Grupo B (estrutura)
│   ├── grupo_c/             # notebooks e documentação do Grupo C
│   ├── grupo_d/             # notebooks do Grupo D (frequência/FFT)
│   └── pre_processamento/   # notebooks de pré-processamento
├── output_exemples/         # exemplos de saídas e resultados
├── src/
│   ├── create_metadata.py   # scripts utilitários para arquitetura do fluxo
│   └── pre_processing.py    # pré-processamento e extração de regiões dos vídeos
├── requirements.txt         # dependências do projeto
└── README.md                # este documento
```

## Pré-processamento e extração de regiões

O pipeline inclui o script `src/pre_processing.py` para preparar os vídeos e extrair regiões de interesse por frame:

- **face**: região facial principal;
- **contorno**: borda/periferia da face para análise de transições;
- **fundo**: área não facial para comparação de padrões com o foreground.

> Futuramente serão utilizadas regiões mais descriminativas como olhos, boca, cabelo, tecido...

Essas regiões são usadas para análise espacial, espectral e temporal de forma separada e comparativa.

## Metadados e organização dos arquivos

Os metadados do projeto são organizados em dois formatos complementares:

- **CSV**: tabela consolidada com campos como `label`, `nome` e `link` dos vídeos, facilitando filtragem, auditoria e integração com os experimentos.
- **JSON**: metadados detalhados por vídeo e por região extraída (face, contorno e fundo), preservando estrutura hierárquica e atributos adicionais.

## Formato dos arquivos de vídeo

Atualmente, o projeto utiliza vídeos em `data/videos/` e gera metadados auxiliares em `data/metadata/` e `data/extracted/`.

- `video-metadata-publish-with-links.csv`: tabela com `label`, `nome` e `link` dos vídeos.
- `*_meta.json`: metadados por vídeo com informações da extração de regiões (face, contorno e fundo).
- `metadata.json`: índice consolidado do dataset para consumo nos experimentos.

Essa organização facilita leitura rápida dos dados nos notebooks e padroniza a extração de sinais espaciais, espectrais e temporais.

## Resultados e métricas

Os resultados são salvos em dois níveis:

- **Frame level**: métricas por frame armazenadas em formato de **DataFrame** para análise fina ao longo do tempo.
- **Video level (final)**: métricas agregadas com variação temporal do vídeo inteiro.

As métricas finais de variação temporal em nível de vídeo são as utilizadas como referência principal para comparação entre métodos e tomada de decisão no ensemble. Elas serão em granularidades de méida, desvio padrão e delta. Sendo calculado a diferença entre regiões por vídeo.

Utilização futura Multibranch CNN com Transformers temporal.

Modelo que combina vetores latentes com sinais frame level para Transformers temporal também pode ser uma opção.

## Dataset utilizado

O dataset de referência utilizado neste projeto é o **DigiFakeAV e Deepfake-Eval-2024:**

Para mais informações sobre o mesmo, acessar o seu repositório:

- Hugging Face:[ DigiFakeAV](https://huggingface.co/datasets/cambrain/DigiFakeAV)
- Hugging Face: [Deepfake-Eval-2024](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024/tree/main)

## Como navegar (primeiro acesso)

1. Comece por este `README.md` para entender os objetivos e os grupos metodológicos.
2. Acesse `experimentos/` para consultar os notebooks de experimentação.
3. Verifique `data/extracted/metadata.json` para mapeamento dos dados processados.
4. Use `src/pre_processing.py` para pré-processamento e geração de metadados por região.
5. Use `src/create_metadata.py` para rotinas auxiliares de organização/consolidação de metadados.

## Ambiente e execução

### Requisitos

- Python 3.10+ (recomendado)
- Dependências em `requirements.txt`

### Instalação

```bash
pip install -r requirements.txt
```

### Fluxo sugerido

1. Organizar os vídeos de entrada em `data/videos/`.
2. Executar o pré-processamento e a extração de regiões com `src/pre_processing.py`.
3. Salvar metadados por vídeo/região em `data/metadata/` e consolidar índices em `data/extracted/`.
4. Rodar experimentos por grupo em `experimentos/`.
5. Consolidar métricas frame-level e video-level para comparação entre abordagens.
6. Evoluir para ensemble híbrido final.

## Status do projeto

Projeto em evolução incremental, com base funcional de extração e análise por grupos.

Estado atual observado no repositório:

- **Estrutura de experimentos ativa**: 5 notebooks em `experimentos/` (grupos A, B, C, D e pré-processamento).
- **Documentação técnica por grupo**:
	- grupos A, B, C e D com `README.md` detalhando escopo, métricas e limitações;
- **Pipeline de pré-processamento disponível**:
	- script em `src/pre_processing.py` para detecção facial, extração de regiões (face/contorno/fundo) e geração de metadados por vídeo.
- **Base de dados local populada**:
	- `data/videos/` com 2042 arquivos;
	- `data/metadata/` com 11 arquivos para teste `*_meta.json`;
	- `data/extracted/metadata.json` consolidado.

Principais pendências para as próximas etapas:

- consolidar agregações **video-level** de forma padronizada entre todos os grupos;
- integrar A/B/C/D em um fluxo único de ensemble com avaliação comparativa;
- revisar `src/create_metadata.py`, que atualmente contém caminhos absolutos de ambiente Windows.

## Referências

- **Textura, frequência e inconsistência física**:
	[Deepfake forensics: a survey of digital forensic methods for multimodal deepfake identification on social media](https://www.researchgate.net/publication/399736959_DeepFake_Detection_Through_Deep_Learning_A_Comprehensive_Review)

- **Textura (LBP-like), sinais espaciais e features estatísticas**:
	[Deepfake detection: Enhancing performance with spatiotemporal texture and deep learning feature fusion](https://www.sciencedirect.com/science/article/pii/S1110866524000987)

- **FFT (domínio da frequência)**:
	[M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection](https://arxiv.org/abs/2104.09770)

- **FFT (domínio de frequência) + integração multimodal**:
	[Cross-modal deepfake detection: integrating textual and frequency domains](https://www.researchgate.net/publication/403705865_Cross-modal_deepfake_detection_integrating_textual_and_frequency_domains)

- **Pesquisador forense com vdídeos e conteúdos de referência**:
	[Willard S. Ribeiro, PhD](https://www.instagram.com/willardsribeiro.ia/)
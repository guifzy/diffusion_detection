# Grupo A - Textura (LBP, Sobel e Laplacian)

## 1. Contexto forense

Este grupo busca sinais de inconsistencias de textura em videos sinteticos, com foco em diferencas entre tres regioes do frame:

- face (regiao principal de interesse);
- contorno (transicao entre face e resto da imagem);
- fundo (contexto nao facial).

Em IA forense, artefatos de geracao costumam aparecer como suavizacao excessiva, bordas incoerentes ou distribuicoes de alta frequencia pouco naturais.

## 2. Escopo atual do notebook

Implementacao atual em `testes_a.ipynb`:

- leitura do CSV de metadados e construcao de `video_path`;
- carregamento de `*_meta.json` por video;
- extracao das regioes face/contorno/fundo por frame;
- calculo das metricas de textura com LBP, Sobel e Laplacian;
- consolidacao por frame via `all_metrics(video_path, max_frames=500, label=None)`.

## 3. Metricas e motivacao

### 3.1 LBP (Local Binary Patterns)

O que mede:

- padrao local de microtextura;
- complexidade (entropia), uniformidade e esparsidade;
- distancia de histograma entre regioes.

Por que usar:

- geradores de video tendem a homogenizar pele e perder variacoes finas naturais;
- diferencas entre face e fundo podem revelar blending artificial.

Contribuicao para IA forense:

- identifica textura facial artificialmente regular;
- aumenta o poder de separacao entre real e fake quando combinado com metricas de gradiente.

### 3.2 Sobel

O que mede:

- magnitude e orientacao de bordas;
- entropia do gradiente e coerencia direcional;
- diferencas de energia e coerencia entre regioes.

Por que usar:

- composicao sintetica pode criar transicoes de borda pouco fisicas;
- direcao de bordas em regioes faciais pode ficar instavel em deepfake.

Contribuicao para IA forense:

- detecta incoerencia estrutural de contornos e transicoes;
- complementa LBP ao focar em geometria local das bordas.

### 3.3 Laplacian

O que mede:

- resposta de alta frequencia (energia);
- formato da distribuicao (kurtosis);
- diferencas entre regioes em alta frequencia.

Por que usar:

- modelos generativos podem suavizar ou exagerar detalhes finos;
- alteracoes em alta frequencia sao comuns em videos sinteticos comprimidos.

Contribuicao para IA forense:

- capta assinatura de nitidez artificial e ruido estrutural;
- fortalece a deteccao de artefatos que nao aparecem em metrica de baixa frequencia.

## 4. Estrutura do DataFrame por frame

Colunas de identificacao:

- `video_name`
- `label` (opcional)
- `frame`

Principais colunas LBP:

- `lbp_face_entropy`, `lbp_face_uniformity`, `lbp_border_entropy`, `lbp_bg_entropy`
- `face_bg_entropy_diff`, `face_bg_uniformity_diff`, `face_bg_sparsity_diff`, `face_bg_hist_dist`
- `face_border_entropy_diff`, `face_border_uniformity_diff`, `face_border_sparsity_diff`, `face_border_hist_dist`
- `border_bg_entropy_diff`, `border_bg_uniformity_diff`, `border_bg_sparsity_diff`, `border_bg_hist_dist`

Principais colunas Sobel:

- `sobel_face_entropy`, `sobel_face_coherence`
- `face_bg_coherence_diff`, `face_bg_energy_diff`
- `face_border_coherence_diff`, `face_border_energy_diff`
- `border_bg_coherence_diff`, `border_bg_energy_diff`

Principais colunas Laplacian:

- `lap_face_energy`, `lap_face_kurtosis`
- `face_bg_kurtosis_diff`, `face_border_kurtosis_diff`, `border_bg_kurtosis_diff`

## 5. Como este grupo contribui no ensemble forense

- fornece sinais explicaveis de textura e borda por frame;
- ajuda a priorizar evidencias em regioes faciais vs. contexto;
- gera features que podem ser agregadas temporalmente para decisao final em nivel de video.

## 6. Limitacoes atuais

- dependente da qualidade do bbox facial nos metadados;
- videos com forte compressao podem reduzir confiabilidade de alta frequencia;
- ainda nao faz agregacao temporal final dentro do notebook (saida principal e frame-level).

## 7. Proximos passos

- agregar features em nivel de video com estatisticas temporais;
- calibrar thresholds por tipo de conteudo e compressao;
- integrar com grupos B/C para classificacao forense multimodal.
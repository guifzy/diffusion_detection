# Grupo A - Textura (LBP, Sobel e Laplacian)

## 1. Contexto forense

Este grupo busca sinais de inconsistencias de textura em videos sinteticos, com foco em diferencas entre tres regioes do frame:

- face;
- border, isto e, a transicao ao redor da face;
- background, isto e, o contexto nao facial.

Em IA forense, artefatos de geracao costumam aparecer como suavizacao excessiva, bordas incoerentes, perda de microtextura ou distribuicoes de alta frequencia pouco naturais. A comparacao entre face, borda e fundo ajuda a detectar quando a face tem estatisticas visuais diferentes do contexto.

## 2. Estado atual do notebook

Implementacao atual em `testes_a.ipynb`:

- leitura do CSV de metadados e construcao de `video_path`;
- filtragem dos videos que possuem `*_meta.json` disponivel;
- amostragem robusta de frames com preservacao do indice real do frame;
- mapeamento `frame_idx -> metadata_idx`, evitando usar bbox de outro frame quando ha `np.linspace`;
- extracao das regioes `face`, `border` e `background` por mascara espacial;
- calculo das metricas de textura com LBP;
- calculo das metricas de borda e gradiente com Sobel;
- calculo das metricas de alta frequencia com Laplacian;
- diferencas absolutas entre pares de regioes;
- agregacao frame-level para video-level;
- relatorio discriminativo com AUC e Cohen's d.

## 3. Mudancas aplicadas

As principais correcoes feitas no notebook foram:

- corrigido o desalinhamento entre frame amostrado e metadata;
- adicionadas funcoes auxiliares para `sample_frame_indices`, `metadata_index_for_frame`, `scale_bbox` e `clip_bbox`;
- corrigido o fluxo para LBP, Sobel e Laplacian receberem bbox original e escalarem internamente;
- corrigida a colisao de nomes entre LBP, Sobel e Laplacian, adicionando prefixos claros (`lbp`, `sobel`, `lap`);
- normalizada a entropia de LBP e Sobel para facilitar comparacao;
- adicionada protecao para distancia cosseno de histogramas vazios;
- Sobel passou a usar coerencia angular ponderada pela magnitude do gradiente;
- Sobel ganhou metricas diretas de `std` e `p95` da magnitude;
- Laplacian passou a preservar energia absoluta em vez de calcular energia apenas depois de normalizacao por desvio padrao;
- Laplacian ganhou `variance`, `mean_abs`, `p95_abs` e `entropy_norm`;
- adicionada protecao para `kurtosis` instavel quando a variancia e quase zero;
- a ultima celula deixou de sobrescrever `df` com metricas de um unico video;
- adicionada avaliacao `evaluate_group_a(max_frames=120)`.

## 4. Metricas retornadas

### 4.1 LBP

Para cada regiao (`face`, `border`, `background`), o notebook retorna:

- `entropy_norm`: entropia normalizada do histograma LBP;
- `uniformity`: concentracao do histograma;
- `sparsity`: proporcao de bins relevantes no histograma.

As colunas seguem o formato:

```text
lbp_{region}_{metric}
```

As diferencas regionais seguem:

```text
{prefix}_lbp_{metric}_diff
{prefix}_lbp_hist_dist
```

Pares usados:

- `face_bg`;
- `face_border`;
- `border_bg`.

### 4.2 Sobel

Para cada regiao, o notebook retorna:

- `entropy_norm`: entropia angular normalizada;
- `coherence`: coerencia direcional ponderada pela magnitude do gradiente;
- `energy`: magnitude media do gradiente;
- `std`: desvio padrao da magnitude;
- `p95`: percentil 95 da magnitude.

As colunas seguem o formato:

```text
sobel_{region}_{metric}
```

As diferencas regionais seguem:

```text
{prefix}_sobel_{metric}_diff
{prefix}_sobel_hist_dist
```

### 4.3 Laplacian

Para cada regiao, o notebook retorna:

- `energy`: energia media absoluta de alta frequencia;
- `variance`: variancia da resposta Laplacian;
- `mean_abs`: media do valor absoluto;
- `p95_abs`: percentil 95 do valor absoluto;
- `entropy_norm`: entropia normalizada do histograma da resposta padronizada;
- `kurtosis`: curtose com protecao para variancia quase zero;
- `sparsity`: proporcao de resposta padronizada acima de 1 desvio.

As colunas seguem o formato:

```text
lap_{region}_{metric}
```

As diferencas regionais seguem:

```text
{prefix}_lap_{metric}_diff
{prefix}_lap_hist_dist
```

## 5. Estrutura dos DataFrames

### 5.1 Frame-level

`all_metrics(video_path, max_frames=500, label=None)` retorna uma linha por frame amostrado.

Colunas de identificacao:

- `video_name`;
- `label`, quando informado;
- `frame`, indice real do frame no video;
- `metadata_idx`, indice do metadado usado para a bbox.

Colunas de sinais:

- `lbp_face_*`;
- `lbp_border_*`;
- `lbp_background_*`;
- `face_bg_lbp_*`;
- `face_border_lbp_*`;
- `border_bg_lbp_*`;
- `sobel_face_*`;
- `sobel_border_*`;
- `sobel_background_*`;
- `face_bg_sobel_*`;
- `face_border_sobel_*`;
- `border_bg_sobel_*`;
- `lap_face_*`;
- `lap_border_*`;
- `lap_background_*`;
- `face_bg_lap_*`;
- `face_border_lap_*`;
- `border_bg_lap_*`.

### 5.2 Video-level

`evaluate_group_a(max_frames=120)` agrega os sinais por video e retorna:

- `video_metrics`: uma linha por video avaliado;
- `report`: ranking de metricas com:
  - `metric`;
  - `scope`;
  - `context_sensitive`;
  - `direction`;
  - `auc_fake`;
  - `auc_abs`;
  - `auc_abs_ci_low`;
  - `auc_abs_ci_high`;
  - `auc_abs_bootstrap_std`;
  - `cohen_d_fake_minus_real`;
  - `n_fake`;
  - `n_real`;
  - `missing_rate`;
  - `real_mean`;
  - `fake_mean`.

O `auc_fake` usa `Fake` como classe positiva. O `auc_abs` mostra a forca discriminativa independentemente da direcao do sinal.

Atualizacao metodologica: o relatorio agora tambem registra a direcao do sinal (`higher_fake` ou `higher_real`), o escopo regional da metrica e uma flag `context_sensitive` para sinais que usam fundo ou diferencas com fundo. O intervalo `auc_abs_ci_low`/`auc_abs_ci_high` e estimado por bootstrap por classe e serve como alerta de estabilidade em amostra pequena, nao como validacao final.

## 6. Checagem atual de coerencia

Rodada verificada com `max_frames=120`:

- videos no CSV: `2042`;
- videos com metadata disponivel para este grupo: `11`;
- videos avaliados: `6 Fake` e `5 Real`;
- `video_metrics`: `11 x 300`;
- `report`: `99 x 15`;
- principais metricas sem valores nulos.

Top sinais observados na rodada:

```text
border_bg_sobel_std_diff_mean             auc_abs=0.9667  d=-1.4994
border_bg_sobel_p95_diff_mean             auc_abs=0.8667  d=-1.2243
face_bg_sobel_std_diff_mean               auc_abs=0.8333  d=-1.2142
face_bg_lap_hist_dist_mean                auc_abs=0.8333  d=-1.1866
face_bg_sobel_entropy_norm_diff_mean      auc_abs=0.8333  d=-1.1835
border_bg_lap_hist_dist_mean              auc_abs=0.8333  d=-1.1778
face_border_lbp_hist_dist_mean            auc_abs=0.8333  d=-1.0514
```

Interpretacao:

- Os melhores sinais estao concentrados em diferencas regionais, especialmente `border_bg` e `face_bg`.
- Sobel aparece como o sinal mais forte nesta amostra, indicando que diferencas de magnitude/variacao de borda entre regioes sao relevantes.
- Laplacian e LBP tambem trazem sinais coerentes, principalmente distancias de histograma entre regioes.
- O Grupo A e forte nesta amostra, mas ainda deve ser validado em mais videos antes de ser tratado como desempenho final.

## 7. Limitacoes

- A avaliacao atual usa apenas os videos que possuem `*_meta.json` compativel: `11` videos no total.
- AUC alto em amostra pequena nao deve ser interpretado como resultado final.
- LBP, Sobel e Laplacian sao sensiveis a compressao, blur, resolucao, iluminacao e qualidade da bbox.
- Diferencas muito fortes em `border_bg` podem refletir enquadramento e fundo, entao devem ser combinadas com sinais de face e outros grupos.
- A proxima etapa recomendada e gerar metadata para mais videos e testar estabilidade dos sinais por codec, resolucao e tipo de conteudo.

## 8. Contribuicao no ensemble forense

O Grupo A adiciona sinais explicaveis de textura, borda e alta frequencia:

- captura microtextura por LBP;
- mede coerencia e energia de bordas por Sobel;
- mede detalhe fino e nitidez por Laplacian;
- compara face, borda e fundo para detectar incoerencia regional;
- fornece features frame-level e video-level para agregacao temporal ou metamodelo final.

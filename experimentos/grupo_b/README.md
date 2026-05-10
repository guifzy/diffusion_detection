# Grupo B - Estrutura (SIFT e Patch Similarity)

## 1. Contexto forense

Este grupo analisa consistencia estrutural do conteudo facial, comparando face, contorno e fundo para encontrar sinais de sintese.

Em IA forense, videos gerados podem manter aparencia global convincente, mas falhar em estabilidade de pontos de interesse, distribuicao de descritores locais e repeticao estrutural de patches. A hipotese principal e que face, borda e fundo devem ter uma relacao estrutural minimamente coerente em videos reais.

## 2. Estado atual do notebook

Implementacao atual em `testes_b.ipynb`:

- leitura do CSV de metadados e construcao de `video_path`;
- filtragem dos videos que possuem `*_meta.json` disponivel;
- amostragem robusta de frames com preservacao do indice real do frame;
- mapeamento `frame_idx -> metadata_idx`, evitando usar bbox de outro frame quando ha `np.linspace`;
- extracao das regioes `face`, `border` e `background` por mascara espacial;
- calculo de metricas estruturais com SIFT;
- calculo de Patch Similarity por regiao, agora respeitando a mascara;
- diferencas absolutas entre pares de regioes;
- agregacao frame-level para video-level;
- relatorio discriminativo com AUC e Cohen's d.

## 3. Mudancas aplicadas

As principais correcoes feitas no notebook foram:

- corrigido o bug em `all_metrics`, que tentava desempacotar `compute_sift_metrics` como tupla;
- corrigido o desalinhamento entre frame amostrado e metadata;
- adicionadas funcoes auxiliares para `sample_frame_indices`, `metadata_index_for_frame`, `scale_bbox` e `clip_bbox`;
- corrigido o fluxo para SIFT e patch receberem bbox original e escalarem internamente;
- corrigida a entropia de descritores SIFT, removendo o uso de histograma `density=True` como se fosse probabilidade;
- adicionadas metricas SIFT diretas para `face`, `border` e `background`, nao apenas face;
- adicionadas metricas de resposta SIFT (`response_mean`, `response_std`) e contagem de keypoints;
- limitada a comparacao de descritores para evitar custo quadratico excessivo;
- corrigido `extract_patches` para respeitar a mascara da regiao;
- normalizados patches antes da similaridade para reduzir efeito de brilho absoluto;
- limitado o numero de patches por regiao para manter o custo controlado;
- integrada Patch Similarity ao `all_metrics`;
- adicionada avaliacao `evaluate_group_b(max_frames=40, include_patch=True)`.

## 4. Metricas retornadas

### 4.1 Features SIFT por regiao

Para cada regiao (`face`, `border`, `background`), o notebook retorna:

- `kp_count`: numero de keypoints detectados;
- `kp_density`: densidade de keypoints por pixel da mascara;
- `desc_entropy_norm`: entropia normalizada dos descritores;
- `desc_self_similarity`: similaridade media interna entre descritores;
- `desc_mean`: media dos valores dos descritores;
- `desc_std`: desvio padrao dos descritores;
- `response_mean`: resposta media dos keypoints;
- `response_std`: variacao da resposta dos keypoints;
- `has_kp`: indicador de presenca de keypoints.

As colunas seguem o formato:

```text
sift_{region}_{metric}
```

Exemplos:

```text
sift_face_kp_count
sift_background_kp_density
sift_border_response_mean
```

### 4.2 Diferencas SIFT entre regioes

As diferencas absolutas comparam pares de regioes:

- `face_bg`;
- `face_border`;
- `border_bg`.

As colunas seguem o formato:

```text
{prefix}_sift_{metric}_diff
```

A distancia media entre descritores usa:

```text
{prefix}_sift_desc_dist
```

### 4.3 Patch Similarity

Para cada regiao, o notebook extrai patches locais dentro da mascara e calcula:

- `sim_mean`: similaridade media entre patches;
- `sim_std`: variacao da similaridade;
- `sim_median`: mediana da similaridade;
- `sim_p95`: percentil 95 da similaridade;
- `patch_count`: quantidade de patches validos.

As colunas seguem o formato:

```text
patch_{region}_{metric}
```

As diferencas regionais seguem:

```text
{prefix}_patch_{metric}_diff
```

Observacao: `patch_count` pode capturar area util da mascara, blur e textura disponivel. Ele e util como sinal auxiliar, mas deve ser interpretado com cautela para nao virar proxy de enquadramento ou tamanho da face.

## 5. Estrutura dos DataFrames

### 5.1 Frame-level

`all_metrics(video_path, max_frames=500, label=None, include_patch=True)` retorna uma linha por frame amostrado.

Colunas de identificacao:

- `video_name`;
- `label`, quando informado;
- `frame`, indice real do frame no video;
- `metadata_idx`, indice do metadado usado para a bbox.

Colunas de sinais:

- `sift_face_*`;
- `sift_border_*`;
- `sift_background_*`;
- `face_bg_sift_*`;
- `face_border_sift_*`;
- `border_bg_sift_*`;
- `patch_face_*`;
- `patch_border_*`;
- `patch_background_*`;
- `face_bg_patch_*`;
- `face_border_patch_*`;
- `border_bg_patch_*`.

### 5.2 Video-level

`evaluate_group_b(max_frames=40, include_patch=True)` agrega os sinais por video e retorna:

- `video_metrics`: uma linha por video avaliado;
- `report`: ranking de metricas com:
  - `metric`;
  - `auc_fake`;
  - `auc_abs`;
  - `cohen_d_fake_minus_real`;
  - `real_mean`;
  - `fake_mean`.

O `auc_fake` usa `Fake` como classe positiva. O `auc_abs` mostra a forca discriminativa independentemente da direcao do sinal.

## 6. Checagem atual de coerencia

Rodada verificada com `max_frames=40` e `include_patch=True`:

- videos no CSV: `2042`;
- videos com metadata disponivel para este grupo: `11`;
- videos avaliados: `6 Fake` e `5 Real`;
- `video_metrics`: `11 x 264`;
- `report`: `87 x 6`;
- principais metricas sem valores nulos.

Top sinais observados na rodada:

```text
sift_background_kp_density_mean             auc_abs=0.8667  d=-1.4906
border_bg_sift_kp_count_diff_mean           auc_abs=0.8667  d=-1.3883
face_bg_sift_response_mean_diff_mean        auc_abs=0.8667  d=-1.2366
face_bg_sift_kp_count_diff_mean             auc_abs=0.8333  d=-1.3791
sift_background_kp_count_mean               auc_abs=0.8333  d=-1.2980
sift_background_desc_entropy_norm_mean      auc_abs=0.8333  d=-1.2305
sift_face_response_std_mean                 auc_abs=0.8333  d= 1.2287
```

Interpretacao:

- Os sinais mais fortes estao ligados a densidade/contagem de keypoints e resposta SIFT no fundo versus face/borda.
- Isso e coerente com a hipotese de que videos sinteticos podem alterar a distribuicao estrutural local e a relacao entre rosto e contexto.
- Patch Similarity apareceu como sinal complementar, principalmente em `patch_count`, mas deve ser usado com cautela por poder refletir geometria da mascara e nao apenas repeticao estrutural.

## 7. Limitacoes

- A avaliacao atual usa apenas os videos que possuem `*_meta.json` compativel: `11` videos no total.
- AUC em amostra pequena nao deve ser interpretado como desempenho final.
- SIFT e sensivel a blur, compressao, escala, iluminacao e qualidade da bbox.
- Patch Similarity pode ficar caro em muitos frames; por isso ha limite de patches por regiao.
- A rodada documentada usa `max_frames=40` por custo computacional. Para resultado mais estavel, aumentar `max_frames` ou salvar/cachear features frame-level.

## 8. Contribuicao no ensemble forense

O Grupo B adiciona evidencia estrutural complementar aos grupos de textura, ruido e frequencia:

- mede distribuicao de keypoints e descritores;
- compara consistencia estrutural entre face, borda e fundo;
- captura sinais de repeticao local via patches;
- fornece features frame-level e video-level para agregacao temporal ou metamodelo final.

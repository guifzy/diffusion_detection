# Grupo E - Fisica de Iluminacao

## 1. Contexto forense

Este grupo investiga sinais fisicos de iluminacao para diferenciar videos reais de videos sinteticos.

Em IA forense, videos gerados podem preservar textura, estrutura e frequencia de forma convincente, mas ainda apresentar incoerencias opticas: luminancia facial pouco compativel com o fundo, sombras artificiais, highlights inconsistentes, saturacao anomala ou direcao de iluminacao instavel entre face, borda e contexto.

Como os metadados atuais possuem `bbox` e `bbox_expanded`, mas nao possuem landmarks de olhos, a primeira versao do Grupo E foca em iluminacao facial e coerencia regional. Reflexos oculares ficam como proxima etapa dependente de landmarks ou detector especifico de olhos.

## 2. Estado atual do notebook

Implementacao atual em `testes_e.ipynb`:

- leitura do CSV de metadados e construcao de `video_path`;
- filtragem dos videos que possuem `*_meta.json` disponivel;
- amostragem robusta de frames com preservacao do indice real do frame;
- mapeamento `frame_idx -> metadata_idx`, evitando usar bbox de outro frame quando ha `np.linspace`;
- extracao das regioes `face`, `border` e `background` por mascara espacial;
- extracao de sub-regioes faciais aproximadas: esquerda, direita, topo, baixo e quadrantes;
- conversao para Lab e HSV para medir luminancia, saturacao e cor;
- estimativa de campo de iluminacao por Gaussian blur no canal L;
- calculo de gradiente de baixa frequencia da iluminacao;
- diferencas absolutas entre pares de regioes;
- agregacao frame-level para video-level;
- relatorio discriminativo com AUC e Cohen's d.

## 3. Metricas retornadas

### 3.1 Features fisicas por regiao

Para cada regiao (`face`, `border`, `background`), o notebook retorna:

- `l_mean`: media da luminancia no canal L do Lab;
- `l_std`: desvio padrao da luminancia;
- `l_contrast`: contraste relativo de luminancia;
- `l_range`: intervalo robusto de luminancia entre percentis 5 e 95;
- `l_entropy_norm`: entropia normalizada da luminancia;
- `highlight_ratio`: proporcao de pixels muito claros;
- `shadow_ratio`: proporcao de pixels muito escuros;
- `a_mean`, `b_mean`: medias cromaticas no Lab;
- `a_std`, `b_std`: variacao cromatica no Lab;
- `sat_mean`, `sat_std`: media e variacao de saturacao no HSV;
- `value_mean`: brilho medio no HSV;
- `illum_mean`, `illum_std`: estatisticas do campo de iluminacao suavizado;
- `grad_energy`: energia do gradiente de iluminacao;
- `grad_std`: variacao do gradiente de iluminacao;
- `grad_coherence`: coerencia direcional do gradiente;
- `grad_direction`: direcao media do gradiente.

As colunas seguem o formato:

```text
phys_{region}_{metric}
```

Exemplos:

```text
phys_face_l_mean
phys_border_grad_energy
phys_background_sat_mean
```

### 3.2 Assimetria facial aproximada

Sem landmarks, o notebook divide a bbox facial em sub-regioes retangulares e retorna:

- `phys_face_lr_luma_diff`: diferenca normalizada de luminancia entre lado esquerdo e direito da face;
- `phys_face_tb_luma_diff`: diferenca normalizada entre topo e base da face;
- `phys_face_luma_quadrant_imbalance`: desequilibrio de luminancia entre quadrantes.

Essas metricas buscam inconsistencias grosseiras de direcao de luz e sombreamento facial.

### 3.3 Diferencas entre regioes

As diferencas absolutas comparam pares de regioes:

- `face_bg`;
- `face_border`;
- `border_bg`.

As colunas seguem o formato:

```text
{prefix}_phys_{metric}_diff
```

Exemplos:

```text
face_bg_phys_l_contrast_diff
face_border_phys_grad_energy_diff
border_bg_phys_shadow_ratio_diff
```

Essas diferencas sao importantes porque um fake pode ter iluminacao facial plausivel isoladamente, mas incoerente com borda e fundo.

## 4. Estrutura dos DataFrames

### 4.1 Frame-level

`all_metrics(video_path, max_frames=500, label=None)` retorna uma linha por frame amostrado.

Colunas de identificacao:

- `video_name`;
- `label`, quando informado;
- `frame`, indice real do frame no video;
- `metadata_idx`, indice do metadado usado para a bbox.

Colunas de sinais:

- `phys_face_*`;
- `phys_border_*`;
- `phys_background_*`;
- `phys_face_lr_luma_diff`;
- `phys_face_tb_luma_diff`;
- `phys_face_luma_quadrant_imbalance`;
- `face_bg_phys_*`;
- `face_border_phys_*`;
- `border_bg_phys_*`.

### 4.2 Video-level

`evaluate_group_e(max_frames=120)` agrega os sinais por video e retorna:

- `video_metrics`: uma linha por video avaliado;
- `report`: ranking de metricas com:
  - `metric`;
  - `auc_fake`;
  - `auc_abs`;
  - `cohen_d_fake_minus_real`;
  - `real_mean`;
  - `fake_mean`.

O `auc_fake` usa `Fake` como classe positiva. O `auc_abs` mostra a forca discriminativa independentemente da direcao do sinal.

## 5. Checagem atual de coerencia

Rodada verificada com `max_frames=120`:

- videos no CSV: `2042`;
- videos com metadata disponivel para este grupo: `11`;
- videos avaliados: `6 Fake` e `5 Real`;
- `video_metrics`: `11 x 336`;
- `report`: `111 x 6`;
- principais metricas sem valores nulos.

Top sinais observados na rodada:

```text
phys_background_sat_mean_mean              auc_abs=1.0000  d= 2.5546
phys_background_grad_direction_mean        auc_abs=0.9333  d= 2.2907
border_bg_phys_l_contrast_diff_mean        auc_abs=0.9333  d=-1.5194
face_bg_phys_shadow_ratio_diff_mean        auc_abs=0.9333  d=-1.2007
phys_border_a_mean_mean                    auc_abs=0.9000  d= 1.6611
phys_background_a_mean_mean                auc_abs=0.9000  d= 1.6132
border_bg_phys_l_entropy_norm_diff_mean    auc_abs=0.9000  d=-1.6072
```

Interpretacao:

- O Grupo E apresenta sinais fortes nesta amostra, especialmente saturacao/cor do fundo e diferencas de contraste/sombra entre regioes.
- As metricas `border_bg` e `face_bg` sao coerentes com a hipotese de iluminacao fisica regional.
- Sinais como `sat_mean`, `a_mean`, `b_mean` e `value_mean` podem refletir vies de dataset, cenario ou fonte do video. Devem ser tratados como auxiliares ate haver validacao maior.
- As metricas mais proximas da hipotese fisica sao diferencas de luminancia, contraste, sombra, highlight e gradiente de iluminacao entre face, borda e fundo.

## 6. Limitacoes

- A avaliacao atual usa apenas os videos que possuem `*_meta.json` compativel: `11` videos no total.
- AUC alto em amostra pequena nao deve ser interpretado como resultado final.
- Sem landmarks, a divisao facial esquerda/direita/topo/base e aproximada.
- Cor e saturacao podem capturar vies de dataset, iluminacao de cena ou compressao, nao apenas sintese.
- Reflexos oculares ainda nao foram implementados porque os metadados atuais nao incluem landmarks de olhos.
- A proxima etapa recomendada e gerar landmarks faciais e avaliar olhos, highlights especulares e coerencia de iluminacao temporal.

## 7. Contribuicao no ensemble forense

O Grupo E adiciona sinais fisicos e opticos ao ensemble:

- mede coerencia de iluminacao entre face, borda e fundo;
- captura assimetrias grosseiras de luz na face;
- avalia contraste, sombras, highlights e gradiente de iluminacao;
- complementa textura, estrutura, ruido e frequencia com uma camada fisica interpretavel;
- fornece features frame-level e video-level para agregacao temporal ou metamodelo final.

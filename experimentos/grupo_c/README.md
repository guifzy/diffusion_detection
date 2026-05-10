# Grupo C - Ruido Residual

## 1. Contexto forense

Este grupo investiga assinaturas de ruido residual para diferenciar videos reais de videos sinteticos.

Em IA forense, modelos generativos podem produzir padroes de ruido diferentes de sensores reais, especialmente nas transicoes entre face, contorno e fundo. Mesmo quando a imagem parece visualmente convincente, o residuo depois de uma suavizacao pode revelar diferencas de energia, dispersao, caudas estatisticas e coerencia regional.

## 2. Estado atual do notebook

Implementacao atual em `testes_c.ipynb`:

- leitura do CSV de metadados e construcao de `video_path`;
- filtragem dos videos que possuem `*_meta.json` disponivel;
- amostragem robusta de frames com preservacao do indice real do frame;
- mapeamento `frame_idx -> metadata_idx`, evitando usar bbox de outro frame quando ha `np.linspace`;
- extracao das regioes `face`, `border` e `background` por mascara espacial;
- geracao do residual por filtro bilateral (`compute_residual`);
- calculo de metricas robustas de ruido por regiao;
- diferencas absolutas entre pares de regioes;
- agregacao frame-level para video-level;
- relatorio discriminativo com AUC e Cohen's d.

## 3. Mudancas aplicadas

As principais correcoes feitas no notebook foram:

- corrigido o desalinhamento entre frame amostrado e metadata;
- adicionadas funcoes auxiliares para `sample_frame_indices`, `metadata_index_for_frame`, `scale_bbox` e `clip_bbox`;
- corrigido o fluxo para `compute_noise_metrics` sempre receber bbox no frame original e escalar internamente;
- adicionada protecao contra `kurtosis` instavel quando a variancia e muito baixa;
- normalizada a entropia do histograma para ficar mais comparavel entre regioes;
- expandidas as metricas diretas para `face`, `border` e `background`, nao apenas face;
- adicionadas metricas robustas como `mad`, `rms`, `p95_abs` e `skewness`;
- adicionada avaliacao `evaluate_group_c(max_frames=120)` no mesmo formato do Grupo D.

## 4. Metricas retornadas

### 4.1 Features por regiao

Para cada regiao (`face`, `border`, `background`), o notebook retorna:

- `mean`: media do residuo;
- `variance`: variancia do residuo;
- `std`: desvio padrao;
- `energy`: media do valor absoluto do residuo;
- `rms`: raiz da media quadratica;
- `entropy_norm`: entropia normalizada do histograma do residuo;
- `skewness`: assimetria da distribuicao;
- `kurtosis`: cauda/curtose da distribuicao, com protecao para variancia quase zero;
- `mad`: desvio absoluto mediano;
- `p95_abs`: percentil 95 do valor absoluto do residuo.

As colunas seguem o formato:

```text
noise_{region}_{metric}
```

Exemplos:

```text
noise_face_variance
noise_border_energy
noise_background_entropy_norm
```

### 4.2 Diferencas entre regioes

As diferencas absolutas comparam pares de regioes:

- `face_bg`;
- `face_border`;
- `border_bg`.

As colunas seguem o formato:

```text
{prefix}_noise_{metric}_diff
```

Exemplos:

```text
face_border_noise_std_diff
face_bg_noise_mad_diff
border_bg_noise_entropy_norm_diff
```

Essas diferencas sao importantes porque um fake pode ter ruido facial estatisticamente diferente do contorno e do fundo, mesmo quando as metricas absolutas da face parecem normais.

## 5. Estrutura dos DataFrames

### 5.1 Frame-level

`all_metrics(video_path, max_frames=500, label=None)` retorna uma linha por frame amostrado.

Colunas de identificacao:

- `video_name`;
- `label`, quando informado;
- `frame`, indice real do frame no video;
- `metadata_idx`, indice do metadado usado para a bbox.

Colunas de sinais:

- `noise_face_*`;
- `noise_border_*`;
- `noise_background_*`;
- `face_bg_noise_*_diff`;
- `face_border_noise_*_diff`;
- `border_bg_noise_*_diff`.

### 5.2 Video-level

`evaluate_group_c(max_frames=120)` agrega os sinais por video e retorna:

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

Rodada verificada com `max_frames=120`:

- videos no CSV: `2042`;
- videos com metadata disponivel para este grupo: `11`;
- videos avaliados: `6 Fake` e `5 Real`;
- `video_metrics`: `11 x 174`;
- `report`: `57 x 6`;
- principais metricas sem valores nulos.

Top sinais observados na rodada:

```text
face_bg_noise_mad_diff_mean               auc_abs=0.7333  d=-0.4318
border_bg_noise_p95_abs_diff_mean         auc_abs=0.7333  d=-0.1597
face_border_noise_std_diff_mean           auc_abs=0.7000  d= 0.8563
face_border_noise_rms_diff_mean           auc_abs=0.7000  d= 0.8362
face_border_noise_p95_abs_diff_mean       auc_abs=0.7000  d= 0.7187
face_border_noise_variance_diff_mean      auc_abs=0.7000  d= 0.7107
```

Interpretacao:

- Os melhores sinais estao em diferencas regionais, principalmente entre face e contorno ou face e fundo.
- `std`, `rms`, `variance`, `mad` e `p95_abs` sao coerentes para ruido residual, pois medem intensidade/dispersao do residuo.
- A separacao atual e moderada: o Grupo C parece util como sinal complementar, mas nao deve ser usado isoladamente como detector final.

## 7. Limitacoes

- A avaliacao atual usa apenas os videos que possuem `*_meta.json` compativel: `11` videos no total.
- AUC em amostra pequena nao deve ser interpretado como desempenho final.
- Ruido residual e sensivel a compressao, reencodificacao, resolucao, iluminacao e parametros do filtro bilateral.
- O filtro bilateral preserva bordas, mas seus parametros podem favorecer ou esconder certos residuos.
- A proxima etapa recomendada e gerar metadata para mais videos e testar estabilidade dos sinais com diferentes codecs e parametros de residual.

## 8. Contribuicao no ensemble forense

O Grupo C adiciona evidencia estatistica complementar aos grupos de textura, estrutura e frequencia:

- captura diferencas estocasticas pouco visiveis a olho nu;
- mede coerencia de ruido entre face e contexto;
- fornece sinais frame-level e video-level para agregacao temporal;
- aumenta cobertura do ensemble em casos em que textura e estrutura parecem convincentes.

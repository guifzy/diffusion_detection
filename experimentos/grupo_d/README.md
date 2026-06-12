# Grupo D - Frequencia (FFT)

## 1. Contexto forense

Este grupo investiga sinais no dominio da frequencia para diferenciar videos reais de videos sinteticos.

Em IA forense, geradores de video podem preservar boa aparencia no espaco da imagem, mas ainda deixar assinaturas espectrais anormais: distribuicao diferente de baixa/media/alta frequencia, direcionalidade excessiva, perda de coerencia entre face e contexto, ou textura facial espectralmente diferente do contorno e do fundo.

## 2. Estado atual do notebook

Implementacao atual em `testes_d.ipynb`:

- leitura do CSV de metadados e construcao de `video_path`;
- filtragem dos videos que possuem `*_meta.json` disponivel;
- amostragem robusta de frames com preservacao do indice real do frame;
- mapeamento `frame_idx -> metadata_idx`, evitando usar bbox de outro frame quando ha `np.linspace`;
- extracao de crops espaciais reais para `face`, `border` e `background`;
- calculo de FFT por crop, nao por mascara espacial aplicada no espectro global;
- extracao de features espectrais por regiao;
- diferencas absolutas entre pares de regioes;
- agregacao frame-level para video-level;
- relatorio discriminativo com AUC e Cohen's d.

Observacao importante: a versao anterior aplicava mascaras de face/borda/fundo diretamente sobre a magnitude da FFT do frame inteiro. Isso misturava coordenadas espaciais com coordenadas de frequencia. A versao atual corrige esse ponto calculando a FFT separadamente em regioes recortadas no dominio espacial.

## 3. Metricas retornadas

### 3.1 Features por regiao

Para cada regiao (`face`, `border`, `background`), o notebook retorna:

- `low_freq_ratio`: proporcao de energia em baixa frequencia, sem o componente DC imediato;
- `mid_freq_ratio`: proporcao de energia em frequencia media;
- `high_freq_ratio`: proporcao de energia em alta frequencia;
- `entropy_norm`: entropia espectral normalizada, comparavel entre regioes;
- `anisotropy`: concentracao direcional da energia espectral;
- `radial_centroid`: centroide radial normalizado da energia no espectro;
- `flatness`: planicidade espectral, util para medir quao espalhada ou tonal e a energia;
- `mean_intensity`: media de intensidade do crop, usada como sinal auxiliar fotometrico;
- `std_intensity`: desvio padrao de intensidade do crop, tambem auxiliar.

As colunas seguem o formato:

```text
fft_{region}_{metric}
```

Exemplos:

```text
fft_face_high_freq_ratio
fft_border_entropy_norm
fft_background_radial_centroid
```

### 3.2 Diferencas entre regioes

As diferencas absolutas comparam pares de regioes:

- `face_bg`;
- `face_border`;
- `border_bg`.

As colunas seguem o formato:

```text
{prefix}_fft_{metric}_diff
```

Exemplos:

```text
face_border_fft_entropy_norm_diff
face_bg_fft_radial_centroid_diff
border_bg_fft_anisotropy_diff
```

Essas diferencas sao especialmente importantes porque o objetivo forense nao e apenas medir o espectro da face isoladamente, mas verificar se a face mantem coerencia espectral com o contorno e o fundo.

## 4. Estrutura dos DataFrames

### 4.1 Frame-level

`all_metrics(video_path, max_frames=500, label=None)` retorna uma linha por frame amostrado.

Colunas de identificacao:

- `video_name`;
- `label`, quando informado;
- `frame`, indice real do frame no video;
- `metadata_idx`, indice do metadado usado para a bbox.

Colunas de sinais:

- `fft_face_*`;
- `fft_border_*`;
- `fft_background_*`;
- `face_bg_fft_*_diff`;
- `face_border_fft_*_diff`;
- `border_bg_fft_*_diff`.

### 4.2 Video-level

`evaluate_group_d(max_frames=120)` agrega os sinais por video e retorna:

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

## 5. Checagem atual de coerencia

Rodada verificada com `max_frames=120`:

- videos no CSV: `2042`;
- videos com metadata disponivel para este grupo: `11`;
- videos avaliados: `6 Fake` e `5 Real`;
- `video_metrics`: `11 x 156`;
- `report`: `51 x 15`;
- principais metricas sem valores nulos.

Top sinais observados na rodada:

```text
face_border_fft_entropy_norm_diff_mean       auc_abs=0.8667  d=-1.5831
face_border_fft_radial_centroid_diff_mean    auc_abs=0.8667  d=-1.1972
fft_border_flatness_mean                     auc_abs=0.8667  d= 1.0045
face_bg_fft_mid_freq_ratio_diff_mean         auc_abs=0.8667  d=-0.7822
border_bg_fft_std_intensity_diff_mean        auc_abs=0.8333  d=-1.3067
face_border_fft_low_freq_ratio_diff_mean     auc_abs=0.8333  d=-1.1211
```

Interpretacao:

- Os melhores sinais estao concentrados em diferencas entre face/contorno/fundo, o que e coerente com a hipotese forense do grupo.
- `entropy_norm`, `radial_centroid`, `mid_freq_ratio`, `low_freq_ratio` e `flatness` sao sinais espectrais coerentes.
- `mean_intensity` e `std_intensity` devem ser tratados como sinais auxiliares, pois medem fotometria do crop e podem capturar diferencas de iluminacao, compressao ou dataset.

## 6. Limitacoes

- A avaliacao atual usa apenas os videos que possuem `*_meta.json` compativel: `11` videos no total.
- AUC alto em amostra pequena nao deve ser interpretado como resultado final.
- Os sinais de intensidade podem ser discriminativos, mas tambem podem refletir vies de dataset.
- As chamadas visuais de `debug_fft_video` ficam comentadas para manter a execucao de avaliacao limpa e reprodutivel.
- A proxima etapa recomendada e gerar metadata para mais videos e repetir o relatorio em uma amostra maior, idealmente com separacao entre validacao exploratoria e teste.

## 7. Contribuicao no ensemble forense

O Grupo D adiciona sinais espectrais explicaveis ao ensemble hibrido:

- cobre artefatos que podem nao aparecer claramente em textura, estrutura ou ruido;
- mede coerencia entre face e contexto;
- produz features frame-level e video-level compativeis com agregacao temporal;
- oferece metricas interpretaveis para combinacao posterior com os grupos A, B e C.

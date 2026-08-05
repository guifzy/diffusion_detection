# Contrato cientifico dos sinais v0.2.0

Este documento define a fonte de verdade dos atributos extraidos no nivel de
frame pelos grupos A-E. O objetivo e impedir divergencia entre notebooks,
pipeline e documentacao durante a etapa de consolidacao metodologica.

| Campo | Definicao |
| --- | --- |
| Versao do pipeline | `0.2.0` |
| Codigo canonico | `src/shared/features/` |
| Unidade atual de extracao | Frame, com uma face dominante e regioes `face`, `border` e `background` |
| Unidade atual de agregacao | Video, por media, desvio-padrao e mediana dos atributos por frame |
| Fora do escopo atual | Reflexos oculares, temporalidade ordenada e sombra geometrica 3D |

## Principios gerais

Os notebooks experimentais importam os extratores canonicos e nao devem manter
formulas independentes. Qualquer mudanca de formula, nome de coluna ou
interpretacao exige nova versao ou registro explicito de migracao.

A versao `0.2.0` implementa sinais estaticos. As agregacoes por video sao
resumos estatisticos invariantes a ordem dos frames; portanto, nao devem ser
descritas como analise temporal.

## Regioes e padronizacao

Todo frame e limitado a 640 pixels no maior lado antes da extracao. A caixa
facial e escalada pelo mesmo fator. A partir da caixa padronizada, sao
construidas tres regioes:

| Regiao | Definicao operacional | Interpretacao |
| --- | --- | --- |
| `face` | Area interna da caixa facial | Conteudo facial principal |
| `border` | Anel ao redor da face dentro da caixa expandida | Transicao face-contexto |
| `background` | Area externa a caixa expandida | Contexto visual local/global disponivel no frame |

## Contrastes regionais

Para uma metrica escalar \(x\), os contrastes entre as regioes \(A\) e \(B\)
sao definidos como:

\[
d_s = x_A - x_B
\]

\[
d_a = |x_A - x_B|
\]

\[
d_n = \frac{x_A - x_B}{|x_A| + |x_B| + 10^{-6}}
\]

| Sufixo | Formula | Interpretacao |
| --- | --- | --- |
| `signed_diff` | \(d_s\) | Preserva direcao da diferenca |
| `abs_diff` | \(d_a\) | Mede magnitude da discrepancia |
| `norm_diff` | \(d_n\) | Normaliza a diferenca pela escala das regioes |

Distancias entre orientacoes usam distancia circular, e nao subtracao linear de
angulos.

## Familias de sinais

| Grupo | Familia | Papel metodologico | Status |
| --- | --- | --- | --- |
| A | LBP, Sobel e Laplaciano | Textura, bordas, orientacao e altas frequencias espaciais | Consolidado com ressalvas pontuais |
| B | SIFT e similaridade de patches | Estrutura local e repeticao de padroes | Exploratorio controlado |
| C | Residuo bilateral | Baseline de alta passagem | Baseline, sem interpretacao como ruido de sensor |
| D | FFT espacial | Distribuicao de potencia em frequencia | Consolidado para sinais estaticos |
| E | Fotometria e candidatos de sombra | Estatisticas regionais de luz, cor e baixa iluminacao | Proxy fotometrico, nao modelo fisico completo |

---

## Grupo A: textura e derivadas espaciais

### LBP multiescala

O Local Binary Pattern e tratado como descritor categorico de microtextura. Por
isso, a versao `0.2.0` nao calcula media, energia ou curtose diretamente sobre
os codigos LBP.

| Configuracao | Pontos | Raio | Prefixo |
| --- | ---: | ---: | --- |
| Escala 1 | 8 | 1 | `lbp_r1_p8` |
| Escala 2 | 16 | 2 | `lbp_r2_p16` |
| Escala 3 | 24 | 3 | `lbp_r3_p24` |

Para cada regiao, sao extraidos:

| Atributo | Interpretacao |
| --- | --- |
| `entropy_norm` | Entropia normalizada do histograma LBP |
| `uniformity` | Soma dos quadrados das probabilidades do histograma |
| `active_bin_ratio` | Fracao de bins com probabilidade superior a 0,01 |

Entre regioes, tambem e emitida `hist_js_distance`, correspondente a distancia
de Jensen-Shannon entre histogramas.

### Sobel

O operador Sobel estima derivadas de primeira ordem. Para derivadas \(G_x\) e
\(G_y\), a magnitude \(m\) e a orientacao \(\theta\) sao:

\[
m = \sqrt{G_x^2 + G_y^2}
\]

\[
\theta = \operatorname{atan2}(G_y, G_x)
\]

A coerencia de orientacao ponderada pela magnitude e:

\[
C_\theta =
\left|
\frac{\sum_i m_i e^{i\theta_i}}{\sum_i m_i + \varepsilon}
\right|
\]

| Atributo | Interpretacao |
| --- | --- |
| `magnitude_mean`, `magnitude_std`, `magnitude_median`, `magnitude_p95` | Distribuicao da intensidade dos gradientes |
| `gradient_energy` | Energia media \(E[m^2]\) |
| `magnitude_entropy_norm` | Dispersao da magnitude dos gradientes |
| `orientation_entropy_norm` | Dispersao das orientacoes |
| `orientation_coherence` | Concentracao direcional dos gradientes |
| `strong_gradient_ratio` | Fracao de pixels acima do limiar medio global do frame |

O nome antigo `coherence` foi removido porque representava grandeza ambigua.

### Laplaciano

O Laplaciano estima derivadas de segunda ordem e e usado como proxy de nitidez,
alta frequencia e resposta a bordas finas.

\[
L = \nabla^2 I
\]

As estatisticas assinadas sao calculadas sobre \(L\), enquanto as estatisticas
de magnitude sao calculadas sobre \(|L|\).

| Classe | Atributos |
| --- | --- |
| Assinados | `signed_mean`, `signed_std`, `signed_variance`, `signed_energy` |
| Magnitude | `abs_mean`, `abs_median`, `abs_p95` |
| Distribuicao | `entropy_norm`, `kurtosis`, `standardized_tail_ratio` |
| Contraste regional | `hist_js_distance` e contrastes escalares |

Ressalva: `signed_std` e `signed_variance` carregam informacao redundante. A
decisao de manter ambos deve ser revisada no EDA.

---

## Grupo B: estrutura local

### SIFT

O SIFT e mantido como descritor regional estatico. A versao `0.2.0` removeu a
comparacao direta entre descritores de face e fundo, pois ela comparava conteudo
semanticamente distinto sem realizar matching geometrico.

| Atributo | Interpretacao |
| --- | --- |
| `kp_count` | Numero de keypoints na regiao |
| `kp_density` | Keypoints por pixel valido |
| `kp_coverage` | Ocupacao espacial em grade 4x4 |
| `response_mean`, `response_std` | Forca media e dispersao das respostas |
| `size_mean`, `size_std` | Escala media e dispersao dos keypoints |
| `orientation_entropy_norm` | Dispersao das orientacoes dos keypoints |
| `orientation_coherence` | Concentracao direcional das orientacoes |
| `descriptor_entropy_norm` | Entropia dos valores dos descritores |
| `descriptor_self_similarity` | Autossimilaridade entre descritores da mesma regiao |

### Similaridade de patches

O tamanho do patch corresponde a aproximadamente 8% da menor dimensao facial,
limitado ao intervalo de 8 a 32 pixels. O stride e metade do patch. Os
candidatos sao distribuidos deterministicamente pela regiao.

| Atributo | Interpretacao |
| --- | --- |
| `sim_mean` | Similaridade media entre patches normalizados |
| `sim_std` | Dispersao das similaridades |
| `sim_median` | Tendencia central robusta |
| `sim_p95` | Similaridade alta entre pares de patches |

Contagens e cobertura de amostragem sao controles de qualidade. Colunas com
prefixo `qc_` sao persistidas no nivel por frame, mas excluidas da agregacao
destinada a modelagem.

---

## Grupo C: residuo bilateral

O sinal e definido como baseline residual de alta passagem:

\[
R = I - \operatorname{Bilateral}(I)
\]

Ele nao deve ser descrito como ruido de sensor, PRNU ou Noiseprint. O prefixo
oficial das colunas e `residual`.

| Atributo | Interpretacao |
| --- | --- |
| `signed_mean` | Tendencia media do residuo |
| `std`, `rms`, `mad`, `abs_p95` | Dispersao e magnitude robusta do residuo |
| `entropy_norm` | Dispersao histogramica da magnitude residual |
| `kurtosis` | Peso de cauda da distribuicao residual |
| `horizontal_lag1_corr`, `vertical_lag1_corr` | Autocorrelacao espacial de primeira defasagem |
| `channel_corr_mean` | Correlacao media entre canais do residuo colorido |

`variance` foi removida por ser exatamente o quadrado de `std`.

---

## Grupo D: frequencia espacial

Cada patch valido e convertido para 128x128, centralizado e multiplicado por uma
janela Hann bidimensional.

\[
\tilde{I} = (I - \bar{I})w
\]

\[
F = \operatorname{fftshift}\left(\mathcal{F}_2\{\tilde{I}\}\right)
\]

\[
P = |F|^2
\]

Todos os patches validos do contorno e do contexto sao analisados, e suas
metricas sao agregadas por media. A versao atual nao escolhe somente o maior
patch.

| Atributo | Interpretacao |
| --- | --- |
| `mean_log_amplitude`, `std_log_amplitude` | Nivel e dispersao da log-amplitude |
| `low_power_ratio`, `mid_power_ratio`, `high_power_ratio` | Particao da potencia espectral sem o DC imediato |
| `radial_centroid` | Centroide radial ponderado por potencia |
| `spectral_entropy_norm` | Dispersao da potencia no espectro |
| `spectral_flatness` | Proximidade entre espectro tonal e espectro plano |
| `angular_anisotropy` | Concentracao direcional da energia espectral |
| `spectral_slope` | Inclinacao radial em escala log-log |
| `patch_count` | Controle de qualidade da quantidade de patches validos |

As razoes de banda sao calculadas sobre potencia e devem formar uma particao do
espectro analisado. O atributo `mean_intensity` foi removido porque nao
representava propriedade frequencial.

---

## Grupo E: fotometria regional e candidatos de sombra

### Fotometria regional

O prefixo oficial e `photo`. A familia mede proxies fotometricos e nao recupera
um modelo fisico completo de iluminacao.

O campo de iluminacao e estimado por suavizacao Gaussiana do canal L do espaco
Lab, com sigma proporcional a escala facial.

| Classe | Atributos |
| --- | --- |
| Luminancia | `l_mean`, `l_std`, `l_contrast_p90_norm`, `l_iqr`, `l_entropy_norm` |
| Cromaticidade Lab | `a_mean`, `a_std`, `b_mean`, `b_std` |
| HSV | `saturation_mean`, `saturation_std`, `value_mean` |
| Campo suavizado | `illumination_mean`, `illumination_std` |
| Gradiente de iluminacao | `illumination_gradient_energy`, `illumination_gradient_std`, `illumination_gradient_coherence`, `illumination_gradient_direction` |
| Clipping e extremos | `dark_pixel_ratio`, `bright_pixel_ratio`, `black_clip_ratio`, `white_clip_ratio` |
| Assimetria facial | `photo_face_lr_luma_asymmetry`, `photo_face_tb_luma_asymmetry`, `photo_face_quadrant_luma_imbalance` |

`shadow_ratio` e `highlight_ratio` baseados em percentis internos foram
removidos, pois permaneciam proximos de 20% por construcao.

### Candidatos de sombra

A luminancia e calculada apos linearizacao de sRGB:

\[
Y = 0.2126R + 0.7152G + 0.0722B
\]

O campo Retinex e estimado no dominio logaritmico:

\[
\hat{L} = \exp \left(G_\sigma * \log(Y + \varepsilon)\right)
\]

O campo e normalizado pela mediana da caixa expandida. Pixels com razao de
iluminacao inferior a 0,65, excluindo pretos clipados, sao marcados como
candidatos de sombra.

| Atributo | Interpretacao |
| --- | --- |
| `candidate_ratio` | Fracao da regiao marcada como candidata |
| `illumination_ratio_mean`, `illumination_ratio_std`, `illumination_ratio_p10` | Estatisticas da iluminacao normalizada |
| `candidate_depth_mean` | Profundidade media dos candidatos |
| `candidate_to_lit_luminance_ratio` | Razao entre luminancia candidata e area iluminada |
| `candidate_chromaticity_shift` | Mudanca de cromaticidade entre candidatos e area iluminada |
| `boundary_density` | Densidade de fronteira dos candidatos |
| `boundary_gradient_mean` | Gradiente medio na fronteira candidata |

Esses atributos modelam baixa iluminacao compativel com sombra, mas nao
resolvem a ambiguidade entre sombra, material escuro e alteracao de exposicao.

## Itens explicitamente adiados

| Item | Motivo do adiamento |
| --- | --- |
| Reflexos e highlights oculares | Exigem regioes oculares confiaveis e resolucao adequada |
| Separacao fisica entre reflectancia e iluminacao | Exige modelo fisico adicional |
| Sombra geometrica 3D | Exige geometria facial, normais e estimativa de luz |
| Primeiras e segundas diferencas temporais | Exigem `timestamp`, `track_id` e tracks estaveis |
| Autocorrelacao e frequencia temporal | Exigem amostragem temporal padronizada |
| Fluxo optico e estabilidade SIFT entre frames | Exigem compensacao de camera e correspondencias temporais |
| rPPG | Sensivel a FPS, movimento, compressao e iluminacao |

## Regras de evolucao

Qualquer mudanca de formula exige:

1. novo nome de coluna ou nova versao do pipeline;
2. atualizacao deste contrato;
3. teste numerico dedicado;
4. regeneracao das camadas Silver e Gold;
5. proibicao de combinar colunas de versoes diferentes sem migracao explicita.

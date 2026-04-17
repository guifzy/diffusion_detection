# Grupo C - Ruido Residual

## 1. Contexto forense

Este grupo investiga assinaturas de ruido residual para diferenciar videos reais de videos sinteticos.

Em IA forense, modelos generativos podem produzir padroes de ruido diferentes de sensores reais, especialmente nas transicoes entre face, contorno e fundo.

## 2. Escopo atual do notebook

Implementacao atual em `testes_c.ipynb`:

- leitura do CSV de metadados e construcao de `video_path`;
- carregamento de `*_meta.json` por video;
- extracao das regioes face/contorno/fundo por frame;
- geracao do residual via filtro bilateral (`compute_residual`);
- calculo de metricas de ruido por regiao e diferencas inter-regionais;
- consolidacao por frame via `all_metrics(video_path, max_frames=500, label=None)`.

## 3. Metricas e motivacao

### 3.1 Ruido residual por regiao

O que mede:

- variancia do ruido (`noise_face_variance`);
- entropia do ruido (`noise_face_entropy`);
- kurtosis do ruido (usada nas diferencas entre regioes).

Por que usar:

- ruido natural costuma ter distribuicao mais coerente com captura por camera;
- sintese pode gerar ruido mais regular, mais instavel ou mal distribuido;
- a comparacao entre face e contexto ajuda a detectar blending e geracao seletiva.

Contribuicao para IA forense:

- detecta inconsistencias estatisticas pouco visiveis a olho nu;
- complementa metricas de textura e estrutura com sinal estocastico.

### 3.2 Diferencas entre regioes

O que mede:

- diferenca de variancia, entropia e kurtosis entre pares de regioes:
  - `face_bg`;
  - `face_border`;
  - `border_bg`.

Por que usar:

- em videos reais, a transicao entre regioes tende a ser fisicamente mais consistente;
- em videos sinteticos, a regiao facial pode ter processo gerativo diferente do fundo.

Contribuicao para IA forense:

- reforca a deteccao de anomalias de composicao espacial;
- melhora a discriminacao quando agregada temporalmente.

## 4. Estrutura do DataFrame por frame

Colunas de identificacao:

- `video_name`
- `label` (opcional)
- `frame`

Colunas diretas de ruido:

- `noise_face_variance`
- `noise_face_entropy`

Colunas de diferencas entre regioes:

- `face_bg_noise_var_diff`
- `face_bg_noise_entropy_diff`
- `face_bg_noise_kurtosis_diff`
- `face_border_noise_var_diff`
- `face_border_noise_entropy_diff`
- `face_border_noise_kurtosis_diff`
- `border_bg_noise_var_diff`
- `border_bg_noise_entropy_diff`
- `border_bg_noise_kurtosis_diff`

## 5. Como este grupo contribui no ensemble forense

- adiciona evidencia estatistica complementar aos grupos de textura e estrutura;
- aumenta robustez contra casos em que o fake esta visualmente convincente;
- oferece sinais frame-level para consolidacao em metricas temporais por video.

## 6. Limitacoes atuais

- residual depende dos parametros do filtro bilateral;
- ruido pode ser afetado por compressao forte e reencodificacao;
- ainda nao inclui agregacao temporal final no notebook.

## 7. Proximos passos

- validar sensibilidade dos parametros do residual em diferentes codecs;
- incluir estatisticas video-level para decisao final;
- integrar com grupos A e B em pipeline unico de classificacao forense.

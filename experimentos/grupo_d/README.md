# Grupo D - Frequencia (FFT)

## 1. Contexto forense

Este grupo investiga sinais no dominio da frequencia para diferenciar videos reais de videos sinteticos.

Em IA forense, geradores de video podem manter qualidade visual no espaco da imagem, mas ainda deixar assinaturas espectrais anormais, como concentracao de energia, padroes direcionais e inconsistencias entre face, contorno e fundo.

## 2. Escopo atual do notebook

Implementacao atual em `testes_d.ipynb`:

- leitura do CSV de metadados e construcao de `video_path`;
- carregamento de `*_meta.json` por video;
- extracao das regioes face/contorno/fundo por frame;
- calculo de magnitude espectral via FFT (`compute_fft`);
- extracao de features espectrais por regiao (`compute_fft_region`);
- comparacoes entre regioes (`fft_region_differences`);
- visualizacao de depuracao com sobreposicao de regioes e mapa de FFT (`debug_fft_video`).

## 3. Metricas e motivacao

### 3.1 Features FFT por regiao

O que mede:

- razao de energia central (baixa frequencia): `dc_ratio`;
- razao de energia periferica (alta frequencia): `high_freq_ratio`;
- entropia espectral: `entropy`;
- anisotropia espectral (energia por direcao): `anisotropy`;
- simetria espectral horizontal e vertical: `symmetry_h` e `symmetry_v`.

Por que usar:

- videos sinteticos podem concentrar energia de forma diferente do padrao natural de captura;
- artefatos de geracao/compressao podem alterar distribuicao de alta frequencia;
- direcionalidade e simetria espectral ajudam a revelar padroes artificiais.

Contribuicao para IA forense:

- adiciona evidencia complementar aos grupos de textura, estrutura e ruido;
- melhora interpretabilidade com sinais diretamente ligados ao espectro de frequencia.

### 3.2 Diferencas entre regioes

O que mede:

- diferencas absolutas entre pares de regioes para:
  - `*_fft_dc_diff`
  - `*_fft_high_freq_diff`
  - `*_fft_entropy_diff`
  - `*_fft_anisotropy_diff`

Pares usados:

- `face_bg`;
- `face_border`;
- `border_bg`.

Por que usar:

- em videos reais, as transicoes entre regioes tendem a manter maior coerencia fisica;
- em videos sinteticos, a regiao facial pode exibir espectro diferente do contexto.

Contribuicao para IA forense:

- reforca deteccao de anomalias de composicao espacial no dominio da frequencia;
- cria features frame-level candidatas para agregacao temporal video-level.

## 4. Estrutura esperada do DataFrame por frame

Colunas de identificacao:

- `video_name`
- `label` (opcional)
- `frame`

Colunas diretas FFT (face):

- `fft_face_dc`
- `fft_face_high_freq`
- `fft_face_entropy`
- `fft_face_anisotropy`

Colunas de diferencas entre regioes (prefixos `face_bg`, `face_border`, `border_bg`):

- `{prefix}_fft_dc_diff`
- `{prefix}_fft_high_freq_diff`
- `{prefix}_fft_entropy_diff`
- `{prefix}_fft_anisotropy_diff`

## 5. Como este grupo contribui no ensemble forense

- incorpora sinais espectrais explicaveis para complementar o ensemble hibrido;
- melhora cobertura de artefatos que nao aparecem com clareza no dominio espacial;
- fornece base para combinacao com agregacoes temporais e metamodelo final.
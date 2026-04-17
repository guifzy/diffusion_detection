# Grupo B - Estrutura (SIFT e Patch Similarity)

## 1. Contexto forense

Este grupo analisa consistencia estrutural do conteudo facial, comparando face, contorno e fundo para encontrar sinais de sintese.

Em IA forense, videos gerados podem manter aparencia global convincente, mas falhar em estabilidade de pontos de interesse e consistencia dos descritores locais.

## 2. Escopo atual do notebook

Implementacao atual em `testes_b.ipynb`:

- leitura do CSV de metadados e construcao de `video_path`;
- carregamento de `*_meta.json` por video;
- extracao das regioes face/contorno/fundo por frame;
- calculo de metricas estruturais com SIFT;
- bloco de Patch Similarity mantido para exploracao;
- consolidacao por frame via `all_metrics(video_path, max_frames=500, label=None)`.

## 3. Metricas e motivacao

### 3.1 SIFT (ativo no `all_metrics`)

O que mede:

- densidade de keypoints na face (`sift_face_kp_density`);
- entropia dos descritores (`sift_face_entropy`);
- auto-similaridade media dos descritores (`sift_face_self_sim`);
- diferencas inter-regionais (`face_bg`, `face_border`, `border_bg`) para densidade, entropia, presenca de keypoints e distancia de descritores.

Por que usar:

- estruturas faciais reais tendem a manter distribuicao de keypoints mais natural;
- videos sinteticos podem exibir repeticao ou instabilidade de descritores;
- diferencas entre face e contexto ajudam a detectar composicao artificial.

Contribuicao para IA forense:

- evidencia inconsistencia estrutural que nao aparece apenas em textura;
- adiciona robustez ao ensemble quando combinado com sinais de ruido e frequencia.

### 3.2 Patch Similarity (exploratorio)

O que mede:

- similaridade media e desvio de similaridade entre patches locais.

Por que usar:

- modelos gerativos podem repetir padroes locais em excesso.

Contribuicao para IA forense:

- potencial para detectar repeticao artificial de textura/estrutura;
- no estado atual, ainda nao entra no DataFrame final de `all_metrics`.

## 4. Estrutura do DataFrame por frame

Colunas de identificacao:

- `video_name`
- `label` (opcional)
- `frame`

Colunas SIFT diretas:

- `sift_face_kp_density`
- `sift_face_entropy`
- `sift_face_self_sim`

Colunas SIFT por diferencas entre regioes (prefixos `face_bg`, `face_border`, `border_bg`):

- `{prefix}_kp_density_diff`
- `{prefix}_desc_entropy_diff`
- `{prefix}_kp_presence_diff`
- `{prefix}_desc_dist`

## 5. Como este grupo contribui no ensemble forense

- adiciona uma camada de analise estrutural para complementar textura e ruido;
- melhora interpretabilidade com comparacoes regionais objetivas;
- fornece sinais frame-level prontos para agregacao temporal em nivel de video.

## 6. Limitacoes atuais

- forte dependencia da deteccao facial e da qualidade dos metadados de bbox;
- sensivel a blur/compressao, que reduzem keypoints utilmente discriminativos;
- Patch Similarity ainda nao integrado ao fluxo final.

## 7. Proximos passos

- avaliar integracao de Patch Similarity no `all_metrics`;
- criar agregacoes temporais por video para classificacao final;
- combinar features do grupo B com A e C em um metamodelo forense.

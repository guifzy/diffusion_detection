# (A) Densidade de estrutura

### Número de keypoints (`num_keypoints`)

**O que mede:**

* quantidade de estruturas detectáveis

**Interpretação:**

* alto → imagem rica em detalhes estruturais
* baixo → imagem “suavizada” ou artificial

👉 Deepfake:

* pode ter **menos keypoints** (suavização da pele)
* ou **mais keypoints artificiais** (ruído estruturado)

---

## (B) Qualidade dos pontos

### Response (força do keypoint)

**O que mede:**

* quão “forte” é um ponto de interesse

**Métricas úteis:**

* média (`mean_response`)
* variância (`std_response`)

**Interpretação:**

* real → distribuição mais natural
* deepfake → pode ter:
  * muitos pontos fracos
  * ou padrões artificiais consistentes

---

## (C) Escala (tamanho dos keypoints)

### `mean_size`

**O que mede:**

* escala das estruturas detectadas

**Interpretação:**

* real → mistura de escalas
* deepfake → pode ter:
  * uniformização
  * perda de microdetalhes

---

## (D) Variabilidade dos descritores

### Variância dos descritores (`descriptor_variance`)

**O que mede:**

* diversidade dos padrões locais

**Interpretação:**

* real → alta diversidade
* deepfake → padrões repetitivos ou artificiais

👉 Muito importante para difusão:

* modelos gerativos tendem a repetir padrões

---

## (E) Distribuição espacial (CRÍTICO)

### dispersão dos keypoints (ex: std em X/Y)

**O que mede:**

* como os pontos estão distribuídos no rosto

**Interpretação:**

* real → distribuição coerente com anatomia
* deepfake → concentração estranha (ex: boca/olhos)

---

## (A) Variação do número de keypoints

### Instabilidade estrutural

**O que observar:**

* flutuações grandes frame a frame

**Interpretação:**

* real → variação suave
* deepfake → variação irregular

---

## (B) Variação da resposta

### instabilidade da força estrutural

* mudanças abruptas indicam:
  * artefatos
  * inconsistência de textura

---

## (C) Drift estrutural

### mudança gradual nos padrões

* descritores mudando sem motivo físico
* indica geração frame a frame sem coerência

---

## (D) Consistência de matching (muito forte)

Se você compara frames consecutivos:

quantos keypoints “se mantêm”?

**Interpretação:**

* real → alta correspondência
* deepfake → baixa correspondência

---

## (E) Distribuição espacial ao longo do tempo

> os pontos continuam nas mesmas regiões do rosto?

* real → sim (estrutura anatômica)
* deepfake → não (instabilidade)

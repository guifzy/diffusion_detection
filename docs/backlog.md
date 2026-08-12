# Backlog metodologico e tecnico

Este documento consolida as proximas atividades do projeto antes da modelagem
experimental. A ordem proposta reflete dependencias metodologicas: primeiro
garantir que as regioes sao corretas e rastreaveis, depois congelar os sinais,
em seguida estabilizar o pipeline de dados e, por fim, executar a analise
exploratoria.

| Campo | Valor |
| --- | --- |
| Versao de referencia | `0.2.0` |
| Escopo atual | Sinais estaticos dos grupos A-E e pipeline Bronze-Silver-Gold |
| Proximo marco | Preparacao para EDA e modelagem experimental |
| Situacao dos dados Gold atuais | Artefato de smoke test; insuficiente para analise estatistica |

## Visao geral

| Ordem | Etapa | Prioridade | Complexidade | Dependencias |
| ---: | --- | --- | --- | --- |
| 1 | Consolidar deteccao MediaPipe, rastreabilidade e auditoria de regioes | P0 | Alta | Nenhuma |
| 2 | Validar e congelar o conjunto de sinais | P1 | Alta | Etapa 1 |
| 3 | Padronizar e validar o pipeline de engenharia de dados | P1 | Alta | Etapas 1 e 2 |
| 4 | Realizar EDA e analise estatistica dos sinais | P2 | Alta | Etapas 1, 2 e 3 |

## Estado atual

O repositorio possui uma implementacao funcional dos sinais estaticos dos grupos
A-E, identificada como versao `0.2.0`, alem de um pipeline Bronze-Silver-Gold
capaz de gerar metadados, atributos por frame, atributos agregados por video e
dataset Gold.

Entretanto, os dados Gold atualmente persistidos devem ser tratados apenas como
validacao operacional. Eles pertencem a um smoke test, possuem cobertura
amostral minima e nao devem sustentar EDA, selecao de atributos ou modelagem
experimental.

---

## 1. Consolidar deteccao MediaPipe e rastreabilidade de regioes

| Campo | Definicao |
| --- | --- |
| Prioridade | P0 |
| Complexidade | Alta |
| Natureza | Metodologia e implementacao |
| Objetivo | Garantir que cada regiao MediaPipe seja associada ao video, frame, tempo, identidade facial e origem da deteccao |

### Justificativa

O pipeline foi adaptado para gerar regioes com MediaPipe, incluindo rosto
completo, olhos, boca, corpo e fundo. Antes do EDA definitivo, essa geometria
precisa ser auditada em amostra real e estabilizada contra multiplas pessoas,
mudancas de cena, oclusoes, perda de track e alternancia entre faces.

### Tarefas

- Auditar se todas as faces elegiveis estao sendo detectadas pelo MediaPipe.
- Validar a associacao da mesma identidade facial entre frames por meio de
  `track_id`.
- Reiniciar ou encerrar tracks apos corte de cena, desaparecimento prolongado,
  perda de rastreamento ou mudanca abrupta de escala/posicao.
- Registrar `timestamp`, `frame_id`, `track_id`, `region_id`, `region_type`,
  score de deteccao, origem da caixa e indicadores de qualidade.
- Definir politica formal para FaceLandmarker, ImageSegmenter, fallback
  geometrico de corpo e `fallback_center`.
- Marcar ou excluir regioes com baixa qualidade, face muito pequena, blur
  excessivo, oclusao relevante ou pose fora do regime aceitavel.
- Gerar auditoria visual estratificada por classe, fonte, dificuldade e tipo de
  falha.
- Definir objetivamente as regioes `rosto_completo`, `olhos`, `boca`, `corpo` e
  `fundo` para cada face rastreada.

### Estrutura de dados esperada

A representacao recomendada para metadados e sinais por frame e uma estrutura
longa, com uma linha por regiao associada a cada identidade facial.

| video_id | frame_id | timestamp_s | track_id | region_id | region_type |
| --- | ---: | ---: | --- | --- | --- |
| video_01 | 120 | 4.00 | face_1 | rosto_completo_1 | rosto_completo |
| video_01 | 120 | 4.00 | face_1 | olhos_1 | olhos |
| video_01 | 120 | 4.00 | face_1 | boca_1 | boca |
| video_01 | 120 | 4.00 | face_1 | corpo_1 | corpo |
| video_01 | 120 | 4.00 | face_2 | rosto_completo_2 | rosto_completo |
| video_01 | 120 | 4.00 | global | fundo | fundo |

O fundo associado a uma face deve representar seu contexto espacial controlado.
Caso tambem exista um fundo global compartilhado, ele deve ser armazenado como
regiao propria para evitar duplicacao semantica.

### Resultado esperado

Todas as regioes devem ser rastreaveis ate video, frame, tempo, identidade,
origem da deteccao e politica de fallback. Frames baseados em fallback nao devem
ser tratados silenciosamente como deteccoes faciais validas.

### Criterio de conclusao

A etapa sera considerada concluida quando a auditoria visual e as metricas de
qualidade demonstrarem identidade estavel, regioes corretas e ausencia de troca
indevida entre faces.

---

## 2. Validar e congelar o conjunto de sinais

| Campo | Definicao |
| --- | --- |
| Prioridade | P1 |
| Complexidade | Alta |
| Natureza | Revisao bibliografica, metodologia e validacao numerica |
| Dependencia | Etapa 1 |
| Objetivo | Definir quais sinais entram no EDA experimental e quais permanecem como secundarios, controles ou pesquisa futura |

### Justificativa

Cada atributo precisa ter uma hipotese falsificavel, formula conhecida,
interpretacao permitida, fatores de confusao e teste controlado. Nenhum sinal
deve entrar no Gold experimental apenas por plausibilidade visual ou conveniencia
de implementacao.

### Revisao da literatura

- Revisar artigos primarios e surveys relevantes.
- Conferir as referencias cientificas em suas publicacoes originais.
- Registrar bases consultadas, termos de busca e criterios de selecao.
- Construir matriz relacionando artigo, sinal, mecanismo, dataset, resultados e
  limitacoes.
- Verificar se cada trabalho analisou a mesma grandeza fisica ou estatistica
  utilizada no projeto.
- Confirmar se o regime de formacao de imagem descrito na literatura e
  compativel com videos reais, videos sinteticos, compressao e redes sociais.
- Evitar sustentar argumentos centrais em fontes sem revisao cientifica.

### Decisoes a validar

| Familia | Decisao atual | Pontos de verificacao |
| --- | --- | --- |
| LBP | Manter | Escala, alinhamento, compressao e estabilidade por regiao |
| Sobel | Manter | Limiar de gradiente forte, resposta direcional e sensibilidade a blur |
| Laplaciano | Manter com ajuste | Redundancia entre `signed_std` e `signed_variance` |
| SIFT estatico | Manter como exploratorio | Controle por area, escala, quantidade de keypoints e pose |
| Patch similarity | Manter como secundario | Estabilidade e ganho incremental sobre textura/frequencia |
| Residuo bilateral | Manter como baseline | Nao interpretar como PRNU, Noiseprint ou ruido de sensor isolado |
| FFT espacial | Manter condicionalmente | Resize, interpolacao, codec, compressao e bandas de frequencia |
| Fotometria regional | Manter | Interpretar como proxy estatistico, nao como iluminacao fisica completa |
| Candidatos de sombra | Manter como exploratorio | Calibracao de limiar, materiais escuros e exposicao |
| Reflexos oculares | Adiar | Exige regioes oculares confiaveis e resolucao adequada |
| Sombra geometrica 3D | Adiar | Exige geometria facial, normais e estimativa de iluminacao |
| SRM ou wavelets | Avaliar | Possivel alternativa ao residuo bilateral |
| rPPG | Pesquisa futura | Alta sensibilidade a movimento, FPS, compressao e iluminacao |

### Temporalidade

A temporalidade deve ser adicionada somente depois que existirem `timestamp`,
`track_id`, cenas e regioes estaveis. As operacoes temporais candidatas sao:

- primeiras e segundas diferencas dos sinais por tempo fisico;
- variacao robusta, mudancas abruptas e autocorrelacao;
- coerencia temporal entre face, contorno e fundo;
- persistencia de fotometria e candidatos de sombra;
- estabilidade de SIFT ou correspondencias locais entre frames;
- fluxo optico e movimento relativo apos compensacao de camera.

Nao se deve aplicar automaticamente toda operacao temporal a todas as colunas.
Apenas sinais com interpretacao temporal plausivel devem gerar derivados, para
evitar explosao dimensional e ruido estatistico.

### Validacao controlada

Testar os sinais com imagens e sequencias sinteticas de comportamento conhecido:
textura, bordas, blur, sharpening, frequencia, ruido, iluminacao, movimento,
compressao, resize e transformacoes geometricas.

### Resultado esperado

Um catalogo versionado deve classificar cada atributo como `manter`,
`secundario`, `controle de qualidade`, `remover` ou `adiar`.

### Criterio de conclusao

A etapa sera considerada concluida quando cada sinal responder ao fenomeno que
afirma medir, possuir limites de interpretacao documentados e apresentar teste
numerico minimo.

---

## 3. Padronizar e validar o pipeline de engenharia de dados

| Campo | Definicao |
| --- | --- |
| Prioridade | P1 |
| Complexidade | Alta |
| Natureza | Engenharia de dados e reprodutibilidade |
| Dependencias | Etapas 1 e 2 |
| Objetivo | Garantir que todos os videos passem pelo mesmo processo de amostragem, padronizacao, extracao, versionamento e controle de qualidade |

### Frames e temporalidade

Nao e obrigatorio que todos os videos tenham a mesma quantidade original de
frames. O requisito metodologico e que a politica de amostragem seja comparavel,
documentada e aplicada de forma identica entre classes.

Pontos a definir:

- FPS de analise ou intervalo temporal entre amostras.
- Quantidade minima e maxima de frames validos.
- Amostragem por tempo ou por janelas de duracao fixa.
- Tratamento de videos curtos, longos e com FPS variavel.
- Tratamento de frames duplicados, ausentes ou irregulares.
- Numero minimo de frames por face e segmento para temporalidade.
- Agregacao que preserve o video como unidade experimental.

Uma quantidade fixa de frames distribuida por todo o video pode representar
intervalos temporais muito diferentes. Para analise temporal, timestamps e
janelas em segundos sao mais apropriados do que somente indices de frames.

### Resolucao e padronizacao espacial

O extrator atual limita o maior lado do frame a 640 pixels, sem ampliar videos
menores. Essa politica precisa ser validada e registrada.

Pontos a documentar:

- resolucao original e resolucao processada;
- fator e metodo de redimensionamento;
- escala da face e cobertura das regioes;
- dimensoes especificas usadas por sinais como FFT;
- impacto da interpolacao em textura, borda, residuo e frequencia.

Padronizacao nao significa forcar todos os videos a mesma resolucao original.
Significa aplicar uma transformacao conhecida, reproduzivel e igual entre
classes, preservando metadados para analise de confundimento.

### Governanca e execucao

- Separar configuracoes de smoke test e experimento completo.
- Substituir a fonte `smoke_source.csv` na execucao experimental.
- Propagar parametros por `params.yaml` e `dvc.yaml`.
- Impedir mistura entre artefatos das versoes `0.1.0` e `0.2.0`.
- Versionar esquema, parametros, manifesto, codigo e dados.
- Impedir que arquivos orfaos de execucoes anteriores entrem no Gold.
- Enriquecer o manifesto com origem, identidade, gerador, codec, resolucao,
  FPS, duracao e cadeia de derivacoes.
- Detectar duplicatas binarias e perceptuais.
- Criar splits agrupados por identidade, origem e derivados.
- Manter o conjunto de teste final congelado.
- Regenerar Silver e Gold depois do congelamento dos sinais.
- Validar NaN, infinitos, constantes, faixas numericas e cobertura das regioes.
- Registrar falhas, exclusoes, custo e tempo por video.

### Resultado esperado

Dataset Silver/Gold experimental reproduzivel, com ambas as classes, uma unica
versao de sinais e rastreabilidade completa.

### Criterio de conclusao

Uma execucao limpa deve reproduzir o mesmo esquema e resultados numericamente
compativeis, sem vazamento experimental, mistura de versoes ou entrada de
artefatos antigos.

---

## 4. Realizar EDA e analise estatistica dos sinais

| Campo | Definicao |
| --- | --- |
| Prioridade | P2 |
| Complexidade | Alta |
| Natureza | Estatistica e analise de dados |
| Dependencias | Etapas 1, 2 e 3 |
| Objetivo | Avaliar qualidade, separacao entre classes, redundancia, estabilidade e fatores de confusao antes da modelagem |

### Planejamento experimental

Executar inicialmente o estudo sobre uma amostra representativa e
estratificada, preservando um conjunto de teste que nao sera consultado durante
o EDA, a selecao de atributos ou a calibracao de hiperparametros.

### EDA de qualidade

- Quantidade de videos, faces, frames, segmentos e regioes.
- Equilibrio entre reais e falsos.
- Duracao, FPS, resolucao, codec, origem e gerador.
- Confianca da deteccao, fallback, pose, blur, oclusao e tamanho facial.
- Missingness, infinitos, constantes e valores fora das faixas esperadas.
- Estabilidade dos sinais dentro do video, por face e por segmento.
- Inspecao visual dos outliers de qualidade.

Problemas de medicao devem ser corrigidos antes de procurar separacao entre as
classes.

### EDA discriminativo

- Visualizar distribuicoes por classe, familia, regiao e dominio.
- Medir tamanhos de efeito e intervalos de confianca.
- Usar testes estatisticos adequados, com correcao por multiplas comparacoes.
- Analisar correlacoes de Spearman e relacoes nao lineares.
- Localizar atributos duplicados, redundantes, quase constantes ou instaveis.
- Avaliar outliers extremos com retorno ao video e ao frame original.
- Verificar se a separacao permanece dentro de cada fonte, codec, gerador e
  nivel de compressao.
- Comparar sinais estaticos e temporais.
- Comparar face, contorno, fundo e contrastes regionais.
- Evitar tratar frames da mesma sequencia como observacoes independentes.

O conjunto atual pode produzir milhares de colunas agregadas por video. A
reducao deve remover redundancias exatas, controlar correlacao e executar
selecao de atributos somente dentro dos folds de treino.

### Entregaveis

- Relatorio de qualidade e exclusoes.
- Catalogo de outliers.
- Mapa de correlacoes e redundancias.
- Ranking de sinais com tamanho de efeito e estabilidade.
- Identificacao de confundidores.
- Decisao final sobre sinais mantidos e removidos.
- Subconjunto de atributos candidato a modelagem.
- Plano de ablacao por familia, regiao e temporalidade.

### Criterio de conclusao

Um sinal so deve seguir para modelagem quando sua separacao entre real e falso
nao for explicada exclusivamente por fonte, codec, resolucao, identidade, erro
de regiao ou outro fator de aquisicao.

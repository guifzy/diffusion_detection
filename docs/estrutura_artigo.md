# Estrutura proposta e avaliação de viabilidade do artigo

> Avaliação realizada em 26 de setembro de 2026 sobre o commit `d639fb2`.
> Este documento é um plano de pesquisa e redação. Ele não apresenta resultados
> experimentais ainda inexistentes.

## Parecer executivo

O repositório sustenta **um projeto de artigo cientificamente viável**, mas ainda
não sustenta **uma submissão de artigo experimental concluído**.

A proposta mais defensável não é alegar a invenção de LBP, SIFT, FFT, resíduos
ou estatísticas temporais. Esses componentes são conhecidos. O artigo deve
investigar se uma representação forense **explícita, regional, temporal e
auditável** consegue generalizar para geradores de vídeo não vistos depois que
atalhos de aquisição, conteúdo e movimento são controlados.

### Diagnóstico por dimensão

| Dimensão | Estado | Parecer |
| --- | --- | --- |
| Relevância do problema | Forte | Detecção sob mudança de gerador é um problema atual e ainda não resolvido. |
| Hipótese científica | Promissora | Sinais interpretáveis e de baixo nível podem complementar detectores profundos e facilitar auditoria. |
| Implementação de sinais | Avançada | Há cinco famílias espaciais/fotométricas, regiões semânticas e derivados temporais. |
| Engenharia e rastreabilidade | Avançada | Há camadas Bronze/Silver/Gold, contratos, DVC e metadados de governança. |
| Dados experimentais disponíveis localmente | Insuficiente | O repositório contém o catálogo de links, mas não contém o DF26 processado nem uma Gold experimental. |
| Modelagem | Ausente | `src/ml` não contém treinamento, seleção, calibração ou inferência de classificadores. |
| Resultados científicos | Ausente | Os notebooks não têm execuções ou saídas persistidas; não há tabelas de desempenho ou ablações. |
| Protocolo de avaliação | Precisa de correção | O LOGOS atual não inclui reais no teste e não separa conteúdo/identidade por `df26_clip_id`. |
| Reprodutibilidade executável | Parcial | A estrutura existe, mas o ambiente inspecionado não executou os testes e o DVC não estava instalado. |
| Prontidão para submissão | Baixa no estado atual | É necessário executar dados, auditoria, modelagem, ablações e comparação com baselines. |

**Veredito:** vale desenvolver o artigo. O repositório já contém uma boa base
metodológica e de software, mas o resultado publicável dependerá de uma
avaliação empírica rigorosa. Resultados negativos também podem ser relevantes,
desde que revelem quais sinais falham, sob quais mudanças e por quais atalhos.

## Enquadramento recomendado

### Título provisório em português

**Sinais forenses interpretáveis para detecção de vídeos gerados por IA sob
mudança de gerador**

### Título provisório em inglês

**Interpretable Forensic Signals for AI-Generated Video Detection under
Generator Shift**

### Alternativa com ênfase em auditoria

**From Texture to Temporal Consistency: Auditing Interpretable Detectors of
AI-Generated Videos**

### Tese central

Uma combinação controlada de sinais texturais, estruturais, residuais,
espectrais, fotométricos e temporais, extraídos de regiões semanticamente
definidas, pode oferecer evidência discriminativa auditável. A validade dessa
evidência deve ser medida em geradores não vistos e após controlar conteúdo,
identidade, movimento, compressão, resolução, FPS e origem dos vídeos.

### Tipo de contribuição

O artigo deve se apresentar como um **estudo experimental de método e
auditoria**, com quatro contribuições possíveis:

1. Uma representação tabular forense explícita, organizada por família de
   sinal, região e comportamento temporal.
2. Um protocolo de avaliação agrupado por conteúdo e por gerador, com um
   holdout comercial mantido intocado até a avaliação final.
3. Uma análise de ablação que localiza o ganho de cada família, região e
   componente temporal.
4. Uma auditoria de atalhos que mede quanto do desempenho é explicado por
   metadados, movimento, amostragem, resolução, codec e compressão.

A quarta contribuição é essencial. Apenas concatenar descritores conhecidos e
treinar um classificador tabular tende a ser uma contribuição fraca.

## Perguntas de pesquisa

**RQ1.** Sinais forenses explícitos discriminam vídeos reais e sintéticos
quando o gerador de teste não aparece no treinamento?

**RQ2.** Quais famílias de sinais — textura e bordas, estrutura local, resíduo,
frequência, fotometria ou temporalidade — permanecem estáveis entre geradores?

**RQ3.** Quais regiões — rosto, olhos, boca, corpo e fundo — oferecem evidência
complementar, e quais apenas capturam conteúdo ou erro de segmentação?

**RQ4.** Os derivados temporais acrescentam informação além dos sinais
espaciais ou reproduzem diferenças espúrias de FPS, duração e movimento?

**RQ5.** O desempenho permanece após controlar compressão, resize, codec,
resolução, FPS, cenário, identidade/conteúdo e intensidade de movimento?

**RQ6.** Um detector interpretável oferece uma relação útil entre desempenho,
custo e auditabilidade quando comparado a baselines profundos recentes?

## Hipóteses falsificáveis

- **H1:** a representação completa supera um baseline que usa apenas
  metadados e variáveis de aquisição nos folds com gerador não visto.
- **H2:** sinais espectrais e residuais são mais transferíveis entre geradores
  que descritores dependentes de conteúdo, mas perdem desempenho sob
  recompressão e resize.
- **H3:** a temporalidade produz ganho incremental somente quando a amostragem
  usa tempo físico e o movimento é balanceado entre as classes.
- **H4:** contrastes entre regiões acrescentam informação além de métricas
  globais, sem depender predominantemente de fallback ou falha de detecção.
- **H5:** nenhum grupo isolado mantém o mesmo desempenho em todos os geradores;
  a utilidade da fusão está na complementaridade e na estabilidade, não apenas
  no melhor resultado médio.

As hipóteses devem ser mantidas mesmo se os resultados as refutarem.

## Escopo dos dados

### Base principal: DF26

O DF26 é a base principal recomendada porque reúne vídeos reais e vídeos
sintéticos de geradores abertos e comerciais recentes em cenários de fala com
uma pessoa. O artigo original relata 271 vídeos reais e 2.420 sintéticos de
sete famílias de geradores. O uso precisa respeitar a licença controlada; os
geradores comerciais devem permanecer exclusivamente na avaliação final.

O DF26, por si só, restringe a validade externa a vídeos curtos de fala com uma
pessoa. Essa restrição deve aparecer no resumo, na discussão e nas limitações.

### Catálogo por links

O manifesto versionado contém 2.039 links, sendo 1.072 rotulados como reais e
967 como falsos. Ele só deve ser usado como validação externa depois de:

- confirmar disponibilidade, licença e proveniência de cada item;
- auditar a confiabilidade dos rótulos;
- remover duplicatas binárias, perceptuais e versões derivadas;
- registrar gerador, fonte, codec e cadeia de recompressão;
- construir splits agrupados por identidade, origem e vídeo-base.

Sem essas verificações, esse catálogo não deve sustentar o resultado principal.

## Correção obrigatória do protocolo experimental

O arquivo `df26_logos_splits.csv` planejado pelo código atual coloca todos os
vídeos reais em `train` nos folds LOGOS e somente os falsos da família omitida
em `test`. Isso impede uma avaliação binária válida no teste. Além disso, a
chave de conteúdo `df26_clip_id` é preservada, mas não é usada para impedir que
conteúdo relacionado apareça nos dois lados do experimento.

### Protocolo recomendado

Usar um protocolo aninhado em duas dimensões:

1. **Mudança de gerador:** omitir uma família open-weight do treinamento.
2. **Mudança de conteúdo:** separar `df26_clip_id` em grupos disjuntos de
   treino, validação e teste.

Em cada repetição:

- treino: reais e fakes de geradores permitidos, somente de clipes de treino;
- validação: reais e fakes de geradores permitidos, somente de clipes de
  validação;
- teste LOGOS: reais e fakes do gerador omitido, somente de clipes de teste;
- holdout comercial: reais e fakes comerciais de clipes não usados na seleção
  do modelo, consultados uma única vez após congelar todo o pipeline.

Todas as derivações do mesmo vídeo-base, prompt, identidade ou clipe devem
ficar no mesmo grupo. Seleção de atributos, imputação, normalização,
balanceamento, calibração e ajuste de hiperparâmetros devem ocorrer somente nos
dados de treino de cada repetição.

Como o número de vídeos reais é menor que o de sintéticos, usar pesos de classe
ou amostragem apenas no treino. Não balancear artificialmente o teste; reportar
métricas robustas ao desbalanceamento.

## Desenho experimental mínimo

### Etapa 1 — Auditoria dos dados e das regiões

- Contagens por classe, gerador, modo de geração e cenário.
- Distribuições de duração, FPS, resolução, codec, bitrate e movimento.
- Duplicatas e relações derivadas entre vídeos.
- Taxa de sucesso, confiança e fallback por região.
- Estabilidade de `track_id`, tamanho facial, pose, oclusão e cortes de cena.
- Inspeção visual estratificada e catálogo de falhas.

### Etapa 2 — Auditoria das features

- Missingness, infinitos, constantes e faixas inválidas.
- Redundância exata e correlação de Spearman.
- Tamanhos de efeito com intervalos de confiança por vídeo.
- Estabilidade do sinal por gerador, cenário e perturbação.
- Comparação entre separação aparente e separação após controlar
  confundidores.

Não usar frames do mesmo vídeo como observações estatisticamente independentes.

### Etapa 3 — Baselines tabulares

Usar pelo menos:

- preditor aleatório e classe majoritária;
- regressão logística regularizada;
- Random Forest;
- gradient boosting tabular;
- baseline de metadados/atalhos sem sinais forenses.

O baseline de metadados deve incluir somente controles como FPS, duração,
resolução, codec e movimento. Se ele obtiver desempenho elevado, isso é um
achado de viés do dataset, não sucesso do detector forense.

### Etapa 4 — Ablacões principais

| Experimento | Objetivo |
| --- | --- |
| A, B, C, D e E isoladamente | Medir a contribuição de cada família de sinais. |
| Estático vs. estático + temporal | Medir o ganho temporal incremental. |
| Cada região isolada | Localizar a fonte da evidência. |
| Rosto vs. fundo vs. conjunto completo | Testar dependência de conteúdo/cenário. |
| Sem contrastes regionais | Medir o valor dos contrastes. |
| Sem features correlacionadas | Testar robustez à largura da Gold. |
| Sem amostras com fallback | Testar dependência de falhas do MediaPipe. |
| Metadados somente | Quantificar atalhos de aquisição. |

### Etapa 5 — Robustez e controles negativos

- Recompressão H.264 em níveis predefinidos.
- Resize com diferentes interpolações.
- Alteração controlada de FPS e política de amostragem.
- Blur, sharpening e ruído leve.
- Balanceamento por intensidade de movimento.
- Equalização por resolução, duração e cenário.
- Embaralhamento de rótulos como controle negativo.

As transformações devem ser aplicadas de forma simétrica às classes e nunca
escolhidas a partir do desempenho do holdout comercial.

### Etapa 6 — Comparação externa

Comparar, quando código e licença permitirem, com:

- um baseline frame-based de frequência;
- um baseline temporal recente;
- resultados oficiais do benchmark DF26, deixando claro quando a comparação é
  apenas com números publicados e usa protocolo diferente.

Uma comparação direta só é válida quando dados, splits e pré-processamento são
equivalentes.

## Métricas e inferência estatística

### Métricas principais

- AUROC por gerador e macro-AUROC entre geradores;
- balanced accuracy;
- AUPRC, sempre acompanhada da prevalência da classe positiva;
- TPR em FPR operacional fixo;
- ECE e Brier score para calibração;
- custo de extração por vídeo, uso de memória e dimensão final.

### Incerteza e testes

- Intervalos de confiança por bootstrap agrupado em `df26_clip_id`.
- Diferenças pareadas entre modelos sobre os mesmos grupos de teste.
- Correção de múltiplas comparações na análise univariada.
- Média e dispersão entre sementes e partições de conteúdo.
- Resultados individuais por gerador, além da média global.

O vídeo é a unidade mínima de avaliação; clipes relacionados formam o grupo de
reamostragem quando houver dependência entre eles.

## Estrutura sugerida do manuscrito

### Resumo

Estruturar em cinco movimentos:

1. problema de generalização de detectores para geradores recentes;
2. lacuna de auditabilidade e risco de atalhos;
3. representação proposta e protocolo agrupado;
4. principais resultados quantitativos, somente após obtê-los;
5. conclusão delimitada ao domínio avaliado.

Evitar expressões como “detector universal” ou “prova física”.

### 1. Introdução

- Evolução dos geradores e falha fora de distribuição.
- Limitação de benchmarks legados e de pistas semânticas.
- Motivação para sinais explícitos e auditáveis.
- Risco de confundimento por FPS, movimento, compressão e origem.
- Pergunta central, hipótese e lista de contribuições.

### 2. Trabalhos relacionados

#### 2.1 Detecção de vídeos totalmente gerados

Cobrir detectores frame-based, espaço-temporais, foundation models e métodos
training-free.

#### 2.2 Forense de baixo nível

Relacionar textura, bordas, resíduos, frequência e compressão às grandezas
efetivamente calculadas pelo projeto.

#### 2.3 Coerência temporal e física

Distinguir estatísticas temporais explícitas de alegações fortes sobre física,
geometria 3D ou causalidade.

#### 2.4 Generalização, datasets e atalhos

Discutir mudança de gerador, vazamento de conteúdo e vieses de movimento,
amostragem, resolução e compressão.

### 3. Método

#### 3.1 Visão geral

Apresentar o fluxo vídeo → amostragem → regiões → sinais por frame → agregação
temporal/regional → vetor por vídeo → classificador tabular.

#### 3.2 Regiões e rastreamento

Definir rosto, olhos, boca, corpo e fundo; explicar `track_id`, confiança,
fallback e exclusões de qualidade.

#### 3.3 Famílias de sinais

- Grupo A: LBP, Sobel e Laplaciano.
- Grupo B: SIFT e similaridade de patches.
- Grupo C: resíduo bilateral, sem interpretá-lo como PRNU.
- Grupo D: distribuição de potência e estatísticas espectrais.
- Grupo E: fotometria regional e candidatos de sombra.

As fórmulas e os limites interpretativos já documentados em
[`contrato_sinais_v0_2.md`](contrato_sinais_v0_2.md) devem ser a fonte da seção.

#### 3.4 Descritores temporais

Definir derivadas em tempo físico, autocorrelação, frequência temporal e
requisitos mínimos de amostragem. Explicar por que nem toda feature espacial
gera automaticamente uma feature temporal.

#### 3.5 Representação por vídeo

Descrever agregação por região, contrastes, tratamento de múltiplos tracks,
missingness e exclusão de colunas de governança da matriz preditiva.

#### 3.6 Classificadores e interpretabilidade

Definir modelos, pré-processamento dentro dos folds e explicações globais e
locais. Importância por permutação agrupada e coeficientes de modelos lineares
são preferíveis a interpretar automaticamente a importância interna de árvores.

### 4. Protocolo experimental

- Bases, licenças, critérios de inclusão e exclusão.
- Splits agrupados por clipe/conteúdo e por gerador.
- Holdout comercial e momento de abertura.
- Hiperparâmetros e seleção aninhada.
- Baselines, ablações, perturbações e métricas.
- Hardware, tempo, versões e sementes.

### 5. Resultados

#### 5.1 Qualidade dos dados e regiões

Reportar cobertura, falhas e exclusões antes de desempenho preditivo.

#### 5.2 Generalização para geradores não vistos

Tabela principal por fold e gerador, com intervalos de confiança.

#### 5.3 Ablacão de famílias e regiões

Mostrar ganho incremental e estabilidade, não apenas ranking médio.

#### 5.4 Valor da temporalidade

Comparar temporalidade antes e depois de balancear movimento e FPS.

#### 5.5 Robustez e atalhos

Mostrar degradação sob perturbações e o desempenho do baseline de metadados.

#### 5.6 Holdout comercial

Apresentar uma única avaliação final, sem ajustes posteriores disfarçados de
análise exploratória.

### 6. Discussão

- Quais sinais parecem compartilhados entre geradores?
- Quais resultados desaparecem sob controles?
- Em que condições a interpretabilidade ajuda a diagnosticar falhas?
- Qual é o custo de desempenho em relação a métodos profundos?
- O que os resultados não permitem afirmar?

### 7. Limitações, ética e reprodutibilidade

- Domínio restrito do DF26.
- Corrida evolutiva entre geradores e detectores.
- Possíveis erros de rótulo, segmentação e rastreamento.
- Risco de uso dual e necessidade de comunicação probabilística.
- Restrições de redistribuição dos dados.
- Disponibilização de código, splits, hashes, parâmetros e resultados
  agregados permitidos pela licença.

### 8. Conclusão

Responder diretamente às perguntas de pesquisa, sem extrapolar além dos
geradores, cenários e perturbações testados.

### Apêndices

- Catálogo completo de features e fórmulas.
- Parâmetros e versões do pipeline.
- Resultados por gerador, cenário e semente.
- Auditoria de regiões e critérios de exclusão.
- Estudos adicionais de correlação e estabilidade.
- Checklist de reprodutibilidade.

## Figuras e tabelas planejadas

### Figuras

1. Visão geral do método e pontos de auditoria.
2. Exemplos das cinco regiões com detecção válida e fallback.
3. Desenho do split em dois eixos: conteúdo e gerador.
4. Desempenho por gerador e família de sinal com intervalos de confiança.
5. Queda de desempenho sob compressão, resize, FPS e balanceamento de
   movimento.
6. Mapa família × região mostrando contribuição e estabilidade.

### Tabelas

1. Comparação com trabalhos relacionados e tipo de evidência usada.
2. Composição dos dados após filtros e exclusões.
3. Catálogo resumido das famílias de sinais.
4. Resultado principal LOGOS por gerador.
5. Ablacão de famílias, regiões e temporalidade.
6. Auditoria de atalhos e robustez.
7. Holdout comercial e comparação com baselines.
8. Custo computacional e dimensionalidade.

## Lacunas concretas do repositório

Antes da redação dos resultados, ainda faltam:

1. Corrigir e testar os splits agrupados do DF26.
2. Obter o dataset sob a licença aplicável e materializar os manifestos.
3. Executar a pipeline completa, não apenas o smoke test de um vídeo registrado
   no `dvc.lock`.
4. Auditar visualmente regiões e tracks em amostra estratificada.
5. Implementar EDA versionada e relatórios persistidos.
6. Implementar o módulo de modelagem em `src/ml`.
7. Impedir explicitamente que colunas `df26_*`, caminhos, nomes e metadados de
   governança entrem como features.
8. Executar baselines, ablações, controles de atalho e robustez.
9. Comparar com pelo menos um método externo sob protocolo compatível.
10. Congelar dependências e restaurar a execução limpa dos testes. Na inspeção,
    `pytest` falhou durante a coleta por incompatibilidade NumPy/pyarrow,
    ausência de OpenCV e problemas de importação; `dvc` não estava disponível.
11. Persistir tabelas, métricas, sementes, logs e versões que sustentem cada
    afirmação do artigo.
12. Revisar e confirmar todas as referências em suas fontes primárias.

## Critérios para decidir pela submissão

Prosseguir para submissão quando:

- o teste contiver ambas as classes e não houver sobreposição indevida de
  conteúdo, identidade ou derivados;
- a auditoria demonstrar cobertura regional adequada e documentar exclusões;
- os resultados forem reproduzíveis em execução limpa;
- o ganho sobre metadados e baselines simples persistir em geradores não vistos;
- as conclusões sobreviverem aos controles de movimento, compressão,
  resolução, FPS e cenário;
- o holdout comercial tiver sido usado somente depois do congelamento;
- intervalos de confiança e resultados por gerador acompanharem as médias;
- houver uma conclusão informativa mesmo que a hipótese principal seja
  refutada.

Se o desempenho desaparecer após os controles, ainda pode existir um artigo de
auditoria/resultado negativo, desde que o fenômeno seja demonstrado com rigor e
comparado à literatura.

## Sequência recomendada de trabalho

1. **Protocolo:** corrigir splits e definir o plano estatístico antes de abrir
   resultados.
2. **Reprodutibilidade:** estabilizar ambiente, testes e versões.
3. **Dados:** processar piloto estratificado, auditar regiões e só então rodar o
   DF26 completo.
4. **EDA:** eliminar falhas de medição e quantificar confundidores.
5. **Modelagem:** executar baselines, seleção interna aos folds e ablações.
6. **Robustez:** aplicar controles e perturbações simétricas.
7. **Congelamento:** escolher o modelo e abrir o holdout comercial uma vez.
8. **Redação:** preencher a estrutura acima com resultados rastreáveis aos
   artefatos versionados.

## Referências primárias prioritárias

- Shykula et al. (2026), [DF26: We Cannot Tell Fake From Real
  Anymore](https://arxiv.org/abs/2609.07369).
- Michels, Jorissen e Michiels (2026), [Dataset Biases and Shortcut Learning in
  Motion-Based AI-Generated Video Detection](https://arxiv.org/abs/2607.00948).
- Corvi et al. (2025), [Seeing What Matters: Generalizable AI-generated Video
  Detection with Forensic-Oriented Augmentation](https://arxiv.org/abs/2506.16802).
- Chen et al. (2024), [DeMamba: AI-Generated Video Detection on Million-Scale
  GenVideo Benchmark](https://arxiv.org/abs/2405.19707).
- Kundu et al. (2025), [Towards a Universal Synthetic Video Detector: From Face
  or Background Manipulations to Fully AI-Generated
  Content](https://openaccess.thecvf.com/content/CVPR2025/html/Kundu_Towards_a_Universal_Synthetic_Video_Detector_From_Face_or_Background_CVPR_2025_paper.html).

Esta lista é um ponto de partida, não uma revisão sistemática completa.

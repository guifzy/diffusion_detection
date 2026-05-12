# Material Orientador do Pitch

## Detecção Automatizada de Deepfakes em Conteúdos Audiovisuais Relacionados a Eleições

**Projeto:** Ensemble híbrido de análise espacial, espectral, física e temporal para detecção de vídeos gerados por IA  
**Objetivo do documento:** sintetizar o problema, a metodologia, o pipeline de dados, os resultados alcançados, o estágio atual de desenvolvimento e os próximos passos do projeto.

---

## 1. Problema e contexto do projeto

O projeto atua no contexto da disseminação de conteúdos audiovisuais sintéticos, especialmente vídeos gerados ou manipulados por inteligência artificial em cenários de alta sensibilidade pública, como eleições. O avanço de modelos generativos torna cada vez mais difícil distinguir, apenas por inspeção humana, um vídeo real de um vídeo sintético.

Esse problema é relevante porque deepfakes podem afetar confiança pública, circulação de desinformação, reputação de candidatos, interpretação de discursos e tomada de decisão do eleitorado. A dificuldade técnica está no fato de que os artefatos visuais estão ficando mais sutis: nem sempre há falhas óbvias de rosto, boca ou movimento.

**Problema central:** como detectar, de forma reprodutível e explicável, sinais visuais que diferenciem vídeos reais de vídeos gerados por IA?

**Hipótese do projeto:** vídeos sintéticos podem ser visualmente convincentes, mas ainda tendem a carregar inconsistências estatísticas, estruturais, espectrais, temporais ou físicas entre face, contorno e fundo.

---

## 2. Solução proposta

A solução proposta é uma pipeline de detecção baseada em um **ensemble híbrido de sinais forenses**. Em vez de depender de um único classificador caixa-preta, o projeto combina diferentes famílias de evidências:

- textura e bordas;
- estrutura local;
- ruído residual;
- frequência;
- física de iluminação;
- futura agregação temporal e modelos de aprendizado.

A ideia é que cada grupo de sinais capture uma faceta diferente do problema. Um vídeo falso pode não apresentar artefatos fortes em textura, mas pode mostrar inconsistência de iluminação, distribuição anormal de frequência ou ruído diferente entre face e fundo.

Essa abordagem favorece três pontos avaliativos importantes:

- **explicabilidade:** cada métrica tem interpretação forense;
- **robustez:** sinais complementares reduzem dependência de um único método;
- **reprodutibilidade:** os experimentos produzem métricas frame-level e video-level padronizadas.

---

## 3. Pipeline de dados

O pipeline atual parte de vídeos classificados como reais ou falsos, com metadados associados em CSV e JSON. O CSV consolida informações como nome do arquivo, rótulo de verdade de vídeo e origem. Os JSONs armazenam metadados de extração por vídeo.

### Fluxo metodológico atual

```mermaid
flowchart LR
    A[Videos reais e fake] --> B[Leitura dos metadados]
    B --> C[Amostragem de frames]
    C --> D[Deteccao de face]
    D --> E[Extracao de regioes]
    E --> F[Calculo de sinais forenses]
    F --> G[Metricas por frame]
    G --> H[Agregacao por video]
    H --> I[Ranking exploratorio por AUC e Cohen's d]
    I --> J[Base para ensemble final]
```

### Etapas principais

1. **Ingestão e organização**

   Os vídeos ficam organizados em `data/videos/`. Os metadados ficam em `data/metadata/`, incluindo o arquivo `video-metadata-publish-with-links.csv` e arquivos `*_meta.json`.

2. **Pré-processamento**

   A pipeline usa OpenCV e RetinaFace para detectar a face. Cada frame é padronizado, a face é localizada e são extraídas três regiões principais:

   - `face`: região facial principal;
   - `border` ou contorno: área ao redor da face;
   - `background` ou fundo: região fora da face expandida.

3. **Extração de sinais**

   Cada grupo metodológico calcula métricas específicas para as regiões. A comparação entre face, contorno e fundo é central, pois o objetivo não é apenas medir a face isoladamente, mas verificar se ela é coerente com o restante da cena.

4. **Agregação**

   As métricas são calculadas por frame e depois agregadas por vídeo com média, desvio padrão e variação. Isso gera uma visão temporal resumida do comportamento do vídeo.

5. **Avaliação exploratória**

   Os grupos geram relatórios com AUC e Cohen's d para ranquear quais sinais têm maior poder discriminativo entre vídeos reais e falsos.

### Arquitetura de dados planejada

Além do pipeline experimental, o projeto propõe uma arquitetura de engenharia de dados baseada em **Lakehouse com padrão Medalhão**:

- **Bronze:** vídeos e metadados brutos;
- **Silver:** frames, regiões e metadados processados;
- **Gold:** features e datasets prontos para modelagem.

Tecnologias planejadas:

- Python, OpenCV, pandas e numpy para processamento;
- Parquet para dados estruturados;
- MinIO como Data Lake compatível com S3;
- DVC para versionamento e reprodutibilidade;
- Prefect para orquestração;
- Great Expectations e Pytest para validação;
- GitHub Actions para CI/CD.

---

## 4. Metodologia aplicada

O projeto está dividido em grupos de sinais, todos seguindo o mesmo padrão: calcular métricas por região, comparar pares de regiões e agregar por vídeo.

### Grupo A: Textura, bordas e alta frequência

Este grupo busca inconsistências de microtextura e bordas.

Métricas principais:

- **LBP:** padrões locais de textura;
- **Sobel:** energia, coerência e variação de gradientes;
- **Laplacian:** nitidez e componentes de alta frequência;
- diferenças entre `face-background`, `face-border` e `border-background`.

Resultado exploratório observado: os melhores sinais apareceram em diferenças regionais, principalmente `border-background` e `face-background`. Sobel foi o sinal mais forte nessa amostra, com AUC exploratória de até **0,9667** em uma das métricas.

### Grupo B: Estrutura local

Este grupo avalia a consistência geométrica e estrutural.

Métricas principais:

- **SIFT:** contagem e densidade de keypoints, entropia de descritores e resposta dos pontos;
- **Patch Similarity:** similaridade entre pequenos blocos locais;
- diferenças estruturais entre face, contorno e fundo.

Resultado exploratório observado: sinais ligados à densidade e contagem de keypoints no fundo e às diferenças entre face e fundo ficaram entre os mais discriminativos, com AUC exploratória de até **0,8667**.

### Grupo C: Ruído residual

Este grupo procura assinaturas estatísticas de ruído que podem diferir entre câmera real e geração sintética.

Métricas principais:

- média, variância, desvio padrão e energia do resíduo;
- RMS, MAD, percentil 95, assimetria e curtose;
- diferenças de ruído entre face, contorno e fundo.

Resultado exploratório observado: a separação foi moderada, com AUC exploratória de até **0,7333**. Isso indica que ruído residual parece mais adequado como sinal complementar do que como detector isolado.

### Grupo D: Frequência

Este grupo analisa o domínio espectral por meio de FFT calculada em crops espaciais.

Métricas principais:

- proporção de baixa, média e alta frequência;
- entropia espectral;
- anisotropia;
- centroide radial;
- flatness espectral;
- diferenças espectrais entre regiões.

Resultado exploratório observado: diferenças entre face e contorno, especialmente entropia espectral e centroide radial, chegaram a AUC exploratória de **0,8667**.

### Grupo E: Física de iluminação

Este grupo avalia coerência óptica e iluminação.

Métricas principais:

- luminância em Lab;
- contraste, sombras e highlights;
- saturação e cor;
- gradiente de iluminação;
- assimetria facial aproximada;
- diferenças de iluminação entre face, contorno e fundo.

Resultado exploratório observado: o grupo apresentou sinais fortes na amostra, com algumas métricas chegando a AUC exploratória de **1,0000**. Porém, esses resultados exigem cautela porque podem refletir viés de dataset, cenário, cor ou fonte do vídeo.

---

## 5. Resultados alcançados e estágio atual

O projeto já possui uma estrutura experimental funcional, com:

- repositório organizado por dados, scripts e experimentos;
- pipeline de pré-processamento com detecção de face e extração de regiões;
- metadados em CSV e JSON;
- notebooks separados por grupo metodológico;
- métricas frame-level e video-level;
- avaliação exploratória com AUC e Cohen's d;
- exemplos de saída em vídeo na pasta `output_exemples/`.

### Estado atual dos dados processados

O CSV principal possui **2042 vídeos cadastrados**. Para os experimentos documentados até aqui, foram utilizados os vídeos com metadados compatíveis já gerados:

- **11 vídeos avaliados**;
- **6 Fake**;
- **5 Real**.

### Principais achados técnicos

1. Os sinais mais fortes aparecem quando comparamos regiões, não apenas quando analisamos a face isoladamente.

2. Textura, estrutura, frequência e iluminação já mostram capacidade discriminativa exploratória.

3. Ruído residual apresenta separação mais moderada, mas adiciona evidência complementar.

4. A pipeline já gera dados em formato adequado para um ensemble final.

5. Os resultados ainda são exploratórios, pois a amostra processada com metadados compatíveis é pequena.

### Leitura científica dos resultados

O estágio atual não permite afirmar desempenho final do detector. O que já é possível afirmar é que a metodologia está coerente: vários sinais independentes apontam para diferenças entre vídeos reais e sintéticos, especialmente na relação entre face, borda e fundo.

---

## 6. Limitações atuais

As principais limitações são:

- baixa quantidade de vídeos com metadados compatíveis nos experimentos atuais;
- risco de AUC inflado por amostra pequena;
- sensibilidade das métricas a compressão, resolução, iluminação e qualidade da bbox;
- possibilidade de viés de dataset em métricas de cor, saturação e fundo;
- ausência de landmarks detalhados para análise de olhos, boca e reflexos oculares;
- necessidade de separar avaliação exploratória, validação e teste.

Essas limitações não invalidam o avanço; elas indicam o que precisa ser controlado para transformar os resultados exploratórios em evidência experimental mais robusta.

---

## 7. Próximos passos planejados

Os próximos passos são:

1. **Expandir o pré-processamento**

   Gerar metadados compatíveis para uma quantidade maior de vídeos do dataset, reduzindo o risco de conclusão baseada em amostra pequena.

2. **Consolidar as features**

   Unificar as saídas dos grupos A, B, C, D e E em uma tabela Gold por vídeo, com padronização de nomes, tipos e agregações.

3. **Construir o ensemble**

   Testar agregação por votação ponderada, score combinado ou metamodelo supervisionado usando as features mais estáveis de cada grupo.

4. **Adicionar validação robusta**

   Separar treino, validação e teste; avaliar estabilidade por fonte, compressão, resolução e tipo de conteúdo.

5. **Incluir sinais temporais**

   Modelar a sequência de frames com CNN + Transformer ou outra abordagem temporal, usando os sinais por frame como entrada complementar.

6. **Evoluir a análise física**

   Adicionar landmarks faciais para olhos, boca e reflexos, permitindo estudar coerência de highlights e iluminação especular.

---

## 8. Considerações finais

Em síntese, o projeto já avançou da formulação do problema para uma pipeline técnica funcional. A contribuição central é propor uma detecção de deepfakes baseada em múltiplos sinais interpretáveis, comparando face, contorno e fundo em diferentes domínios.

O estágio atual demonstra preparo técnico em três dimensões:

- **dados:** organização de vídeos, metadados e regiões;
- **método:** extração de sinais forenses explicáveis;
- **avaliação:** relatórios exploratórios por vídeo com métricas discriminativas.

O próximo salto do projeto é escalar o processamento, consolidar as features e validar um ensemble final em uma base maior e mais controlada. Assim, a continuidade do trabalho deve transformar os resultados exploratórios atuais em evidências experimentais mais robustas, com separação adequada entre experimentação, validação e teste.

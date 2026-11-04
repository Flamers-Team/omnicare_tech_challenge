# OmniCare AI

Sistema de monitoramento hospitalar multimodal do Tech Challenge da Fase 4 (Pós-Tech IA para Devs, FIAP). É um protótipo acadêmico, sem validação clínica, para apoio à decisão.

O guia completo, escrito para quem está começando, está em [docs/Guia_do_Projeto_OmniCare.docx](docs/Guia_do_Projeto_OmniCare.docx).

## 1. Fluxo multimodal

| Dado | Base | Processamento | Situação |
|---|---|---|---|
| Sinais vitais em série temporal | MIMIC-IV Demo 2.2 | Comparação com as 24 h anteriores do próprio paciente | Pronto |
| Prescrições | MIMIC-IV Demo 2.2 | Mudanças fortes de dose e inícios tardios | Pronto |
| Marcha (esqueleto de sensores) | Multi-Gait-Posture | 13 medidas e pontuação z robusta | Pronto |
| Marcha (vídeo) | Vídeo do grupo | MediaPipe Pose e a mesma referência | Código pronto, sem validação com vídeo real |
| Voz (som) | TAME Pain | Medidas acústicas e regressão logística | Modelo treinado, ainda não ligado |
| Fala (conteúdo) | Consulta gravada pelo grupo | Azure Speech to Text e Azure Text Analytics | Código pronto, faltam chaves e teste |
| Vídeo de cirurgia | Endoscapes2023 | YOLOv8s e ResNet18, um frame a cada 5 segundos | Pronto, com as limitações descritas na seção 4 |

Cada resultado da internação entra em uma pontuação de risco que mostra o motivo de cada ponto. Quando o risco sobe, o sistema grava um alerta com laudo, e uma pessoa da equipe confirma que viu. Um indício de desvio na cirurgia gera um alerta próprio, de nível atenção. Os pesos foram definidos pelo projeto.

## 2. Como executar

Use Python 3.11. No PowerShell, na pasta do projeto:

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Abra http://localhost:8000 no navegador.

As chaves do Azure são opcionais para rodar a demonstração. Sem elas, o sistema informa que não há transcrição. Para usá-las, preencha `backend/.env`, que nunca deve ir para o Git:

```
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_SPEECH_LANGUAGE=pt-BR
AZURE_LANGUAGE_KEY=
AZURE_LANGUAGE_ENDPOINT=
```

### Modelos de cirurgia

Os pesos foram treinados com uma base de licença restrita e não fazem parte do repositório. Sem eles, a aba de cirurgia mostra as métricas, mas a análise fica desligada. Para ligá-la, gere os arquivos com `training/omnicare_endoscapes_colab.ipynb` e copie `endoscapes_yolov8s.pt` e `cvs_resnet18.pt` para `backend/models/`.

Para analisar os vídeos de teste, quem tiver acesso à base pode gerar o zip com `training/prepare_endoscapes.py`. O sistema procura `endoscapes_omnicare.zip` na pasta Downloads; em outro local, informe o caminho na variável de ambiente `ENDOSCAPES_ZIP`. Nenhuma imagem é salva ou exibida.

## 3. Testes

```powershell
cd backend
pip install -r requirements-dev.txt
python -m pytest tests
```

## 4. Resultados

Os arquivos completos estão em `training/resultados/`. As anomalias de sinais vitais, prescrições e marcha são simuladas sobre dados reais, então esses números são testes de engenharia e não medem acerto clínico.

**Sinais vitais** (metade dos pacientes, separada do ajuste do limiar):

| Medida | Comparação com o paciente | Regras fixas | Isolation Forest |
|---|---|---|---|
| Eventos por paciente por dia | 0,60 | 2,91 | 0,36 |
| Disparo por acaso em 6 h | 4,8% | 19,3% | 7,5% |
| Detecção de alteração moderada | 46,2% | 55,1% | 25,3% |
| Detecção de alteração forte | 67,2% | 70,5% | 51,6% |

**Prescrições:** em 250 internações, 872 mudanças de dose, 330 inícios tardios e 49 eventos de gravidade alta. Uma dose 10 vezes maior, simulada, gerou evento de gravidade alta em 88,7% dos casos (266 de 300).

**Alerta combinado**, só com sinais vitais e prescrições: 0,50 aviso de atenção e 0,24 de risco alto por paciente por dia.

**Marcha** (cada pessoa fora da referência): alarme falso de 7,1% em linha reta; detecção de 100% com o tronco 20° à frente e de 38,1% com o tronco 10° à frente. Na pior pessoa saudável, o alarme falso foi de 66,7%.

**Voz** (51 pessoas, sempre testando em pessoas novas): AUROC de 0,72 para dor ≥ 4 com gravação de base e de 0,65 sem ela. Dor e frio quase coincidem nessa base, então o resultado é um sinal vocal associado à dor induzida, e não uma medida de dor.

**Cirurgia, detecção de estruturas** (YOLOv8s, 312 frames de teste):

| Estrutura | AP50 |
|---|---|
| Vesícula | 0,95 |
| Instrumento | 0,91 |
| Triângulo de Calot | 0,55 |
| Ducto cístico | 0,53 |
| Artéria cística | 0,49 |
| Placa cística | 0,25 |

A média geral (mAP 50-95) foi de 38,2 no teste e de 33,5 na validação. O relatório técnico da base publica de 29,3 a 32,7 com outros modelos. A diferença entre validação e teste já é parecida com essa distância, e a ferramenta de cálculo pode não ser a mesma, então o resultado é compatível com o publicado, sem indicar superioridade.

**Cirurgia, critérios por frame** (ResNet18, 1.799 frames de teste): AP médio de 46,1, contra 51,5 da ResNet50 publicada (C1 42,5, C2 39,7 e C3 56,2). O melhor modelo apareceu na época 2 de 15, com sinal de sobreajuste nas seguintes.

**Cirurgia, alerta de desvio por vídeo** (40 vídeos de teste, 26 com desvio). Um desvio é um vídeo que chega ao primeiro clipe sem C1, C2 e C3 marcados pelos especialistas, o que não significa erro da equipe.

| Medida | Regra adotada (C2 e C3) | Regra anterior (C1, C2 e C3) |
|---|---|---|
| Desvios detectados | 20 de 26 (77%; IC 95%: 58% a 89%) | 22 de 26 (85%) |
| Vídeos com CVS sem falso alarme | 10 de 14 (71%; IC 95%: 45% a 88%) | 7 de 14 (50%) |
| AUROC | 0,78 (IC 95%: 0,60 a 0,94) | 0,76 (IC 95%: 0,58 a 0,91) |
| Acurácia, contra 65% de sempre dizer "desvio" | 75% (p = 0,45) | 72,5% (p = 0,55) |

A regra foi revista em `training/avaliar_regra_cvs.py`, com validação cruzada nos 41 vídeos de validação. Ela troca menos falsos alarmes por mais desvios perdidos, e a amostra não permite dizer que é melhor no geral. A candidata sem C1 só foi considerada depois de vermos o resultado do C1 no teste (AUROC 0,50, contra 0,67 na validação), e o teste foi olhado uma segunda vez, então estes números tendem a ser um pouco otimistas.

Limitações do alerta:

* O C1 não entra na decisão, então um desvio só em C1 passa sem aviso. Por isso o sistema não afirma que a Visão Crítica foi atingida.
* Vídeos mais longos ainda tendem a parecer mais seguros (correlação de -0,41 entre o número de frames e a pontuação de desvio).
* A pontuação pode ser alta antes de o critério ser atingido, então o alerta usa o vídeo até o primeiro clipe e não serve como aviso durante a cirurgia.
* A base vem de um único centro, e os especialistas discordam entre si (kappa de 0,38 no CVS).
* O caminho de envio de vídeo só foi testado com arquivos montados a partir dos frames da base.

## 5. Bases

1. **MIMIC-IV Clinical Database Demo 2.2:** 100 pacientes e 140 internações em UTI, com sinais vitais e prescrições.
2. **Multi-Gait-Posture:** 14 adultos saudáveis com andador, em 367 caminhadas.
3. **TAME Pain:** 7.044 áudios de 51 pessoas saudáveis, com dor induzida por água gelada.
4. **Endoscapes2023:** 201 vídeos de colecistectomia laparoscópica, com 1.933 frames anotados com caixas e 11.090 com os critérios da Visão Crítica de Segurança.

## 6. Citações das Bases
TAME Pain e Endoscapes2023 têm acesso restrito na PhysioNet: cada pessoa precisa criar a própria conta e assinar o termo de uso, e esses dados não fazem parte deste repositório. O repositório só traz trechos derivados das bases abertas, em `backend/sample_data/`.

**Endoscapes2023** (PhysioNet Restricted Health Data License 1.5.0)
* Mascagni, P., Alapatt, D., Murali, A., Vardazaryan, A., Garcia Vazquez, A., Okamoto, N., Costamagna, G., Mutter, D., Marescaux, J., Dallemagne, B., & Padoy, N. (2024). Endoscapes2023, A Critical View of Safety and Surgical Scene Segmentation Dataset for Laparoscopic Cholecystectomy (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/czwq-jh81
* Murali, A., Alapatt, D., Mascagni, P., Vardazaryan, A., Garcia, A., Okamoto, N., et al. (2023). Latent graph representations for critical view of safety assessment. IEEE Transactions on Medical Imaging. https://ieeexplore.ieee.org/abstract/document/10319763
* Murali, A., Alapatt, D., Mascagni, P., Vardazaryan, A., Garcia, A., Okamoto, N., Costamagna, G., Mutter, D., Marescaux, J., Dallemagne, B., & Padoy, N. (2023). The Endoscapes Dataset for Surgical Scene Segmentation, Object Detection, and Critical View of Safety Assessment: Official Splits and Benchmark. arXiv:2312.12429. Fonte da regra de CVS por vídeo e dos resultados de referência.

**TAME Pain** (PhysioNet Restricted Health Data License 1.5.0)
* Dao, T., Schneiders, E., Williams, J., Bautista, J. R., Seabrooke, T., Vigneswaran, G., Kolpekwar, R., Vashistha, R., & Farahi, A. (2025). TAME Pain: Trustworthy AssessMEnt of Pain from Speech and Audio for the Empowerment of Patients (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/20e2-1g10

**A Multi-Camera and Multimodal Dataset for Posture and Gait Analysis** (Creative Commons Attribution 4.0)
* Palermo, M., Mendes Lopes, J., André, J., Cerqueira, J., & Santos, C. (2021). A multi-camera and multimodal dataset for posture and gait analysis (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/fyxw-n385

**MIMIC-IV Clinical Database Demo** (Open Data Commons Open Database License 1.0)
* Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/dp1f-ex47
* Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2021). MIMIC-IV (version 1.0). PhysioNet.

**PhysioNet** (citação padrão pedida por todas as bases acima)
* Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z
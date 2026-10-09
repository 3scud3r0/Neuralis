# AxisLab — Research Toolkit v0.3.0

**Pacote real, executável em CPU**, derivado da prova de conceito anterior e inspirado em princípios publicados de *representation engineering*, interpretabilidade mecanicista, autoencoders esparsos e edição de memória. Não baixa nem modifica pesos de terceiros automaticamente.

Este pacote implementa algoritmos independentes em PyTorch e **adaptadores opcionais** que chamam as bibliotecas oficiais. Ele **não contém cópias** de nnsight, SAELens, RepE, ROME, MEMIT ou TransformerLens. A integração completa com cada biblioteca/modelo precisa ser validada em seu ambiente.

## Instalação

Requer Python >= 3.10 e PyTorch >= 2.3. No diretório extraído:

```bash
python -m pip install -e '.[test]'
python demo.py
python demo_advanced.py
python -m pytest -q
```

Instalação opcional, **dependente dos requisitos dos seus modelos**:

```bash
python -m pip install -e '.[nnsight,saelens,transformerlens,test]'
```

A biblioteca RepE e as implementações completas de ROME e MEMIT devem ser instaladas a partir dos projetos originais (links abaixo). **Não** estão incluídas em extras porque suas dependências e versões podem entrar em conflito com PyTorch/Transformers atuais. Execute em ambientes virtuais separados quando necessário.

O ambiente de desenvolvimento usado para gerar este pacote tinha PyTorch CPU, **mas não possuía** as dependências opcionais. Por isso, testes da implementação independente e dos contratos simulados de adaptadores foram executados; os backends externos não foram testados ponta a ponta.

## Capacidades prontas (PyTorch local)

| Módulo | API | O que faz |
|---|---|---|
| `core` | `fit_axis` | Diferença de médias com supressão PCA de controles, como a técnica explorada em Pain-axis |
| `representations` | `fit_paired_axis`, `fit_pca_axis`, `pca_subspace`, `fit_ridge_probe` | Contrastes pareados, direção PC1, subespaço PCA, *probe* linear |
| `representations` | `roc_auc`, `evaluate_axis`, `AxisBank` | AUC, avaliação em dados independentes, comparação/cosine e composição de eixos |
| `interventions` | `activation_patch`, `multi_axis_steering`, `projection_ablation` | Substituição de ativações, injeção de vários vetores e ablação por projeção |
| `sparse` | `TopKSAE`, `train_topk_sae`, `sae_delta_intervention` | SAE esparso treinável e intervenção sobre features com preservação do erro de reconstrução |
| `memory` | `rank_one_update`, `batched_ridge_update`, `weight_edit` | Solução de edição linear exata/de mínimos quadrados, limite de norma e rollback transacional |
| `streaming` | `ActivationMoments`, `forward_moments` | Médias e variâncias de ativação sem manter o histórico integral na memória |

Todos os métodos usam dimensões explícitas. As operações de *steering*, ablação, *patching* e *weight_edit* restauram a situação original ao sair do bloco `with` (a menos que `weight_edit(..., keep=True)` seja declarado conscientemente).

## Integrações opcionais

| Biblioteca upstream | Adaptador AxisLab | Limite atual |
|---|---|---|
| [TransformerLens 4](https://github.com/TransformerLensOrg/TransformerLens) | `transformer_lens_capture`, `transformer_lens_steer` | Steering local via `run_with_hooks`; backends remotos não permitem mutação por hook comum |
| [nnsight / NDIF](https://github.com/ndif-team/nnsight) | `nnsight_capture`, `nnsight_steer` | Usa `trace()`; execução `remote=True` requer conta, disponibilidade de modelo e validação externa |
| [SAELens](https://github.com/decoderesearch/SAELens) | `saelens_encode_decode`, `sae_delta_intervention` | Aceita objetos SAE com `encode`/`decode`; não treina SAEs de produção |
| [RepE](https://github.com/andyzoujm/representation-engineering) | `repe_registry` | Registra pipelines oficiais; métodos pareados/PCA locais são independentes |
| [ROME](https://github.com/kmeng01/rome) | `upstream_model_edit(algorithm="ROME")` | Usa implementação upstream se instalada, com requests/hparams específicos |
| [MEMIT](https://github.com/kmeng01/memit) | `upstream_model_edit(algorithm="MEMIT")` | Usa implementação upstream se instalada, com requests/hparams específicos |

### Exemplo: intervenção causal em rede pequena

```python
import torch
from torch import nn
from axislab_poc import fit_paired_axis, temporary_steering

baseline = torch.tensor([[-1.,0.], [0.,0.], [1.,0.]])
positive = baseline + torch.tensor([2.,0.])
axis = fit_paired_axis(positive, baseline)  # aprende direção +feature0

network = nn.Sequential(nn.Identity(), nn.Linear(2, 1, bias=False))
with torch.no_grad():
    network[1].weight.copy_(torch.tensor([[1.,0.]]))

x = torch.zeros(1, 2)
print(network(x).item())  # 0.0
with temporary_steering(network, "0", axis, coefficient=2.0):
    print(network(x).item())  # 2.0
print(network(x).item())  # 0.0: hook removido
```

**Entenda cada operação:** `baseline` são exemplos de controle com duas features; `positive` desloca apenas a primeira; `fit_paired_axis` calcula a média de `positive - baseline` e normaliza a direção; `nn.Identity` fornece uma camada intermediária observável; `nn.Linear(2,1)` lê apenas a primeira feature. O `with` registra um hook temporário que adiciona duas unidades na direção descoberta. O último `print` verifica que a rede recuperou o comportamento original.

### Exemplo: edição de pesos, com rollback

```python
import torch
from torch import nn
from axislab_poc import rank_one_update, weight_edit

layer = nn.Linear(2, 1, bias=False)
key = torch.tensor([1., 0.])
target = torch.tensor([3.])
delta = rank_one_update(layer, key, target)
with weight_edit(layer, delta, max_relative_norm=100):
    print(layer(key))  # valor alvo 3 (aprox. erro de ponto flutuante)
# Agora os pesos da camada são restaurados.
```

`rank_one_update` calcula o menor ajuste de norma Frobenius que faz uma **única camada linear** mapear `key` para `target`. `weight_edit` aplica a alteração e restaura os pesos ao final. Isto **não** localiza automaticamente um fato dentro de um Transformer, nem implementa todo o ROME.

### Exemplo: SAE Top-K

```python
import torch
from axislab_poc import TopKSAE, train_topk_sae

x = torch.randn(200, 16)                      # ativações sintéticas
sae = TopKSAE(d_model=16, d_sae=64, k=4)     # 4 de 64 features ativas por amostra
metrics = train_topk_sae(sae, x, steps=100)   # otimiza reconstrução
reconstruction, features = sae(x[:3])         # 3 linhas de entrada
print(reconstruction.shape, features.shape)   # (3, 16), (3, 64)
```

O encoder projeta 16 entradas em 64 ativações com ReLU; Top-K retém até quatro valores positivos por amostra; o decoder reconstrói as 16 dimensões; a otimização Adam reduz o erro quadrático médio. Não presuma que cada feature corresponda a um conceito interpretável: isso exige rotulação, avaliações, controles e replicação.

### Exemplo: TransformerLens local

```python
from transformer_lens.model_bridge import TransformerBridge
from axislab_poc import transformer_lens_capture

bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
print(list(bridge.hook_dict)[:20])  # inspecione nomes reais de hooks
# Escolha um dos nomes expostos por hook_dict e execute:
# logits, activation = transformer_lens_capture(bridge, "Hello world", hook_name)
```

O `TransformerBridge` baixa um modelo quando necessário. Neste projeto, não foi testado no ambiente de desenvolvimento. Os nomes de hooks podem variar conforme arquitetura e versão.

### Exemplo: nnsight local/remoto (opcional)

```python
# Requer: pip install nnsight
# from nnsight import TransformersModel
# from axislab_poc import nnsight_capture
# model = TransformersModel("openai-community/gpt2", dispatch=True)
# activation = nnsight_capture(model, "Hello", "transformer.h.0")
```

`remote=True` é aceito no adaptador, mas só funciona se o serviço NDIF autenticar sua conta e hospedar o modelo solicitado. Nenhum resultado remoto foi testado neste pacote. `nnsight_steer` altera a saída completa do módulo durante uma trace, e não implementa ainda seleção de tokens para todas as arquiteturas.

## Matemática de edição de memória

Para camada afim `y = Wx + b`, a correção de posto 1 satisfaz um par chave/valor:

\[\Delta W = \frac{(y_*-Wk-b)k^\top}{k^\top k}.\]

Para múltiplos pares `K` (colunas de chaves) e `Y` (colunas de alvos):

\[\Delta W = (Y-WK-b) (K^\top K+\lambda I)^{-1}K^\top.\]

A implementação usa `torch.linalg.solve` em vez de formar explicitamente a inversa e conta com um termo de regularização `ridge`. Este **não é o algoritmo MEMIT completo**: MEMIT distribui atualizações entre camadas e inclui estimativas de estatísticas internas para preservação de conhecimento.

## Garantias que não estão sendo feitas

- Não se demonstrou criação de conhecimentos novos, melhoria universal de raciocínio ou controle de consciência/dor subjetiva.
- Não há benchmark em LLMs de trilhões de parâmetros nem integração distribuída multisservidor testada.
- Não existe isolamento automático entre edições: uma mudança pode afetar entradas não relacionadas. Avalie `preservation_drift` e baselines.
- `TopKSAE` não substitui o treinamento e as validações da SAELens.
- Vetores lineares e *probes* são aproximações e não necessariamente correspondem a mecanismos causais únicos.
- A extração de ativações de um modelo completo em memória pode ser cara; para produção, capturar apenas pontos necessários e agregar por GPU.

## Protocolos de pesquisa sugeridos

1. Separe dados por grupos/assuntos para evitar vazamento e avalie em exemplos inéditos.
2. Compare contra direções aleatórias com norma equiparada, PCA não supervisionada, métodos de prompt e LoRA.
3. Faça dose-response, ablação de direções, *activation patching* e preserve tarefas não-alvo.
4. Documente cada versão do modelo, tokenizer, conjunto de dados, nome de hook, seed, hardware e hiperparâmetros.
5. Não execute intervenções em agentes com acesso a ferramentas externas sem medidas de segurança e testes específicos.

## Atribuição e licenciamento

A implementação neste pacote é independente, usando fórmulas e interfaces públicas. Projeto próprio sob licença MIT (arquivo `LICENSE`). Bibliotecas opcionais continuam pertencendo a seus respectivos autores e licenças. Consulte os repositórios originais para detalhes de licenciamento, restrições de uso e citações acadêmicas.

- [Pain-axis](https://github.com/valen-research/Pain-axis)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [NNsight](https://github.com/ndif-team/nnsight)
- [SAELens](https://github.com/decoderesearch/SAELens)
- [RepE](https://github.com/andyzoujm/representation-engineering)
- [ROME](https://github.com/kmeng01/rome)
- [MEMIT](https://github.com/kmeng01/memit)


## v0.3 — Observabilidade de raciocínio em modelos **locais e autorizados**

### O que mudou

- `lenses.logit_lens`: projeta ativações `[... , d]` via pesos de *unembedding* `[vocab,d]`, retornando IDs, probabilidades, entropia e rank de token-alvo, sem confundir projeção com transcrição de pensamentos. Aplique a normalização de saída correta da arquitetura (LayerNorm/RMSNorm) antes da cabeça de linguagem.
- `lenses.coda_lens`: interface genérica para aplicar o bloco/sufixo Coda do próprio modelo antes da projeção; exige que o chamador forneça a função arquitetural correta. **Não** é uma reprodução testada da implementação específica de Huginn.
- `lenses.TunedLens` e `fit_tuned_lens`: aprende transformação afim intermediária sob a distribuição produzida pela camada final, sem modificar os pesos do modelo base. Exige verificação com dados de teste separados para garantir utilidade.
- `traces.LocalActivationTrace`: captura ordenada de ativações com hooks PyTorch, múltiplas chamadas a módulos recorrentes, orçamento de memória e exportação local `.npz` mais metadados `.json`. **Não** reconstrói uma cadeia de pensamento textual.
- `traces.capture_hf_hidden_states`: solicita estados ocultos de um modelo Hugging Face que roda **localmente** usando `output_hidden_states=True`; captura toda camada ou somente o token não-padded final.
- `causal_reasoning.causal_patch_effect`: comparação entre execução limpa, corrompida e patch com medida de recuperação de desempenho. `sae_feature_patch` transfere apenas características selecionadas entre duas ativações usando **o mesmo** SAE.
- `causal_reasoning.component_logit_contributions`: projeção linear de componentes aditivos para explicar diferenças de logit, **não** interpretação exata após normalizações não-lineares.
- `interpretation.FeatureHypothesis`, `inspect_sae_features`, `inspect_axis`: estatísticas associadas a rótulos de características com origem (manual/heurístico/validado) claramente definida.
- `interpretation.vocabulary_partition`: distingue ID de token normal e ID de token de vocabulário latente em modelos duais que você controla, sem fingir que token latente tem uma tradução natural.
- `circuits.recurrent_attention_summary`: mede massa de atenção das cabeças para posições de memória em passos recorrentes (observação estatística, não prova de mecanismo). `export_circuit_html` gera um SVG interativo offline com filtros de pesos e rótulos de evidência causal/descritiva.
- `reporting.export_trace_json`/`export_trace_html`: relatório offline que exibe *ranks* e distribuições projetadas, com advertência sobre fidelidade.

### Exemplo: trajetória de predição em um modelo local

```python
import torch
from axislab_poc import decode_states, export_trace_html, trace_token_ranks

# Três estados da MESMA representação [batch=1, width=3].
states = [torch.tensor([[0., 1., 0.]]),
          torch.tensor([[0.5, 0.5, 0.]]),
          torch.tensor([[1., 0., 0.]])] 

# A cabeça de saída tem 3 tokens e lê diretamente as 3 dimensões.
unembed = 8 * torch.eye(3)  # [vocab=3, width=3]
target = torch.tensor([0])
readouts = decode_states(states, unembed, target_ids=target, top_k=3)
print(trace_token_ranks(readouts))  # [2, 1, 1] (empates têm o mesmo rank)
export_trace_html('reasoning_report.html', readouts, stage_labels=['t0', 't1', 't2'])
```

Os três estados são construídos artificialmente para ilustrar como evolui a distribuição projetada de um token-alvo. Um rank decrescente **não prova** que o modelo calculou verbalmente aquele token durante o processamento.

### Critério experimental

Para alegar interpretação de raciocínio, compare predições em dados inéditos, intervenções causais com controles aleatórios, aproximações de normalização, replicabilidade por arquitetura e diferenças entre descrições em linguagem natural e computação efetiva. Uma SAE treinada em dados CoT e outra em dados não-CoT podem ter features desalinhadas — não transplante índices entre ambas sem verificar alinhamento.

### Limites de acesso e confidencialidade

Estes módulos capturam ativações apenas de modelos locais ou explicitamente controlados. **Não** contornam APIs para acessar traces ocultos, não reutilizam envelopes criptografados, não automatizam prompt injection ou replay de tool calls e não transmitem raciocínios privados a servidores externos. Abordagens de auditoria de APIs podem trabalhar com respostas, metadados, resumos disponibilizados pelo provedor e saídas de testes autorizados.

### Referências de pesquisa

- https://arxiv.org/abs/2507.22928 — SAEs + patching em CoT/GSM8K.
- https://arxiv.org/abs/2507.02199 — Huginn, Logit Lens e Coda Lens; resultados em grande parte limitados.
- https://github.com/jialiang19/latent-cot-thinking — estudo e notebooks de latent CoT.
- https://github.com/wenquanlu/huginn-latent-cot — estudos de trajetórias recorrentes e múltiplas lentes.
- https://github.com/LE-WH/LatentReasoning — arquitetura experimental de vocabulário dual.
- https://arxiv.org/abs/2608.20055 — estudo de riscos de vazamento de CoT por API; não incorporado como funcionalidade de extração.


### Exemplo prático com um Transformer local

`examples/hf_local_lens.py` demonstra a inspeção camada a camada de `openai-community/gpt2`, com `transformers`, os pesos da cabeça de saída e a normalização final do GPT-2. Instale `transformers` separadamente e rode `python examples/hf_local_lens.py`. O modelo só será baixado durante essa execução, não durante o teste normal da AxisLab. Os experimentos em CPU deste projeto não incluem benchmark nesse modelo.

### Exemplo de gráfico de circuitos offline

O arquivo gerado por `demo_reasoning.py` em `axislab_reasoning_output/circuit.html` inclui nós, arestas rotuladas e controle interativo de intensidade. É um mapa de arquitetura e **não** demonstra um circuito de raciocínio aprendido. Adicione atribuições causais apenas depois de executar intervenções e registrar seus efeitos.

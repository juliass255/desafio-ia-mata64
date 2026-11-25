 # Rede neural que aprende onde os sinais costumam ficar

MATA64 · Inteligência Artificial · UFBA · 2026.2

A rede (MLP) aprende **entre execuções** onde os sinais costumam ficar. O Q-learning continua aprendendo **dentro** da execução, a partir do que o robô pisa. A previsão da rede só define o valor inicial da Q-table. Na execução avaliada, cada sinal continua sendo conhecido só quando o robô passa por cima dele.

## Estrutura do projeto

```
grade_inteligente/
├── worlds/grade_inteligente.wbt          (não modificar)
├── controllers/
│   ├── grade_supervisor/                 (não modificar)
│   └── grade_controller/
│       ├── grade_controller.py           Webots: sensores, motores, supervisor, registros
│       ├── agente.py                     Q-learning (sem Webots)
│       ├── prior_sinais.py               rede: atributos, passagem para a frente (numpy), Q inicial
│       └── prior_sinais.npz              pesos treinados (gerado pelo treinar_prior.py)
├── ferramentas/                          fora do Webots, roda no terminal
│   ├── coletar.py                        roda o Webots N vezes e guarda os dados
│   ├── treinar_prior.py                  treina a rede (numpy) e gera curvas e figuras
│   └── sim_grade.py                      bancada offline para testes rápidos
├── dados/
│   ├── mapas/                            um mapa_*.csv por execução (dados de treino)
│   └── resultados/                       resultado_avaliacao.txt de cada execução
└── resultados/                           curvas_treino.csv/png, comparacao_teste.csv, mapa_prior.png
```

Dependências: Python 3 e **numpy** (o controlador usa só numpy). O **matplotlib** é opcional e serve só para as figuras.

## Roteiro passo a passo

1. **Coloque os arquivos no lugar.** Copie `agente.py`, `prior_sinais.py` e o novo `grade_controller.py` para `controllers/grade_controller/`. Copie a pasta `ferramentas/` para a raiz de `grade_inteligente/`.

2. **Teste se nada quebrou.** Abra o mundo no Webots normalmente. Ainda não existe `prior_sinais.npz`, então o console deve mostrar `prior=não (otimista)` e o robô deve se comportar como antes.

3. **Teste o pipeline na bancada (5 minutos, sem Webots):**
   ```
   cd ferramentas
   python sim_grade.py coletar --n 60
   python treinar_prior.py --dados ../dados/mapas_sim
   python sim_grade.py avaliar --n 300
   ``` rede funcionam. Depois **apague** `prior_sinais.npz` e `dados/mapas_sim/`: a rede da entrega deve ser treinada com dados do Webots.

   Isso confirma que coleta, treino e uso da
4. **Colete dados reais no Webots:**
   ```
   python coletar.py --n 40
   ```
   Cada execução usa uma semente nova. O script liga o modo coleta, que usa ε = 0,2 e não usa o prior, para variar as rotas. Os mapas vistos ficam em `dados/mapas/`. Se o comando `webots` não estiver no PATH, use `--webots "<caminho do executável>"`. Também dá para coletar à mão: crie `modo_coleta.flag` na pasta do controlador, rode o Webots várias vezes e apague o arquivo no fim.

5. **Treine a rede:**
   ```
   python treinar_prior.py
   ```
   O script separa as execuções em treino, validação e teste (60/20/20, por execução). Ele treina com parada antecipada, compara a rede no teste com a frequência global e com a tabela por segmento, e salva `prior_sinais.npz` na pasta do controlador. As curvas e o mapa do prior vão para `resultados/`.

6. **Confira o que a rede aprendeu.** Olhe `resultados/mapa_prior.png`: os segmentos com sinal fixo devem aparecer escuros. Se alguma região estiver mal coberta, colete mais execuções (passo 4) e treine de novo.

7. **Escolha o peso do prior em sementes de validação** (na bancada, porque no Webots é lento):
   ```
   python sim_grade.py avaliar --n 300 --semente-ini 200000 --pesos 0.02,0.05,0.1,0.25,0.5,1
   ```
   Ajuste `PESO_PRIOR` em `prior_sinais.py` se outro valor for melhor. O teste final (passo 8) usa outras sementes.

8. **Meça no Webots, no modo de avaliação:**
   ```
   python coletar.py --n 10 --modo avaliacao
   ```
   Para comparar, renomeie `prior_sinais.npz` temporariamente e rode de novo. Assim o controlador volta ao otimista. Os `dados/resultados/resultado_*.txt` e o `resumo_execucoes.csv` (com a coluna `usa_prior`) são a evidência dos itens 2 e 5.

9. **Prepare a entrega.**
   - `prior_sinais.npz` **precisa** estar na pasta do controlador.
   - `modo_coleta.flag` **não pode** estar lá: com ele, o professor rodaria o modo coleta.
   - Declare numpy como dependência (item 6).

## Resultados na bancada (simulação, não Webots)

São 300 sementes de teste que nem a rede nem a escolha do peso viram. A bancada segue as regras do supervisor, mas não simula física, tempo nem timeout. Servem para decidir; as figuras do vídeo devem vir do Webots.

| Configuração | Tentativas (média) | Até 8 tentativas | Segmentos (média) | Sinais pisados |
| --- | --- | --- | --- | --- |
| Otimista (controlador atual) | 6,9 | 76% | 72 | 16,0 |
| Com prior da rede (peso 0,05) | 3,6 | 91% | 42 | 7,7 |

Rede × tabela de frequência por segmento como prior, nas mesmas sementes de teste:

| Execuções de treino | Tabela: tentativas | Rede: tentativas | Tabela: segmentos | Rede: segmentos |
| --- | --- | --- | --- | --- |
| 12 | 3,70 | 3,51 | 44,5 | 41,7 |
| 36 | 4,27 | 3,63 | 56,2 | 41,7 |

## O que a bancada mostrou (e já está corrigido no código)

- **Prior pesado demais deixa o robô tímido.** Com peso 1, a média passou de 120 segmentos: o robô andava em círculos sobre segmentos conhecidos em vez de arriscar um novo. O peso de 0,05 vence na validação. Com ele, a rede quase nunca justifica um desvio. O robô continua no caminho curto, mas, entre caminhos de mesmo tamanho, escolhe o que a rede considera mais seguro. Essa é a frase para explicar o papel da rede no vídeo.
- **O risco de reset saiu do prior.** Somá-lo ao valor inicial piorava o problema anterior. O Q-learning já aprende esse risco com `R_RESET`.
- **Bug no controlador antigo.** O supervisor penaliza cada sinal uma vez por tentativa. Quando o robô cruzava de novo um segmento com sinal na mesma tentativa, não chegava `pen_flag`, e o controlador gravava o segmento como livre. `agente.aprender_segmento` agora ignora essa passagem.
- **Cobertura dos dados.** Na coleta simulada, o robô quase nunca passou pelos dois sinais fixos da linha de baixo, então a rede não os aprendeu. Ela só aprende onde os dados chegam.

## Limites para dizer no vídeo

- O ganho existe porque o supervisor tem regularidade (sinais fixos e corredores livres nas bordas). Se os sinais fossem 100% aleatórios, a rede aprenderia "28% em todo lugar" e o comportamento quase não mudaria.
- Uma tabela de frequência por segmento faz algo parecido. A rede generaliza melhor com poucas execuções porque recebe atributos do segmento, não o identificador dele. Essa é a comparação do item 4.
- Uma seed como a 200065 da bancada é impossível para qualquer política: as duas saídas de A têm vermelho, então toda tentativa termina em reset.

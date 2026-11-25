"""
prior_sinais.py — rede neural que estima onde os sinais costumam ficar
======================================================================
MATA64 - Inteligência Artificial | UFBA | Semestre 2026.2

Papel no projeto
  O Q-learning aprende DENTRO de uma execução (o que o robô pisou).
  Esta rede aprende ENTRE execuções: a partir dos mapas de várias sementes,
  ela estima, para cada segmento, a probabilidade de ter sinal verde,
  amarelo, vermelho ou nenhum. Essa estimativa vira o valor inicial da
  Q-table, no lugar da inicialização otimista ("todo segmento é livre").

  A rede NÃO vê os sinais da execução atual: ela só usa o que foi
  aprendido em execuções anteriores (modelo pré-treinado). Na execução,
  cada sinal continua sendo conhecido só quando o robô passa por cima.

Arquitetura: MLP 7 -> 16 -> 16 -> 4, ReLU nas ocultas, softmax na saída.
  O treino fica em ferramentas/treinar_prior.py. Aqui só a passagem para a
  frente, em numpy puro: o controlador não precisa de torch/tensorflow.
"""

import os
import numpy as np

from agente import (GRID, B, CUSTO_PASSO, R_CHEGADA, N_FAIXAS,
                    acoes_validas, vizinho, segmento)

# Classes da saída da rede e a penalidade de cada uma
CLASSES  = ("nenhum", "verde", "amarelo", "vermelho")
PEN_CLS  = np.array([0.0, -10.0, -20.0, -50.0])
PEN_PARA_CLASSE = {0.0: 0, -10.0: 1, -20.0: 2, -50.0: 3}

ARQ_PESOS = "prior_sinais.npz"


# ── Representação do segmento (entrada da rede) ───────────────
def todos_segmentos():
    """Os 180 segmentos da grade 10x10 (90 horizontais + 90 verticais)."""
    segs = [("H", c, l) for l in range(GRID) for c in range(GRID - 1)]
    segs += [("V", c, l) for c in range(GRID) for l in range(GRID - 1)]
    return segs


def atributos(seg):
    """Vetor de 7 atributos de um segmento, todos entre 0 e 1.

    Só geometria da grade, nada sobre a execução atual:
      x, y do meio do segmento; se é horizontal; e se está em cada
      uma das quatro bordas (as bordas são onde a regularidade aparece).
    """
    o, c, l = seg
    if o == "H":
        xm, ym = c + 0.5, float(l)
    else:
        xm, ym = float(c), l + 0.5
    m = GRID - 1
    return [
        xm / m,
        ym / m,
        1.0 if o == "H" else 0.0,
        1.0 if (o == "V" and c == 0) else 0.0,       # borda esquerda (oeste)
        1.0 if (o == "V" and c == m) else 0.0,       # borda direita (leste)
        1.0 if (o == "H" and l == 0) else 0.0,       # borda de cima (norte)
        1.0 if (o == "H" and l == m) else 0.0,       # borda de baixo (sul)
    ]


N_ATRIBUTOS = 7


# ── A rede (passagem para a frente) ───────────────────────────
class RedeMLP:
    def __init__(self, pesos):
        self.p = {k: np.asarray(v, dtype=float) for k, v in pesos.items()}

    @staticmethod
    def inicial(n_oculta=16, semente=0):
        """Pesos pequenos e aleatórios (aula 18: pesos iguais travam a rede)."""
        rng = np.random.default_rng(semente)
        def he(n_in, n_out):
            return rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        return RedeMLP({
            "W1": he(N_ATRIBUTOS, n_oculta), "b1": np.zeros(n_oculta),
            "W2": he(n_oculta, n_oculta),    "b2": np.zeros(n_oculta),
            "W3": he(n_oculta, len(CLASSES)), "b3": np.zeros(len(CLASSES)),
        })

    def frente(self, X):
        """Retorna as probabilidades (n, 4) e os valores intermediários."""
        p = self.p
        z1 = X @ p["W1"] + p["b1"];  a1 = np.maximum(z1, 0.0)
        z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(z2, 0.0)
        z3 = a2 @ p["W3"] + p["b3"]
        z3 = z3 - z3.max(axis=1, keepdims=True)
        e = np.exp(z3)
        prob = e / e.sum(axis=1, keepdims=True)
        return prob, (X, z1, a1, z2, a2)

    def prever(self, X):
        return self.frente(np.asarray(X, dtype=float))[0]

    def salvar(self, caminho):
        np.savez(caminho, **self.p)

    @staticmethod
    def carregar(caminho):
        with np.load(caminho) as dados:
            return RedeMLP({k: dados[k] for k in dados.files})


def carregar_rede(pasta="."):
    """Carrega os pesos treinados; None se o arquivo não existir."""
    caminho = os.path.join(pasta, ARQ_PESOS)
    if not os.path.exists(caminho):
        return None
    return RedeMLP.carregar(caminho)


def probabilidades(rede):
    """Probabilidade de cada classe de sinal, para os 180 segmentos."""
    segs = todos_segmentos()
    P = rede.prever([atributos(s) for s in segs])
    return {s: P[i] for i, s in enumerate(segs)}


# ── Do prior à Q-table inicial ────────────────────────────────
# Peso do prior: 0 = otimista puro (como antes), 1 = penalidade esperada inteira.
# Escolhido na bancada com sementes de validação (ver README_rede.md).
# Com peso pequeno, a penalidade prevista de um segmento (no máximo 50 x 0,05
# = 2,5 pontos) quase nunca justifica um desvio: o robô continua indo pelo
# caminho curto, mas, entre caminhos de mesmo tamanho, escolhe o que a rede
# considera mais seguro. Com peso 1 o robô fica pessimista e dá voltas.
PESO_PRIOR = 0.05


def q_inicial_com_prior(prob_seg, peso=PESO_PRIOR, tol=1e-6, max_iter=1000):
    """Valor inicial de Q(col, lin, faixa, direção) a partir do prior.

    Reward esperado de um segmento ainda não visitado:
        -1 + peso * soma_k p_k * pen_k
    ou seja, o custo do passo mais uma fração da penalidade média prevista.
    O peso < 1 mantém parte do otimismo, que é o que faz o robô explorar
    (Sutton & Barto 2.6). O risco de reset fica com o Q-learning, que já o
    aprende com R_RESET; colocá-lo também aqui deixou o robô tímido demais.

    Uma iteração de valor sobre as 100 interseções calcula o melhor valor
    esperado até B. Com peso = 0 o resultado é exatamente a inicialização
    otimista de antes: 100 - (1 + distância até B).
    """
    estados = [(c, l) for c in range(GRID) for l in range(GRID) if (c, l) != B]
    r = {}
    for (c, l) in estados:
        for d in acoes_validas(c, l):
            p = prob_seg[segmento(c, l, d)]
            r[(c, l, d)] = CUSTO_PASSO + peso * float(p @ PEN_CLS)

    # começa do valor otimista (limite superior) e desce até o ponto fixo
    V = {(c, l): R_CHEGADA + CUSTO_PASSO * ((B[0] - c) + (B[1] - l))
         for (c, l) in estados}
    for _ in range(max_iter):
        delta = 0.0
        for (c, l) in estados:
            melhor = -np.inf
            for d in acoes_validas(c, l):
                nc, nl = vizinho(c, l, d)
                alvo = R_CHEGADA if (nc, nl) == B else V[(nc, nl)]
                melhor = max(melhor, r[(c, l, d)] + alvo)
            delta = max(delta, abs(melhor - V[(c, l)]))
            V[(c, l)] = melhor
        if delta < tol:
            break

    Q0 = {}
    for (c, l) in estados:
        for d in acoes_validas(c, l):
            nc, nl = vizinho(c, l, d)
            alvo = R_CHEGADA if (nc, nl) == B else V[(nc, nl)]
            for f in range(N_FAIXAS):            # o prior não depende da faixa
                Q0[(c, l, f, d)] = r[(c, l, d)] + alvo
    return Q0


def preparar_q_inicial(pasta=".", peso=PESO_PRIOR):
    """Atalho para o controlador: rede -> prior -> Q inicial (ou None)."""
    rede = carregar_rede(pasta)
    if rede is None:
        return None
    return q_inicial_com_prior(probabilidades(rede), peso)

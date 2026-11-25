"""
agente.py — Q-learning da grade, sem dependência do Webots
==========================================================
MATA64 - Inteligência Artificial | UFBA | Semestre 2026.2

É a mesma lógica que estava dentro do grade_controller.py, movida para
um módulo próprio. Assim ela pode ser usada:
  - pelo controlador no Webots (grade_controller.py);
  - pela bancada offline (ferramentas/sim_grade.py), para testar rápido.

Única mudança de comportamento: o valor inicial da Q-table pode vir de
fora (q_inicial). Sem ele, o agente usa a inicialização otimista de antes.
"""

import random

# ── Grade ─────────────────────────────────────────────────────
GRID = 10
B = (9, 9)
DELTA  = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}   # 0=L 1=S 2=O 3=N
NOME_D = {0: "Leste", 1: "Sul", 2: "Oeste", 3: "Norte"}

# ── Q-learning ────────────────────────────────────────────────
ALPHA       = 1.0     # sinais fixos durante a execução: uma passagem já dá o custo real
GAMMA       = 1.0     # episódio finito com custo por passo: não precisa descontar
CUSTO_PASSO = -1.0
R_CHEGADA   = 100.0
R_RESET     = -100.0  # cada volta a A custa uma tentativa e tempo
LIMIAR      = -60.0   # pontuação que faz o supervisor devolver o robô a A

# Faixas de orçamento: o que ainda cabe antes do reset em -60
#   0: pontuação 0          -> cabe até um vermelho
#   1: pontuação -10 a -30  -> cabe até um amarelo
#   2: pontuação -40        -> só cabe um verde
#   3: pontuação -50        -> nenhum sinal cabe
N_FAIXAS  = 4
REP_FAIXA = {0: 0.0, 1: -20.0, 2: -40.0, 3: -50.0}


def faixa(score):
    if score > -10: return 0
    if score > -40: return 1
    if score > -50: return 2
    return 3


def acoes_validas(col, lin):
    return [d for d in range(4)
            if 0 <= col + DELTA[d][0] < GRID
            and 0 <= lin + DELTA[d][1] < GRID]


def dist_b(col, lin):
    return (B[0] - col) + (B[1] - lin)


def vizinho(col, lin, d):
    return col + DELTA[d][0], lin + DELTA[d][1]


def segmento(col, lin, d):
    """Chave canônica do segmento, igual nos dois sentidos.

    ("H", c, l): segmento horizontal entre (c, l) e (c+1, l)
    ("V", c, l): segmento vertical   entre (c, l) e (c, l+1)
    col cresce para Leste, lin cresce para Sul (A = (0,0), B = (9,9)).
    """
    nc, nl = vizinho(col, lin, d)
    if d in (0, 2):
        return ("H", min(col, nc), lin)
    return ("V", col, min(lin, nl))


class AgenteQ:
    """Q-learning tabular com memória entre tentativas.

    State  = (coluna, linha, faixa de orçamento)
    Action = Leste, Sul, Oeste, Norte (só as que ficam dentro da grade)
    Reward = -1 por segmento + penalidade do sinal
             + 100 ao chegar em B, -100 quando o robô é devolvido a A.
    """

    def __init__(self, epsilon=0.0, q_inicial=None, semente=None):
        self.Q = {}            # (col, lin, faixa, direcao) -> valor
        self.mapa = {}         # segmento canônico -> penalidade (0 = livre)
        self.epsilon = epsilon
        self.q_inicial = q_inicial   # dict com o mesmo formato de Q, ou None
        self.rng = random.Random(semente)

    # ── Leitura da Q-table ────────────────────────────────────
    def q(self, col, lin, f, d):
        """Lê Q. Na primeira consulta usa o valor inicial.

        Sem q_inicial: valor otimista, como se o caminho até B estivesse livre.
        Com q_inicial: valor esperado calculado a partir da rede (prior).
        """
        k = (col, lin, f, d)
        if k not in self.Q:
            if self.q_inicial is not None:
                self.Q[k] = self.q_inicial[k]
            else:
                nc, nl = vizinho(col, lin, d)
                self.Q[k] = R_CHEGADA + CUSTO_PASSO * (1 + dist_b(nc, nl))
        return self.Q[k]

    def max_q(self, col, lin, f):
        if (col, lin) == B:
            return 0.0
        return max(self.q(col, lin, f, d) for d in acoes_validas(col, lin))

    # ── Política ──────────────────────────────────────────────
    def escolher(self, col, lin, f):
        """ε-greedy sobre a Q-table, com desempate aleatório."""
        validas = acoes_validas(col, lin)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(validas)
        melhor = max(self.q(col, lin, f, d) for d in validas)
        empatadas = [d for d in validas if self.q(col, lin, f, d) == melhor]
        return self.rng.choice(empatadas)

    # ── Aprendizado ───────────────────────────────────────────
    def atualizar(self, col, lin, d, pen, score_antes, resetou):
        """Atualização do Q-learning para o segmento (col, lin) -> direção d.

        A penalidade pertence ao segmento, não à faixa. Por isso o mesmo
        segmento é atualizado nas 4 faixas: na faixa real com a pontuação
        real, e nas outras com um valor representativo.
        score_antes=None atualiza só com valores representativos.
        """
        nc, nl = vizinho(col, lin, d)
        f_real = faixa(score_antes) if score_antes is not None else None
        for f in range(N_FAIXAS):
            s0 = score_antes if f == f_real else REP_FAIXA[f]
            novo = s0 + pen
            r = CUSTO_PASSO + pen
            if (f == f_real and resetou) or novo <= LIMIAR:
                r += R_RESET
                alvo = self.max_q(0, 0, 0)        # o episódio continua a partir de A
            elif (nc, nl) == B:
                r += R_CHEGADA
                alvo = 0.0                        # estado terminal
            else:
                alvo = self.max_q(nc, nl, faixa(novo))
            atual = self.q(col, lin, f, d)
            self.Q[(col, lin, f, d)] = atual + ALPHA * (r + GAMMA * alvo - atual)

    def aprender_segmento(self, col, lin, d, pen, score_antes, resetou):
        """Guarda o segmento no mapa e atualiza Q nos dois sentidos.

        O supervisor só penaliza um sinal UMA vez por tentativa. Se o robô
        cruza de novo um segmento com sinal conhecido na mesma tentativa,
        chega pen = 0, mas o sinal continua lá para as próximas tentativas.
        Nesse caso o mapa e a Q-table não são alterados.
        """
        seg = segmento(col, lin, d)
        if pen == 0.0 and self.mapa.get(seg, 0.0) != 0.0:
            return
        nc, nl = vizinho(col, lin, d)
        inv = (d + 2) % 4
        self.mapa[seg] = pen                     # o sinal vale nos dois sentidos
        self.atualizar(col, lin, d, pen, score_antes, resetou)
        if (nc, nl) != B:
            self.atualizar(nc, nl, inv, pen, None, False)

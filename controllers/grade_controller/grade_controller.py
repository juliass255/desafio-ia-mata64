"""
grade_controller.py — Q-learning com memória entre tentativas
=============================================================
MATA64 - Inteligência Artificial | UFBA | Semestre 2026.2
Desafio IA — Grade Inteligente

Base: controlador de movimento aleatório fornecido pelo professor.
A navegação (girar_para, navegar_para), a leitura dos sensores e a
comunicação com o supervisor foram mantidas sem alteração.

O que mudou:
  - escolher_direcao() usa uma Q-table (Aprendizado por Reforço).
  - State  = (coluna, linha, faixa de orçamento).
  - Actions = Leste, Sul, Oeste, Norte (só as que ficam dentro da grade).
  - Reward = -1 por segmento + penalidade do sinal
             + 100 ao chegar em B, -100 quando o robô é devolvido a A.
  - A Q-table começa otimista: cada ação vale o que valeria se o
    caminho até B estivesse livre. Por isso o robô já sai em direção
    a B e explora sozinho os segmentos que ainda não conhece.
  - A Q-table e o mapa de sinais ficam na memória durante toda a
    execução: o reset devolve o robô a A, mas não apaga o que aprendeu.
  - Cada segmento percorrido é gravado em CSV para a análise.

REGRA DO ENUNCIADO (respeitada aqui):
  A câmera NÃO é usada para avistar sinais à frente. As decisões
  dependem apenas das penalidades já recebidas.
"""

from controller import Robot
import math, os, random, csv

# ── Parâmetros de navegação (do professor) ────────────────────
MAX_V      = 15.0    # velocidade máxima das rodas (rad/s)
TOL_DIST   = 0.06   # tolerância de chegada na célula (m)
TOL_ANG    = 2.5    # tolerância de alinhamento (graus)
SPD_GIRO   = 2.5    # velocidade de giro no lugar (rad/s)

RESET_FILE = "reset_flag.txt"
PEN_FILE   = "pen_flag.txt"

# ── Parâmetros do Q-learning ──────────────────────────────────
ALPHA       = 1.0     # sinais fixos durante a execução: uma passagem já dá o custo real
GAMMA       = 1.0     # episódio finito com custo por passo: não precisa descontar
EPSILON     = 0.0     # a inicialização otimista já faz a exploração (Sutton & Barto, 2.6)
CUSTO_PASSO = -1.0
R_CHEGADA   = 100.0
R_RESET     = -100.0  # cada volta a A custa uma tentativa e tempo
LIMIAR      = -60.0   # pontuação que faz o supervisor devolver o robô a A
SINAIS      = (-10.0, -20.0, -50.0)
SEMENTE_AGENTE = None # fixe um inteiro para repetir os desempates nos testes

# ── Arquivos de registro (ficam na pasta do controlador) ──────
LOG_PASSOS = "log_passos.csv"
LOG_MAPA   = "mapa_sinais.csv"
LOG_RESUMO = "resumo_execucoes.csv"

# ── Constantes da grade ───────────────────────────────────────
GRID = 10
ORIG_X, ORIG_Y = -4.5, 4.5
B = (9, 9)

# Direções: 0=Leste 1=Sul 2=Oeste 3=Norte
DELTA  = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}
NOME_D = {0:"Leste", 1:"Sul", 2:"Oeste", 3:"Norte"}

# Ângulos do Compass calibrados para o Pioneer3dx neste mundo
ANG_DIR = {0:358.0, 1:268.0, 2:178.0, 3:88.0}

# ── Inicialização do robô ─────────────────────────────────────
robot    = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor  = robot.getDevice("left wheel")
right_motor = robot.getDevice("right wheel")
left_motor.setPosition(float("inf"));  left_motor.setVelocity(0)
right_motor.setPosition(float("inf")); right_motor.setVelocity(0)

gps     = robot.getDevice("gps");     gps.enable(timestep)
compass = robot.getDevice("compass"); compass.enable(timestep)
camera  = robot.getDevice("camera");  camera.enable(timestep)  # usada só para a gravação da demonstração

random.seed(SEMENTE_AGENTE)

# ── Funções utilitárias (do professor) ────────────────────────

def set_velocidade(esq, dir):
    left_motor.setVelocity( max(-MAX_V, min(MAX_V, esq)))
    right_motor.setVelocity(max(-MAX_V, min(MAX_V, dir)))

def parar():
    set_velocidade(0, 0)

def get_posicao():
    v = gps.getValues(); return v[0], v[1]

def get_angulo():
    v = compass.getValues()
    return math.degrees(math.atan2(v[0], v[1])) % 360

def posicao_para_grade(x, y):
    col = int(round((x + 4.5) / 1.0))
    lin = int(round((4.5 - y) / 1.0))
    return max(0, min(9, col)), max(0, min(9, lin))

def grade_para_posicao(col, lin):
    return -4.5 + col * 1.0, 4.5 - lin * 1.0

def verificar_reset():
    if os.path.exists(RESET_FILE):
        os.remove(RESET_FILE); return True
    return False

def verificar_penalidade():
    if os.path.exists(PEN_FILE):
        try:
            pen = float(open(PEN_FILE).read().strip())
            os.remove(PEN_FILE); return pen
        except:
            if os.path.exists(PEN_FILE): os.remove(PEN_FILE)
    return 0.0

# ── Controle de movimento (do professor) ──────────────────────

def erro_angular(alvo):
    e = alvo - get_angulo()
    if e >  180: e -= 360
    if e < -180: e += 360
    return e

def girar_para(direcao):
    """Gira no lugar até alinhar. Retorna True quando pronto."""
    e = erro_angular(ANG_DIR[direcao])
    if abs(e) < TOL_ANG:
        parar(); return True
    set_velocidade(-SPD_GIRO if e > 0 else  SPD_GIRO,
                    SPD_GIRO if e > 0 else -SPD_GIRO)
    return False

def navegar_para(alvo_x, alvo_y):
    """Move em linha reta até o alvo. Retorna True quando chegou."""
    x, y   = get_posicao()
    dx, dy = alvo_x - x, alvo_y - y
    if math.hypot(dx, dy) < TOL_DIST:
        parar(); return True
    ang_alvo = math.degrees(math.atan2(dy, dx)) % 360
    e = ang_alvo - get_angulo()
    if e >  180: e -= 360
    if e < -180: e += 360
    correcao = 0.06 * e * MAX_V
    set_velocidade(MAX_V - correcao, MAX_V + correcao)
    return False

# ════════════════════════════════════════════════════════════
# AGENTE DE APRENDIZADO POR REFORÇO (Q-learning)
# ════════════════════════════════════════════════════════════

# Faixas de orçamento: o que ainda cabe antes do reset em -60
#   0: pontuação 0          -> cabe até um vermelho
#   1: pontuação -10 a -30  -> cabe até um amarelo
#   2: pontuação -40        -> só cabe um verde
#   3: pontuação -50        -> nenhum sinal cabe
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

Q = {}      # (col, lin, faixa, direcao) -> valor
mapa = {}   # (col, lin, direcao) -> penalidade do segmento (0 = livre)

def q(col, lin, f, d):
    """Lê Q. Na primeira consulta, usa o valor otimista: caminho livre até B."""
    k = (col, lin, f, d)
    if k not in Q:
        nc, nl = col + DELTA[d][0], lin + DELTA[d][1]
        Q[k] = R_CHEGADA + CUSTO_PASSO * (1 + dist_b(nc, nl))
    return Q[k]

def max_q(col, lin, f):
    if (col, lin) == B:
        return 0.0
    return max(q(col, lin, f, d) for d in acoes_validas(col, lin))

def escolher_direcao(col, lin, f):
    """Política ε-greedy sobre a Q-table, com desempate aleatório."""
    validas = acoes_validas(col, lin)
    if random.random() < EPSILON:
        return random.choice(validas)
    melhor = max(q(col, lin, f, d) for d in validas)
    empatadas = [d for d in validas if q(col, lin, f, d) == melhor]
    return random.choice(empatadas)

def atualizar(col, lin, d, pen, score_antes, resetou):
    """Atualização do Q-learning para o segmento (col, lin) -> direção d.

    A penalidade pertence ao segmento, não à faixa. Por isso o mesmo
    segmento é atualizado nas 4 faixas: na faixa real com a pontuação
    real, e nas outras com um valor representativo. Assim o robô não
    precisa pisar no mesmo sinal 4 vezes para aprender que ele existe.
    score_antes=None atualiza só com valores representativos.
    """
    nc, nl = col + DELTA[d][0], lin + DELTA[d][1]
    f_real = faixa(score_antes) if score_antes is not None else None
    for f in range(4):
        s0 = score_antes if f == f_real else REP_FAIXA[f]
        novo = s0 + pen
        r = CUSTO_PASSO + pen
        if (f == f_real and resetou) or novo <= LIMIAR:
            r += R_RESET
            alvo = max_q(0, 0, 0)          # o episódio continua a partir de A
        elif (nc, nl) == B:
            r += R_CHEGADA
            alvo = 0.0                     # estado terminal
        else:
            alvo = max_q(nc, nl, faixa(novo))
        atual = q(col, lin, f, d)
        Q[(col, lin, f, d)] = atual + ALPHA * (r + GAMMA * alvo - atual)

def aprender_segmento(col, lin, d, pen, score_antes, resetou):
    """Guarda o segmento no mapa e atualiza Q nos dois sentidos."""
    nc, nl = col + DELTA[d][0], lin + DELTA[d][1]
    inv = (d + 2) % 4
    mapa[(col, lin, d)] = pen
    mapa[(nc, nl, inv)] = pen            # o sinal fica no segmento, vale nos dois sentidos
    atualizar(col, lin, d, pen, score_antes, resetou)
    if (nc, nl) != B:
        atualizar(nc, nl, inv, pen, None, False)

# ── Registro em CSV (dados para gráficos e análise) ───────────

arq_passos = open(LOG_PASSOS, "w", newline="")
log = csv.writer(arq_passos)
log.writerow(["tentativa", "tempo_s", "col", "lin", "direcao",
              "alvo_col", "alvo_lin", "pen_segmento", "score_tentativa",
              "faixa", "resetou"])

def salvar_mapa():
    with open(LOG_MAPA, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["col", "lin", "direcao", "penalidade"])
        for (c, l, d), p in sorted(mapa.items()):
            w.writerow([c, l, NOME_D[d], p])

def salvar_resumo(chegou):
    novo = not os.path.exists(LOG_RESUMO)
    with open(LOG_RESUMO, "a", newline="") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["tentativas", "tempo_total_s", "score_final",
                        "segmentos_percorridos", "chegou_B",
                        "alpha", "gamma", "epsilon", "r_reset"])
        w.writerow([tentativa, round(robot.getTime(), 1), pontuacao,
                    total_segmentos, chegou, ALPHA, GAMMA, EPSILON, R_RESET])

def registrar_segmento(resetou):
    """Fecha o segmento em andamento: aprende e grava no log."""
    global total_segmentos
    aprender_segmento(col, lin, dir_alvo, pen_segmento, score_antes, resetou)
    log.writerow([tentativa, round(robot.getTime(), 1), col, lin,
                  NOME_D[dir_alvo], alvo_col, alvo_lin, pen_segmento,
                  pontuacao, faixa(score_antes), int(resetou)])
    arq_passos.flush()
    total_segmentos += 1

# ── Estado do agente ──────────────────────────────────────────
col, lin  = 0, 0
direcao   = 1          # Sul inicialmente
fase      = "decidindo"
dir_alvo  = 1
alvo_col  = 0; alvo_lin = 0
alvo_x    = 0.0; alvo_y = 0.0
pontuacao = 0.0        # pontuação da tentativa atual (zera no reset)
score_antes  = 0.0     # pontuação no início do segmento em andamento
pen_segmento = 0.0     # soma dos sinais pisados no segmento em andamento
tentativa = 1
total_segmentos = 0

print("=" * 55)
print("  Desafio IA - MATA64 | Q-learning")
print("=" * 55 + "\n")

# ── Loop principal ────────────────────────────────────────────
while robot.step(timestep) != -1:
    x, y = get_posicao()

    # 1. Penalidade? Lida ANTES do reset: o sinal que causou o reset
    #    precisa ser atribuído ao segmento em que o robô estava.
    pen = verificar_penalidade()
    if pen != 0.0:
        pontuacao += pen
        if pen in SINAIS:              # a penalidade de timeout (-30) não é um sinal
            pen_segmento += pen
        print(f"  pen={pen:.0f} | score={pontuacao:.0f}")

    # 2. Reset pelo supervisor? O episódio continua; a Q-table fica.
    if verificar_reset():
        parar()
        if fase in ("girando", "movendo"):
            registrar_segmento(resetou=True)
        salvar_mapa()
        col, lin = 0, 0; direcao = 1; fase = "decidindo"
        pontuacao = 0.0; pen_segmento = 0.0
        tentativa += 1
        print(f"  Voltou para A — tentativa {tentativa}\n")
        continue

    # 3. Máquina de estados
    if fase == "decidindo":
        col, lin = posicao_para_grade(x, y)
        d = escolher_direcao(col, lin, faixa(pontuacao))
        dc, dl   = DELTA[d]
        alvo_col, alvo_lin = col + dc, lin + dl
        alvo_x, alvo_y = grade_para_posicao(alvo_col, alvo_lin)
        dir_alvo = d
        score_antes = pontuacao
        pen_segmento = 0.0
        print(f"  [{col},{lin}] -> {NOME_D[d]} -> [{alvo_col},{alvo_lin}]"
              f" | score={pontuacao:.0f} tentativa={tentativa}")
        fase = "girando" if d != direcao else "movendo"

    elif fase == "girando":
        if girar_para(dir_alvo):
            direcao = dir_alvo; fase = "movendo"

    elif fase == "movendo":
        if navegar_para(alvo_x, alvo_y):
            registrar_segmento(resetou=False)
            col, lin = alvo_col, alvo_lin
            fase = "decidindo"

            # 4. Chegou em B? Só conta ao chegar na interseção (dentro de
            #    TOL_DIST = 0,22 m), dentro do raio de 0,35 m do supervisor.
            if (col, lin) == B:
                parar()
                pontuacao += R_CHEGADA
                salvar_mapa()
                salvar_resumo(chegou=1)
                arq_passos.close()
                print(f"\n{'='*55}")
                print(f"  CHEGOU EM B! Tentativa {tentativa} | "
                      f"tempo {robot.getTime():.1f} s | score {pontuacao:.0f}")
                print(f"{'='*55}\n")
                break
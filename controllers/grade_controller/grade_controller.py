"""
grade_controller.py — Q-learning com memória entre tentativas + prior neural
============================================================================
MATA64 - Inteligência Artificial | UFBA | Semestre 2026.2
Desafio IA — Grade Inteligente

Base: controlador de movimento aleatório fornecido pelo professor.
A navegação (girar_para, navegar_para), a leitura dos sensores e a
comunicação com o supervisor foram mantidas sem alteração.

Organização:
  - agente.py        Q-learning (state, actions, reward, atualização).
  - prior_sinais.py  Rede neural (MLP) treinada com execuções anteriores.
                     Estima onde os sinais costumam ficar e define o valor
                     inicial da Q-table. Sem o arquivo prior_sinais.npz, o
                     agente volta à inicialização otimista de antes.
  - este arquivo     Webots: sensores, motores, supervisor, registros.

Modos (escolhidos pela variável de ambiente GRADE_MODO ou pelo arquivo
modo_coleta.flag nesta pasta):
  - "avaliacao" (padrão, é o que o professor roda): ε = 0, usa o prior.
  - "coleta": ε = 0,2 e sem prior, para variar as rotas e cobrir mais
    segmentos nos dados de treino da rede.

REGRA DO ENUNCIADO (respeitada aqui):
  A câmera NÃO é usada para avistar sinais à frente. As decisões dependem
  das penalidades já recebidas nesta execução e do prior aprendido em
  execuções anteriores (modelo pré-treinado).
"""

from controller import Robot
import math, os, csv, time

import agente as ag
from agente import AgenteQ, B, DELTA, NOME_D, faixa, segmento
import prior_sinais

# ── Parâmetros de navegação (do professor) ────────────────────
MAX_V      = 5.0    # velocidade máxima das rodas (rad/s)
TOL_DIST   = 0.22   # tolerância de chegada na célula (m)
TOL_ANG    = 2.5    # tolerância de alinhamento (graus)
SPD_GIRO   = 2.5    # velocidade de giro no lugar (rad/s)

RESET_FILE = "reset_flag.txt"
PEN_FILE   = "pen_flag.txt"
FIM_FILE   = "fim_execucao.txt"      # avisa o ferramentas/coletar.py
SINAIS     = (-10.0, -20.0, -50.0)

# ── Modo de execução ──────────────────────────────────────────
MODO = os.environ.get("GRADE_MODO", "").strip().lower()
if not MODO:
    MODO = "coleta" if os.path.exists("modo_coleta.flag") else "avaliacao"
EPSILON_COLETA = 0.2
SEMENTE_AGENTE = None   # fixe um inteiro para repetir os desempates nos testes

# ── Arquivos de registro ──────────────────────────────────────
LOG_PASSOS = "log_passos.csv"
LOG_MAPA   = "mapa_sinais.csv"
LOG_RESUMO = "resumo_execucoes.csv"
PASTA_MAPAS = os.path.join("..", "..", "dados", "mapas")   # dados de treino da rede

# Ângulos do Compass calibrados para o Pioneer3dx neste mundo
ANG_DIR = {0: 358.0, 1: 268.0, 2: 178.0, 3: 88.0}

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

# ── Agente ────────────────────────────────────────────────────
if MODO == "coleta":
    q0 = None
    agente = AgenteQ(epsilon=EPSILON_COLETA, q_inicial=None, semente=SEMENTE_AGENTE)
else:
    q0 = prior_sinais.preparar_q_inicial(".")
    agente = AgenteQ(epsilon=0.0, q_inicial=q0, semente=SEMENTE_AGENTE)
USA_PRIOR = q0 is not None

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

# ── Registro em CSV (dados para gráficos, análise e treino) ───

if os.path.exists(FIM_FILE):
    os.remove(FIM_FILE)

arq_passos = open(LOG_PASSOS, "w", newline="")
log = csv.writer(arq_passos)
log.writerow(["tentativa", "tempo_s", "col", "lin", "direcao",
              "alvo_col", "alvo_lin", "pen_segmento", "score_tentativa",
              "faixa", "resetou"])

ID_EXECUCAO = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"

def salvar_mapa():
    """mapa_sinais.csv (desta execução) + cópia em dados/mapas para o treino."""
    linhas = sorted(agente.mapa.items())
    with open(LOG_MAPA, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["orientacao", "col", "lin", "penalidade"])
        for (o, c, l), p in linhas:
            w.writerow([o, c, l, p])
    try:
        os.makedirs(PASTA_MAPAS, exist_ok=True)
        with open(os.path.join(PASTA_MAPAS, f"mapa_{ID_EXECUCAO}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["orientacao", "col", "lin", "penalidade"])
            for (o, c, l), p in linhas:
                w.writerow([o, c, l, p])
    except OSError as e:
        print(f"  [aviso] não salvou o mapa em {PASTA_MAPAS}: {e}")

CAB_RESUMO = ["id_execucao", "modo", "usa_prior", "tentativas", "tempo_total_s",
              "score_final", "segmentos_percorridos", "chegou_B",
              "alpha", "gamma", "epsilon", "r_reset"]

def salvar_resumo(chegou):
    if os.path.exists(LOG_RESUMO):
        with open(LOG_RESUMO, newline="") as f:
            cab = next(csv.reader(f), [])
        if cab != CAB_RESUMO:                       # formato antigo: guarda e recomeça
            os.replace(LOG_RESUMO, LOG_RESUMO + ".antigo")
    novo = not os.path.exists(LOG_RESUMO)
    with open(LOG_RESUMO, "a", newline="") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(CAB_RESUMO)
        w.writerow([ID_EXECUCAO, MODO, int(USA_PRIOR), tentativa,
                    round(robot.getTime(), 1), pontuacao, total_segmentos, chegou,
                    ag.ALPHA, ag.GAMMA, agente.epsilon, ag.R_RESET])

def registrar_segmento(resetou):
    """Fecha o segmento em andamento: aprende e grava no log."""
    global total_segmentos
    agente.aprender_segmento(col, lin, dir_alvo, pen_segmento, score_antes, resetou)
    log.writerow([tentativa, round(robot.getTime(), 1), col, lin,
                  NOME_D[dir_alvo], alvo_col, alvo_lin, pen_segmento,
                  pontuacao, faixa(score_antes), int(resetou)])
    arq_passos.flush()
    total_segmentos += 1

# ── Estado do controlador ─────────────────────────────────────
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
print("  Desafio IA - MATA64 | Q-learning + prior neural")
print(f"  modo={MODO} | prior={'sim' if USA_PRIOR else 'não (otimista)'}")
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
        d = agente.escolher(col, lin, faixa(pontuacao))
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
                pontuacao += ag.R_CHEGADA
                salvar_mapa()
                salvar_resumo(chegou=1)
                arq_passos.close()
                with open(FIM_FILE, "w") as f:
                    f.write(ID_EXECUCAO)
                print(f"\n{'='*55}")
                print(f"  CHEGOU EM B! Tentativa {tentativa} | "
                      f"tempo {robot.getTime():.1f} s | score {pontuacao:.0f}")
                print(f"{'='*55}\n")
                break

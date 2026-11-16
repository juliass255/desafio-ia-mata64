"""
grade_supervisor.py - Grade Inteligente - MATA64
Monitora pontuacao, sinais coloridos e comunica
penalidades ao controlador via arquivo.
Registra metricas de avaliacao em arquivo ao final.
"""
from controller import Supervisor
import random, math, time, os

LIMIAR_RETORNO = -60
RECOMPENSA_B   = 100
TEMPO_MAX      = 180
RAIO_B         = 0.35
RAIO_SINAL     = 0.25

PENALIDADE = {"verde":-10, "amarelo":-20, "vermelho":-50}
COR_RGB    = {
    "verde":   [0.1, 0.85, 0.2],
    "amarelo": [0.95,0.85,0.05],
    "vermelho":[0.9, 0.1, 0.1],
}
POS_A = [-4.5, 4.5, 0.0]
ROT_A = [0.0, 0.0, 1.0, -1.5708]
POS_B = (4.5,-4.5)

TODOS_SINAIS = [
    "SV_8_6", "SH_8_7", "SH_2_2", "SV_9_6", "SH_8_5", "SH_8_6",
    "SH_3_1", "SH_5_7", "SV_8_8", "SV_6_3", "SV_7_7", "SV_3_7",
    "SH_6_0", "SH_3_4", "SV_2_6", "SH_4_6", "SV_6_4", "SH_3_6",
    "SH_3_8", "SH_4_3", "SH_8_8", "SH_2_1", "SH_4_5", "SV_1_1",
    "SH_0_4", "SV_9_7", "SV_9_5", "SV_9_2", "SV_2_4", "SV_9_0",
    "SV_7_2", "SH_4_8", "SV_6_1", "SH_2_6", "SH_2_7", "SH_5_0",
    "SH_7_4", "SH_0_5", "SH_0_3", "SH_7_0", "SH_8_4", "SV_8_5",
    "SH_4_1", "SH_6_1", "SV_4_0", "SH_5_3", "SV_6_7", "SH_1_4",
    "SH_5_6", "SH_0_6",
]

RESET_FILE   = "../grade_controller/reset_flag.txt"
PEN_FILE     = "../grade_controller/pen_flag.txt"
LOG_FILE     = "../grade_controller/resultado_avaliacao.txt"

import os, sys

def caminho(f):
    """
    Resolve o caminho do arquivo de forma robusta
    tanto no Windows quanto no Linux/Mac.
    Tenta o caminho relativo primeiro; se falhar,
    usa a pasta do proprio supervisor.
    """
    # Tenta relativo (estrutura padrao do Webots)
    rel = os.path.join(os.path.dirname(__file__), "..", "grade_controller",
                       os.path.basename(f))
    rel = os.path.normpath(rel)
    return rel

sv   = Supervisor(); ts = int(sv.getBasicTimeStep())
robo = sv.getFromDef("ROBO")
trans= robo.getField("translation")
rot  = robo.getField("rotation")

painel = sv.getDevice("painel")   # Display do proprio Supervisor
PW, PH = 200, 80

# ── Semente de randomização ───────────────────────────────────
# Para avaliacao padronizada, defina a semente abaixo.
# None = semente aleatoria (desenvolvimento normal do aluno)
SEMENTE = None
if SEMENTE is None:
    SEMENTE = random.randint(0, 999999)
random.seed(SEMENTE)
print(f"  Semente de randomizacao: {SEMENTE}")

# ── Métricas de avaliação ────────────────────────────────────
tempo_total   = 0.0    # tempo acumulado desde a 1a tentativa (nao zera)
tempo_ep      = 0.0    # tempo da tentativa atual (zera ao voltar A)
pontuacao     = 0.0    # pontuacao da tentativa atual
tentativas    = 0      # numero de reposicionamentos em A
chegou_B      = False  # se o robo chegou ao ponto B
score_final   = 0.0    # pontuacao da tentativa em que chegou em B

# Contadores de sinais pisados (acumulado em todas as tentativas)
sinais_pisados = {"verde": 0, "amarelo": 0, "vermelho": 0}

cores_ativas  = {}
sinais_visit  = set()
retornando    = False
passos_ret    = 0
ESPERA_RET    = 60
episodio      = 1
ultimo_seg    = POS_A[:2]

N_VERDE    = 16
N_AMARELO  = 16
N_VERMELHO = 18

# ── Painel visual ─────────────────────────────────────────────
def desenhar_painel(tempo, score):
    try:
        restante = max(0.0, float(TEMPO_MAX) - float(tempo))
        minutos  = int(tempo) // 60
        segundos = int(tempo) % 60
        painel.setFont("Arial", 10, True)
        painel.setColor(0x000000)
        painel.fillRectangle(0, 0, 199, 79)
        painel.setColor(0x6666FF)
        painel.drawText("Desafio IA - MATA64", 4, 4)
        if restante > 60:   painel.setColor(0x00FF88)
        elif restante > 30: painel.setColor(0xFFCC00)
        else:               painel.setColor(0xFF4444)
        painel.drawText(f"Tempo: {minutos:02d}:{segundos:02d}", 4, 22)
        if score < -30:   painel.setColor(0xFF4444)
        elif score < 0:   painel.setColor(0xFFCC00)
        else:             painel.setColor(0x00FF88)
        painel.drawText(f"Score: {score:+.0f} pts", 4, 42)
        painel.setColor(0xFF8800)
        painel.drawText(f"Limite: {LIMIAR_RETORNO}  Restam: {int(restante)}s", 4, 62)
    except Exception as e:
        print(f"  [painel] erro: {e}")

# ── Registro de avaliação ─────────────────────────────────────
def registrar_resultado():
    """
    Grava resultado_avaliacao.txt com todas as metricas
    necessarias para aferir os itens 1 e 2 do barema.
    """
    linhas = [
        "=" * 50,
        "  RESULTADO DA AVALIACAO — MATA64 2026.2",
        "=" * 50,
        f"  Semente de randomizacao : {SEMENTE}",
        f"  Chegou ao ponto B       : {'SIM' if chegou_B else 'NAO'}",
        "-" * 50,
        "  ITEM 1 — Chegada",
        f"  Tentativas realizadas   : {tentativas}",
        f"  Pontuacao final (ult.   ",
        f"  tentativa ao chegar B)  : {score_final:+.1f} pts",
        "-" * 50,
        "  ITEM 2 — Eficiencia",
        f"  Tempo total ate B       : {tempo_total:.1f} s  "
        f"({int(tempo_total)//60:02d}:{int(tempo_total)%60:02d})",
        f"  Num. reposicionamentos  : {tentativas}",
        "-" * 50,
        "  SINAIS PISADOS (todas as tentativas)",
        f"  Verde    (-10 pts) : {sinais_pisados['verde']:3d} "
        f"= {sinais_pisados['verde']*10:4d} pts de penalidade",
        f"  Amarelo  (-20 pts) : {sinais_pisados['amarelo']:3d} "
        f"= {sinais_pisados['amarelo']*20:4d} pts de penalidade",
        f"  Vermelho (-50 pts) : {sinais_pisados['vermelho']:3d} "
        f"= {sinais_pisados['vermelho']*50:4d} pts de penalidade",
        f"  Total de sinais    : {sum(sinais_pisados.values()):3d}",
        "=" * 50,
    ]
    texto = "\n".join(linhas)
    print("\n" + texto)
    with open(caminho(LOG_FILE), "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(f"\n  Resultado salvo em: {caminho(LOG_FILE)}")

# ── Posições originais dos sinais ─────────────────────────────
pos_orig = {}
for d in TODOS_SINAIS:
    n = sv.getFromDef(d)
    if n: pos_orig[d] = list(n.getField("translation").getSFVec3f())

# ── Funções de navegação ──────────────────────────────────────
def gerar_posicoes_aleatorias():
    ORIG_X=-4.5; ORIG_Y=4.5; CELL=1.0
    posicoes = []
    for lin in range(9):
        for col in range(9):
            posicoes.append((ORIG_X+col*CELL+0.5, ORIG_Y-lin*CELL))
    for col in range(1, 10):
        for lin in range(9):
            posicoes.append((ORIG_X+col*CELL, ORIG_Y-lin*CELL-0.5))
    return posicoes

def randomizar():
    global cores_ativas
    cores_ativas = {}
    total = N_VERDE + N_AMARELO + N_VERMELHO   # 16+16+18 = 50

    # ── 4 posicoes reservadas no meio das bordas esq e inf ────────
    # Borda esquerda col=0 (segmentos verticais): x=-4.5, y=4.5-lin-0.5
    # Borda inferior lin=9 (segmentos horizontais): y=-4.5, x=-4.5+col+0.5
    POS_BORDA  = [(-4.5, 4.0), (-4.5, 3.0), (-4.0, -4.5), (-3.0, -4.5)]
    CORES_BORDA = ["vermelho", "amarelo", "vermelho", "amarelo"]

    # ── Restante totalmente aleatorio ─────────────────────────────
    pool = [p for p in gerar_posicoes_aleatorias() if p not in set(POS_BORDA)]
    pos_alea = random.sample(pool, total - len(POS_BORDA))   # 46 posicoes

    # Cores restantes descontando as 2VM+2AM ja usadas nas bordas
    cores_alea = (["verde"] * N_VERDE +
                  ["amarelo"] * (N_AMARELO - 2) +
                  ["vermelho"] * (N_VERMELHO - 2))
    random.shuffle(cores_alea)

    # Junta tudo: bordas + aleatorios
    pos_final  = POS_BORDA + pos_alea
    cor_final  = CORES_BORDA + cores_alea

    sinais_usados = TODOS_SINAIS[:total]
    for i, def_nome in enumerate(sinais_usados):
        n = sv.getFromDef(def_nome)
        if not n: continue
        x, y = pos_final[i]
        cor   = cor_final[i]
        cores_ativas[def_nome] = cor
        shape = n.getField("children").getMFNode(0)
        shape.getField("appearance").getSFNode()\
             .getField("baseColor").setSFColor(COR_RGB[cor])
        n.getField("translation").setSFVec3f([x, y, 0.003])

    for def_nome in TODOS_SINAIS[total:]:
        n = sv.getFromDef(def_nome)
        if n:
            t = n.getField("translation").getSFVec3f()
            n.getField("translation").setSFVec3f([t[0], t[1], -1.0])

    print(f"  Sinais: 🟢{N_VERDE} 🟡{N_AMARELO} 🔴{N_VERMELHO} (total {total})"
          f" — bordas garantidas + restantes aleatorios")

def voltar_A():
    global retornando, passos_ret, tentativas
    trans.setSFVec3f(POS_A); rot.setSFRotation(ROT_A)
    robo.resetPhysics()
    retornando = True; passos_ret = 0; tentativas += 1
    with open(caminho(RESET_FILE), "w") as f: f.write("reset")
    print(f"  ↩  Volta para A | Score: {pontuacao:.1f} "
          f"| Tentativa #{tentativas} | Tempo total: {tempo_total:.0f}s")

def dist_B():
    p = trans.getSFVec3f()
    return math.hypot(p[0]-POS_B[0], p[1]-POS_B[1])

def checar_sinal():
    p = trans.getSFVec3f()
    for d, cor in cores_ativas.items():
        if d in sinais_visit: continue
        n = sv.getFromDef(d)
        if not n: continue
        t = n.getField("translation").getSFVec3f()
        if math.hypot(p[0]-t[0], p[1]-t[1]) < RAIO_SINAL:
            return PENALIDADE[cor], cor, d
    return 0, None, None

def notificar_pen(pen):
    with open(caminho(PEN_FILE), "w") as f: f.write(str(pen))

# ── Início ────────────────────────────────────────────────────
print("\n" + "="*55)
print("  Grade Inteligente - Supervisor iniciado")
print("="*55)
randomizar()

# ── Loop principal ────────────────────────────────────────────
while sv.step(ts) != -1:
    tempo_ep    += ts / 1000.0
    tempo_total += ts / 1000.0   # tempo total nunca zera
    pos = trans.getSFVec3f()

    # Painel visual
    if int(tempo_ep * 1000) % (ts * 10) < ts:
        try:
            desenhar_painel(tempo_ep, pontuacao)
        except:
            pass

    # Aguarda estabilizacao apos retorno
    if retornando:
        passos_ret += 1
        if passos_ret >= ESPERA_RET:
            retornando = False
            pontuacao  = 0.0
            tempo_ep   = 0.0
            sinais_visit.clear()
            ultimo_seg = POS_A[:2]
            print(f"  ✅ Estabilizado em A — tentativa {tentativas+1} iniciando\n")
        continue

    # Chegou em B?
    if dist_B() < RAIO_B:
        pontuacao  += RECOMPENSA_B
        chegou_B    = True
        score_final = pontuacao
        print(f"\n  🏁 Chegou em B! Ep {episodio} | "
              f"Score: {pontuacao:+.1f} | "
              f"Tempo total: {tempo_total:.1f}s | "
              f"Tentativas: {tentativas}")
        try:
            painel.setFont("Arial", 10, True)
            painel.setColor(0x000000)
            painel.fillRectangle(0, 0, 199, 79)
            painel.setColor(0x6666FF)
            painel.drawText("Desafio IA - MATA64", 4, 4)
            painel.setColor(0x00FF88)
            minutos  = int(tempo_total) // 60
            segundos = int(tempo_total) % 60
            painel.drawText(f"Total: {minutos:02d}:{segundos:02d}", 4, 22)
            painel.drawText(f"Score: {pontuacao:+.0f} pts", 4, 42)
            painel.setColor(0xFF8800)
            painel.drawText("CHEGOU EM B! Parabens!", 4, 62)
        except:
            pass
        registrar_resultado()
        episodio += 1
        break

    # Timeout?
    if tempo_ep >= TEMPO_MAX:
        pontuacao -= 30
        print(f"  ⏱  Timeout | Score: {pontuacao:.1f} | "
              f"Tempo total: {tempo_total:.1f}s")
        voltar_A()
        continue

    # Sinal colorido?
    pen, cor, d = checar_sinal()
    if pen:
        pontuacao += pen
        sinais_visit.add(d)
        sinais_pisados[cor] += 1   # acumula em todas as tentativas
        e = {"verde":"🟢","amarelo":"🟡","vermelho":"🔴"}[cor]
        print(f"  {e} {cor:8s} {pen:4d} | Score: {pontuacao:.1f} "
              f"| {cor.capitalize()}s pisados: {sinais_pisados[cor]}")
        notificar_pen(pen)

    dx = pos[0]-ultimo_seg[0]; dy = pos[1]-ultimo_seg[1]
    if math.hypot(dx, dy) >= 1.0:
        ultimo_seg = [pos[0], pos[1]]

    if pontuacao <= LIMIAR_RETORNO:
        voltar_A()

# Se nao chegou em B, registra mesmo assim
if not chegou_B:
    registrar_resultado()

"""
sim_grade.py — bancada offline (sem Webots) para testar o agente e a rede
=========================================================================
MATA64 - Inteligência Artificial | UFBA | Semestre 2026.2

O QUE É: uma simulação no nível de segmentos (sem física) que segue as
regras públicas do grade_supervisor.py: 50 sinais (16 verdes, 16 amarelos,
18 vermelhos), sorteio igual ao do supervisor, reset em -60, +100 em B.
O agente é exatamente o mesmo (agente.py) usado no Webots.

PARA QUE SERVE:
  - testar o pipeline (coleta -> treino -> prior) em minutos;
  - comparar configurações em centenas de sementes (itens 4 e 5).

O QUE NÃO É: não substitui as execuções no Webots. O tempo de cada
segmento, o timeout de 180 s e erros de navegação não estão aqui.
As figuras da apresentação devem vir das execuções reais.

Uso (a partir desta pasta):
  python sim_grade.py coletar --n 60 --saida ../dados/mapas_sim
  python sim_grade.py avaliar --n 300 --semente-ini 200000 --pesos 0.1,0.25,0.5,1   (validação)
  python sim_grade.py avaliar --n 300 --semente-ini 100000                          (teste)
"""

import argparse, csv, os, random, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "controllers", "grade_controller"))

from agente import AgenteQ, B, LIMIAR, faixa, segmento, vizinho   # noqa: E402

N_VERDE, N_AMARELO, N_VERMELHO = 16, 16, 18
PEN = {"verde": -10.0, "amarelo": -20.0, "vermelho": -50.0}


def gerar_mapa(semente):
    """Sorteia os sinais como o randomizar() do supervisor.

    Devolve {segmento canônico: penalidade} só com os segmentos que têm sinal.
    """
    rng = random.Random(semente)
    ORIG_X, ORIG_Y = -4.5, 4.5
    posicoes = []
    for lin in range(9):
        for col in range(9):
            posicoes.append((ORIG_X + col + 0.5, ORIG_Y - lin))
    for col in range(1, 10):
        for lin in range(9):
            posicoes.append((ORIG_X + col, ORIG_Y - lin - 0.5))

    pos_borda  = [(-4.5, 4.0), (-4.5, 3.0), (-4.0, -4.5), (-3.0, -4.5)]
    cores_borda = ["vermelho", "amarelo", "vermelho", "amarelo"]
    total = N_VERDE + N_AMARELO + N_VERMELHO
    pool = [p for p in posicoes if p not in set(pos_borda)]
    pos_alea = rng.sample(pool, total - len(pos_borda))
    cores_alea = (["verde"] * N_VERDE + ["amarelo"] * (N_AMARELO - 2)
                  + ["vermelho"] * (N_VERMELHO - 2))
    rng.shuffle(cores_alea)

    mapa = {}
    for (x, y), cor in zip(pos_borda + pos_alea, cores_borda + cores_alea):
        c_f, l_f = x + 4.5, 4.5 - y
        if abs(c_f - round(c_f)) > 0.25:          # meio horizontal: x = c + 0,5
            seg = ("H", int(round(c_f - 0.5)), int(round(l_f)))
        else:                                     # meio vertical: y = l + 0,5
            seg = ("V", int(round(c_f)), int(round(l_f - 0.5)))
        mapa[seg] = PEN[cor]
    return mapa


def simular(mapa, agente, max_segmentos=3000):
    """Roda um episódio até B. Devolve as métricas do supervisor."""
    col, lin, score = 0, 0, 0.0
    tentativas, segmentos = 1, 0
    pisados = {-10.0: 0, -20.0: 0, -50.0: 0}
    visitados = set()          # como no supervisor: cada sinal penaliza 1 vez por tentativa
    while segmentos < max_segmentos:
        d = agente.escolher(col, lin, faixa(score))
        seg = segmento(col, lin, d)
        pen = 0.0 if seg in visitados else mapa.get(seg, 0.0)
        if pen:
            visitados.add(seg)
        score_antes = score
        score += pen
        if pen:
            pisados[pen] += 1
        resetou = score <= LIMIAR
        agente.aprender_segmento(col, lin, d, pen, score_antes, resetou)
        segmentos += 1
        if resetou:
            tentativas += 1
            col, lin, score = 0, 0, 0.0
            visitados.clear()
            continue
        col, lin = vizinho(col, lin, d)
        if (col, lin) == B:
            return {"chegou": 1, "tentativas": tentativas, "segmentos": segmentos,
                    "score_final": score + 100.0,
                    "verdes": pisados[-10.0], "amarelos": pisados[-20.0],
                    "vermelhos": pisados[-50.0]}
    return {"chegou": 0, "tentativas": tentativas, "segmentos": segmentos,
            "score_final": float("nan"), "verdes": pisados[-10.0],
            "amarelos": pisados[-20.0], "vermelhos": pisados[-50.0]}


def salvar_mapa_visto(agente, caminho):
    """Mesmo formato que o controlador grava em dados/mapas."""
    with open(caminho, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["orientacao", "col", "lin", "penalidade"])
        for (o, c, l), p in sorted(agente.mapa.items()):
            w.writerow([o, c, l, p])


def cmd_coletar(a):
    os.makedirs(a.saida, exist_ok=True)
    for i in range(a.n):
        semente = a.semente_ini + i
        ag = AgenteQ(epsilon=a.epsilon, semente=semente)
        simular(gerar_mapa(semente), ag)
        salvar_mapa_visto(ag, os.path.join(a.saida, f"mapa_sim_{semente:06d}.csv"))
    print(f"{a.n} mapas salvos em {a.saida}")


def cmd_avaliar(a):
    import prior_sinais
    configs = [("otimista", None)]
    rede = prior_sinais.RedeMLP.carregar(a.prior)
    prob = prior_sinais.probabilidades(rede)
    pesos = [float(w) for w in a.pesos.split(",")] if a.pesos else [prior_sinais.PESO_PRIOR]
    for w in pesos:
        configs.append((f"prior {w:g}", prior_sinais.q_inicial_com_prior(prob, w)))

    linhas = []
    for i in range(a.n):
        semente = a.semente_ini + i
        mapa = gerar_mapa(semente)
        for nome, q in configs:
            r = simular(mapa, AgenteQ(epsilon=0.0, q_inicial=q, semente=semente))
            r.update({"semente": semente, "config": nome})
            linhas.append(r)

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            w.writeheader(); w.writerows(linhas)

    def resumo(nome):
        rs = [r for r in linhas if r["config"] == nome]
        if not rs:
            return
        n = len(rs)
        ok = [r for r in rs if r["chegou"]]
        med = lambda k: sum(r[k] for r in ok) / max(1, len(ok))
        ate8 = sum(1 for r in ok if r["tentativas"] <= 8)
        sinais = sum(r["verdes"] + r["amarelos"] + r["vermelhos"] for r in ok) / max(1, len(ok))
        print(f"{nome:12s} | chegou {len(ok)}/{n} | tentativas média {med('tentativas'):5.2f}"
              f" | até 8 tentativas {100*ate8/n:5.1f}% | segmentos média {med('segmentos'):6.1f}"
              f" | sinais pisados {sinais:5.2f} | score final {med('score_final'):6.1f}")
    print(f"Sementes {a.semente_ini}..{a.semente_ini + a.n - 1} (simulação, sem física)")
    for nome, _ in configs:
        resumo(nome)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coletar", help="gera mapas vistos pelo agente (dados de treino)")
    c.add_argument("--n", type=int, default=60)
    c.add_argument("--saida", default=os.path.join(AQUI, "..", "dados", "mapas_sim"))
    c.add_argument("--epsilon", type=float, default=0.2)
    c.add_argument("--semente-ini", type=int, default=1)
    c.set_defaults(f=cmd_coletar)
    v = sub.add_parser("avaliar", help="compara otimista × prior em sementes novas")
    v.add_argument("--n", type=int, default=300)
    v.add_argument("--prior", default=os.path.join(
        AQUI, "..", "controllers", "grade_controller", "prior_sinais.npz"))
    v.add_argument("--semente-ini", type=int, default=100000,
                   help="use sementes que não estão nos dados de treino")
    v.add_argument("--pesos", default="",
                   help="lista de pesos do prior para comparar, ex.: 0.25,0.5,1")
    v.add_argument("--csv", default="")
    v.set_defaults(f=cmd_avaliar)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()

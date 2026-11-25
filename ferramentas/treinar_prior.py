"""
treinar_prior.py — treina a rede que estima onde os sinais costumam ficar
=========================================================================
MATA64 - Inteligência Artificial | UFBA | Semestre 2026.2

Entrada : dados/mapas/mapa_*.csv, um arquivo por execução (gravados pelo
          grade_controller.py). Cada linha é um segmento que o robô
          percorreu e a penalidade que encontrou (0 = livre).
Saída   : controllers/grade_controller/prior_sinais.npz (pesos da rede)
          + resultados/ com curvas de treino, comparação e figuras.

Protocolo (aulas 15 e 18):
  - Treino, validação e teste separados POR EXECUÇÃO (semente): segmentos
    da mesma execução nunca aparecem em dois conjuntos (sem vazamento).
  - Descida do gradiente com retropropagação, em numpy (sem bibliotecas
    de redes neurais), otimizador Adam, lote completo.
  - Parada antecipada: guarda os pesos da época de menor perda de validação.
  - O teste é medido uma única vez, no final, contra duas alternativas:
      (a) frequência global das classes (ignora a posição);
      (b) tabela de frequência por segmento (sem generalização).

Uso (a partir desta pasta):
  python treinar_prior.py --dados ../dados/mapas
"""

import argparse, csv, glob, os, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
PASTA_CTRL = os.path.join(AQUI, "..", "controllers", "grade_controller")
sys.path.insert(0, PASTA_CTRL)

from prior_sinais import (RedeMLP, atributos, todos_segmentos, CLASSES,   # noqa: E402
                          PEN_CLS, PEN_PARA_CLASSE, ARQ_PESOS)


# ── Dados ─────────────────────────────────────────────────────
def ler_execucoes(pasta):
    """Lista de execuções; cada uma é uma lista de (segmento, classe)."""
    execucoes = []
    for arq in sorted(glob.glob(os.path.join(pasta, "*.csv"))):
        linhas = []
        with open(arq, newline="") as f:
            for r in csv.DictReader(f):
                pen = float(r["penalidade"])
                if pen not in PEN_PARA_CLASSE:
                    continue
                seg = (r["orientacao"], int(r["col"]), int(r["lin"]))
                linhas.append((seg, PEN_PARA_CLASSE[pen]))
        if linhas:
            execucoes.append((os.path.basename(arq), linhas))
    return execucoes


def separar(execucoes, semente=0, frac=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(semente)
    idx = rng.permutation(len(execucoes))
    n_tr = int(round(frac[0] * len(idx)))
    n_va = int(round(frac[1] * len(idx)))
    partes = (idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:])
    return [[execucoes[i] for i in p] for p in partes]


def para_matrizes(execucoes):
    segs = [s for _, linhas in execucoes for s, _ in linhas]
    y = np.array([c for _, linhas in execucoes for _, c in linhas], dtype=int)
    X = np.array([atributos(s) for s in segs], dtype=float)
    return X, y, segs


# ── Perda e retropropagação ───────────────────────────────────
def entropia_cruzada(prob, y):
    return float(-np.mean(np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1.0))))


def gradientes(rede, X, y, l2):
    prob, (X, z1, a1, z2, a2) = rede.frente(X)
    n = len(y)
    p = rede.p
    d3 = prob.copy(); d3[np.arange(n), y] -= 1.0; d3 /= n     # softmax + entropia cruzada
    g = {"W3": a2.T @ d3 + l2 * p["W3"], "b3": d3.sum(0)}
    d2 = (d3 @ p["W3"].T) * (z2 > 0)                          # volta pela ReLU
    g["W2"] = a1.T @ d2 + l2 * p["W2"]; g["b2"] = d2.sum(0)
    d1 = (d2 @ p["W2"].T) * (z1 > 0)
    g["W1"] = X.T @ d1 + l2 * p["W1"];  g["b1"] = d1.sum(0)
    return g, entropia_cruzada(prob, y)


def treinar(Xtr, ytr, Xva, yva, n_oculta, epocas, lr, l2, paciencia, semente):
    rede = RedeMLP.inicial(n_oculta, semente)
    m = {k: np.zeros_like(v) for k, v in rede.p.items()}
    v = {k: np.zeros_like(w) for k, w in rede.p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    melhor, melhor_p, melhor_ep, sem_melhora = np.inf, None, 0, 0
    curva = []
    for ep in range(1, epocas + 1):
        g, perda_tr = gradientes(rede, Xtr, ytr, l2)
        for k in rede.p:                                       # Adam
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            mh, vh = m[k] / (1 - b1 ** ep), v[k] / (1 - b2 ** ep)
            rede.p[k] -= lr * mh / (np.sqrt(vh) + eps)
        perda_va = entropia_cruzada(rede.prever(Xva), yva)
        curva.append((ep, perda_tr, perda_va))
        if perda_va < melhor - 1e-6:
            melhor, melhor_ep, sem_melhora = perda_va, ep, 0
            melhor_p = {k: w.copy() for k, w in rede.p.items()}
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                break
    return RedeMLP(melhor_p), curva, melhor_ep


# ── Alternativas para comparação ──────────────────────────────
def freq_global(ytr):
    c = np.bincount(ytr, minlength=len(CLASSES)).astype(float) + 1.0
    return c / c.sum()


def freq_por_segmento(segs_tr, ytr, global_p, forca=4.0):
    """Tabela de frequência por segmento, suavizada pela frequência global."""
    cont = {}
    for s, c in zip(segs_tr, ytr):
        cont.setdefault(s, np.zeros(len(CLASSES)))[c] += 1
    tabela = {s: (n + forca * global_p) / (n.sum() + forca) for s, n in cont.items()}
    return lambda s: tabela.get(s, global_p)


def erro_penalidade(prob, y):
    """Erro absoluto médio entre a penalidade esperada e a real (pontos)."""
    return float(np.mean(np.abs(prob @ PEN_CLS - PEN_CLS[y])))


# ── Figuras ───────────────────────────────────────────────────
def figuras(curva, rede, pasta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib import colors as mcolors
    except ImportError:
        print("  (matplotlib não instalado: figuras não geradas)")
        return

    tinta, tinta2, grade = "#0b0b0b", "#52514e", "#e1e0d9"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": tinta2, "xtick.color": tinta2,
                         "ytick.color": tinta2, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb"})

    # 1. Curvas de treino e validação (diagnóstico da aula 18)
    ep = [c[0] for c in curva]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ep, [c[1] for c in curva], color="#2a78d6", lw=2, label="treino")
    ax.plot(ep, [c[2] for c in curva], color="#eb6834", lw=2, label="validação")
    i = int(np.argmin([c[2] for c in curva]))
    ax.axvline(ep[i], color="#898781", lw=1, ls="--")
    ax.annotate(f"menor validação\népoca {ep[i]}", (ep[i], curva[i][2]),
                xytext=(10, 18), textcoords="offset points", color=tinta2, fontsize=9)
    ax.set_xlabel("época"); ax.set_ylabel("entropia cruzada")
    ax.set_title("Perda da rede ao longo do treino", color=tinta, loc="left")
    ax.grid(axis="y", color=grade, lw=0.8); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(pasta, "curvas_treino.png"), dpi=150)
    plt.close(fig)

    # 2. Mapa da penalidade esperada por segmento (o que a rede aprendeu)
    segs = todos_segmentos()
    P = rede.prever([atributos(s) for s in segs])
    esperado = -(P @ PEN_CLS)                       # pontos perdidos, positivo
    linhas = []
    for (o, c, l) in segs:
        if o == "H":
            linhas.append([(c, -l), (c + 1, -l)])
        else:
            linhas.append([(c, -l), (c, -(l + 1))])
    rampa = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = mcolors.LinearSegmentedColormap.from_list("azul", rampa)
    norma = mcolors.Normalize(vmin=0, vmax=max(30.0, float(esperado.max())))
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    lc = LineCollection(linhas, cmap=cmap, norm=norma, linewidths=5, capstyle="round")
    lc.set_array(esperado); ax.add_collection(lc)
    ax.scatter([0], [0], s=90, color="#2a78d6", zorder=3)
    ax.scatter([9], [-9], s=90, color="#eb6834", zorder=3)
    ax.annotate("A", (0, 0), xytext=(-14, 6), textcoords="offset points", color=tinta)
    ax.annotate("B", (9, -9), xytext=(8, -12), textcoords="offset points", color=tinta)
    ax.set_xlim(-0.6, 9.6); ax.set_ylim(-9.6, 0.6); ax.set_aspect("equal"); ax.axis("off")
    cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("penalidade esperada (pontos)", color=tinta2); cb.outline.set_visible(False)
    ax.set_title("Penalidade esperada por segmento, segundo a rede", color=tinta, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(pasta, "mapa_prior.png"), dpi=150)
    plt.close(fig)


# ── Principal ─────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dados", default=os.path.join(AQUI, "..", "dados", "mapas"))
    p.add_argument("--saida", default=os.path.join(PASTA_CTRL, ARQ_PESOS))
    p.add_argument("--resultados", default=os.path.join(AQUI, "..", "resultados"))
    p.add_argument("--oculta", type=int, default=16)
    p.add_argument("--epocas", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--paciencia", type=int, default=300)
    p.add_argument("--semente", type=int, default=0)
    a = p.parse_args()

    execucoes = ler_execucoes(a.dados)
    if len(execucoes) < 5:
        sys.exit(f"Só {len(execucoes)} execuções em {a.dados}. Colete mais antes de treinar.")
    tr, va, te = separar(execucoes, a.semente)
    Xtr, ytr, str_ = para_matrizes(tr)
    Xva, yva, _ = para_matrizes(va)
    Xte, yte, ste = para_matrizes(te)
    print(f"Execuções: {len(tr)} treino, {len(va)} validação, {len(te)} teste")
    print(f"Segmentos: {len(ytr)} treino, {len(yva)} validação, {len(yte)} teste")

    rede, curva, melhor_ep = treinar(Xtr, ytr, Xva, yva, a.oculta, a.epocas,
                                     a.lr, a.l2, a.paciencia, a.semente)
    print(f"Parada antecipada: pesos da época {melhor_ep} de {len(curva)}")

    # Teste: uma única vez, contra as duas alternativas
    gp = freq_global(ytr)
    tab = freq_por_segmento(str_, ytr, gp)
    P_rede = rede.prever(Xte)
    P_glob = np.tile(gp, (len(yte), 1))
    P_tab = np.array([tab(s) for s in ste])
    print("\nTeste (execuções que a rede nunca viu):")
    print(f"  {'modelo':28s} {'entropia cruzada':>17s} {'erro médio (pts)':>17s}")
    comparacao = []
    for nome, P in (("frequência global", P_glob),
                    ("tabela por segmento", P_tab),
                    ("rede neural (MLP)", P_rede)):
        ce, ea = entropia_cruzada(P, yte), erro_penalidade(P, yte)
        comparacao.append((nome, ce, ea))
        print(f"  {nome:28s} {ce:17.4f} {ea:17.2f}")

    rede.salvar(a.saida)
    print(f"\nPesos salvos em {os.path.normpath(a.saida)}")

    os.makedirs(a.resultados, exist_ok=True)
    with open(os.path.join(a.resultados, "curvas_treino.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoca", "perda_treino", "perda_validacao"])
        w.writerows(curva)
    with open(os.path.join(a.resultados, "comparacao_teste.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["modelo", "entropia_cruzada", "erro_medio_pts"])
        w.writerows(comparacao)
    figuras(curva, rede, a.resultados)
    print(f"Curvas, comparação e figuras em {os.path.normpath(a.resultados)}")


if __name__ == "__main__":
    main()

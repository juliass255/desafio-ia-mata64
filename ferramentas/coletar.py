"""
coletar.py — roda o Webots várias vezes seguidas para gerar dados de treino
===========================================================================
MATA64 - Inteligência Artificial | UFBA | Semestre 2026.2

Cada execução do Webots usa uma semente nova (o supervisor sorteia com
SEMENTE = None) e o controlador grava o mapa visto em dados/mapas/.
Este script repete isso N vezes, sem ninguém clicar em nada:

  1. cria controllers/grade_controller/modo_coleta.flag (ε = 0,2, sem prior);
  2. abre o Webots em modo rápido, sem renderização;
  3. espera o controlador criar fim_execucao.txt (chegou em B) ou o tempo
     limite estourar;
  4. copia resultado_avaliacao.txt para dados/resultados/ e fecha o Webots;
  5. no fim, apaga modo_coleta.flag (a entrega precisa rodar em avaliação).

Uso (a partir desta pasta):
  python coletar.py --n 40
  python coletar.py --n 40 --webots "C:/Program Files/Webots/msys64/mingw64/bin/webots.exe"
  python coletar.py --n 10 --modo avaliacao      (execuções com o prior, para a análise)

O caminho do Webots também pode vir da variável de ambiente WEBOTS.
"""

import argparse, os, shutil, subprocess, sys, time

AQUI  = os.path.dirname(os.path.abspath(__file__))
RAIZ  = os.path.normpath(os.path.join(AQUI, ".."))
CTRL  = os.path.join(RAIZ, "controllers", "grade_controller")
MUNDO = os.path.join(RAIZ, "worlds", "grade_inteligente.wbt")
FLAG_COLETA = os.path.join(CTRL, "modo_coleta.flag")
FIM   = os.path.join(CTRL, "fim_execucao.txt")
RESULTADO = os.path.join(CTRL, "resultado_avaliacao.txt")
PASTA_RES = os.path.join(RAIZ, "dados", "resultados")


def rodar_uma(webots, timeout_s, i, n, modo):
    for f in (FIM, RESULTADO):
        if os.path.exists(f):
            os.remove(f)
    cmd = [webots, "--mode=fast", "--no-rendering", "--minimize", "--batch",
           "--stdout", "--stderr", MUNDO]
    env = dict(os.environ, GRADE_MODO=modo)
    log = open(os.path.join(PASTA_RES, f"webots_{i:03d}.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    inicio, chegou = time.time(), False
    try:
        while time.time() - inicio < timeout_s:
            if os.path.exists(FIM):
                chegou = True
                time.sleep(3)          # dá tempo ao supervisor de gravar o resultado
                break
            if proc.poll() is not None:
                break
            time.sleep(1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if os.path.exists(RESULTADO):
        shutil.copy(RESULTADO, os.path.join(PASTA_RES, f"resultado_{i:03d}.txt"))
    dur = time.time() - inicio
    print(f"[{i}/{n}] {'chegou em B' if chegou else 'NÃO chegou (tempo limite)'} "
          f"em {dur:.0f} s de relógio")
    return chegou


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--webots", default=os.environ.get("WEBOTS", "webots"))
    p.add_argument("--timeout", type=float, default=900, help="segundos de relógio por execução")
    p.add_argument("--modo", choices=["coleta", "avaliacao"], default="coleta")
    a = p.parse_args()

    if shutil.which(a.webots) is None and not os.path.exists(a.webots):
        sys.exit(f"Webots não encontrado em '{a.webots}'. Use --webots ou a variável WEBOTS.")
    os.makedirs(PASTA_RES, exist_ok=True)

    if a.modo == "coleta":
        open(FLAG_COLETA, "w").write("coleta\n")
    elif os.path.exists(FLAG_COLETA):
        os.remove(FLAG_COLETA)
    try:
        ok = sum(rodar_uma(a.webots, a.timeout, i + 1, a.n, a.modo) for i in range(a.n))
    finally:
        if os.path.exists(FLAG_COLETA):
            os.remove(FLAG_COLETA)
    print(f"\n{ok}/{a.n} execuções chegaram em B. Mapas em dados/mapas, "
          f"resultados em dados/resultados.")


if __name__ == "__main__":
    main()

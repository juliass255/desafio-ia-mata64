/*
 * grade_controller.c
 * ==================
 * MATA64 - Inteligência Artificial | UFBA | Semestre 2026.2
 * Desafio IA — Grade Inteligente
 *
 * OBJETIVO
 * --------
 * Programar o robô Pioneer3dx para navegar de forma autônoma
 * da posição A (canto superior esquerdo) até B (canto inferior
 * direito) numa grade 10x10, aprendendo a evitar os segmentos
 * penalizados pelos sinais coloridos.
 *
 * DESCRIÇÃO DA GRADE
 * ------------------
 *   Grade 10x10, espaçamento 1m (arena 11x11m)
 *   Ponto A : coluna 0, linha 0 → posição (-4.5,  4.5)
 *   Ponto B : coluna 9, linha 9 → posição ( 4.5, -4.5)
 *   Decidir direção SOMENTE nas interseções.
 *
 * SINAIS E PENALIDADES
 * --------------------
 *   VERDE    → -10 pts    AMARELO → -20 pts
 *   VERMELHO → -50 pts    Timeout → -30 pts → volta para A
 *   Pontuação <= -60      → volta para A (pontuação zera)
 *   Chegar em B           → +100 pts → encerra
 *
 * SENSORES DISPONÍVEIS
 * --------------------
 *   GPS     : wb_gps_get_values()     → posição (x, y, z)
 *   Compass : wb_compass_get_values() → orientação
 *   Camera  : wb_camera_get_image()   → imagem RGB 64x64
 *
 * MOTORES
 * -------
 *   "left wheel"  → wb_motor_set_velocity()
 *   "right wheel" → wb_motor_set_velocity()
 *   Velocidade máxima recomendada: 5.0 rad/s
 *
 *   O Pioneer3dx é um robô DIFERENCIAL:
 *     Frente       → ambas positivas e iguais
 *     Ré           → ambas negativas e iguais
 *     Giro direita → esq. positiva, dir. negativa
 *     Giro esq.    → esq. negativa, dir. positiva
 *
 * COMUNICAÇÃO COM O SUPERVISOR
 * -----------------------------
 *   reset_flag.txt → robô reposicionado em A
 *                    Ao detectar: apague e redefina estado.
 *   pen_flag.txt   → penalidade recebida (float em texto)
 *                    Ao detectar: leia, registre, apague.
 *
 * CONVERSÃO DE COORDENADAS
 * -------------------------
 *   grade → posição:
 *     x = -4.5 + coluna * 1.0
 *     y =  4.5 - linha  * 1.0
 *
 *   posição → grade:
 *     coluna = (int) round((x + 4.5) / 1.0)
 *     linha  = (int) round((4.5 - y) / 1.0)
 *
 * ÂNGULOS DO COMPASS
 * -------------------
 *   Leste → 358°   Sul   → 268°
 *   Oeste → 178°   Norte →  88°
 *
 * TÉCNICAS SUGERIDAS (livre escolha)
 * ------------------------------------
 *   Redes Bayesianas | Lógica Fuzzy | Redes Neurais
 *   Lógica 1ª Ordem  | Aprendizado por Reforço
 */

#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/gps.h>
#include <webots/compass.h>
#include <webots/camera.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Constantes ──────────────────────────────────────────── */
#define MAX_V      5.0
#define GRID_SIZE  10
#define ORIG_X    -4.5
#define ORIG_Y     4.5
#define CELL_SIZE  1.0

/* ── Dispositivos ────────────────────────────────────────── */
static WbDeviceTag motor_esq, motor_dir;
static WbDeviceTag gps_dev, compass_dev, camera_dev;
static int timestep;

/* ── Funções utilitárias ─────────────────────────────────── */

/** Define velocidade das rodas com limite de segurança. */
void set_velocidade(double esq, double dir) {
    if (esq >  MAX_V) esq =  MAX_V;
    if (esq < -MAX_V) esq = -MAX_V;
    if (dir >  MAX_V) dir =  MAX_V;
    if (dir < -MAX_V) dir = -MAX_V;
    wb_motor_set_velocity(motor_esq, esq);
    wb_motor_set_velocity(motor_dir, dir);
}

/** Para o robô. */
void parar() {
    set_velocidade(0.0, 0.0);
}

/** Retorna ângulo atual em graus (0-360) pelo Compass. */
double get_angulo() {
    const double *v = wb_compass_get_values(compass_dev);
    double ang = atan2(v[0], v[1]) * 180.0 / M_PI;
    if (ang < 0.0) ang += 360.0;
    return ang;
}

/**
 * Converte posição (x, y) para célula da grade.
 * Resultado armazenado em *col e *lin.
 */
void posicao_para_grade(double x, double y, int *col, int *lin) {
    *col = (int)round((x - ORIG_X) / CELL_SIZE);
    *lin = (int)round((ORIG_Y - y) / CELL_SIZE);
    if (*col < 0) *col = 0;
    if (*col >= GRID_SIZE) *col = GRID_SIZE - 1;
    if (*lin < 0) *lin = 0;
    if (*lin >= GRID_SIZE) *lin = GRID_SIZE - 1;
}

/**
 * Converte célula da grade para posição (x, y).
 * Resultado armazenado em *x e *y.
 */
void grade_para_posicao(int col, int lin, double *x, double *y) {
    *x = ORIG_X + col * CELL_SIZE;
    *y = ORIG_Y - lin * CELL_SIZE;
}

/**
 * Verifica se o supervisor reposicionou o robô em A.
 * Retorna 1 se houve reset — redefina seu estado interno!
 */
int verificar_reset() {
    FILE *f = fopen("reset_flag.txt", "r");
    if (f) {
        fclose(f);
        remove("reset_flag.txt");
        return 1;
    }
    return 0;
}

/**
 * Lê penalidade enviada pelo supervisor.
 * Retorna o valor (ex: -50.0) ou 0.0 se não houver.
 * Use para atualizar seu histórico de decisões.
 */
double verificar_penalidade() {
    FILE *f = fopen("pen_flag.txt", "r");
    if (f) {
        double pen = 0.0;
        fscanf(f, "%lf", &pen);
        fclose(f);
        remove("pen_flag.txt");
        return pen;
    }
    return 0.0;
}

/* ════════════════════════════════════════════════════════════
 *
 *   IMPLEMENTE SUA SOLUÇÃO ABAIXO
 *
 *   Sugestão de estrutura:
 *
 *   1. Declare seu estado global:
 *      - Posição atual na grade (col, lin)
 *      - Direção atual do robô
 *      - Histórico: float historico[10][10][4]
 *        onde [col][lin][dir] acumula penalidades
 *
 *   2. No loop principal, a cada passo:
 *      a. Chame verificar_reset() — trate retorno a A
 *      b. Chame verificar_penalidade() — registre
 *      c. Leia GPS/Compass
 *      d. Nas interseções: decida a direção
 *      e. Gire (use Compass) e avance (use GPS)
 *      f. Repita até col==9 && lin==9 (ponto B)
 *
 *   IMPORTANTE: mude de direção APENAS nas interseções!
 *
 * ════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {

    /* Inicialização */
    wb_robot_init();
    timestep = (int)wb_robot_get_basic_time_step();

    /* Motores */
    motor_esq = wb_robot_get_device("left wheel");
    motor_dir = wb_robot_get_device("right wheel");
    wb_motor_set_position(motor_esq, INFINITY);
    wb_motor_set_position(motor_dir, INFINITY);
    wb_motor_set_velocity(motor_esq, 0.0);
    wb_motor_set_velocity(motor_dir, 0.0);

    /* Sensores — habilite os que for usar */
    gps_dev     = wb_robot_get_device("gps");
    compass_dev = wb_robot_get_device("compass");
    camera_dev  = wb_robot_get_device("camera");

    wb_gps_enable(gps_dev,         timestep);
    wb_compass_enable(compass_dev, timestep);
    wb_camera_enable(camera_dev,   timestep);

    printf("================================================\n");
    printf("  Desafio IA - MATA64 | UFBA 2026.2\n");
    printf("  Implemente sua solucao aqui!\n");
    printf("================================================\n");

    /* TODO: inicialize seu estado aqui */

    /* Loop principal */
    while (wb_robot_step(timestep) != -1) {

        /* Sempre verifique estes eventos a cada passo */
        if (verificar_reset()) {
            /* TODO: redefina seu estado (posição, direção, histórico) */
            printf("  Robo reposicionado em A - redefina seu estado!\n");
        }

        double pen = verificar_penalidade();
        if (pen != 0.0) {
            /* TODO: registre a penalidade e atualize sua estratégia */
            printf("  Penalidade recebida: %.0f\n", pen);
        }

        /* Leitura de posição e orientação */
        const double *pos = wb_gps_get_values(gps_dev);
        int col, lin;
        posicao_para_grade(pos[0], pos[1], &col, &lin);
        double angulo = get_angulo();

        /* TODO: implemente sua lógica de navegação aqui */
        parar();
    }

    wb_robot_cleanup();
    return 0;
}

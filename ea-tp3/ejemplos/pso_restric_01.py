import numpy as np
import random


def meets_all_restrictions(particle, restrictions):
    return all(g(particle) for g in restrictions)

def linear_w(t, t_max, w_max=0.9, w_min=0.4):
    return w_max - ((w_max-w_min)/t_max)*t


def pso(n_particles, n_dimensions, max_iterations, c1, c2, w, restrictions, target_function, print_sol=False, w_algorithm=None, phi=None):
    if phi is not None:
        c1 = c2 = phi/2
    # inicialización de particulas
    x = np.zeros((n_particles, n_dimensions))  # matriz para las posiciones de las particulas
    v = np.zeros((n_particles, n_dimensions))  # matriz para las velocidades de las particulas
    pbest = np.zeros((n_particles, n_dimensions))  # matriz para los mejores valores personales
    pbest_fit = -np.inf * np.ones(n_particles)  # mector para las mejores aptitudes personales (inicialmente -infinito)
    gbest = np.zeros(n_dimensions)  # mejor solución global
    gbest_fit = -np.inf  # mejor aptitud global (inicialmente -infinito)
    historic_gbest = []
    # inicializacion de particulas factibles

    for i in range(n_particles):
        while True:  # bucle para asegurar que la particula sea factible
            x[i] = np.random.uniform(0, 10, n_dimensions)  # inicializacion posicion aleatoria en el rango [0, 10]
            # if g1(x[i]) and g2(x[i]) and g3(x[i]):  # se comprueba si la posicion cumple las restricciones
            if meets_all_restrictions(x[i], restrictions):
                break  # Salir del bucle si es factible

        v[i] = np.random.uniform(-1, 1, n_dimensions)  # inicializar velocidad aleatoria
        pbest[i] = x[i].copy()  # ee establece el mejor valor personal inicial como la posicion actual
        fit = target_function(x[i])  # calculo la aptitud de la posicion inicial
        if fit > pbest_fit[i]:  # si la aptitud es mejor que la mejor conocida
            pbest_fit[i] = fit  # se actualiza el mejor valor personal

    # Optimizacion
    for it in range(max_iterations):  # Repetir hasta el número máximo de iteraciones
        for i in range(n_particles):
            fit = target_function(x[i])  # Se calcula la aptitud de la posicion actual
            # Se comprueba si la nueva aptitud es mejor y si cumple las restricciones
            if fit > pbest_fit[i] and meets_all_restrictions(x[i], restrictions):
                pbest_fit[i] = fit  # Se actualiza la mejor aptitud personal
                pbest[i] = x[i].copy()  # Se actualizar la mejor posicion personal
                if fit > gbest_fit:  # Si la nueva aptitud es mejor que la mejor global
                    gbest_fit = fit  # Se actualizar la mejor aptitud global
                    gbest = x[i].copy()  # Se actualizar la mejor posicion global

            # actualizacion de la velocidad de la particula
            if w_algorithm is None:
                v[i] = w * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
            elif w_algorithm == 'linear':
                v[i] = linear_w(it, max_iterations) * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
            elif w_algorithm == 'constriction':
                chi = 2/(abs(2-phi-np.sqrt(phi**2 -4*phi)))
                v[i] = chi*(v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i]))

            x[i] += v[i]  # Se actualiza la posicion de la particula

            # se asegura de que la nueva posicion esté dentro de las restricciones
            if not meets_all_restrictions(x[i], restrictions):
                # Si la nueva posicion no es válida, revertir a la mejor posicion personal
                x[i] = pbest[i].copy()
        historic_gbest.append(gbest)
    # Se imprime la mejor solucion encontrada y también su valor optimo
    if print_sol:
        pretty_gbest = [float(f"{el:.2f}") for el in gbest]
        print(f"Mejor solucion: {pretty_gbest}")
        print(f"Valor optimo: {gbest_fit}")
    return np.array(historic_gbest)

import random
import numpy as np
import matplotlib.pyplot as plt
# setting for seed reproducibility and to compare with original file
random.seed(42)


# parametros
TAMANIO_POBLACION = 4
LONGITUD_CROMOSOMA = 10
TASA_MUTACION = 0.1
TASA_CRUCE = 0.92
GENERACIONES = 10
X_MIN = -31
X_MAX = 31
EPSILON = 0.001  # Valor pequeño para evitar división por cero en la funcion fitness


#  -----------------------------------------------------------------
# funcion para mapear el valor binario a un rango [-31, 31]
#  -----------------------------------------------------------------
def binario_a_decimal(cromosoma):
    decimal = int(cromosoma, 2)
    x = X_MIN + decimal * (X_MAX - X_MIN) / ((2 ** LONGITUD_CROMOSOMA) - 1)
    return x

#  -----------------------------------------------------------------
# Aqui en las proximas lineas se puede ver que mi funcion objetivo es
# a veces diferente de mi funcion fitness, depende del problema a resolver
#  -----------------------------------------------------------------


#  -----------------------------------------------------------------
# funcion objetivo x^2
#  -----------------------------------------------------------------
def funcion_objetivo(x):
    return x ** 2


#  -----------------------------------------------------------------
# funcion fitness o tambien llamada funcion de aptitud (1/(x^2 + epsilon))
#  -----------------------------------------------------------------
def aptitud(cromosoma):
    x = binario_a_decimal(cromosoma)
    return 1 / (funcion_objetivo(x) + EPSILON)


#  -----------------------------------------------------------------
# se inicializa la poblacion
#  -----------------------------------------------------------------

def generar_cromosoma(longitud_cromosoma):
    random_bits = [random.randint(0, 1) for _ in range(longitud_cromosoma)]
    cromosoma = "".join(map(str, random_bits))
    return cromosoma

def inicializar_poblacion(tamanio_poblacion, longitud_cromosoma):
    return [generar_cromosoma(longitud_cromosoma) for _ in range(tamanio_poblacion)]

#  -----------------------------------------------------------------
# seleccion por ruleta
#  -----------------------------------------------------------------
def seleccion_ruleta(poblacion, aptitud_total):
    probabilidades = [aptitud(individuo)/aptitud_total for individuo in poblacion]
    probabilidades_acumuladas = np.cumsum(probabilidades)

    r = random.uniform(0, 1)
    i = next(i for i,acumulada in enumerate(probabilidades_acumuladas) if r <= acumulada)
    return poblacion[i]


#  -----------------------------------------------------------------
# cruce monopunto con probabilidad de cruza pc = 0.92
#  -----------------------------------------------------------------
def cruce_mono_punto(progenitor1, progenitor2, tasa_cruce):
    if random.uniform(0,1) < tasa_cruce:
        punto_cruce = random.randint(1, len(progenitor1) - 1)
        descendiente1 = progenitor1[:punto_cruce] + progenitor2[punto_cruce:]
        descendiente2 = progenitor2[:punto_cruce] + progenitor1[punto_cruce:]
    else:
        descendiente1, descendiente2 = progenitor1, progenitor2
    return [descendiente1, descendiente2]


#  -----------------------------------------------------------------
# mutacion
#  -----------------------------------------------------------------
def mutacion(cromosoma, tasa_mutacion):
    cromosoma_mutado = [int(not int(bit)) if random.uniform(0,1) < tasa_mutacion else bit for bit in cromosoma]
    return "".join(map(str,cromosoma_mutado))


#  -----------------------------------------------------------------
# aplicación de operadores geneticos
#  -----------------------------------------------------------------
def algoritmo_genetico(tamanio_poblacion, longitud_cromosoma, tasa_mutacion, tasa_cruce, generaciones):
    poblacion = inicializar_poblacion(tamanio_poblacion, longitud_cromosoma)
    mejor_funcion_objetivo_generaciones = []  # Lista para almacenar la aptitud del mejor individuo y grficar luego

    for generacion in range(generaciones):
        print("Generación:", generacion + 1)

        # se calcula aptitud total para luego
        aptitud_total = sum(aptitud(cromosoma) for cromosoma in poblacion)

        print("Aptitud total:", aptitud_total)

        #  -----------------------------------------------------------------
        # seleccion de progenitores con el metodo ruleta
        progenitores = [seleccion_ruleta(poblacion, aptitud_total) for _ in range(tamanio_poblacion)]

        #  -----------------------------------------------------------------
        # Cruce
        grouped_progenitores = zip(progenitores[0::2], progenitores[1::2])
        unflattened_descendientes = [cruce_mono_punto(p1,p2, tasa_cruce) for p1,p2 in grouped_progenitores]
        descendientes = sum(unflattened_descendientes, [])
        #  -----------------------------------------------------------------
        # Mutacion
        descendientes_mutados = [mutacion(descendiente, tasa_mutacion) for descendiente in descendientes]

        # Aquí se aplica elitismo
        # Se reemplazan los peores cromosomas con los mejores progenitores
        poblacion.sort(key=aptitud)  # se ordena la poblacion por aptitud en forma ascendente
        # se ordena los descendientes por aptitud en forma descendente
        descendientes_mutados.sort(key=aptitud, reverse=True)
        for i in range(len(descendientes_mutados)):
            if aptitud(descendientes_mutados[i]) > aptitud(poblacion[i]):
                poblacion[i] = descendientes_mutados[i]

        # Mostrar el mejor individuo de la generacion
        mejor_individuo = max(poblacion, key=aptitud)  # Buscar el maximo para la aptitud
        mejor_funcion_objetivo_generaciones.append(funcion_objetivo(binario_a_decimal(mejor_individuo)))

        print("mi", mejor_individuo)
        print("Mejor individuo:", binario_a_decimal(mejor_individuo), "Aptitud:", aptitud(mejor_individuo))
        print("_________________________________________________________________________________")

    # Graficar la evolución de la aptitud
    plt.plot(range(1, generaciones + 1), mejor_funcion_objetivo_generaciones, marker='o')
    plt.xlabel('Generación')
    plt.ylabel('Valor de la Función Objetivo')
    plt.title('Curva de Convergencia del Algoritmo Genético')
    plt.grid(True)
    plt.show()
    return max(poblacion, key=aptitud)  # se retorna el mejor individuo


#  -----------------------------------------------------------------
# ejecucion principal del algoritmo genetico
#  -----------------------------------------------------------------
print("_________________________________________________________________________________")
print()
mejor_solucion = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES)
print("Mejor solución:", binario_a_decimal(mejor_solucion), "Aptitud:", aptitud(mejor_solucion))
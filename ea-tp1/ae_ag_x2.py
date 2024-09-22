#  -----------------------------------------------------------------
# Algoritmo Genetico que encuentra el maximo de la funcion x^2
# en el intervalo [0, 31]
# Seleccion por ruleta
# Pc = 0.92
# Pm = 0.1
#  -----------------------------------------------------------------


import random

# Parametros
TAMANIO_POBLACION = 4
LONGITUD_CROMOSOMA = 5
TASA_MUTACION = 0.1
TASA_CRUCE = 0.92
GENERACIONES = 10


#  -----------------------------------------------------------------
# Aptitud (y = x^2)
#  -----------------------------------------------------------------
def aptitud(cromosoma):
    x = int(cromosoma, 2)
    return x ** 2


#  -----------------------------------------------------------------
# Inicializar la población
#  -----------------------------------------------------------------
def inicializar_poblacion(tamanio_poblacion, longitud_cromosoma):
    poblacion = []
    for tp in range(tamanio_poblacion):
        cromosoma = ""
        for lc in range(longitud_cromosoma):
            #  se crean los cromosomas alelo por alelo en forma aleatoria
            #  hasta completar toda la poblacion
            cromosoma = cromosoma + str(random.randint(0, 1))
        poblacion.append(cromosoma)
    return poblacion


#  -----------------------------------------------------------------
# Seleccion por ruleta
#  -----------------------------------------------------------------
def seleccion_ruleta(poblacion, aptitud_total):
    probabilidades = []
    for individuo in poblacion:
        prob = aptitud(individuo) / aptitud_total
        probabilidades.append(prob)

    probabilidades_acumuladas = []
    suma = 0
    for prob in probabilidades:
        suma = suma + prob
        probabilidades_acumuladas.append(suma)

    r = random.random()
    # despues de generar un numero aleatorio entre 0 y 1
    # se itera sobre la lista probabilidades_acumuladas
    # y se obtiene el indice (i) del cromosoma que selecciono para que forme parte de la nueva poblacion
    # tambien se obtiene el valor de probabilidad acumulada en la variable "acumulada"
    for i, acumulada in enumerate(probabilidades_acumuladas):
        if r <= acumulada:
            return poblacion[i]


#  -----------------------------------------------------------------
# Cruce monopunto con probabilidad de cruza pc = 0.92
#  -----------------------------------------------------------------
def cruce_mono_punto(progenitor1, progenitor2, tasa_cruce):
    if random.random() < tasa_cruce:
        punto_cruce = random.randint(1, len(progenitor1) - 1)  # elijo aleatoriamente un punto de cruce
        descendiente1 = progenitor1[:punto_cruce] + progenitor2[punto_cruce:]
        descendiente2 = progenitor2[:punto_cruce] + progenitor1[punto_cruce:]
    else:
        descendiente1, descendiente2 = progenitor1, progenitor2
    return descendiente1, descendiente2


#  -----------------------------------------------------------------
# mutacion
#  -----------------------------------------------------------------
def mutacion(cromosoma, tasa_mutacion):
    cromosoma_mutado = ""
    for bit in cromosoma:  # aqui se itera cada gen del cromosoma recibido
        if random.random() < tasa_mutacion:
            # se produce la mutacion de un alelo si es que el numero aleatorio generado
            # es inferior que tasa_mutacion tambien llamado "pm" (prob.de mutacion)
            cromosoma_mutado = cromosoma_mutado + str(int(not int(bit)))
        else:
            cromosoma_mutado = cromosoma_mutado + bit
    return cromosoma_mutado


#  -----------------------------------------------------------------
# aplicacion de operadores geneticos
#  -----------------------------------------------------------------
def algoritmo_genetico(tamanio_poblacion, longitud_cromosoma, tasa_mutacion, tasa_cruce, generaciones):
    poblacion = inicializar_poblacion(tamanio_poblacion, longitud_cromosoma)

    for generacion in range(generaciones):  # en este caso se definio un maximo de 10 generaciones
        print("Generación:", generacion + 1)

        # se calcula aptitud total (suma de evaluaciones de cada cromosoma) para luego
        # poder obtener la Ps de cada individuo (Ps = f(i) / sumatoria(f(i)) <-ruleta)
        aptitud_total = 0
        for cromosoma in poblacion:
            aptitud_total = aptitud_total + aptitud(cromosoma)

        print("Sumatoria de aptitudes total:", aptitud_total)

        #  -----------------------------------------------------------------
        # seleccion de progenitores con el metodo ruleta
        # se crea una lista vacia de progenitores primero y luego se llama
        # a la funcion seleccion_ruleta para que devuelva de a uno los individuos
        # que se convertiran en futuros progenitores
        progenitores = []
        for _ in range(tamanio_poblacion):
            progenitores.append(seleccion_ruleta(poblacion, aptitud_total))

        #  -----------------------------------------------------------------
        # Cruce
        descendientes = []
        for i in range(0, tamanio_poblacion, 2):
            #  se llama a cruce_mono_punto y se le envia pares de progenitores secuencialmente
            # para que se produzca la cruza (en este caso monopunto) segun la tasa_cruce (o Pc)
            descendiente1, descendiente2 = cruce_mono_punto(progenitores[i], progenitores[i + 1], tasa_cruce)
            descendientes.extend([descendiente1, descendiente2])

        #  -----------------------------------------------------------------
        # mutacion
        descendientes_mutados = []
        for descendiente in descendientes:
            descendientes_mutados.append(mutacion(descendiente, tasa_mutacion))

        # Aqui se aplica elitismo
        # se reemplazar los peores cromosomas con los mejores progenitores
        poblacion.sort(key=aptitud)
        descendientes_mutados.sort(key=aptitud, reverse=True)
        for i in range(len(descendientes_mutados)):
            if aptitud(descendientes_mutados[i]) > aptitud(poblacion[i]):
                poblacion[i] = descendientes_mutados[i]

        # mostrar el mejor individuo de la generacion
        mejor_individuo = max(poblacion, key=aptitud)
        print("Mejor individuo:", int(mejor_individuo, 2), "Aptitud:", aptitud(mejor_individuo))
        print("_________________________________________________________________________________")

    return max(poblacion, key=aptitud)


#  -----------------------------------------------------------------
# algoritmo genetico ejecucion principal
#  -----------------------------------------------------------------
print("_________________________________________________________________________________")
print()
mejor_solucion = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES)
print("Mejor solución:", int(mejor_solucion, 2), "Aptitud:", aptitud(mejor_solucion))

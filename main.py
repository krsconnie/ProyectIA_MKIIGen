import life
import test


# AYUDA
#life.help()

if input("E, para entrenar: ") == "E":
    # EVOLUCION
    life.let_there_be_life(config_file="config-neat")

else:
# TESTS
    #test.jugar_humano(mapa="Level1.LiuKangVsJax.state")
    test.jugar_agente(genoma_file="generaciones_v1/mejor_agente.pkl" ,mapa="Level1.LiuKangVsJax.state")
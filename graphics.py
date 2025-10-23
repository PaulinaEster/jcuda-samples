import matplotlib.pyplot as plt
import numpy as np

tempos_serial = [ ] # kernel
tempos_serial_java = [ ] # kernel
tempos_paralelo = [ ]  # kernel
tempos_paralelo_jcuda = [ ]  # kernel

tempos_serial = [318.261150, 318.261150, 318.261150, 318.261150, 318.261150 ] # total
tempos_serial_java = [390.907917, 402.007764, 391.601574, 396.918816, 394.149500] # total
tempos_paralelo = [26.668809, 26.566759, 26.527186, 27.059151, 26.509537 ] # total
tempos_paralelo_jcuda = [31.455419, 31.563263, 31.287826, 31.459561, 31.819042] # total

media_serial = np.mean(tempos_serial)
media_serial_java = np.mean(tempos_serial_java)
media_paralelo = np.mean(tempos_paralelo)
media_paralela_jcuda = np.mean(tempos_paralelo_jcuda)

desvio_serial = np.std(tempos_serial, ddof=1)
desvio_serial_java = np.std(tempos_serial_java, ddof=1)
desvio_paralelo = np.std(tempos_paralelo, ddof=1)
desvio_jcuda = np.std(tempos_paralelo_jcuda, ddof=1)

speedup_cuda = media_serial / media_paralelo
speedup_jcuda = media_serial / media_paralela_jcuda

print(f"Tempo médio serial     = {media_serial:.6f} ± {desvio_serial:.6f}")
print(f"Tempo médio serial JAVA = {media_serial_java:.6f} ± {desvio_serial_java:.6f}")
print(f"Tempo médio CUDA        = {media_paralelo:.6f} ± {desvio_paralelo:.6f}")
print(f"Tempo médio JCuda       = {media_paralela_jcuda:.6f} ± {desvio_jcuda:.6f}")
print(f"Speedup CUDA            = {speedup_cuda:.2f}x")
print(f"Speedup JCuda           = {speedup_jcuda:.2f}x")

# ============================
# Gráfico 1 - Comparação dos tempos com error bars
# ============================
labels = ["Serial", "Serial Java", "CUDA", "JCUDA"]
tempos_medios = [media_serial, media_serial_java, media_paralelo, media_paralela_jcuda]
desvios = [desvio_serial, desvio_serial_java, desvio_paralelo, desvio_jcuda]

plt.figure(figsize=(8,5))
barras = plt.bar(labels, tempos_medios, yerr=desvios, capsize=5, color=["blue", "green", "orange", "grey"])
plt.title("Comparação dos Tempos Médios")
plt.ylabel("Tempo médio (s)")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Adicionando os valores do desvio padrão no topo das barras
for barra, desvio in zip(barras, desvios):
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2, altura + desvio + 0.001, f"{desvio:.6f}s", 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("./resultados/caomparacao-tempo-kernel.png", dpi=300)

# ============================
# Gráfico 2 - Comparação dos Speedups
# ============================ 
valores_speedup = [speedup_cuda, speedup_jcuda]
labels_speedup = ["CUDA", "JCuda"]
plt.figure(figsize=(8,5))
plt.bar(labels_speedup, valores_speedup, color=["green", "orange"])
# plt.axhline(1.0, color="red", linestyle="--", label="Serial")
plt.title("Comparação de Speedup em relação ao Serial")
plt.ylabel("Speedup")
# plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("./resultados/speedup-kernel.png", dpi=300)

# ============================
#  Gráfico 3 - Tempos de cada etapa
# ============================
# tempos = {
#     "Memory Transfers": np.mean([ ]),
#     "Kernel": np.mean([ ]),
#     "Malloc": np.mean([ ]),
#     "Linearização": np.mean([ ]),
#     "Deslinearização": np.mean([ ])
# }
tempos_jcuda = { 
    "Kernel": np.mean([27.916958, 28.403587, 28.227065, 28.195507, 28.513576]),
    "Memory Transfers": np.mean([0.142848, 0.099479, 0.088403, 0.094375, 0.131267]),
    "Linearização": np.mean([1.761504, 1.612835, 1.546741, 1.651673, 1.642119]),
    "Deslinearização": np.mean([0.213142, 0.214917, 0.211059, 0.261270, 0.212357]),
    "JCudaDriver": np.mean([1.420576, 1.232160, 1.214255, 1.256457, 1.319346])
}

# tempos_cuda = {
#     "Kernel": np.mean([24.927875, 24.949061, 24.930720, 24.968140, 24.902483]),
#     "Memory Transfers": np.mean([0.209460, 0.209048, 0.209069, 0.209397, 0.210827]),
#     "Linearização": np.mean([1.103617, 0.978846, 0.962170, 1.292542, 0.961996]),
#     "Deslinearização": np.mean([0.222016, 0.222374, 0.221895, 0.228050, 0.225564])
# }
 
# Preparar dados
etapas = list(tempos_jcuda.keys())
valores = list(tempos_jcuda.values())
total = sum(valores)

# Calcular porcentagem de cada etapa
percentuais = [v / total * 100 for v in valores]

plt.figure(figsize=(8,5))
plt.bar(etapas, valores, color='skyblue')
plt.title("Contribuição de cada etapa para o tempo total JCUDA")
plt.ylabel("Tempo (s)")
plt.xticks(rotation=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("./resultados/tempo-etapa-jcuda.png", dpi=300)
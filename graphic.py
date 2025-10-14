import matplotlib.pyplot as plt
import numpy as np

# tempos_serial = [ ] # total
# tempos_paralelo = [ ]  # total
# tempos_paralelo_jcuda = [ ]  # total

tempos_serial = [ ] # kernel
tempos_paralelo = [ ] # kernel
tempos_paralelo_jcuda = [ ] # kernel

labels = ["Serial", "CUDA", "JCUDA"]

# Cálculo média tempo
media_serial = np.mean(tempos_serial)
media_paralelo = np.mean(tempos_paralelo)
media_paralela_jcuda = np.mean(tempos_paralelo_jcuda)

# Cálculo desvio padrão
desvio_serial = np.std(tempos_serial, ddof=1)
desvio_paralelo = np.std(tempos_paralelo, ddof=1)
desvio_jcuda = np.std(tempos_paralelo_jcuda, ddof=1)

# Cálculo do Speedup
speedup_cuda = media_serial / media_paralelo
speedup_jcuda = media_serial / media_paralela_jcuda

print(f"Tempo médio serial     = {media_serial:.6f} ± {desvio_serial:.6f}")
print(f"Tempo médio CUDA = {media_paralelo:.6f} ± {desvio_paralelo:.6f}")
print(f"Tempo médio JCuda = {media_paralela_jcuda:.6f} ± {desvio_jcuda:.6f}")
print(f"Speedup CUDA     = {speedup_cuda:.2f}x")
print(f"Speedup JCuda     = {speedup_jcuda:.2f}x")

# ============================
# Gráfico 1 - Comparação dos tempos com error bars
# ============================
tempos_medios = [media_serial, media_paralelo, media_paralela_jcuda]
desvios = [desvio_serial, desvio_paralelo, desvio_jcuda]

plt.figure(figsize=(8,5))
barras = plt.bar(labels, tempos_medios, yerr=desvios, capsize=5, color=["blue", "green", "orange"])
plt.title("Comparação dos Tempos Médios")
plt.ylabel("Tempo médio (s)")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Adicionando os valores do desvio padrão no topo das barras
for barra, desvio in zip(barras, desvios):
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2, altura + desvio + 0.001, f"{desvio:.6f}s", 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("./resultados/speedup-tempo-kernel.png", dpi=300)

# ============================
# Gráfico 2 - Comparação dos Speedups
# ============================ 
valores_speedup = [speedup_cuda, speedup_jcuda]

plt.figure(figsize=(8,5))
plt.bar(labels, valores_speedup, color=["green", "orange"])
plt.axhline(1.0, color="red", linestyle="--", label="Serial")
plt.title("Comparação de Speedup em relação ao Serial")
plt.ylabel("Speedup")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("./resultados/speedup-gray-kernel.png", dpi=300)

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
    "Memory Transfers": np.mean([ ]),
    "Kernel": np.mean([ ]),
    "Malloc": np.mean([ ]),
    "Linearização": np.mean([ ]),
    "Deslinearização": np.mean([ ])
}
 
# Preparar dados
etapas = list(tempos_jcuda.keys())
valores = list(tempos_jcuda.values())
total = sum(valores)

# Calcular porcentagem de cada etapa
percentuais = [v / total * 100 for v in valores]

plt.figure(figsize=(8,5))
plt.bar(etapas, valores, color='skyblue')
plt.title("Contribuição de cada etapa para o tempo total")
plt.ylabel("Tempo (s)")
plt.xticks(rotation=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("./resultados/tempo-etapa-jcuda.png", dpi=300)
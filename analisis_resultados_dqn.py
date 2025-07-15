import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuración ---
# Nuevo nombre del archivo CSV de DQN
DQN_CSV_PATH = 'dqn_training_history_.csv'
OUTPUT_DIR = 'results_dqn' # Directorio para guardar los gráficos específicos de DQN

# Asegúrate de que el directorio de salida exista
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Iniciando análisis de resultados para DQN...")

# --- 1. Cargar los datos ---
try:
    df_dqn = pd.read_csv(DQN_CSV_PATH)
    print(f"Datos de DQN cargados desde: {DQN_CSV_PATH}")
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo CSV de DQN en '{DQN_CSV_PATH}'. Asegúrate de que esté en la misma carpeta o actualiza la ruta.")
    df_dqn = None

if df_dqn is not None:
    print("\n--- Estadísticas Generales (DQN) ---")
    print(f"Número total de episodios: {len(df_dqn)}")
    print(f"Máximo Score alcanzado: {df_dqn['Score'].max()}")
    print(f"Episodio del Máximo Score: {df_dqn['Score'].idxmax() + 1}") # +1 porque los índices son base 0
    print(f"Máximo de Tuberías Superadas: {df_dqn['Pipes_Surpassed'].max()}")
    print(f"Episodio de Máximas Tuberías Superadas: {df_dqn['Pipes_Surpassed'].idxmax() + 1}")
    print(f"Score medio general: {df_dqn['Score'].mean():.2f}")
    print(f"Tuberías superadas medias: {df_dqn['Pipes_Surpassed'].mean():.2f}")


    # --- 2. Visualizaciones para DQN ---
    print("\nGenerando gráficos para DQN...")

    # Configuración de estilo de Seaborn
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(15, 7)) # Aumentamos el tamaño para mejor visualización

    # Gráfico de Score por Episodio (con media móvil)
    plt.subplot(1, 3, 1) # Cambiado a 1 fila, 3 columnas
    plt.plot(df_dqn['Episode'], df_dqn['Score'], label='Score por Episodio', alpha=0.3, color='skyblue')
    plt.plot(df_dqn['Episode'], df_dqn['Mean_Score_Last_100'], color='red', label='Media (cada 100 episodios)')
    plt.title('DQN: Evolución del Score por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # Gráfico de Tuberías Superadas por Episodio (con media móvil)
    plt.subplot(1, 3, 2) # Cambiado a 1 fila, 3 columnas
    plt.plot(df_dqn['Episode'], df_dqn['Pipes_Surpassed'], label='Tuberías Superadas', alpha=0.3, color='lightgreen')
    plt.plot(df_dqn['Episode'], df_dqn['Pipes_Surpassed'].rolling(window=100).mean(), color='green', label='Media (cada 100 episodios)')
    plt.title('DQN: Evolución de Tuberías Superadas')
    plt.xlabel('Episodio')
    plt.ylabel('Tuberías Superadas')
    plt.legend()
    plt.grid(True)

    # Gráfico de Decaimiento de Epsilon
    plt.subplot(1, 3, 3) # Cambiado a 1 fila, 3 columnas
    plt.plot(df_dqn['Episode'], df_dqn['Epsilon'], label='Epsilon', color='purple')
    plt.title('DQN: Decaimiento de Epsilon')
    plt.xlabel('Episodio')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dqn_training_metrics_10000_episodes.png'))
    plt.show()
    print(f"Gráfico 'dqn_training_metrics_10000_episodes.png' guardado en '{OUTPUT_DIR}'")

else:
    print("\nNo se pudo realizar el análisis. Por favor, verifica la ruta del archivo CSV de DQN.")

print("\nAnálisis de resultados para DQN completado.")
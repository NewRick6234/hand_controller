
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carrega os dados
df = pd.read_csv("dataset/gestures.csv")

# Separa entradas e respostas
X = df.drop("label", axis=1)
y = df["label"]

# Verifica quantidade de amostras
print("\nQuantidade de amostras no dataset:")
print(y.value_counts())

gestos_esperados = ["aberta", "fechada", "apontar_esquerda", "apontar_direita"]
contagens = y.value_counts().to_dict()
faltando = [g for g in gestos_esperados if contagens.get(g, 0) < 30]

if faltando:
    print("\n" + "!" * 60)
    print(f"[ATENCAO] Amostras insuficientes para os gestos: {faltando}")
    print("Cada gesto precisa de no minimo 30 amostras para treinar.")
    print("Execute 'python collect_data.py' para coletar os gestos que faltam.")
    print("!" * 60)
    raise SystemExit(1)

# Divide os dados
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Cria o modelo
modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Treina
print("\nTreinando modelo...")

modelo.fit(X_train, y_train)

# Testa
previsoes = modelo.predict(X_test)

acuracia = accuracy_score(y_test, previsoes)

print("\nResultado do treinamento:")
print("Acuracia:", round(acuracia * 100, 2), "%")

print("\nRelatorio:")
print(classification_report(y_test, previsoes))

# Salva o modelo
joblib.dump(modelo, "hand_model.pkl")

print("\nModelo salvo em hand_model.pkl")

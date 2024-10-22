import requests
import pandas as pd
from sklearn.linear_model import LinearRegression

# Função para obter os dados da API
def obter_dados_cidade(cidade):
    api_key = '1dc1f784d3abea26b268cf151165db0f'
    url = f'http://api.openweathermap.org/data/2.5/weather?q={cidade}&appid={api_key}&units=metric'
    response = requests.get(url)

    # Verificando se a resposta da API foi bem-sucedida
    if response.status_code != 200:
        print(f"Erro ao obter dados da API: {response.status_code} - {response.text}")
        return None

    return response.json()

# Função para preparar os dados
def preparar_dados(dados):
    if dados is None:
        return None
    
    cidade = dados['name']
    temperatura = dados['main']['temp']
    umidade = dados['main']['humidity']
    pressao = dados['main']['pressure']
    descricao_clima = dados['weather'][0]['description']  # Acessando a descrição correta

    # Criação do DataFrame com os dados obtidos da API
    df = pd.DataFrame({
        'cidade': [cidade],
        'temperatura': [temperatura],
        'umidade': [umidade],
        'pressao': [pressao],
        'descricao_clima': [descricao_clima]
    })
    
    return df

# Função para treinar o modelo
def treinar_modelo(df):
    if 'temperatura_futura' not in df.columns or df['temperatura_futura'].isnull().any():
        raise ValueError("A coluna 'temperatura_futura' deve estar presente e não pode conter valores NaN.")

    X = df[['temperatura', 'umidade', 'pressao']]
    y = df['temperatura_futura']
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Função para fazer a previsão de temperatura
def prever_temperatura(model, dados_reais):
    nova_previsao = model.predict([dados_reais])  # Verificando que os dados estão no formato correto
    return nova_previsao[0]

# Função Principal
def main():
    cidade = input("Digite o nome da cidade: ")  # Agora o usuário pode escolher a cidade
    dados = obter_dados_cidade(cidade)

    # Preparar os dados
    df_atual = preparar_dados(dados)

    if df_atual is None:
        print("Não foi possível obter os dados para a cidade especificada.")
        return

    # Dados históricos
    historico_dados = pd.DataFrame({
        'temperatura': [30, 29, 28, 27],
        'umidade': [80, 75, 70, 65],
        'pressao': [1013, 1012, 1011, 1010],
        'temperatura_futura': [31, 30, 29, 28]  # Temperaturas futuras correspondentes
    })

    # Concatenar os dados históricos com os dados obtidos
    df = pd.concat([historico_dados, df_atual], ignore_index=True)

    # Preencher a coluna 'temperatura_futura' para a última linha (a linha atual)
    df['temperatura_futura'] = df['temperatura'].shift(-1)
    df['temperatura_futura'].fillna(df['temperatura'].iloc[-1], inplace=True)

    print("Dados utilizados para treinar o modelo:")
    print(df)

    # Treinar o modelo
    try:
        modelo = treinar_modelo(df)
    except ValueError as e:
        print(f"Erro ao treinar o modelo: {e}")
        return

    # Prever a temperatura com os dados atuais
    dados_reais = [df_atual['temperatura'].iloc[0], df_atual['umidade'].iloc[0], df_atual['pressao'].iloc[0]]
    temperatura_prevista = prever_temperatura(modelo, dados_reais)

    # Exibir o resultado
    print(f"Temperatura atual em {df_atual['cidade'].iloc[0]}: {df_atual['temperatura'].iloc[0]}°C")
    print(f"Previsão de temperatura ajustada: {temperatura_prevista:.2f}°C")

# Executar o programa
main()

import pandas as pd
import joblib
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics

# 1. etapa carregar dados
def carregar_dados(caminho_arquivo= "historicoAcademico.csv"):
    try:

        if os.path.exists(caminho_arquivo):

            df = pd.read_csv(caminho_arquivo, encoding="latin1", sep=',')

            print("o arquivo foi carregado com sucesso")

            return df
        else:
            print("o arquivo não foi encotrado dentro da pasta")

            return None
        
    except Exception as e:
        print("erro inesperado ao carregar o arquivo: ", e)

        return None 

#---------chamar a função para armazenar o resultado----- 

dados = carregar_dados()

#------------Etapa 02: preparação e divisão de dados------------------ 
# definição de X (features) e Y (target)


if dados is not None:
    print(f"\ntotal de registros carregados: {len(dados)}")
    print("iniciando pipeline de treinamento")

    TARGET_COLUMN = "Status_Final"

#etapa 2.1 - definição das features e target
    try:
        X = dados.drop(TARGET_COLUMN, axis=1)
        y = dados[TARGET_COLUMN]

        print(f"FEATURES (X) definidas: {list(X.columns)}")
        print(f"features (y) definidas: {TARGET_COLUMN}")
       
    except KeyError:

        print(f"\n----------- erros ciritco --------")
        print(f"a coluna {TARGET_COLUMN} não foi encontrada no CSV")
        print(f"colunas disponiveis: {list(dados.columns)}")
        print(f"por favor, ajuste avariavel 'TARGET_COLUMN' e tente novamente!!")
        
        #se o target não for encotrado, irá encerrar o script
        exit()

    #etapa 2.2 - divisão entre treino e teste 
    print("\n------dividindo dados em treino e teste-----------")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size=0.2,  #20% dos dados serão ultilizados para teste 
        random_state= 42, #vai garantir a reprodutibilidade
        stratify=y     #vai manter a proporção de aprovados e reprovados 
    )


    print(f"dados de treino: {len(X_train)} | dados de teste: {len(X_test)}")

    #etapa 03 : criação de pipeline de ml
    print("\n -------------------------criando a pipeline de ml---------------")
    pipeline_model = pipeline.Pipeline([
        #scaler --> normalização dos dados(colocando tudo na mesma escala)
        #model---> aplica o modelo de regressão logística 
        ('scaler', preprocessing.StandardScaler()),
        ('model', linear_model.LogisticRegression(random_state=42))
    ])

    #etepa 04: treinamento e avaliação dos dados 

    print("\n---------treinamento do modelo---------")
    #treina a pipeline com os dados do treino
    pipeline_model.fit(X_train, y_train)

    print("modelo treinado avaliando com os dados de teste...")
    y_pred = pipeline_model.predict(X_test)

    #avaliação de desempenho 
    accuracy = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred)

    print("\n---------relatorio de avaliação geral----------")
    print(f"acuracia geral: {accuracy * 100:.2f}%")
    print("\n relatório de classficação detalhado: ")
    print(report)

#etapa 05: salvando o modelo 
    mode_filename = 'model_previsto_desempenho.joblib'

    print(f"\nSalvando o pipeline treinando em.. {mode_filename}")
    joblib.dump(pipeline_model, mode_filename)

    print("processo concluído com sucesso!")
    print(f"o arquivo '{mode_filename}' está para ser ultilizado !")

else:
    print("o pipeline não pode continuar pois os dados não foram carregados ")

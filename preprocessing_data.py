from useful_functions import *


# Pré-processamento dos datasets
def preprocessing(file_t):
    # Leitura dos dados
    data = read_data(file_t)

    # Colunas para analisar os dados e deletar as amostras (rows)
    columns_analyze = ['evolucaoCaso', 'sintomas', 'tipoTeste', 'resultadoTeste', 'classificacaoFinal']

    # Colunas para deletar
    columns_del = ['ÿid', 'dataNascimento', 'estadoTeste', 'paisOrigem', 'municipioIBGE', 'origem', 'estadoNotificacao',
                    'estadoNotificacaoIBGE', 'municipioNotificacao', 'estado', 'estadoIBGE', 'excluido', 'validado',
                    'dataEncerramento', 'dataTeste', 'municipio', 'municipioNotificacaoIBGE', 'cbo', 'dataNotificacao',
                    'dataInicioSintomas']

    # Sintomas e condições para virarem colunas
    features = [['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao'],
                ['assintomatico', 'outros', 'tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza', 'gustativos',
                 'olfativos']]
    colunas = ['condicoes', 'sintomas']

    # Deletando alguns rows com base nos valores nulos ou diferentes do necessário para se trabalhar
    data.dropna(how='any', subset=columns_analyze, inplace=True)
    data.drop(data[data['tipoTeste'] != 'RT-PCR'].index, inplace=True)
    data.drop(data[~data['classificacaoFinal'].str.contains(r'Confirmado')].index, inplace=True)
    data.drop(data[data['resultadoTeste'].str.contains(r'^[In].*')].index, inplace=True)
    data.drop(data[data['evolucaoCaso'] == 'Cancelado'].index, inplace=True)
    data.drop(data[data['evolucaoCaso'] == 'Ignorado'].index, inplace=True)
    del columns_analyze

    # Consertando um probleminha de datas com valores ausentes para poder aplicar funções
    data.loc[data.dataNotificacao.isnull(), 'dataNotificacao'] = data['dataInicioSintomas']
    data.loc[data.dataInicioSintomas.isnull(), 'dataInicioSintomas'] = data['dataNotificacao']
    # Chamando função para consertar as datas e aplicar nas colunas correspondentes
    data['dataNotificacao'] = data['dataNotificacao'].apply(lambda dates: re.sub(r'(\d{4})-(\d{2})-(\d{2})(.*)', r'\1-\2-\3', dates))
    data['dataInicioSintomas'] = data['dataInicioSintomas'].apply(lambda dates: re.sub(r'(\d{4})-(\d{2})-(\d{2})(.*)', r'\1-\2-\3', dates))

    # Correção, muda valores de null para empty strings.
    for col in ['sintomas', 'condicoes']:
        data[col].fillna('', inplace=True)

    # Realizar operação entre as datas, subtraindo dataInicioSintomas de dataNotificacao e cria a coluna diasSintomas
    dataNotificacao = pd.to_datetime(data['dataNotificacao'], errors='coerce', format='%Y-%m-%d')
    dataInicioSintomas = pd.to_datetime(data['dataInicioSintomas'], errors='coerce', format='%Y-%m-%d')
    data.insert(2, "diasSintomas", (dataNotificacao - dataInicioSintomas).dt.days, True)

    # Deletando atributos irrelevantes para análise
    data.drop(columns_del, axis=1, inplace=True)
    del columns_del

    # Adiciona os sintomas e condicoes como colunas no dataframe, cada sintoma/condição vira 1 coluna
    add_column(data, features[1] + features[0])

    # Preenche as colunas dos sintomas e condicoes com 1
    for feature, coluna in zip(features, colunas):
        data = data.apply(standarding_features, axis=1, args=feature, coluna=coluna)

    # Deleta todos os rows que não possuem nenhum dos 8 e 6 principais sintomas, descrevem basicamente um assintomático
    data.drop(data[(data['tosse'] == 0) & (data['febre'] == 0) & (data['garganta'] == 0) & (data['coriza'] == 0) &
                   (data['cabeca'] == 0) & (data['dispneia'] == 0) & (data['olfativos'] == 0) &
                   (data['gustativos'] == 0)].index, inplace=True)
    data.drop(data[(data['tosse'] == 0) & (data['febre'] == 0) & (data['garganta'] == 0) & (data['coriza'] == 0) &
                   (data['cabeca'] == 0) & (data['dispneia'] == 0)].index, inplace=True)

    # Mais algumas colunas e linhas para serem limpas, só podem serem feitas depois de algumas limpezas prévias
    features_columns = ['sintomas', 'assintomatico', 'condicoes', 'outros'] # Modificar para add ou remover olfativos e gustativos
    data.drop(features_columns, axis=1, inplace=True)

    # Trocar alguns valores
    cols_change = ['profissionalSaude', 'resultadoTeste', 'sexo', 'evolucaoCaso'] # Índice 0 no zip
    old_values = [['Não', 'Sim'], ['Negativo', 'Positivo'], ['Feminino', 'Masculino'], [r'.*tratamento.*', r'Internado.*']] # Índice 1 no zip
    new_values = [[0, 1], [0, 1], [0, 1], ['Cura', 'Internado']] # Índice 2 no zip
    use_regex = [False, False, False, True] # Índice 3 no zip

    for values in zip(cols_change, old_values, new_values, use_regex):
        data[values[0]].replace(to_replace=values[1][0], value=values[2][0], regex=values[3], inplace=True)
        data[values[0]].replace(to_replace=values[1][1], value=values[2][1], regex=values[3], inplace=True)

    del cols_change, old_values, new_values, use_regex

    # Converte colunas selecionadas para int
    for col in ['sexo', 'idade', 'profissionalSaude']:
        data[col].replace(to_replace=r'\D', value=0, inplace=True, regex=True)
        data[col].fillna(0, inplace=True)
        data[col] = data[col].astype(int)

    # Salvando o arquivo
    saving_data(data, file_t)

    return data
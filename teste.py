import pyodbc

# Informações da conexão
server = '10.0.0.14'  # Exemplo: 'localhost'
database = 'softran_rasador'  
username = 'softran'  
password = 'sof1320'  

# String de conexão
conexao_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Conexão com o banco de dados
conexao = pyodbc.connect(conexao_str)

# Comandos SQL
# Função para ler o conteúdo SQL do arquivo
def ler_sql_do_arquivo(nome_do_arquivo):
    with open(nome_do_arquivo, 'r') as arquivo:
        return arquivo.read()

# Lendo os comandos SQL dos arquivos
sql_comando1 = ler_sql_do_arquivo('C:\\Users\\juare\\Desktop\\TCC\\Dados TCC.sql')
sql_comando2 = ler_sql_do_arquivo('C:\\Users\\juare\\Desktop\\TCC\\Dados TCC Plus.sql')

# Executando os comandos
conexao = pyodbc.connect(conexao_str)
cursor = conexao.cursor()

try:
    cursor.execute(sql_comando1)
    resultado1 = cursor.fetchall()
    print("Resultado 1:", resultado1)

    cursor.execute(sql_comando2)
    resultado2 = cursor.fetchall()
    print("Resultado 2:", resultado2)

finally:
    cursor.close()
    conexao.close()
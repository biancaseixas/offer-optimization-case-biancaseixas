# ifood-offer-optimization

Modelo de recomendação de ofertas com estimativa de impacto incremental de receita - case técnico iFood.

## Estrutura do repositório

```
ifood-offer-optimization/
├── data/                       # Datasets
│   ├── raw/                    # Dados originais (offers.json, profile.json, transactions.json)
│   └── processed/              # Dados processados / dataset unificado
├── notebooks/                  # Jupyter notebooks
│   ├── 1_data_processing.ipynb # Limpeza e preparação do dataset unificado (PySpark)
│   └── 2_modeling.ipynb        # Treino, avaliação e estimativa de impacto do modelo
├── presentation/                # Slides para stakeholders de negócio
├── src/                         # Código fonte reutilizável (funções PySpark de processamento)
│   └── data_processing.py
├── README.md
└── requirements.txt
```

## Como executar

Testado no Windows com Python 3.11 + PySpark 3.5.1. **PySpark 3.5.1 tem um bug conhecido de
incompatibilidade com Python 3.12 no Windows** (o worker do Spark crasha silenciosamente) — use
Python 3.10 ou 3.11.

1. Criar e ativar um ambiente virtual (Python 3.10/3.11) e instalar as dependências:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Baixar os dados brutos e extrair em `data/raw/`:
   - `offers.json`, `profile.json`, `transactions.json`
   (link no PDF do case, em `case/` se você tiver adicionado)
3. **Somente no Windows**, o Spark local precisa de `winutils.exe`/`hadoop.dll` (Hadoop
   LocalFileSystem) para gravar arquivos:
   - Baixe os dois arquivos de uma versão próxima à do Hadoop empacotado no PySpark (ex.:
     [cdarlint/winutils](https://github.com/cdarlint/winutils), pasta `hadoop-3.3.6/bin/` para
     PySpark 3.5.1) e coloque em `C:\hadoop\bin\`.
   - O notebook já configura `HADOOP_HOME`, `JAVA_HOME` e `PYSPARK_PYTHON` automaticamente na
     célula de setup — ajuste os caminhos lá se o seu `JAVA_HOME` for diferente.
4. Registrar o kernel do venv no Jupyter (para o `jupyter nbconvert`/Jupyter Notebook usarem o
   Python certo, com PySpark instalado):
   ```
   python -m ipykernel install --user --name ifood-case --display-name "Python (ifood-case)"
   ```
5. Rodar `notebooks/1_data_processing.ipynb` (selecionando o kernel "Python (ifood-case)") para
   gerar o dataset unificado em `data/processed/opportunities/`.
6. Rodar `notebooks/2_modeling.ipynb` para treinar/avaliar o modelo e gerar as estimativas de
   impacto.

Alternativa recomendada pelo próprio case: rodar em
[Databricks Community Edition](https://community.cloud.databricks.com/), que evita toda a
configuração de ambiente local acima.

## Premissas

Ver a seção "Premissas e observações" ao final de `notebooks/1_data_processing.ipynb` para a
lista completa (tratamento de `age==118`, aleatoriedade da atribuição de ofertas, grão da tabela
de oportunidades, critério de sucesso das ofertas informationais, etc.). Novas premissas de
modelagem serão adicionadas conforme o notebook 2 avança.

## Apresentação

Slides voltados a líderes de negócio (não técnicos) em `presentation/`, com estimativas/projeções de impacto da estratégia proposta.

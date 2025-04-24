# Implementação de Rede Neural Profunda

## Descrição
Esta atividade implementa uma **rede neural profunda** para classificação binária, expandindo os conceitos da regressão logística para múltiplas camadas.

## Notação Matemática

### Elementos Básicos
- `m`: Número de instâncias
- `f`: Número de atributos
- `n[l]`: Unidades na camada `l`
- `X`: Matriz de entrada (m × f)
- `y`: Vetor de classes reais (m × 1)

### Matrizes por Camada
| Símbolo | Descrição | Dimensão |
|---------|-----------|----------|
| `Z[l]`  | Saídas lineares | m × n[l] |
| `A[l]`  | Ativações | m × n[l] |
| `W[l]`  | Pesos | n[l] × n[l-1] |
| `Zd[l]` | Gradientes de Z | m × n[l] |

## Componentes Implementados

### 1. FuncaoAtivacao
```python
sigmoid = FuncaoAtivacao(
    lambda z: 1/(1+np.exp(-z)),
    lambda a, dz_w_prox: a*(1-a)*dz_w_prox,
    dz_ultima_camada=lambda a, y: a-y
)
```

### 2. Unidade
- **Forward propagation**:  
  Cálculo da saída linear:  
  `z = w·A_ant + b`  
  (onde `A_ant` são as ativações da camada anterior)

- **Backward propagation**:  
  Cálculo dos gradientes:
  - `∇w`: Gradiente dos pesos
  - `∇b`: Gradiente do bias

### 3. Camada
```python
@property
def mat_w(self):  # Stack de pesos (n[l] × n[l-1])
@property 
def mat_dz(self): # Gradientes agrupados
```

### 4. Rede Neural
```python
def fit(self, X, y, lr):
    # 1. Forward pass
    # 2. Backward pass 
    # 3. Atualização de pesos
```

## Instruções de Execução

### Instalação de Dependências

Certifique-se de ter o Python e os pacotes necessários instalados:

```bash
apt-get install python3 jupyter python3-pip
pip install -r requirements.txt
```

Atenção: Evite usar sudo com o pip.

### Executando o Jupyter Notebook

```bash
jupyter notebook
```
Abra o notebook fornecido e execute as células conforme as instruções no arquivo.

---

## Créditos

Atividade desenvolvida para a disciplina de Machine Learning do CEFET-MG, Campus Nova Gameleira.

Material baseado nas aulas do professor Daniel Hasan Dalip.
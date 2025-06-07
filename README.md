## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/Teibao2004/AOOP-ComputerVision.git
cd AOOP-ComputerVision
```

### 2. Criar um ambiente virtual (OPCIONAL)
```bash
python -m venv venv
source venv/bin/activate # Se estiver a usar Linux ou macOS
venv\Scripts\activate    # Se estiver a usar Windows
```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

## Utilização

### 1. Adicionar um video de futebol a `videos/` com o nome `video.mp4`
- Poderá também trocar o video_path que se encontra na 6 linha do script.

### 2. Executar o script
```
python main.py
```

### 3. Durante a execução:
- É aberta uma janela com o vídeo e as deteções visuais feitas pelo modelo.
- Pode pressionar a tecla `q` para sair a qualquer momento

### 4. No fim da execução:
Será fechada a janela e apresentada a percentagem de posse de bola de cada equipa:
```bash
=== POSSE DE BOLA FINAL ===
Equipa 1: 64.2%
Equipa 2: 35.8%
```
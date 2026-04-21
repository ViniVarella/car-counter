# 🚗 Contador de Veículos

Contagem eficiente de veículos em vídeos utilizando um programa de visão computacional baseado em YOLOv8 para o monitoramento otimizado do tráfego.

## 🌐 Visão Geral

Este projeto usa técnicas de visão computacional para detectar, rastrear e contar veículos em vídeos de tráfego. A implementação combina o modelo de detecção YOLO (You Only Look Once) com o algoritmo SORT (Simple Online and Realtime Tracking), permitindo acompanhar os objetos ao longo dos frames e registrar a passagem deles por uma linha de contagem.

## 🚀 Recursos

- **Detecção de objetos:** utiliza o YOLO para identificar veículos no vídeo.
- **Rastreamento:** aplica o SORT para manter IDs de rastreamento em tempo real.
- **Contagem:** registra veículos que cruzam a linha configurada no vídeo.
- **Visualização:** exibe o vídeo processado com caixas, IDs, linha de contagem e painel com o total.

## 🛠️ Requisitos

Certifique-se de ter os seguintes componentes instalados:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV (`cv2`)
- `cvzone`
- `SORT` (Simple Online and Realtime Tracking)
- Python `3.14`
- GPU NVIDIA com suporte a CUDA, caso queira inferência mais rápida

## 🏗️ Instalação

1. **Clone o repositório:**

   ```bash
   git clone https://github.com/ViniVarella/car-counter.git
   cd car-counter
   ```

2. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

   Se você já criou um ambiente virtual com dependências antigas ou incompatíveis, atualize o `pip` antes:

   ```bash
   python -m pip install --upgrade pip
   ```

3. **Verifique se a GPU está disponível no PyTorch:**

   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

4. **Defina o arquivo de pesos YOLO (`.pt`) no `MODEL_PATH` dentro do `main.py`.**

   Opcoes comuns de peso:

   - `yolov8n.pt`: mais leve e mais rapido
   - `yolov8s.pt`: equilibrio entre velocidade e precisao
   - `yolov8m.pt`: mais pesado, com tendencia a melhor precisao
   - `yolov8l.pt`: ainda mais pesado, indicado quando desempenho nao for problema
   - `yolov8x.pt`: o mais pesado da linha YOLOv8 padrao

   Se voce tiver acesso a internet, o Ultralytics baixa automaticamente o peso informado na primeira execucao, caso o arquivo ainda nao exista localmente.

5. **Execute o script:**

   ```bash
   python main.py
   ```

   O projeto usa CUDA automaticamente quando `torch.cuda.is_available()` retorna `True`.

## ⚙️ Configuração

Os principais ajustes podem ser feitos diretamente em [main.py](C:/Users/vinic/PycharmProjects/car-counter/main.py):

- `MODEL_PATH`: caminho do arquivo de pesos YOLO.
- `VIDEO_PATH`: caminho do vídeo de entrada.
- `INFERENCE_SIZE`: resolução usada na inferência.
- `PROCESS_EVERY_N_FRAMES`: quantidade de frames pulados entre uma inferência e outra.
- `CONFIDENCE_THRESHOLD`: confiança mínima para considerar uma detecção.
- `USE_MASK`: ativa ou desativa o uso de máscara na inferência.
- `MASK_PATH`: caminho da máscara usada quando `USE_MASK = True`.
- `COUNT_LINES`: lista de dicionários com `x`, `y`, `angle` e `length` para definir uma ou mais linhas de contagem.
  `angle` segue a convenção visual da tela: `0` aponta para a direita, valores positivos sobem e valores negativos descem.

## 📹 Saída

O vídeo processado é salvo como `result.mp4` na raiz do projeto.

## 🙌 Créditos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- `SORT`

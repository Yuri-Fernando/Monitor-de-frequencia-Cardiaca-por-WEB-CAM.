🫀 Monitor de Frequência Cardíaca via Webcam (rPPG)

Python
OpenCV
NumPy
Status

Este projeto implementa um monitor de frequência cardíaca em tempo real utilizando Fotopletismografia Remota (rPPG). Através do processamento de vídeo de uma webcam comum, o sistema detecta variações imperceptíveis na coloração da pele causadas pelo fluxo sanguíneo microvascular, extraindo o batimento cardíaco com alta precisão.

Desenvolvido para a disciplina de Processamento Digital de Sinais em Tempo Discreto no Programa de Pós-Graduação em Engenharia Elétrica da UFSCar.

🛠️ Tecnologias e Conceitos Aplicados

    Fotopletismografia (PPG): Medição de mudanças no volume sanguíneo através da luz.
    Visão Computacional: Utilização da biblioteca OpenCV para captura e manipulação de frames.
    Pirâmides Gaussianas: Implementação de funções buildGauss e reconstructFrame para decomposição espacial e redução de ruído.
    Filtros Digitais FIR: Projeto de um filtro Passa-Banda (1Hz - 2Hz) para isolar a frequência cardíaca humana (45 - 180 BPM).
    Transformada de Fourier (FFT): Análise espectral do sinal extraído para identificação do pico de frequência correspondente ao pulso.

📐 O Projeto do Filtro

Para garantir a fidelidade do sinal biomédico, aplicamos um filtro de resposta de impulso finita (FIR) com janelamento Gaussiano.

A frequência de corte foi definida dinamicamente baseada no FPS da câmera:
f=1.0⋅FPS⋅[0,149]ordem do filtro
f=ordem do filtro1.0⋅FPS⋅[0,149]​

O sistema utiliza uma máscara de frequência que admite apenas valores dentro da banda passante, eliminando ruídos de iluminação e movimentação.


📊 Resultados

O sistema foi validado comparando os dados obtidos via webcam com um smartwatch simultaneamente.

    Frequência Detectada: Entre 79,5 e 83,4 BPM.
    Precisão: Resultados consistentes com a literatura de engenharia biomédica para adultos em repouso.


💻 Como Executar

    Clone o repositório:

    bash
    Copy
    git clone https://github.com/seu-usuario/Monitor-de-frequencia-Cardiaca-por-WEB-CAM.git  

    Instale as dependências:

    bash
    Copy
    pip install numpy opencv-python  

    Execute o script principal:

    bash
    Copy
    python main.py  

👥 Autores

    André L. C. S. Santos
    Thales M. Marques
    Yuri F. Dubbern

📚 Referências

    McDuff, J. D. et al. (2017). Fusing Partial Camera Signal for Non-Contact Pulse Rate Variability Measurement.
    Proakis, J. G. (2007). Digital Signal Processing.


MIT License  
  
Copyright (c) 2026 Yuri F. Dubbern, André L. C. S. Santos, Thales M. Marques  
  
Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:  
  
The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.  
  
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.


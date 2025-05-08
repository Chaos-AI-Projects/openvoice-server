# MIT License
#
# Copyright (c) 2025 Jun Sheng (Aka Chaos Eternal)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM python:3.10
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 ffmpeg\
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/myshell-ai/MeloTTS.git \
    && cd MeloTTS \
    && git checkout 209145371cff8fc3bd60d7be902ea69cbdb7965a \
    && pip install -e .
RUN python -m unidic download
RUN cd MeloTTS && python melo/init_downloads.py

RUN git clone https://github.com/myshell-ai/OpenVoice.git \
    && cd OpenVoice \
    && git checkout 74a1d147b17a8c3092dd5430504bd83ef6c7eb23 \
    && pip install -e .

RUN curl -o checkpoints_v2_0417.zip https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip \
    && unzip checkpoints_v2_0417.zip && rm checkpoints_v2_0417.zip

RUN python3 -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"

RUN python3 -c "from openvoice import se_extractor;from openvoice.api import ToneColorConverter;\
tone_color_converter = ToneColorConverter('checkpoints_v2/converter/config.json', device='cpu');\
tone_color_converter.load_ckpt('checkpoints_v2/converter/checkpoint.pth')"

RUN python3 -c "import numpy; import torch;\
  from openvoice import se_extractor;\
  t = torch.tensor(numpy.array([0.1]*10), dtype=torch.float);\
  se_extractor.get_vad_segments(t)"

RUN python3 -c "from melo.api import TTS; tts = TTS(language='EN', device='cpu'); tts.tts_to_file('hello world', 1, '/tmp/a.wav', speed = 0.8)"

RUN python3 -c "from melo.api import TTS; tts = TTS(language='ZH', device='cpu'); tts.tts_to_file('hello world', 1, '/tmp/a.wav', speed = 0.8)"

COPY requirements.in /app
RUN pip install -r requirements.in

COPY ovserver.py /app
RUN rm /tmp/a.wav
ENTRYPOINT ["python3", "ovserver.py"]
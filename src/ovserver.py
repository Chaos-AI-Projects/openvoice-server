#!/usr/bin/env python3
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

"""An openvoice TTS server."""
import argparse
import json
import os
import shutil
import tempfile

import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

from melo.api import TTS

import numpy as np

import falcon
from falcon.media.multipart import MultipartForm
from falcon import Request, Response
from wsgiref import simple_server

from typing import List
from sqlalchemy import create_engine
from sqlalchemy.orm import (DeclarativeBase,
                            Mapped,
                            Session,
                            mapped_column,
                            sessionmaker)


CHECKPOINT_PATH = os.getenv("CHECKPOINTS_PATH", "checkpoints_v2")
ckpt_converter = os.path.join(CHECKPOINT_PATH, "converter")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def vector2hexid(vector: np.ndarray):
    binarray = np.digitize(vector, [0])
    if len(binarray) != 256:
        raise ValueError("length of vector should be 256")

    hex_string = ""
    for i in range(0, 256, 8):
        byte_array = binarray[i:i+8]
        byte_int = 0
        for bit in byte_array:
            byte_int = (byte_int << 1) + bit
        hex_string += f"{byte_int:02X}"
    return hex_string

def array2json(array):
    return json.dumps(array.tolist())

def json2array(js):
    l = json.loads(js)
    return np.array(l)

class Base(DeclarativeBase):
    pass

class ToneColor(Base):
    __tablename__ = "tone_colors"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    desc: Mapped[str]
    tone_embedding: Mapped[str]

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///tonecolor.db")
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

class TonesController:
    """Falcon Controller for generating speech using tone.
    """
    def __init__(self,
                 tone_converter_config: str,
                 languages: List,
                 source_se_path: str,
                 device: str = "",
                 water_mark: str = "default"):
        """
        Args:
              tone_converter_config: f'{ckpt_converter}/config.json'
              languages: ["EN", "ZH"]
              source_se_path: f'checkpoints_v2/base_speakers/ses/'
              device: cpu or cuda
              water_mark to be added to the generated audio
        """
        self.water_mark = water_mark
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tone_color_converter = ToneColorConverter(tone_converter_config,
                                                       device=self.device)
        self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        self.models = {}
        self.speaker_id = {}
        self.source_se = {}
        self.speaker_map = {}
        for l in languages:
            self.models[l] = TTS(language=l, device=self.device)
            for k, v in self.models[l].hps.data.spk2id.items():
                dialect = k.lower().replace("_", "-")
                self.source_se[dialect] = torch.load(
                    f"{source_se_path}/{dialect}.pth",
                    map_location=self.device
                )
                self.speaker_map[dialect] = l
                self.speaker_id[dialect] = v

    # pylint: disable=W0613
    def on_get_cap(self, req: Request, resp: Response):
        result = []
        for k, v in self.speaker_map.items():
            result.append(dict(lang=v, dialect=k))
        resp.media = dict(desc=("supported languages, "
                                "use dialect as model in openai api"),
                          lang=result)

    # pylint: disable=W0613
    def on_get_status(self, req: Request, resp: Response):
        resp.media = dict(status="ok")

    def on_post(self, req: Request, resp: Response):
        """Create tone embedding by upload refrence audio.
        """
        form: MultipartForm = req.get_media()
        session: Session = SessionLocal()
        tone_desc = ""
        tone_name = None

        with tempfile.TemporaryDirectory() as tempdir:
            uploaded_audio = open(f"{tempdir}/uploaded_audio", "wb")
            for part in form:
                if part.name == "audiofile":
                    shutil.copyfileobj(part.stream, uploaded_audio)
                    uploaded_audio.close()
                if part.name == "name":
                    tone_name = part.text
                if part.name == "desc":
                    tone_desc = part.text
            if tone_name is None:
                raise falcon.HTTPBadRequest(title="missing tone_name")
            if not uploaded_audio.closed:
                raise falcon.HTTPBadRequest(title="missing audio")

            target_se, _ = se_extractor.get_se(uploaded_audio.name,
                                               self.tone_color_converter,
                                               vad=True)
        embedding_vector = target_se.numpy(force=True).reshape(256)
        vector_string = np.array2string(embedding_vector, separator=", ")
        tone_id = vector2hexid(embedding_vector)
        tone_color = (session.query(ToneColor)
                      .filter(ToneColor.id == tone_id)
                      .first())
        if not tone_color:
            tone_color = ToneColor(id=tone_id,
                                   name=tone_name,
                                   desc=tone_desc,
                                   tone_embedding=vector_string)
            session.add(tone_color)
            session.commit()
            resp.status = falcon.HTTP_CREATED
        else:
            resp.status = falcon.HTTP_CONFLICT
        resp.media = {"id": tone_id,
                      "name": tone_name,
                      "desc": tone_desc,
                      # "tone_embedding": vector_string,

                      }
        session.close()

    def on_post_tts(self,
                    req: Request,
                    resp: Response,
                    tone_id: str,
                    dialect: str):
        """Create speech from text and selected tone.
        """
        if not dialect in self.speaker_map:
            raise falcon.HTTPBadRequest(title="dialect not supported")


        session: Session = SessionLocal()
        data = req.media
        text = data["text"]
        speed = float(data.get("speed", 1.0))
        tone = session.query(ToneColor).filter(ToneColor.id == tone_id).first()
        if not tone:
            raise falcon.HTTPNotFound(title="Given tone id not exists")
        self._make_tts(dialect, tone, text, speed, resp)
        session.close()

    def _make_tts(self, dialect, tone, text, speed, resp: Response):

        language = self.speaker_map[dialect]
        tts = self.models[language]
        speaker_id = self.speaker_id[dialect]
        source_se = self.source_se[dialect]

        # pylint: disable=E1121
        target_se_np = (json2array(tone.tone_embedding).
                        reshape(1, 256, 1))
        target_se = torch.tensor(target_se_np,
                                 dtype=torch.float,
                                 device=self.device)
        with tempfile.TemporaryDirectory() as tempdir:
            generated_audio_path = f"{tempdir}/generated.wav"
            toned_audio_path = f"{tempdir}/toned.wav"
            tts.tts_to_file(text, speaker_id, generated_audio_path, speed=speed)

            self.tone_color_converter.convert(
                    audio_src_path=generated_audio_path,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=toned_audio_path,
                    message=self.water_mark)

            resp.downloadable_as = "tts.wav"
            resp.content_type = "audio/x-wav"

            resp.stream = open(toned_audio_path, "rb")
            resp.status = falcon.HTTP_200


    def on_get(self, req: Request, resp: Response):
        """Query tone id
        """
        session: Session = SessionLocal()
        query = session.query(ToneColor)
        name = req.get_param("name")
        if name:
            query = query.filter(ToneColor.name == name)
        desc = req.get_param("desc")
        if desc:
            query = query.filter(ToneColor.desc.ilike(f"%{desc}%"))
        result = []
        for tc in query.all():
            result.append(dict(id=tc.id, name=tc.name, desc=tc.desc))
        resp.media = result
        session.close()

    def on_get_tone(self, req: Request, resp: Response, tone_id: str):
        """get tone by id
        """
        session: Session = SessionLocal()
        tone = session.query(ToneColor).filter(ToneColor.id == tone_id).first()
        if tone:
            resp.media = dict(id=tone.id,
                              name=tone.name,
                              desc=tone.desc,
                              tone_embedding = tone.tone_embedding)
        else:
            raise falcon.HTTPNotFound(title="Given tone id not exists")
        session.close()

    def on_post_openai(self,
                       req: Request,
                       resp: Response,
                       comp: str,
                       func: str):
        if comp != "audio":
            raise falcon.HTTPBadRequest(
                title="Only /v1/audio/speech are supported"
                )
        if func != "speech":
            raise falcon.HTTPBadRequest(
                title="Only /v1/audio/speech are supported"
                )
        data = req.media
        dialect = data["model"]
        input_text = data["input"]
        tone_name = data["voice"]

        speed = float(data.get("speed", 1.0))

        if data.get("response_format", "wav") != "wav":
            raise falcon.HTTPBadRequest(
                title="Only wav is supported in response_format"
            )

        session: Session = SessionLocal()
        tone = (session.
                query(ToneColor).
                filter(ToneColor.name == tone_name).
                first())

        self._make_tts(dialect, tone, input_text, speed, resp)
        session.close()

app = falcon.App()

tones = TonesController(f"{ckpt_converter}/config.json",
                        ["EN", "ZH"],
                        f"{CHECKPOINT_PATH}/base_speakers/ses/"
                        )

app.add_route("/tones/{tone_id}", tones)
app.add_route("/tones/{tone_id}/{dialect}", tones, suffix="tts")
app.add_route("/tones", tones)
app.add_route("/status", tones, suffix="status")
app.add_route("/cap", tones, suffix="cap")
app.add_route("/v1/{comp}/{func}", tones, suffix="openai")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TTS Server using OpenVoice"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Run on this address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Listen on this port"
    )
    args = parser.parse_args()
    httpd = simple_server.make_server(args.host, args.port, app)
    httpd.serve_forever()

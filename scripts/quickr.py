import json
import os,sys
import shutil
import string
import pathlib
from subprocess import Popen, PIPE
import numpy as np
from PIL import Image
from random import randint
import platform
import modules

from modules.script_callbacks import on_cfg_denoiser,remove_current_script_callbacks

import gradio as gr
from modules import scripts
from modules.images import save_image
from modules.sd_samplers import sample_to_image
from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules.processing import Processed, process_images, StableDiffusionProcessingImg2Img
from modules import processing
from modules.shared import state
import json
from moviepy.editor import VideoFileClip

try: # make me know if there is better solution
    import skvideo
except:
    if 'VENV_DIR' in os.environ:
        path = os.path.join(os.environ["VENV_DIR"],'scripts','python')
    elif 'PYTHON' in os.environ:
        path = os.path.join(os.environ["PYTHON"], 'python')
    else:
        from shutil import which
        path = which("python")
    os.system(path+ " -m pip install sk-video")
    import skvideo

class LatentMemory:
    def __init__(self, interp_factor=0.1, scale_factor = 0.95):
        self.latents_now = []
        self.latents_mem = []
        self.flushed = False
        self.ifactor = interp_factor * 0.5 #
        self.nowfactor = self.ifactor
        self.scalefactor = scale_factor

    def put(self,latent):
        self.latents_now.append(latent)

    def get(self):
        return self.latents_mem.pop(0)

    def interpolate(self, latent1, latent2):
        latent = latent1 * (1. - self.nowfactor) + latent2 * self.nowfactor
        self.nowfactor = self.nowfactor * self.scalefactor
        return latent

    def flush(self):
        self.latents_mem = self.latents_now
        self.latents_now = []
        self.nowfactor = self.ifactor
        self.flushed = True


class Script(scripts.Script):
    def title(self):
        return 'Quickr'

    def show(self, is_img2img):
        return scripts.AlwaysVisible
        #return is_img2img

    def __init__(self):
        self.img2img_component = gr.Image()
        self.img2img_inpaint_component = gr.Image()
        self.is_have_callback = False
        #from RIFE_HDv3 import Model
        #model = Model()
        #model.load_model('flownet.pkl', -1)


    # ui components
    def ui(self, is_visible):
            def img_dummy_update(arg):
                if arg is None:
                    dummy_image = "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIAAQMAAADOtka5AAAABlBMVEUAAAD///+l2Z/dAAAHZ0lEQVQYGe3BsW7cyBnA8f98JLQEIniJKxIVwS2BAKlSqEwRaAnkRfQILlMY3pHPQNp7g7vXCBBAIyMPkEcYdVfOuRofaH4huaslJZFr2W3m9yNJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiT5PyNA1uY8IjzYasNKw1rVgmqAjapft2z1NsAmkgN/NG6rGpiRqTp2lrh1mGYVYXujmMgqriJsLSvgrWky1cAME9eBHbzeOEzMImzZYQImZBG2sAYib0wgcCQcxY/n3EIIFdDmDDQAbc7AYiI/AzVHwkiFA6FX4zCevRpKkMBHHhEmbhiUdISOtzg6wkHu6XmOhAlHr4UMHFwDYuk4uIaLmsIReUSY8JZeQcfQCXVOz/BY4Eh4RuEMPMYRUXoe4+iVloZHhIlQ08vpWTrVGQNL5y8VFbSA5Uh4zlJCQKApS3oBYYEwESt6QgPUWFoaBjUWrkuUp4TnHJ1Ir7igF+m1UNMzjIQ5F3Qy0LxmLwPjOBBGwnP3dJqKjg30mopewWDdMBLmXNI5A+SavTNA6aw0MCU8FyzQlnTuGLQlIJbOJ378NWckzLmmcw54x945nfd0rKVlJMyo6RRMFEDG4BaUkfBcNIBSAYGBUoHRdzwnTHkGlQPyyCiPUACGp3ImCgYNPcuEBT6bKxwDy5Ew4zsLyCUjuaRjeE6YKB29lp5jwgHneLzlCWHG7+iYa0bmmqmcI2GisvdwjdLzTHjggmBDTS/nSJjYstfSc0w4pkqOhIm3gAEsncBEAGqoY8UTwsg0BCuOBYFIU9K74Eg4Kl5FYp1b+DedaBlFCxaq9hwocBwJD2QV/ktz+Qe+A5NLDZWlrEHOpYbqpqQo9QfjzlbKnK02YLQx6suNaiOZRqMNbPW2kUy1Zduw0bBV9Szaek7K1JIkSZIk325n+Saq+pM2kKnPFIxavs5G9UbVsr6J7CxZZIEw75e7Qj9VNeXuPQ6ILBCWNPBLxYYP+JplwpIWmpIr7unkLBFOKXhDsFQsE5YotBigbuCMJcIiSycSoWSZcJIAVQvnLBGWFBwVLBNOiQQaFCqWCCcIOVCSE1kiLCkhDwwsy4QlFRSeg0uWCAu2Di5rBh9YJixb/YsH1ywRFtzVkLN3zzJhiecTD4xjibAkwDsOLIuEE+7YC8Ii4RRLJ0BtWSIsieBpSjqRZcIJoQZy4E8sEk6IFR1PwzLhlLJl8HsWCSc0UFA4WpYJJ7SFMmhYJJygOe8pLcoy4RThAxtOEk654Z4r4B98tXXANGzd+i7yN15nka+kLRt1m5Cpz9RVW7V8IyVJkiRJesq8VWRCONLITvVWG1aqLSsc8BbWalef9bOqZ6fq+esZ87aezT9jpurYacsmC2DUkrWYuHUbbVmpeuLGMmvjeGWCievAFd9zRYRMPaYBv7Zrt7OZN96ElWMkHAUG8eM58AtXvIccUPbcj9egVkJTMRKOIrR0VOi94QMURQkWPJ1Y0sl9WzISnrsBGhO5h8+/FRw1BeAKpwUjYVRxTs/REcDSKCAYR6elV0LOSDhqOPAWWgIBSjqOPc0Bj8UyEkYlFUc5UBOxYBBGFQgj4aiFSC/UUDIIdDxYeu/A8IQwKpjyRA4sEwqOkXCkcEkvVkABVECEQMbgjk4NhpEwyhlVTNQcCU8II4vlQc3Ba4gUPMh5ImckePaM/voTo6rlYBMNjwkjxwwPDScIE8axp8ayZ+idc3Bf8IQwMowce+KAkqOGJ4SRF8sDz4GFlgsetDwhTFieyegULBNGgQPPnocCUA6UPWUkTPydQQGEWNF7RSe/ZFCDcscjwijyWAE0dCyP1YyEiT8zKB17Di7oyDWDt+y1jIRRw8QbBpd0HKPC84gwUTKoLBhbUlq4FsAwMA1QYHnDSBi1DYMtIC0943LAO3oSgSrwiPDcWzpasAGxZxzlgU6sTWQkTER6pgEy3nMFuQY6ll7hgcumksBIGCm94lXEcMEH3kBBL9CrXztKY9sy9ywIZE2m6lBt2fwQYKMRWMPGb7RlreppNpYF12z1VhtWqg0rdbDVCJxh9LOqZ6fq2LYsuWbCtDw4Y2odSJIkSZJvlrU5X2ulrdGGlYa1stH/bPXWs4u83E7dxsPOEnlrmlVcBVaWl7tahbWFHbwm8saELJIxS5j1W0HnFoKJ/AyfcxYIszTnQAIfAWGB8AW55yRh3jsOCkcEDAuEl/GWecIXlJYGCDXzhC+ooOUE4WVixTzhC5RBUzJPmPeWg5pBWzBPeBnNmSfMMg1PWOYJs84iRysNgDBPmGNi4OgTPwJ3zBPmqP2ZkbWAt8zKmXXDmqNbOqFmlvBCkXnCFxgOKmYJX+DYa0pmCS/UFswSllV0vGWgObOEZTVTllnCMn8P16HmJGGZZcoxSzjJECv2PLOERYZgxTUle4FZwpL6jFjntj1nL/IVdrBRtVn4HnamzRrjyXi5lQajqtZoAxsNW/2pJeMbbD1kaukZkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiR5uf8BYlCmiXFq3J0AAAAASUVORK5CYII="
                    import io
                    import base64
                    dummy_image = io.BytesIO(base64.b64decode(dummy_image))
                    img = Image.open(dummy_image)
                    return img
                else:
                    return arg
            with gr.Row():
                file = gr.File(label="Upload Video", file_types = ['.*;'], live=True, file_count = "single")
                tmp_path = gr.Textbox(label='Or path to file', lines=1, value='')
            with gr.Row():
                fps = gr.Slider(
                    label="FPS change",
                    minimum=1,
                    maximum=60,
                    step=1,
                    value=24,
                )
            with gr.Row():
                gr.HTML(value='<div class="text-center block">Latent space temporal blending</div></br>')
            with gr.Row():
                with gr.Column(min_width=100):
                        freeze_input_fps = gr.Checkbox(label='Keep input frames', value=False)
                        keep_fps = gr.Checkbox(label='Keep FPS', value=False)
                with gr.Column(min_width=100):
                        sfactor  = gr.Slider(
                            label="Strength",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.2,
                        )
                        sexp = gr.Slider(
                            label="Strength scaling (per step)",
                            minimum=0.9,
                            maximum=1.1,
                            step=0.005,
                            value=1.0,
                        )
            file.upload(fn=img_dummy_update,inputs=[self.img2img_component],outputs=[self.img2img_component])
            tmp_path.change(fn=img_dummy_update,inputs=[self.img2img_component],outputs=[self.img2img_component])
            #self.img2img_component.update(Image.new("RGB",(512,512),0))
            return [tmp_path, fps, file,sfactor,sexp,freeze_input_fps,keep_fps]

    def after_component(self, component, **kwargs):
        if component.elem_id == "img2img_image":
            self.img2img_component = component
            return self.img2img_component

    def run(self, p:StableDiffusionProcessingImg2Img, file_path, fps, file_obj, sfactor, sexp, freeze_input_fps, keep_fps, *args):
        # for now just try cut the video into frames and save them to tmp
        # to see if it works
        print("We are running Quickr...")
        os.makedirs("tmp", exist_ok=True)
        input_video = VideoFileClip(file_path)

        # extract the audio
        audio = input_video.audio
        audio.save_audiofile("./tmp/audio.mp3")
        print("Audio saved")

        # extract the frames
        input_video.write_images_sequence("./tmp/frame%06d.png", fps=fps)
        print("Frames saved")





import reflex as rx
import yt_dlp
import os
import librosa
import numpy as np
import tempfile
import pygame
import shutil
import glob
import threading
import asyncio
from pydub import AudioSegment
from pydub.generators import Sine
import re

class State(rx.State):
    url: str = ""
    status: str = ""
    download_path: str = os.path.expanduser("~/Downloads")
    video_info: dict = {}
    show_thumbnail: bool = False
    audio_file: str = ""
    bpm: float = 0
    beat_times: list = []
    is_playing: bool = False
    audio_duration: float = 0
    temp_dir: str = ""
    manual_bpm: float = 0
    metronome_volume: float = -20
    download_progress: int = 0
    temp_files: list = []
    progress_value: int = 0
    is_processing: bool = False

    @rx.background
    async def get_video_info(self):
        if not self.url:
            async with self:
                self.status = "Por favor, ingresa una URL válida."
            return

        try:
            async with self:
                self.is_processing = True
                self.progress_value = 0
                self.status = "Obteniendo información del video..."
            
            ydl_opts = {'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
            
            async with self:
                self.video_info = {
                    'title': info['title'],
                    'thumbnail': info['thumbnail']
                }
                self.show_thumbnail = True
                self.status = "Información del video obtenida. Listo para analizar."
                self.progress_value = 100
        except Exception as e:
            async with self:
                self.status = f"Error: {str(e)}"
                self.show_thumbnail = False
        finally:
            async with self:
                self.is_processing = False

    @rx.background
    async def analyze_audio(self):
        if not self.video_info:
            async with self:
                self.status = "Por favor, obtén la información del video primero."
            return

        try:
            async with self:
                self.is_processing = True
                self.progress_value = 0
                self.temp_dir = tempfile.mkdtemp()
                self.audio_file = os.path.join(self.temp_dir, 'audio.mp3')
                self.status = "Descargando audio para análisis..."
            
            def progress_hook(d):
                asyncio.create_task(self.download_progress_hook(d))

            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': self.audio_file,
                'progress_hooks': [progress_hook],
                'keepvideo': False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])

            if not os.path.exists(self.audio_file):
                original_file = self.audio_file.rsplit('.', 1)[0] + '.*'
                original_files = glob.glob(original_file)
                if original_files:
                    os.rename(original_files[0], self.audio_file)
                else:
                    raise Exception(f"No se encontró ningún archivo de audio en: {self.temp_dir}")

            async with self:
                self.status = "Analizando el audio..."
                self.progress_value = 50

            y, sr = librosa.load(self.audio_file, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Generar tiempos de beats uniformemente basados en el BPM calculado
            beat_duration = 60 / tempo
            beat_times = np.arange(0, duration, beat_duration).tolist()
            
            async with self:
                self.audio_duration = duration
                self.bpm = round(float(tempo), 2)
                self.beat_times = beat_times
                self.progress_value = 100
                self.status = f"Análisis completado. BPM: {self.bpm}"

        except Exception as e:
            async with self:
                self.status = f"Error en el análisis: {str(e)}"
        finally:
            async with self:
                self.is_processing = False

    async def download_progress_hook(self, d):
        if d['status'] == 'downloading':
            p = d.get('_percent_str', '0%')
            p = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', p)  # Eliminar códigos ANSI
            p = p.replace('%', '').strip()
            try:
                percentage = float(p)
                progress = int(percentage / 2)  # La descarga representa la primera mitad del proceso
                async with self:
                    self.progress_value = progress
            except ValueError:
                print(f"No se pudo convertir el porcentaje: {p}")

    def play_preview(self):
        if not self.audio_file or not self.beat_times:
            self.status = "Por favor, analiza el audio primero."
            return

        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            pygame.mixer.music.load(self.audio_file)
            
            duration = 0.05
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * 1000 * t) * 0.5
            fade = np.linspace(1, 0, len(t))
            tone = tone * fade
            stereo_tone = np.column_stack((tone, tone))
            metronome_sound = pygame.sndarray.make_sound((stereo_tone * 32767).astype(np.int16))

            self.status = "Reproduciendo con metrónomo..."
            self.is_playing = True
            pygame.mixer.music.play()

            def playback_thread():
                start_time = pygame.time.get_ticks()
                for beat_time in self.beat_times:
                    if not self.is_playing:
                        break
                    current_time = pygame.time.get_ticks() - start_time
                    wait_time = int(beat_time * 1000) - current_time
                    if wait_time > 0:
                        pygame.time.wait(wait_time)
                    if self.is_playing:
                        metronome_sound.play()
                
                if self.is_playing:
                    pygame.mixer.music.stop()
                    self.is_playing = False
                    self.status = "Reproducción finalizada."

            threading.Thread(target=playback_thread).start()

        except Exception as e:
            self.status = f"Error en la reproducción: {str(e)}"
            self.is_playing = False

    def pause_playback(self):
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
            self.status = "Reproducción pausada."

    def stop_preview(self):
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.status = "Reproducción detenida."

    @rx.background
    async def download_video(self):
        if not self.video_info:
            self.status = "Por favor, obtén la información del video primero."
            return

        try:
            async with self:
                self.is_processing = True
                self.progress_value = 0
            
            def progress_hook(d):
                asyncio.create_task(self.download_progress_hook(d))

            ydl_opts = {
                'outtmpl': os.path.join(self.download_path, '%(title)s.%(ext)s'),
                'format': 'bestaudio/best',
                'progress_hooks': [progress_hook],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                async with self:
                    self.status = f"Descargando: {self.video_info['title']}"
                ydl.download([self.url])
            async with self:
                self.status = "¡Descarga completada!"
                self.progress_value = 100
        except Exception as e:
            async with self:
                self.status = f"Error en la descarga: {str(e)}"
        finally:
            async with self:
                self.is_processing = False

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = ""
        self.audio_file = ""
        self.cleanup_temp_files()

    def set_manual_bpm(self, value):
        try:
            self.manual_bpm = float(value)
        except ValueError:
            self.status = "Por favor, ingresa un valor numérico válido para BPM."

    def use_manual_bpm(self):
        if self.manual_bpm > 0:
            self.bpm = self.manual_bpm
            beat_duration = 60 / self.bpm
            self.beat_times = [i * beat_duration for i in range(int(self.audio_duration / beat_duration))]
            self.status = f"BPM manual establecido: {self.bpm}"
        else:
            self.status = "Por favor, ingresa un valor válido de BPM antes de usar."

    def download_clean_audio(self):
        if not self.audio_file:
            self.status = "Por favor, analiza el audio primero."
            return
        
        try:
            output_file = os.path.join(self.download_path, f"{self.video_info['title']}_clean.mp3")
            shutil.copy2(self.audio_file, output_file)
            self.status = f"Audio limpio descargado: {output_file}"
        except Exception as e:
            self.status = f"Error al descargar audio limpio: {str(e)}"

    def set_metronome_volume(self, value):
        if isinstance(value, list) and len(value) > 0:
            value = value[0]
        try:
            self.metronome_volume = float(value)
        except ValueError:
            print(f"Error: No se pudo convertir '{value}' a float")

    @rx.background
    async def download_audio_with_metronome(self):
        if not self.audio_file or not self.beat_times:
            self.status = "Por favor, analiza el audio primero."
            return
        
        try:
            async with self:
                self.is_processing = True
                self.progress_value = 0
            
            audio = AudioSegment.from_mp3(self.audio_file)
            
            duration_ms = 20
            metronome_sound = (Sine(880).to_audio_segment(duration=duration_ms)
                               .fade_in(5).fade_out(15)
                               .apply_gain(self.metronome_volume))
            
            total_beats = len(self.beat_times)
            for i, beat_time in enumerate(self.beat_times):
                position_ms = int(beat_time * 1000)
                audio = audio.overlay(metronome_sound, position=position_ms)
                async with self:
                    self.progress_value = int((i + 1) / total_beats * 100)
            
            output_file = os.path.join(self.download_path, f"{self.video_info['title']}_with_metronome.mp3")
            audio.export(output_file, format="mp3")
            
            async with self:
                self.status = f"Audio con metrónomo descargado: {output_file}"
                self.progress_value = 100
        except Exception as e:
            async with self:
                self.status = f"Error al descargar audio con metrónomo: {str(e)}"
        finally:
            async with self:
                self.is_processing = False

    def preview_with_metronome(self):
        if not self.audio_file or not self.beat_times:
            self.status = "Por favor, analiza el audio primero."
            return

        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_filename = temp_file.name

            self.temp_files.append(temp_filename)

            audio = AudioSegment.from_mp3(self.audio_file)

            duration_ms = 20
            metronome_sound = (Sine(880).to_audio_segment(duration=duration_ms)
                               .fade_in(5).fade_out(15)
                               .apply_gain(self.metronome_volume))

            preview_duration = min(10000, len(audio))
            preview_audio = audio[:preview_duration]

            for beat_time in self.beat_times:
                if beat_time * 1000 > preview_duration:
                    break
                position_ms = int(beat_time * 1000)
                preview_audio = preview_audio.overlay(metronome_sound, position=position_ms)

            preview_audio.export(temp_filename, format="mp3")

            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()

            self.status = "Reproduciendo vista previa con metrónomo..."
            self.is_playing = True

            def cleanup():
                pygame.time.wait(int(preview_duration))
                pygame.mixer.music.stop()
                self.is_playing = False
                self.status = "Vista previa finalizada."
                self.cleanup_temp_files()

            threading.Thread(target=cleanup).start()

        except Exception as e:
            self.status = f"Error en la vista previa: {str(e)}"
            self.is_playing = False

    def cleanup_temp_files(self):
        for file in self.temp_files:
            try:
                if os.path.exists(file):
                    os.unlink(file)
            except PermissionError:
                print(f"No se pudo eliminar {file}. Se intentará más tarde.")
        self.temp_files = [f for f in self.temp_files if os.path.exists(f)]

def index():
    return rx.box(
        rx.vstack(
            rx.heading("Descargador de YouTube con Metrónomo", size="lg", color="white"),
            rx.input(
                placeholder="Ingresa la URL del video de YouTube",
                on_change=State.set_url,
                width="100%",
                bg="rgba(255, 255, 255, 0.1)",
                border="1px solid rgba(255, 255, 255, 0.2)",
                color="white",
                _placeholder={"color": "rgba(255, 255, 255, 0.5)"},
            ),
            rx.hstack(
                rx.button(
                    "Obtener Info",
                    on_click=State.get_video_info,
                    bg="#4CAF50",
                    color="white",
                    _hover={"bg": "#45a049"},
                ),
                rx.button(
                    "Analizar Audio",
                    on_click=State.analyze_audio,
                    bg="#FF9800",
                    color="white",
                    _hover={"bg": "#FB8C00"},
                ),
                rx.button(
                    rx.cond(
                        State.is_playing,
                        "Pausar",
                        "Reproducir"
                    ),
                    on_click=State.play_preview,
                    bg="#9C27B0",
                    color="white",
                    _hover={"bg": "#8E24AA"},
                ),
                rx.button(
                    "Detener",
                    on_click=State.stop_preview,
                    bg="#F44336",
                    color="white",
                    _hover={"bg": "#E53935"},
                ),
                rx.button(
                    "Descargar",
                    on_click=State.download_video,
                    bg="#2196F3",
                    color="white",
                    _hover={"bg": "#1E88E5"},
                ),
                rx.button(
                    "Limpiar",
                    on_click=State.cleanup,
                    bg="#607D8B",
                    color="white",
                    _hover={"bg": "#546E7A"},
                ),
                width="100%",
                justify="space-between",
            ),
            rx.hstack(
                rx.input(
                    placeholder="BPM manual",
                    on_change=State.set_manual_bpm,
                    type="number",
                    width="50%",
                ),
                rx.button(
                    "Usar BPM Manual",
                    on_click=State.use_manual_bpm,
                    bg="#009688",
                    color="white",
                    _hover={"bg": "#00897B"},
                ),
                width="100%",
                justify="space-between",
            ),
            rx.hstack(
                rx.button(
                    "Descargar Audio Limpio",
                    on_click=State.download_clean_audio,
                    bg="#3F51B5",
                    color="white",
                    _hover={"bg": "#3949AB"},
                ),
                rx.button(
                    "Descargar con Metrónomo",
                    on_click=State.download_audio_with_metronome,
                    bg="#673AB7",
                    color="white",
                    _hover={"bg": "#5E35B1"},
                ),
                width="100%",
                justify="space-between",
            ),
            rx.vstack(
                rx.text("Volumen del Metrónomo", color="white"),
                rx.slider(
                    min=-40,
                    max=0,
                    step=1,
                    default_value=-20,
                    on_change=State.set_metronome_volume,
                    width="100%",
                ),
                rx.button(
                    "Vista Previa con Metrónomo",
                    on_click=State.preview_with_metronome,
                    bg="#E91E63",
                    color="white",
                    _hover={"bg": "#D81B60"},
                ),
                width="100%",
            ),
            rx.cond(
                State.show_thumbnail,
                rx.vstack(
                    rx.image(
                        src=State.video_info['thumbnail'],
                        width="100%",
                        max_width="480px",
                        border_radius="md",
                    ),
                    rx.text(State.video_info['title'], color="white", font_weight="bold"),
                    padding="4",
                    bg="rgba(0, 0, 0, 0.5)",
                    border_radius="md",
                    width="100%",
                ),
            ),
            rx.cond(
                State.is_processing,
                rx.vstack(
                    rx.text("Procesando...", color="white"),
                    rx.progress(value=State.progress_value),
                    width="100%",
                ),
            ),
            rx.text(State.status, color="rgba(255, 255, 255, 0.7)"),
            rx.text(f"BPM: {State.bpm}", color="rgba(255, 255, 255, 0.7)"),
            rx.cond(
                State.download_progress > 0,
                rx.vstack(
                    rx.text("Progreso de descarga:", color="white"),
                    rx.progress(value=State.download_progress),
                    width="100%",
                ),
            ),
           
            width="100%",
            max_width="600px",
            spacing="4",
            padding="6",
        ),
        width="100%",
        min_height="100vh",
        bg="#1E1E1E",
        display="flex",
        justify_content="center",
        align_items="center",
    )

app = rx.App()
app.add_page(index)

if __name__ == "__main__":
    app.compile()
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

    def get_video_info(self):
        if not self.url:
            self.status = "Por favor, ingresa una URL válida."
            return

        try:
            ydl_opts = {'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.status = "Obteniendo información del video..."
                info = ydl.extract_info(self.url, download=False)
                self.video_info = {
                    'title': info['title'],
                    'thumbnail': info['thumbnail']
                }
                self.show_thumbnail = True
                self.status = "Información del video obtenida. Listo para analizar."
        except Exception as e:
            self.status = f"Error: {str(e)}"
            self.show_thumbnail = False

    def analyze_audio(self):
        if not self.video_info:
            self.status = "Por favor, obtén la información del video primero."
            return

        try:
            # Crear un directorio temporal
            self.temp_dir = tempfile.mkdtemp()
            self.audio_file = os.path.join(self.temp_dir, 'audio.mp3')
            
            print(f"Directorio temporal creado: {self.temp_dir}")
            print(f"Archivo de audio será: {self.audio_file}")

            # Configurar las opciones de descarga
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': self.audio_file,
                'progress_hooks': [self.download_progress_hook],
                'keepvideo': False,
                'verbose': True
            }

            # Descargar el audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.status = "Descargando audio para análisis..."
                print(f"Iniciando descarga de: {self.url}")
                ydl.download([self.url])

            # Verificar si el archivo se descargó correctamente
            if not os.path.exists(self.audio_file):
                # Si el archivo MP3 no existe, buscar el archivo original
                original_file = self.audio_file.rsplit('.', 1)[0] + '.*'
                original_files = glob.glob(original_file)
                if original_files:
                    # Renombrar el archivo original a MP3
                    os.rename(original_files[0], self.audio_file)
                else:
                    raise Exception(f"No se encontró ningún archivo de audio en: {self.temp_dir}")

            file_size = os.path.getsize(self.audio_file)
            if file_size == 0:
                raise Exception(f"El archivo está vacío: {self.audio_file}")
            
            print(f"Archivo descargado correctamente. Tamaño: {file_size} bytes")

            # Analizar el audio
            self.status = "Analizando el audio..."
            y, sr = librosa.load(self.audio_file, sr=None)
            self.audio_duration = librosa.get_duration(y=y, sr=sr)

            print(f"Audio cargado. Duración: {self.audio_duration} segundos")

            # Detectar el tempo y los beats
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            self.bpm = round(float(tempo), 2)
            self.beat_times = librosa.frames_to_time(beats, sr=sr).tolist()

            print(f"Análisis completado. BPM: {self.bpm}")
            self.status = f"Análisis completado. BPM: {self.bpm}"

        except Exception as e:
            self.status = f"Error en el análisis: {str(e)}"
            print(f"Error detallado: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            # No eliminaremos el directorio temporal aquí para permitir la reproducción
            pass

    def download_progress_hook(self, d):
        if d['status'] == 'finished':
            print('Descarga completada')
        elif d['status'] == 'downloading':
            print(f"Descargando: {d['_percent_str']} de {d['_total_bytes_str']}")

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
            
            # Crear un sonido de metrónomo simple (estéreo)
            duration = 0.05  # duración del sonido en segundos
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * 1000 * t) * 0.5
            fade = np.linspace(1, 0, len(tone))
            tone = tone * fade
            stereo_tone = np.column_stack((tone, tone))  # Crear array estéreo
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
            print(f"Error detallado: {e}")
            import traceback
            print(traceback.format_exc())

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

    def download_video(self):
        if not self.video_info:
            self.status = "Por favor, obtén la información del video primero."
            return

        try:
            ydl_opts = {
                'outtmpl': os.path.join(self.download_path, '%(title)s.%(ext)s'),
                'format': 'bestaudio/best',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.status = f"Descargando: {self.video_info['title']}"
                ydl.download([self.url])
            self.status = "¡Descarga completada!"
        except Exception as e:
            self.status = f"Error en la descarga: {str(e)}"

    def cleanup(self):
        # Limpiar archivos temporales
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Directorio temporal eliminado: {self.temp_dir}")
        self.temp_dir = ""
        self.audio_file = ""

    def set_manual_bpm(self, value):
        try:
            self.manual_bpm = float(value)
        except ValueError:
            self.status = "Por favor, ingresa un valor numérico válido para BPM."

    def use_manual_bpm(self):
        if self.manual_bpm > 0:
            self.bpm = self.manual_bpm
            # Recalcular los beat_times basados en el BPM manual
            beat_duration = 60 / self.bpm
            self.beat_times = [i * beat_duration for i in range(int(self.audio_duration / beat_duration))]
            self.status = f"BPM manual establecido: {self.bpm}"
        else:
            self.status = "Por favor, ingresa un valor válido de BPM antes de usar."


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
                    type_="number",
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
            rx.text(State.status, color="rgba(255, 255, 255, 0.7)"),
            rx.text(f"BPM: {State.bpm}", color="rgba(255, 255, 255, 0.7)"),
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
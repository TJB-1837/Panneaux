from contextlib import contextmanager
from time import perf_counter

import cv2
import numpy as np

from .detector import proposals
from .model import Classifier


@contextmanager
def video_frames(source):
    camera = cv2.VideoCapture(source)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source vidéo : {source}")

        def frames():
            while True:
                ok, frame = camera.read()
                if not ok:
                    if isinstance(source, int):
                        raise RuntimeError("Impossible de lire une image de la caméra")
                    break
                yield frame

        yield frames()
    finally:
        camera.release()


@contextmanager
def screen_frames(title):
    import ctypes

    import win32gui
    import win32ui

    window = win32gui.FindWindow(None, title)
    if not window:
        raise RuntimeError(f"Fenêtre introuvable : {title}")

    def frames():
        while win32gui.IsWindow(window):
            left, top, right, bottom = win32gui.GetWindowRect(window)
            width, height = right - left, bottom - top
            dc_handle = win32gui.GetWindowDC(window)
            window_dc = compatible_dc = bitmap = previous = None
            try:
                window_dc = win32ui.CreateDCFromHandle(dc_handle)
                compatible_dc = window_dc.CreateCompatibleDC()
                bitmap = win32ui.CreateBitmap()
                bitmap.CreateCompatibleBitmap(window_dc, width, height)
                previous = compatible_dc.SelectObject(bitmap)
                print_window = ctypes.windll.user32.PrintWindow
                print_window.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
                target = compatible_dc.GetSafeHdc()
                if not (print_window(window, target, 2) or print_window(window, target, 0)):
                    raise RuntimeError(f"Échec de la capture de '{title}'")
                pixels = np.frombuffer(bitmap.GetBitmapBits(True), dtype=np.uint8)
                frame = cv2.cvtColor(pixels.reshape(height, width, 4), cv2.COLOR_BGRA2BGR)
            finally:
                if compatible_dc is not None:
                    if previous is not None:
                        compatible_dc.SelectObject(previous)
                    compatible_dc.DeleteDC()
                if bitmap is not None:
                    win32gui.DeleteObject(bitmap.GetHandle())
                win32gui.ReleaseDC(window, dc_handle)
            yield frame

    yield frames()


def live(args):
    classifier = Classifier(args.bundle)
    detect = proposals(args)
    if args.window:
        source = screen_frames(args.window)
    else:
        source = video_frames(str(args.video) if args.video is not None else args.camera)
    try:
        with source as frames:
            for frame in frames:
                start = perf_counter()
                results = list(classifier.classify(frame, detect(frame)))
                elapsed = 1000 * (perf_counter() - start)
                for (x1, y1, x2, y2), label, score in results:
                    if score < args.threshold:
                        continue
                    name = classifier.names[label]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} {score:.0%}", (x1, max(14, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"{elapsed:.0f} ms", (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                cv2.imshow("Reconnaissance de panneaux", frame)
                if cv2.waitKey(1) & 0xff in (27, ord("q")):
                    break
    finally:
        cv2.destroyAllWindows()

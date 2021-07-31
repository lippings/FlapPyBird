from time import sleep
from random import random
import threading
from typing import Dict, Tuple, Any

import pygame
import cv2

from .detector import BaseDetector, DeepSmileDetector, CascadeSmileDetector, LandmarkBlinkDetector

RAND_EVENT = pygame.event.custom_type()
SMILE_EVENT = pygame.event.custom_type()


class BaseEventGenerator():
    def __init__(self, show_webcam=False, thread_name='Event generator thread'):
        self._job_thread = threading.Thread(
            target=self._thread_job,
            name=thread_name,
            daemon=True
        )
        
        self._running = False
        self._current_frame = None
        self._show = show_webcam
        self.event = pygame.event.custom_type()

    def _fire_event(self):
        ev = pygame.event.Event(self.event)
        pygame.event.post(ev)

    def _thread_job(self):
        ...

    def display(self):
        if self._show:
            if self._current_frame is not None:
                cv2.imshow('', self._current_frame)

    def start(self):
        if self._running:
            print('Already running')
            return
        self._running = True
        self._job_thread.start()
    
    def stop(self):
        self._running = False
        self._job_thread.join()


class RandomEventGenerator(BaseEventGenerator):
    def __init__(self, clicks_per_sec=1):
        super().__init__(show_webcam=False)
        self._cps = clicks_per_sec

    def _thread_job(self):
        period = 0.05
        prob = period * self._cps
        while self._running:
            r = random()

            if r < prob:
                self._fire_event()
            
            sleep(period)


class SmileEventGenerator(BaseEventGenerator):
    def __init__(self, show_webcam=False, method='deep'):
        super().__init__(show_webcam=show_webcam)

        method_dict: Dict[str, Tuple[BaseDetector, Dict[str, Any], str]] = {
            'deep': (
                DeepSmileDetector,
                {
                    'pretrained_name': 'mobilenetv2'
                },
                'Deep network-based detector'),
            'cascade': (
                CascadeSmileDetector,
                {},
                'Haar cascade filter-based detector'
            )
        }
        
        self._detector = None
        for method_name, (method_class, method_kwargs, _) in method_dict.items():
            if method_name == method:
                self._detector = method_class(**method_kwargs)
                break
        
        if self._detector is None:
            method_descs = '\n'.join(
                f'\t{method_name}: {method_desc}' for method_name, (_, _, method_desc) in method_dict.items()
            )

            raise AttributeError(f'Unknown method name for smile detector: {method}\n'
                                 f'Known methods:\n{method_descs}')
    
    def _thread_job(self):
        cap = cv2.VideoCapture(0)

        while self._running:
            _, frame = cap.read()

            pred_label = self._detector(frame)

            if pred_label == 1:
                self._fire_event()
            
            # imshow outside main thread doesn't work
            self._current_frame = frame


class BlinkEventGenerator(BaseEventGenerator):
    def __init__(self, show_webcam=False, method='landmark'):
        super().__init__(show_webcam=show_webcam)

        method_dict: Dict[str, Tuple[BaseDetector, Dict[str, Any], str]] = {
            'landmark': (
                LandmarkBlinkDetector,
                {},
                'Landmark-based detector')
        }
        
        self._detector = None
        for method_name, (method_class, method_kwargs, _) in method_dict.items():
            if method_name == method:
                self._detector = method_class(**method_kwargs)
                break
        
        if self._detector is None:
            method_descs = '\n'.join(
                f'\t{method_name}: {method_desc}' for method_name, (_, _, method_desc) in method_dict.items()
            )

            raise AttributeError(f'Unknown method name for smile detector: {method}\n'
                                 f'Known methods:\n{method_descs}')
    
    def _thread_job(self):
        cap = cv2.VideoCapture(0)

        while self._running:
            _, frame = cap.read()

            pred_label = self._detector(frame)

            if pred_label == 1:
                self._fire_event()
            
            # imshow outside main thread doesn't work
            self._current_frame = frame

# EOF

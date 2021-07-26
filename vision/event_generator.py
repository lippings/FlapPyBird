from time import sleep
from random import random
import threading
from typing import Dict, Tuple, Any

import pygame
from cv2 import VideoCapture, imshow

from .detector import BaseDetector, DeepSmileDetector, CascadeSmileDetector

RAND_EVENT = pygame.event.custom_type()
SMILE_EVENT = pygame.event.custom_type()


class RandomEventGenerator():
    def __init__(self, clicks_per_sec=1):
        self._cps = clicks_per_sec
        self._job_thread = threading.Thread(
            target=self._thread_job,
            name='Random event generator thread',
            daemon=True
        )

    def _thread_job(self):
        period = 0.05
        prob = period * self._cps
        while self._running:
            r = random()

            if r < prob:
                ev = pygame.event.Event(RAND_EVENT)
                pygame.event.post(ev)
            
            sleep(period)

    def start(self):
        if self._running:
            print('Already running')
            return
        self._running = True
        self._job_thread.start()
    
    def stop(self):
        self._running = False
        self._job_thread.join()


class SmileEventGenerator():
    def __init__(self, show_webcam=False, method='deep'):
        self._show = show_webcam
        self._running = False
        self._job_thread = threading.Thread(
            target=self._thread_job,
            name='Smile detection event generator',
            daemon=True
        )

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
        
        if method == 'deep':
            self._detector = DeepSmileDetector()
        self._current_frame = None
    
    def _thread_job(self):
        cap = VideoCapture(0)

        while self._running:
            _, frame = cap.read()

            pred_label = self._detector(frame)

            if pred_label == 1:
                ev = pygame.event.Event(SMILE_EVENT)
                pygame.event.post(ev)
            
            # imshow outside main thread doesn't work
            self._current_frame = frame

    def display(self):
        if self._show:
            if self._current_frame is not None:
                imshow('', self._current_frame)
    
    def start(self):
        if self._running:
            print('Already running')
            return
        self._running = True
        self._job_thread.start()
    
    def stop(self):
        self._running = False
        self._job_thread.join()

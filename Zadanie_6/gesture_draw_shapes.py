"""
Title: Gesture-Controlled Shape Drawing (OpenCV + MediaPipe)
Authors:
- Wiktor Rapacz
- Hania Paczoska

Problem:
Create a real-time hand-gesture detection program that uses a webcam stream and maps at least
four different hand gestures to application actions.

Solution (this script):
This program uses MediaPipe Hands to detect 21 hand landmarks in real time and then applies a
lightweight finger-state heuristic to classify a small set of stable gestures. The detected gesture
controls a simple drawing canvas rendered with OpenCV.

How it works (high level):
1) OpenCV reads frames from the webcam.
2) Frames are passed to MediaPipe Hands, which returns hand landmarks (no user training needed).
3) A heuristic checks which fingers are extended to recognize gestures.
4) A smoothing layer requires the same gesture to be stable for N frames.
5) The gesture triggers a drawing action on an OpenCV canvas.

Environment setup:
- Python: 3.10
- Install dependencies (in your virtualenv):
    pip install opencv-python mediapipe numpy

Usage:
1) Run:
    python gesture_draw_shapes.py
2) Show your RIGHT hand to the camera (recommended for stability).
3) Gestures:
   - OPEN_PALM  (all fingers)            -> draw circle
   - V_SIGN     (index + middle)         -> draw square
   - FIST       (all folded)             -> clear canvas
   - INDEX_ONLY (index only, hold)       -> increase size + cycle color of the last shape
   - PINKY_ONLY (pinky only, hold)       -> decrease size of the last shape
4) Controls:
   - Press 'H' to toggle help overlay
   - Press 'Q' or ESC to quit (click the OpenCV window first so it has focus)

"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class AppConfig:
    """
    Application configuration for the gesture drawing demo.

    Attributes:
        width: Frame/canvas width in pixels.
        height: Frame/canvas height in pixels.
        stable_frames: Number of consecutive frames required to accept a gesture as stable.
        draw_cooldown_sec: Cooldown for one-shot actions (draw circle/square/clear).
        hold_repeat_sec: Repeat interval for hold actions (resize/recolor).
        max_shape_size: Upper bound for shape size.
        min_shape_size: Lower bound for shape size.
        size_step: Size increment/decrement per repeat.
        show_help: Whether to show help text overlay by default.
        camera_index: Webcam index for OpenCV VideoCapture.
    """
    width: int = 640
    height: int = 480
    stable_frames: int = 6
    draw_cooldown_sec: float = 0.8
    hold_repeat_sec: float = 0.15
    max_shape_size: int = 220
    min_shape_size: int = 20
    size_step: int = 5
    show_help: bool = True
    camera_index: int = 0


class GestureDrawingApp:
    """
    Real-time gesture-controlled shape drawing application.

    This class owns:
    - webcam capture (OpenCV)
    - hand landmark detection (MediaPipe Hands)
    - gesture classification (heuristics)
    - smoothing (stable-frames filter)
    - an OpenCV canvas where the "last shape" is drawn and can be resized/recolored
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the app state and allocate the drawing canvas.

        Args:
            config: Application configuration.
        """
        self.cfg = config

        # MediaPipe modules
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        # Drawing canvas and drawing state
        self.canvas = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        self.shape_size = 60
        self.color_index = 0
        self.colors = [
            (0, 255, 0),    # green
            (255, 0, 0),    # blue
            (0, 0, 255),    # red
            (255, 255, 0),  # cyan
        ]

        # Always scale/recolor ONLY the last drawn shape:
        # last_shape = ("CIRCLE" or "SQUARE", color_index_at_draw_time)
        self.last_shape: Optional[Tuple[str, int]] = None

        # Smoothing / timing
        self.stable_gesture: Optional[str] = None
        self.stable_count: int = 0
        self.last_draw_time: float = 0.0
        self.last_hold_time: float = 0.0

        self.show_help: bool = self.cfg.show_help

    def run(self) -> None:
        """
        Run the main loop: capture frames, detect hand, classify gesture, and draw.

        Raises:
            RuntimeError: If the webcam cannot be opened.
        """
        cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam (VideoCapture({self.cfg.camera_index})).")

        with self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_hands=1,
        ) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)  # selfie view
                gesture = self._detect_gesture(frame, hands)

                self._update_stability(gesture)

                now = time.time()
                cx, cy = self.cfg.width // 2, self.cfg.height // 2

                self._apply_actions(now, cx, cy)

                combined = cv2.addWeighted(frame, 0.75, self.canvas, 0.35, 0)
                self._draw_overlay(combined)

                cv2.imshow("Gesture Drawing Demo (OpenCV Canvas)", combined)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("h"):
                    self.show_help = not self.show_help

        cap.release()
        cv2.destroyAllWindows()

    def _detect_gesture(self, frame_bgr: np.ndarray, hands) -> str:
        """
        Detect and classify the gesture for a single frame.

        Args:
            frame_bgr: Current webcam frame (BGR).
            hands: An initialized MediaPipe Hands context.

        Returns:
            A gesture label string (e.g., OPEN_PALM, V_SIGN, FIST, INDEX_ONLY, PINKY_ONLY, UNKNOWN, NO_HAND).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if not res.multi_hand_landmarks:
            return "NO_HAND"

        lm = res.multi_hand_landmarks[0]
        self.mp_draw.draw_landmarks(frame_bgr, lm, self.mp_hands.HAND_CONNECTIONS)
        return self._classify_gesture(lm)

    def _classify_gesture(self, lm) -> str:
        """
        Classify gestures based on simple finger extension rules.

        The heuristic uses fingertip vs PIP landmark Y coordinates:
        - If tip is above PIP (smaller y in image coords), the finger is considered extended.

        Returns:
            Gesture label:
            - OPEN_PALM, V_SIGN, FIST, INDEX_ONLY, PINKY_ONLY, or UNKNOWN.
        """
        def is_extended(tip: int, pip: int) -> bool:
            return lm.landmark[tip].y < lm.landmark[pip].y

        index = is_extended(8, 6)
        middle = is_extended(12, 10)
        ring = is_extended(16, 14)
        pinky = is_extended(20, 18)

        # Priority order matters.
        if index and middle and ring and pinky:
            return "OPEN_PALM"
        if index and middle and (not ring) and (not pinky):
            return "V_SIGN"
        if (not index) and (not middle) and (not ring) and (not pinky):
            return "FIST"
        if index and (not middle) and (not ring) and (not pinky):
            return "INDEX_ONLY"
        if pinky and (not index) and (not middle) and (not ring):
            return "PINKY_ONLY"

        return "UNKNOWN"

    def _update_stability(self, gesture: str) -> None:
        """
        Update stable-gesture tracking.

        Args:
            gesture: Current raw gesture label.
        """
        if gesture == self.stable_gesture:
            self.stable_count += 1
        else:
            self.stable_gesture = gesture
            self.stable_count = 1

    def _apply_actions(self, now: float, cx: int, cy: int) -> None:
        """
        Apply drawing actions based on a stable gesture.

        Args:
            now: Current timestamp in seconds.
            cx: Canvas center x.
            cy: Canvas center y.
        """
        if self.stable_count < self.cfg.stable_frames:
            return
        if self.stable_gesture in ("UNKNOWN", "NO_HAND", None):
            return

        # One-shot actions (draw/clear)
        if self.stable_gesture in ("OPEN_PALM", "V_SIGN", "FIST"):
            if (now - self.last_draw_time) >= self.cfg.draw_cooldown_sec:
                if self.stable_gesture == "OPEN_PALM":
                    self._clear_canvas()
                    self._draw_circle(cx, cy, self.shape_size, self.colors[self.color_index])
                    self.last_shape = ("CIRCLE", self.color_index)

                elif self.stable_gesture == "V_SIGN":
                    self._clear_canvas()
                    self._draw_square(cx, cy, self.shape_size, self.colors[self.color_index])
                    self.last_shape = ("SQUARE", self.color_index)

                elif self.stable_gesture == "FIST":
                    self._clear_canvas()
                    self.last_shape = None

                self.last_draw_time = now

        # Hold actions (resize/recolor last shape)
        if self.stable_gesture == "INDEX_ONLY":
            if (now - self.last_hold_time) >= self.cfg.hold_repeat_sec:
                self.shape_size = min(self.cfg.max_shape_size, self.shape_size + self.cfg.size_step)
                self.color_index = (self.color_index + 1) % len(self.colors)
                self.last_hold_time = now

                if self.last_shape is not None:
                    # Update last shape color to current selection so the change is visible live.
                    self.last_shape = (self.last_shape[0], self.color_index)
                    self._clear_canvas()
                    self._redraw_last_shape(cx, cy)

        elif self.stable_gesture == "PINKY_ONLY":
            if (now - self.last_hold_time) >= self.cfg.hold_repeat_sec:
                self.shape_size = max(self.cfg.min_shape_size, self.shape_size - self.cfg.size_step)
                self.last_hold_time = now

                if self.last_shape is not None:
                    self._clear_canvas()
                    self._redraw_last_shape(cx, cy)

    def _draw_circle(self, cx: int, cy: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a filled circle on the canvas."""
        cv2.circle(self.canvas, (cx, cy), size, color, -1)

    def _draw_square(self, cx: int, cy: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a filled square on the canvas."""
        cv2.rectangle(self.canvas, (cx - size, cy - size), (cx + size, cy + size), color, -1)

    def _clear_canvas(self) -> None:
        """Clear the drawing canvas."""
        self.canvas[:] = 0

    def _redraw_last_shape(self, cx: int, cy: int) -> None:
        """
        Re-draw the last shape using the CURRENT size and the stored color index.

        Args:
            cx: Canvas center x.
            cy: Canvas center y.
        """
        if self.last_shape is None:
            return
        shape_type, c_idx = self.last_shape
        color = self.colors[c_idx]

        if shape_type == "CIRCLE":
            self._draw_circle(cx, cy, self.shape_size, color)
        elif shape_type == "SQUARE":
            self._draw_square(cx, cy, self.shape_size, color)

    def _draw_overlay(self, frame_bgr: np.ndarray) -> None:
        """
        Draw UI overlay with the current stable gesture and app state.

        Args:
            frame_bgr: Frame to draw on (BGR).
        """
        cv2.rectangle(frame_bgr, (10, 10), (630, 140), (0, 0, 0), -1)
        cv2.putText(
            frame_bgr,
            f"Gesture: {self.stable_gesture}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_bgr,
            f"Size: {self.shape_size}   Color: {self.color_index + 1}/{len(self.colors)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        if self.show_help:
            cv2.putText(
                frame_bgr,
                "OPEN_PALM: circle | V_SIGN: square | FIST: clear",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame_bgr,
                "INDEX_ONLY: bigger+color | PINKY_ONLY: smaller | H: help",
                (20, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )


def main() -> None:
    """
    Program entry point.

    Creates the application and runs the main loop.
    """
    app = GestureDrawingApp(AppConfig())
    app.run()


if __name__ == "__main__":
    main()

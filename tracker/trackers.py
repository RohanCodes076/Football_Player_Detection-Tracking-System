# Import All Required Libraries
import cv2
import os
import numpy as np
import pickle
from collections import defaultdict, deque

from ultralytics import YOLO
import supervision as sv

from utils import get_center_of_bbox


class Tracker:

    def __init__(self, model_path):

        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        # Store player trails
        self.trail_history = defaultdict(lambda: deque(maxlen=25))

    # ---------------------------------------------------
    # DETECT FRAMES
    # ---------------------------------------------------

    def detect_frames(self, frames):

        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):

            detections_batch = self.model.predict(
                frames[i:i + batch_size],
                conf=0.1
            )

            detections += detections_batch

        return detections

    # ---------------------------------------------------
    # GET TRACKS
    # ---------------------------------------------------

    def get_object_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):

        if (
            read_from_stub
            and stub_path is not None
            and os.path.exists(stub_path)
        ):

            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)

            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):

            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision format
            detection_supervision = sv.Detections.from_ultralytics(
                detection
            )

            # Convert goalkeeper -> player
            for object_ind, class_id in enumerate(
                detection_supervision.class_id
            ):

                if cls_names[class_id] == "goalkeeper":

                    detection_supervision.class_id[
                        object_ind
                    ] = cls_names_inv["player"]

            # Tracking
            detection_with_tracks = (
                self.tracker.update_with_detections(
                    detection_supervision
                )
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Players + Referees
            for frame_detection in detection_with_tracks:

                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:

                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox
                    }

                if cls_id == cls_names_inv['referee']:

                    tracks["referees"][frame_num][track_id] = {
                        "bbox": bbox
                    }

            # Ball
            for frame_detection in detection_supervision:

                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:

                    tracks["ball"][frame_num][1] = {
                        "bbox": bbox
                    }

        # Save stub
        if stub_path is not None:

            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    # ---------------------------------------------------
    # ROUNDED RECTANGLE
    # ---------------------------------------------------

    def draw_rounded_rect(
        self,
        frame,
        top_left,
        bottom_right,
        color,
        thickness=2,
        radius=10
    ):

        x1, y1 = top_left
        x2, y2 = bottom_right

        # Lines
        cv2.line(
            frame,
            (x1 + radius, y1),
            (x2 - radius, y1),
            color,
            thickness
        )

        cv2.line(
            frame,
            (x1 + radius, y2),
            (x2 - radius, y2),
            color,
            thickness
        )

        cv2.line(
            frame,
            (x1, y1 + radius),
            (x1, y2 - radius),
            color,
            thickness
        )

        cv2.line(
            frame,
            (x2, y1 + radius),
            (x2, y2 - radius),
            color,
            thickness
        )

        # Corners
        cv2.ellipse(
            frame,
            (x1 + radius, y1 + radius),
            (radius, radius),
            180,
            0,
            90,
            color,
            thickness
        )

        cv2.ellipse(
            frame,
            (x2 - radius, y1 + radius),
            (radius, radius),
            270,
            0,
            90,
            color,
            thickness
        )

        cv2.ellipse(
            frame,
            (x1 + radius, y2 - radius),
            (radius, radius),
            90,
            0,
            90,
            color,
            thickness
        )

        cv2.ellipse(
            frame,
            (x2 - radius, y2 - radius),
            (radius, radius),
            0,
            0,
            90,
            color,
            thickness
        )

        return frame

    # ---------------------------------------------------
    # TRANSPARENT LABEL
    # ---------------------------------------------------

    def draw_label(
        self,
        frame,
        text,
        x,
        y,
        color
    ):

        overlay = frame.copy()

        # Transparent rectangle
        cv2.rectangle(
            overlay,
            (x, y - 28),
            (x + 50, y),
            color,
            -1
        )

        alpha = 0.45

        cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0,
            frame
        )

        # Text
        cv2.putText(
            frame,
            text,
            (x + 10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        return frame

    # ---------------------------------------------------
    # PLAYER TRAIL
    # ---------------------------------------------------

    def draw_trail(
        self,
        frame,
        track_id,
        center,
        color
    ):

        self.trail_history[track_id].append(center)

        points = self.trail_history[track_id]

        for i in range(1, len(points)):

            thickness = int(np.sqrt(25 / float(i + 1)) * 2)

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                color,
                thickness
            )

        return frame

    # ---------------------------------------------------
    # DRAW PLAYER
    # ---------------------------------------------------

    def draw_player(
        self,
        frame,
        bbox,
        track_id
    ):

        BLUE = (255, 0, 0)

        x1, y1, x2, y2 = map(int, bbox)

        # Rounded BBox
        frame = self.draw_rounded_rect(
            frame,
            (x1, y1),
            (x2, y2),
            BLUE,
            thickness=2
        )

        # Trail
        center_x, center_y = get_center_of_bbox(bbox)

        frame = self.draw_trail(
            frame,
            track_id,
            (center_x, center_y),
            BLUE
        )

        # Label
        frame = self.draw_label(
            frame,
            str(track_id),
            x1,
            y1,
            BLUE
        )

        return frame

    # ---------------------------------------------------
    # DRAW REFEREE
    # ---------------------------------------------------

    def draw_referee(
        self,
        frame,
        bbox,
        track_id
    ):

        YELLOW = (0, 255, 255)

        x1, y1, x2, y2 = map(int, bbox)

        # Rounded BBox
        frame = self.draw_rounded_rect(
            frame,
            (x1, y1),
            (x2, y2),
            YELLOW,
            thickness=2
        )

        # No trail for referee

        # Label
        frame = self.draw_label(
            frame,
            str(track_id),
            x1,
            y1,
            YELLOW
        )

        return frame

    # ---------------------------------------------------
    # DRAW BALL
    # ---------------------------------------------------

    def draw_ball(
        self,
        frame,
        bbox
    ):

        center_x, center_y = get_center_of_bbox(bbox)

        # Glow Effect
        for radius in range(20, 5, -5):

            overlay = frame.copy()

            cv2.circle(
                overlay,
                (center_x, center_y),
                radius,
                (255, 255, 255),
                -1
            )

            alpha = 0.08

            cv2.addWeighted(
                overlay,
                alpha,
                frame,
                1 - alpha,
                0,
                frame
            )

        # Main Ball
        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (255, 255, 255),
            -1
        )

        return frame

    # ---------------------------------------------------
    # DRAW ALL ANNOTATIONS
    # ---------------------------------------------------

    def draw_annotations(
        self,
        video_frames,
        tracks
    ):

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):

            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # ---------------- PLAYERS ----------------

            for track_id, player in player_dict.items():

                frame = self.draw_player(
                    frame,
                    player["bbox"],
                    track_id
                )

            # ---------------- REFEREES ----------------

            for track_id, referee in referee_dict.items():

                frame = self.draw_referee(
                    frame,
                    referee["bbox"],
                    track_id
                )

            # ---------------- BALL ----------------

            for _, ball in ball_dict.items():

                frame = self.draw_ball(
                    frame,
                    ball["bbox"]
                )

            output_video_frames.append(frame)

        return output_video_frames
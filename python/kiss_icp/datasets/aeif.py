# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to mse, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.*
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE msE OR OTHER DEALINGS IN THE
# SOFTWARE.

import glob
import os
from aeifdataset import Dataloader, get_projection_img, Lidar, Points

import numpy as np

class AEIFDataset:
    def __init__(self, data_dir, sequence: str, *_, **__):
        self.sequence_id = sequence
        try:
            # Zerlege die Sequenz in Name und ID-Bereich
            self._sequence_name, file_range = sequence.split("#")
            start_id, end_id = map(int, file_range.split("-"))
        except ValueError:
            # Kein Bereich angegeben, alles verarbeiten
            self._sequence_name = sequence
            start_id, end_id = None, None
            print(
                "No maneuver specified. Processing all files at once. "
                "To process a specific range, specify `sequence#id-id`."
            )

            # Verzeichnis der Sequenz
        self._aeif_sequence_dir = os.path.join(data_dir, self._sequence_name)

        # Berechne die Start- und End-Indizes und initialisiere den Dataloader
        self._start_file, self._end_file = self._filter_file_range(start_id, end_id)
        self.scan_files = Dataloader(self._aeif_sequence_dir)[self._start_file:self._end_file]

        # Initialisiere Buffer und Frames
        self._files_buffer = self._create_buffer()
        self._frames_per_file = self._get_num_frames()

    def _filter_file_range(self, start_id, end_id):
        """
        Bestimmt die Start- und Enddateien basierend auf der ID im Dateinamen.
        """
        # Lade alle Dateien im Verzeichnis
        all_files = sorted(glob.glob(os.path.join(self._aeif_sequence_dir, "*.4mse")))

        # Standardwerte für Indizes
        start_file = 0
        end_file = len(all_files)

        if start_id is None or end_id is None:
            return start_file, end_file

        # Suche die Start- und End-Indizes
        for idx, file_path in enumerate(all_files):
            # Extrahiere die ID aus dem Dateinamen
            file_name = os.path.basename(file_path)  # Beispiel: id123_2024-09-27_10-31-32.4mse
            file_id = int(file_name.split('_')[0][2:])  # Annahme: "id123_..."

            if file_id == start_id and start_file == 0:
                start_file = idx
            if file_id == end_id:
                end_file = idx + 1
                break

        return start_file, end_file

    def __getitem__(self, idx):
        file_idx, frame_idx = self._find_file_and_frame(idx)
        return self.scans(file_idx, frame_idx)

    def __len__(self):
        return sum(self._frames_per_file)

    def _create_buffer(self):
        buffer = []
        for record in self.scan_files:
            buffer.append(record)
        return buffer

    def _find_file_and_frame(self, target):
        total = 0
        for file_idx, num in enumerate(self._frames_per_file):
            total += num
            if target < total:
                frame_idx = target - (total - num)
                return file_idx, frame_idx
        raise ValueError("Target out of range")

    def _get_num_frames(self):
        num_frames = []
        for record in self._files_buffer:
            num_frames.append(record.num_frames)
        return num_frames

    def scans(self, file_idx, frame_idx):
        return self.read_point_cloud(self._files_buffer[file_idx], frame_idx)

    def read_point_cloud(self, scan_file, frame_idx):
        points = scan_file[frame_idx].vehicle.lidars.TOP
        xyz_points = np.stack(
            (points['x'], points['y'], points['z']),
            axis=-1)
        return xyz_points, self._get_timestamps(points)

    @staticmethod
    def _get_timestamps(points):
        points_ts = points['t']
        normalized_points_ts = (points_ts - points_ts.min()) / (points_ts.max() - points_ts.min())
        return normalized_points_ts

    def get_frames_timestamps(self) -> np.ndarray:
        timestamps = []
        for file in self._files_buffer:
            for frame in file:
                timestamps.append(frame.timestamp)
        timestamps = np.array(timestamps).reshape(-1, 1)
        return timestamps

    def project_points(self, idx, points=None):
        file_idx, frame_idx = self._find_file_and_frame(idx)
        img = self._files_buffer[file_idx][frame_idx+1].vehicle.cameras.STEREO_LEFT
        if points is None:
            lidar = self._files_buffer[file_idx][frame_idx].vehicle.lidars.TOP
        else:
            lidar_info = self._files_buffer[file_idx][frame_idx].vehicle.lidars.TOP.info
            lidar = Lidar(lidar_info, Points(points))
        proj_img = get_projection_img(img, lidar)
        proj_img.save(f'{idx}.png')

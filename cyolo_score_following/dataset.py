
import glob
import os
import random
import torch
import torchvision

import numpy as np


from cyolo_score_following.utils.data_utils import load_sequences, SAMPLE_RATE, FPS, FRAME_SIZE, HOP_SIZE, load_msseq, load_masked_bipage_sequences, load_bipage_sequences, load_blank_bipage_sequences
from cyolo_score_following.utils.dist_utils import is_main_process
from cyolo_score_following.utils.general import load_wav, AverageMeter, get_max_box, xywh2xyxy, box_iou
from cyolo_score_following.augmentations.impulse_response import ImpulseResponse
from cyolo_score_following.utils.general import load_yaml

from multiprocessing import get_context
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

CLASS_MAPPING = {0: 'Note', 1: 'Bar', 2: 'System'}

Repeat_List = [
    "AndreJ__O34__andre-sonatine_synth",
    "BachJS__BWV117a__BWV-117a_synth",
    "BachJS__BWV511__BWV-511_synth",
    "BachJS__BWV512__BWV-512_synth",
    "BachJS__BWV516__BWV-516_synth",
    "BachJS__BWV817__bach-french-suite-6-menuet_synth",
    "BachJS__BWV825__15title-hub_synth",
    "BachJS__BWV825__16title-hub_synth",
    "BachJS__BWV829__55title-hub_synth",
    "BachJS__BWV830__BWV-830-2_synth",
    "BachJS__BWV988__bwv-988-v09_synth",
    "BachJS__BWV988__bwv-988-v09_synth",
    "BachJS__BWV988__bwv-988-v12_synth",
    "BachJS__BWV988__bwv-988-v13_synth",
    "BachJS__BWV994__bach-applicatio_synth",
    "BachJS__BWV1006a__bwv-1006a_5_synth",
    "BachJS__BWVAnh113__anna-magdalena-03_synth",
    "BachJS__BWVAnh116__anna-magdalena-07_synth",
    "BachJS__BWVAnh120__BWV-120_synth",
    "BachJS__BWVAnh131__air_synth",
    "BachJS__BWVAnh691__BWV-691_synth",
    "BeethovenLv__O79__LVB_Sonate_79_1_synth",
    "BurgmullerJFF__O100__25EF-02_synth",
    "HandelGF__Aylesford__10-menuetii_synth",
    "HandelGF__Aylesford__16-airmitvar_synth",
    "HandelGF__Aylesford__19-menuet_synth",
    "MozartWA__KV331__KV331_1_2_var1_synth",
    "MozartWA__KV331__KV331_1_5_var4_synth",
    "MuellerAE__muller-siciliano__muller-siciliano_synth",
    "SatieE__gymnopedie_1__gymnopedie_1_synth",
    "SchumannR__O68__schumann-op68-01-melodie_synth",
    "SchumannR__O68__schumann-op68-06-pauvre-orpheline_synth",
    "SchumannR__O68__schumann-op68-08-cavalier-sauvage_synth",
    "SchumannR__O68__schumann-op68-16-premier-chagrin_synth",
    "SchumannR__O68__schumann-op68-26-sans-titre_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.1.2__traditioner_af_swenska_folk_dansar.1.2_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.1.3__traditioner_af_swenska_folk_dansar.1.3_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.1.4__traditioner_af_swenska_folk_dansar.1.4_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.1.5__traditioner_af_swenska_folk_dansar.1.5_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.1.6__traditioner_af_swenska_folk_dansar.1.6_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.1.12__traditioner_af_swenska_folk_dansar.1.12_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.1.14__traditioner_af_swenska_folk_dansar.1.14_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.1.19__traditioner_af_swenska_folk_dansar.1.19_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.1.20__traditioner_af_swenska_folk_dansar.1.20_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.1.22__traditioner_af_swenska_folk_dansar.1.22_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.1.26__traditioner_af_swenska_folk_dansar.1.26_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.1.30__traditioner_af_swenska_folk_dansar.1.30_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.2.27__traditioner_af_swenska_folk_dansar.2.27_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.3.6__traditioner_af_swenska_folk_dansar.3.6_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.3.7__traditioner_af_swenska_folk_dansar.3.7_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.3.8__traditioner_af_swenska_folk_dansar.3.8_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.3.9__traditioner_af_swenska_folk_dansar.3.9_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.3.12__traditioner_af_swenska_folk_dansar.3.12_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.3.13__traditioner_af_swenska_folk_dansar.3.13_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.3.17__traditioner_af_swenska_folk_dansar.3.17_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.3.18__traditioner_af_swenska_folk_dansar.3.18_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.3.19__traditioner_af_swenska_folk_dansar.3.19_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.3.22__traditioner_af_swenska_folk_dansar.3.22_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.3.24__traditioner_af_swenska_folk_dansar.3.24_synth",
    "Traditional__traditioner_af_swenska_folk_dansar.3.34__traditioner_af_swenska_folk_dansar.3.34_synth",
    "Yaniewicz__leslanciers__leslanciers_synth"
]
class SequenceDataset(Dataset):
    def __init__(self, scores, performances, sequences, piece_names, interpol_c2o,
                 staff_coords, add_per_staff, predict_sb=False, augment=False, transform=None):

        self.scores = scores
        self.performances = performances
        self.sequences = []
        self.rand_perf_indices = {}
        self.sequences = sequences
        self.augment = augment
        self.piece_names = piece_names
        self.interpol_c2o = interpol_c2o
        self.staff_coords = staff_coords
        self.add_per_staff = add_per_staff

        self.predict_sb = predict_sb

        self.fps = FPS
        self.sample_rate = SAMPLE_RATE
        self.frame_size = FRAME_SIZE
        self.hop_length = HOP_SIZE

        self.length = len(self.sequences)
        self.gt_width = 30
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        seq = self.sequences[item]

        piece_id = seq['piece_id']
        score = self.scores[piece_id]

        is_onset = seq['is_onset']

        signal = self.performances[piece_id]

        # if signal is provided as a path it should be loaded from the disk
        if isinstance(signal, str):
            signal = load_wav(signal, SAMPLE_RATE)

        start_frame = int(seq['start_frame'])
        frame = int(seq['frame'])
        scale_factor = seq['scale_factor']

        start_t = int(start_frame * self.hop_length)
        t = self.frame_size + int(frame * self.hop_length)

        truncated_signal = signal[start_t:t]

        true_position, page_nr = seq['true_position'][:2], seq['true_position'][-1]
        
        notes_available = true_position[0] >= 0
        max_y_shift = seq['max_y_shift']
        max_x_shift = seq['max_x_shift']

        page_nr = int(page_nr)

        s = score[page_nr]

        system = np.asarray(seq['true_system'], dtype=np.float32)
        system /= scale_factor

        bar = np.asarray(seq['true_bar'], dtype=np.float32)
        bar /= scale_factor

        true_pos = np.copy(true_position)
        true_pos = true_pos / scale_factor
        
        width = self.gt_width / scale_factor
        height = seq['height'] / scale_factor
        
        

        if self.augment:

            yshift = random.randint(int(max_y_shift[0]/scale_factor), int(max_y_shift[1]/scale_factor))
            xshift = random.randint(int(max_x_shift[0] / scale_factor), int(max_x_shift[1] / scale_factor))

            true_pos[0] += yshift
            true_pos[1] += xshift

            s = np.roll(s, yshift, 0)
            s = np.roll(s, xshift, 1)

            # System [center_x, center_y, width, height]
            system[0] += xshift
            system[1] += yshift

            bar[0] += xshift
            bar[1] += yshift

            # pad signal randomly by 0-20 frames (0-1seconds)
            truncated_signal = np.pad(truncated_signal, (random.randint(0, int(self.fps)) * self.hop_length, 0),
                                      mode='constant')

        center_y, center_x = true_pos
        target = []

        if notes_available:
            target.append([0, 0, center_x/s.shape[1], center_y/s.shape[0], width/s.shape[1], height/s.shape[0]])
        
        if self.predict_sb:
            target.append([0, 1, bar[0] / s.shape[1], bar[1] / s.shape[0], bar[2] / s.shape[1], bar[3] / s.shape[0]])
            target.append([0, 2, system[0]/s.shape[1], system[1]/s.shape[0], system[2]/s.shape[1], system[3]/s.shape[0]])

        target = np.asarray(target, dtype=np.float32)

        unscaled_targets = np.copy(target[target[:, 1] != 3])
        unscaled_targets[:, 2] *= s.shape[1]
        unscaled_targets[:, 3] *= s.shape[0]
        unscaled_targets[:, 4] *= s.shape[1]
        unscaled_targets[:, 5] *= s.shape[0]

        unscaled_targets[:, 2:] *= scale_factor

        interpol_c2o = self.interpol_c2o[piece_id][page_nr]
        add_per_staff = [self.staff_coords[piece_id][page_nr], self.add_per_staff[piece_id][page_nr]]
        piece_name = f"{self.piece_names[piece_id]}_page_{page_nr}"

        # only use two minutes of audio to avoid GPU memory issues
        truncated_signal = truncated_signal[- (120 * 2 * self.sample_rate):]

        sample = {'performance': truncated_signal,  'score': s[None], 'target': target,
                  'file_name': piece_name, 'is_onset': is_onset, 'interpol_c2o': interpol_c2o,
                  'add_per_staff': add_per_staff, 'scale_factor': scale_factor, 'unscaled_target': unscaled_targets,
                  't': t}

        if self.transform:
            # print("do transform")
            sample = self.transform(sample)

        return sample

class MaskedSequenceDataset(Dataset):
    def __init__(self, scores, performances, sequences, piece_names, interpol_c2o,
                 staff_coords, add_per_staff, predict_sb=False, augment=False, transform=None):

        self.scores = scores
        self.performances = performances
        self.sequences = []
        self.rand_perf_indices = {}
        self.sequences = sequences
        self.augment = augment
        self.piece_names = piece_names
        self.interpol_c2o = interpol_c2o
        self.staff_coords = staff_coords
        self.add_per_staff = add_per_staff

        self.predict_sb = predict_sb

        self.fps = FPS
        self.sample_rate = SAMPLE_RATE
        self.frame_size = FRAME_SIZE
        self.hop_length = HOP_SIZE

        self.length = len(self.sequences)
        self.gt_width = 30
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        seq = self.sequences[item]

        piece_id = seq['piece_id']
        mask_id = seq['mask_id']
        score = self.scores[piece_id]

        is_onset = seq['is_onset']

        signal = self.performances[piece_id]

        # if signal is provided as a path it should be loaded from the disk
        if isinstance(signal, str):
            signal = load_wav(signal, SAMPLE_RATE)

        start_frame = int(seq['start_frame'])
        frame = int(seq['frame'])
        scale_factor = seq['scale_factor']

        start_t = int(start_frame * self.hop_length)
        t = self.frame_size + int(frame * self.hop_length)

        truncated_signal = signal[start_t:t]

        true_position, page_nr = seq['true_position'][:2], seq['true_position'][-1]
        
        notes_available = true_position[0] >= 0

        page_nr = int(page_nr)
        mask_id = int(mask_id)
        s = score[mask_id]

        system = np.asarray(seq['true_system'], dtype=np.float32)
        system /= scale_factor

        bar = np.asarray(seq['true_bar'], dtype=np.float32)
        bar /= scale_factor

        true_pos = np.copy(true_position)
        true_pos = true_pos / scale_factor
        
        width = self.gt_width / scale_factor
        height = seq['height'] / scale_factor

        center_y, center_x = true_pos
        target = []
        # pretarget = []
        # target = [[0, 0, center_x/s.shape[1], center_y/s.shape[0], width/s.shape[1], height/s.shape[0], close_to_page_end]]
        if notes_available:
            target.append([0, 0, center_x/s.shape[1], center_y/s.shape[0], width/s.shape[1], height/s.shape[0]])
        
        if self.predict_sb:
            target.append([0, 1, bar[0] / s.shape[1], bar[1] / s.shape[0], bar[2] / s.shape[1], bar[3] / s.shape[0]])
            target.append([0, 2, system[0]/s.shape[1], system[1]/s.shape[0], system[2]/s.shape[1], system[3]/s.shape[0]])

        target = np.asarray(target, dtype=np.float32)

        unscaled_targets = np.copy(target[target[:, 1] != 3])
        unscaled_targets[:, 2] *= s.shape[1]
        unscaled_targets[:, 3] *= s.shape[0]
        unscaled_targets[:, 4] *= s.shape[1]
        unscaled_targets[:, 5] *= s.shape[0]

        unscaled_targets[:, 2:] *= scale_factor

        interpol_c2o = self.interpol_c2o[piece_id][mask_id]
        add_per_staff = [self.staff_coords[piece_id][mask_id], self.add_per_staff[piece_id][mask_id]]
        piece_name = f"{self.piece_names[piece_id]}_page_{page_nr}_masked_{mask_id}"

        # only use two minutes of audio to avoid GPU memory issues
        truncated_signal = truncated_signal[- (120 * 2 * self.sample_rate):]

        sample = {'performance': truncated_signal,  'score': s[None], 'target': target,
                  'file_name': piece_name, 'is_onset': is_onset, 'interpol_c2o': interpol_c2o,
                  'add_per_staff': add_per_staff, 'scale_factor': scale_factor, 'unscaled_target': unscaled_targets,
                  't': t}

        if self.transform:
            # print("do transform")
            sample = self.transform(sample)

        return sample

class BiSequenceDataset(Dataset):
    def __init__(self, scores, performances, sequences, piece_names, interpol_c2o,
                 staff_coords, add_per_staff, predict_sb=False, augment=False, transform=None):

        self.scores = scores
        self.performances = performances
        self.sequences = []
        self.rand_perf_indices = {}
        self.sequences = sequences
        self.augment = augment
        self.piece_names = piece_names
        self.interpol_c2o = interpol_c2o
        self.staff_coords = staff_coords
        self.add_per_staff = add_per_staff

        self.predict_sb = predict_sb

        self.fps = FPS
        self.sample_rate = SAMPLE_RATE
        self.frame_size = FRAME_SIZE
        self.hop_length = HOP_SIZE

        self.length = len(self.sequences)
        self.gt_width = 30
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        
        seq = self.sequences[item]
        
        piece_id = seq['piece_id']
        score = self.scores[piece_id]

        is_onset = seq['is_onset']

        signal = self.performances[piece_id]

        # if signal is provided as a path it should be loaded from the disk
        if isinstance(signal, str):
            signal = load_wav(signal, SAMPLE_RATE)

        scale_factor = seq['scale_factor']
        truncated_signal = []
        if isinstance(seq['start_frame'], list) and isinstance(seq['frame'], list):
            for start_frame, frame in zip(seq['start_frame'], seq['frame']):
                start_t = int(start_frame * self.hop_length)
                t = self.frame_size + int(frame * self.hop_length)
                
                
                if truncated_signal == []:
                    truncated_signal = signal[start_t:t]
                else:
                    truncated_signal = np.concatenate((truncated_signal, signal[start_t:t]))
                    

        else:
            start_frame = int(seq['start_frame'])
            frame = int(seq['frame'])
            

            start_t = int(start_frame * self.hop_length)
            t = self.frame_size + int(frame * self.hop_length)

            truncated_signal = signal[start_t:t]
            
        true_position, page_nr = seq['true_position'][:2], seq['true_position'][-1]
        notes_available = true_position[0] >= 0
        max_y_shift = seq['max_y_shift']
        max_x_shift = seq['max_x_shift']

        page_nr = int(page_nr)
        s = score[page_nr]

        system = np.asarray(seq['true_system'], dtype=np.float32)
        system /= scale_factor

        bar = np.asarray(seq['true_bar'], dtype=np.float32)
        bar /= scale_factor

        true_pos = np.copy(true_position)

        true_pos = true_pos / scale_factor
        width = self.gt_width / scale_factor
        height = seq['height'] / scale_factor

        if self.augment:

            yshift = random.randint(int(max_y_shift[0]/scale_factor), int(max_y_shift[1]/scale_factor))
            xshift = random.randint(int(max_x_shift[0] / scale_factor), int(max_x_shift[1] / scale_factor))

            true_pos[0] += yshift
            true_pos[1] += xshift

            s = np.roll(s, yshift, 0)
            s = np.roll(s, xshift, 1)

            # System [center_x, center_y, width, height]
            system[0] += xshift
            system[1] += yshift

            bar[0] += xshift
            bar[1] += yshift

            # pad signal randomly by 0-20 frames (0-1seconds)
            truncated_signal = np.pad(truncated_signal, (random.randint(0, int(self.fps)) * self.hop_length, 0),
                                      mode='constant')

        center_y, center_x = true_pos

        target = []

        # target = [[0, 0, center_x/s.shape[1], center_y/s.shape[0], width/s.shape[1], height/s.shape[0], close_to_page_end]]
        if notes_available:
            target.append([0, 0, center_x/s.shape[1], center_y/s.shape[0], width/s.shape[1], height/s.shape[0]])
        # target = []

        if self.predict_sb:
            target.append([0, 1, bar[0] / s.shape[1], bar[1] / s.shape[0], bar[2] / s.shape[1], bar[3] / s.shape[0]])
            target.append([0, 2, system[0]/s.shape[1], system[1]/s.shape[0], system[2]/s.shape[1], system[3]/s.shape[0]])

        target = np.asarray(target, dtype=np.float32)

        unscaled_targets = np.copy(target)
        unscaled_targets[:, 2] *= s.shape[1]
        unscaled_targets[:, 3] *= s.shape[0]
        unscaled_targets[:, 4] *= s.shape[1]
        unscaled_targets[:, 5] *= s.shape[0]

        unscaled_targets[:, 2:] *= scale_factor

        interpol_c2o = self.interpol_c2o[piece_id][page_nr]
        add_per_staff = [self.staff_coords[piece_id][page_nr], self.add_per_staff[piece_id][page_nr]]
        
        piece_name = f"{self.piece_names[piece_id]}_page_{page_nr}"
        # print(piece_name, seq['start_frame'], seq['frame'])
        # only use two minutes of audio to avoid GPU memory issues
        truncated_signal = truncated_signal[- (120 * self.sample_rate):]
        
        sample = {'performance': truncated_signal,  'score': s[None], 'target': target,
                  'file_name': piece_name, 'is_onset': is_onset, 'interpol_c2o': interpol_c2o,
                  'add_per_staff': add_per_staff, 'scale_factor': scale_factor, 'unscaled_target': unscaled_targets,
                  't': t}

        if self.transform:
            sample = self.transform(sample)

        return sample

class MaskedBiSequenceDataset(Dataset):
    def __init__(self, scores, performances, sequences, piece_names, interpol_c2o,
                 staff_coords, add_per_staff, predict_sb=False, augment=False, transform=None):

        self.scores = scores
        self.performances = performances
        self.sequences = []
        self.rand_perf_indices = {}
        self.sequences = sequences
        self.augment = augment
        self.piece_names = piece_names
        self.interpol_c2o = interpol_c2o
        self.staff_coords = staff_coords
        self.add_per_staff = add_per_staff

        self.predict_sb = predict_sb

        self.fps = FPS
        self.sample_rate = SAMPLE_RATE
        self.frame_size = FRAME_SIZE
        self.hop_length = HOP_SIZE

        self.length = len(self.sequences)
        self.gt_width = 30
        self.transform = transform
        
        # for i in range(500):
        #     print(i, self.sequences[i]['loc'], self.sequences[i]['start_frame'], self.sequences[i]['frame'])

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        
        seq = self.sequences[item]
        
        piece_id = seq['piece_id']
        score = self.scores[piece_id]

        is_onset = seq['is_onset']

        signal = self.performances[piece_id]

        # if signal is provided as a path it should be loaded from the disk
        if isinstance(signal, str):
            signal = load_wav(signal, SAMPLE_RATE)

        scale_factor = seq['scale_factor']
        truncated_signal = []
        
        # print(item, seq['start_frame'], seq['frame'])
        

        start_frame = seq['start_frame']
        frame = int(seq['frame'])
        start_t = int(start_frame * self.hop_length)
        t = self.frame_size + int(frame * self.hop_length)

        truncated_signal = signal[start_t:t]

            
        true_position, page_nr = seq['true_position'][:2], seq['mask_score']
        notes_available = true_position[0] >= 0
        max_y_shift = seq['max_y_shift']
        max_x_shift = seq['max_x_shift']

        page_nr = int(page_nr)
        # mask_id = seq['mask_score']
        s = score[page_nr]

        system = np.asarray(seq['true_system'], dtype=np.float32)
        system /= scale_factor

        bar = np.asarray(seq['true_bar'], dtype=np.float32)
        bar /= scale_factor

        true_pos = np.copy(true_position)

        true_pos = true_pos / scale_factor
        width = self.gt_width / scale_factor
        height = seq['height'] / scale_factor

        if self.augment:

            yshift = random.randint(int(max_y_shift[0]/scale_factor), int(max_y_shift[1]/scale_factor))
            xshift = random.randint(int(max_x_shift[0] / scale_factor), int(max_x_shift[1] / scale_factor))

            true_pos[0] += yshift
            true_pos[1] += xshift

            s = np.roll(s, yshift, 0)
            s = np.roll(s, xshift, 1)

            # System [center_x, center_y, width, height]
            system[0] += xshift
            system[1] += yshift

            bar[0] += xshift
            bar[1] += yshift

            # pad signal randomly by 0-20 frames (0-1seconds)
            truncated_signal = np.pad(truncated_signal, (random.randint(0, int(self.fps)) * self.hop_length, 0),
                                      mode='constant')

        center_y, center_x = true_pos

        target = []

        # target = [[0, 0, center_x/s.shape[1], center_y/s.shape[0], width/s.shape[1], height/s.shape[0], close_to_page_end]]
        if notes_available:
            target.append([0, 0, center_x/s.shape[1], center_y/s.shape[0], width/s.shape[1], height/s.shape[0]])
        # target = []

        if self.predict_sb:
            target.append([0, 1, bar[0] / s.shape[1], bar[1] / s.shape[0], bar[2] / s.shape[1], bar[3] / s.shape[0]])
            target.append([0, 2, system[0]/s.shape[1], system[1]/s.shape[0], system[2]/s.shape[1], system[3]/s.shape[0]])

        target = np.asarray(target, dtype=np.float32)

        unscaled_targets = np.copy(target)
        unscaled_targets[:, 2] *= s.shape[1]
        unscaled_targets[:, 3] *= s.shape[0]
        unscaled_targets[:, 4] *= s.shape[1]
        unscaled_targets[:, 5] *= s.shape[0]

        unscaled_targets[:, 2:] *= scale_factor

        interpol_c2o = self.interpol_c2o[piece_id][page_nr]
        add_per_staff = [self.staff_coords[piece_id][page_nr], self.add_per_staff[piece_id][page_nr]]
        
        piece_name = f"{self.piece_names[piece_id]}_bipage_{page_nr}_mask_{page_nr}"
        # print(piece_name, seq['start_frame'], seq['frame'])
        # only use two minutes of audio to avoid GPU memory issues
        truncated_signal = truncated_signal[- (120 * self.sample_rate):]
        
        sample = {'performance': truncated_signal,  'score': s[None], 'target': target,
                  'file_name': piece_name, 'is_onset': is_onset, 'interpol_c2o': interpol_c2o,
                  'add_per_staff': add_per_staff, 'scale_factor': scale_factor, 'unscaled_target': unscaled_targets}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class CustomBatch:
    def __init__(self, batch):
        self.file_names = [x['file_name'] for x in batch]
        self.perf = [torch.as_tensor(x['performance'], dtype=torch.float32) for x in batch]
        targets = []
        unscaled_targets = []
        for i, x in enumerate(batch):
            # add image idx to targets for loss computation
            if x['target'] is not None:
                target = x['target']

                target[:, 0] = i
                targets.append(target)

                unscaled_target = x['unscaled_target']
                unscaled_target[:, 0] = i
                unscaled_targets.append(unscaled_target)

        self.targets = torch.as_tensor(np.concatenate(targets), dtype=torch.float32)
        self.unscaled_targets = torch.as_tensor(np.concatenate(unscaled_targets), dtype=torch.float32)

        self.interpols = [x['interpol_c2o'] for x in batch]
        self.add_per_staff = [x['add_per_staff'] for x in batch]

        self.scores = torch.as_tensor(np.stack([x['score'] for x in batch]), dtype=torch.float32)
        self.scale_factors = torch.FloatTensor([x['scale_factor'] for x in batch]).float().unsqueeze(-1)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.scores = self.scores.pin_memory()
        self.perf = [p.pin_memory() for p in self.perf]
        self.targets = self.targets.pin_memory()
        self.unscaled_targets = self.unscaled_targets.pin_memory()
        self.scale_factors = self.scale_factors.pin_memory()
        return self


def collate_wrapper(batch):
    return CustomBatch(batch)


def compute_batch_stats(detections, true_positions, piece_stats, file_names, file_interpols, file_add_per_staff):
    gt = true_positions.float().cpu()
    pred = detections[:, :2].detach().cpu()


    for num, fname in enumerate(file_names):

        gt_note = gt[((gt[:, 0] == num) & (gt[:, 1] == 0))]

        if fname not in piece_stats:
            piece_stats[fname] = {}

        if 'frame_diff' not in piece_stats[fname]:
            piece_stats[fname]['frame_diff'] = []

        if len(gt_note) == 1:

            gt_note = gt_note[0, 2:4]

            staff_coords, add_per_staff = file_add_per_staff[num]

            staff_id_pred = np.argwhere(min(staff_coords, key=lambda y: abs(y - pred[num][1])) == staff_coords).item()
            staff_id_gt = np.argwhere(min(staff_coords, key=lambda y: abs(y - gt_note[1])) == staff_coords).item()

            # unroll x coord
            x_coord_gt = gt_note[0] + add_per_staff[staff_id_gt]
            x_coord_pred = pred[num][0] + add_per_staff[staff_id_pred]

            # calculate difference of onset frames
            frame_diff = abs(file_interpols[num](x_coord_pred) - file_interpols[num](x_coord_gt))

            piece_stats[fname]['frame_diff'].append(frame_diff)

    return piece_stats


def eval_class(prediction, targets, piece_stats, file_names, scale_factors, class_id=1, th=0.8):
    gt_boxes = targets[targets[:, 1] == class_id][:, 2:]

    pred_boxes = get_max_box(prediction, class_id=class_id)

    pred_boxes *= scale_factors
    # print(scale_factors)
    pred_boxes = xywh2xyxy(pred_boxes)
    gt_boxes = xywh2xyxy(gt_boxes)
    
    iou = np.diagonal(box_iou(pred_boxes, gt_boxes).cpu().numpy())
    
    # print(class_id, pred_boxes.shape, gt_boxes.shape, iou.shape)
    for num, fname in enumerate(file_names):

        if fname not in piece_stats:
            piece_stats[fname] = {}

        if class_id not in piece_stats[fname]:
            piece_stats[fname][class_id] = []

        # count as a correct prediction if iou is over a certain threshold
        piece_stats[fname][class_id].append(int(iou[num] > th))

    return piece_stats, pred_boxes/scale_factors

            
def load_dataset(paths, augment=False, scale_width=416, split_files=None, ir_path=None,
                 only_onsets=False, load_audio=True, predict_sb=False, score_type="basic"):

    scores = {}
    piece_names = {}
    all_sequences = []
    performances = {}
    interpol_c2os = {}
    staff_coords_all = {}
    add_per_staff_all = {}
    params = []

    files = []
    if split_files is not None:
        assert len(split_files) == len(paths)

        for idx, split_file in enumerate(split_files):
            split = load_yaml(split_file)
            files.extend([os.path.join(paths[idx], f'{file}.npz') for file in split['files']])

    else:
        for path in paths:
            files.extend(glob.glob(os.path.join(path, '*.npz')))

    for i, score_path in enumerate(files):
        # if os.path.basename(score_path)[:-4] in Repeat_List:
        if "ChopinFF__O9__nocturne_in_b-flat_minor_synth" not in score_path:
            params.append(dict(
                i=i,
                piece_name=os.path.basename(score_path)[:-4],
                path=os.path.dirname(score_path),
                scale_width=scale_width,
                load_audio=load_audio
            ))
        # else:
        #     print(os.path.basename(score_path)[:-4])

    print(f'Loading {len(params)} file(s)...')

    # results = [load_piece_sequences(params[0])]
    with get_context("spawn").Pool(16) as pool:
        if score_type == "basic":   
            results = list(tqdm(pool.imap_unordered(load_sequences, params), total=len(params)))
        elif score_type == "masked": 
            results = list(tqdm(pool.imap_unordered(load_msseq, params), total=len(params)))
        elif score_type == "bipage": 
            results = list(tqdm(pool.imap_unordered(load_bipage_sequences, params), total=len(params)))
        elif score_type == "masked_bipage": 
            results = list(tqdm(pool.imap_unordered(load_masked_bipage_sequences, params), total=len(params)))
        elif score_type == "blank_bipage": 
            results = list(tqdm(pool.imap_unordered(load_blank_bipage_sequences, params), total=len(params)))
            
        print("result", len(results))
        for result in results:
            i, score, signals, piece_name, sequences, interpol_c2o, staff_coords, add_per_staff = result
            scores[i] = score
            performances[i] = signals
            piece_names[i] = piece_name
            interpol_c2os[i] = interpol_c2o
            staff_coords_all[i] = staff_coords
            add_per_staff_all[i] = add_per_staff

            all_sequences.extend([seq for seq in sequences if (seq['is_onset'] or not only_onsets)])

    print('Done loading.')

    if ir_path is not None:
        print('Using Impulse Response Augmentation')
        ir_aug = ImpulseResponse(ir_paths=ir_path, ir_prob=0.5)
        transform = torchvision.transforms.Compose([ir_aug])
    else:
        transform = None

    if score_type == "basic": 
        return SequenceDataset(scores, performances, all_sequences, piece_names, interpol_c2os, staff_coords_all,
                            add_per_staff_all, augment=augment, transform=transform, predict_sb=predict_sb)
    elif score_type == "masked": 
        return MaskedSequenceDataset(scores, performances, all_sequences, piece_names, interpol_c2os, staff_coords_all,
                            add_per_staff_all, augment=augment, transform=transform, predict_sb=predict_sb)
    elif score_type == "bipage": 
        return BiSequenceDataset(scores, performances, all_sequences, piece_names, interpol_c2os, staff_coords_all,
                        add_per_staff_all, augment=augment, transform=transform, predict_sb=predict_sb)
    elif score_type == "masked_bipage": 
        return MaskedBiSequenceDataset(scores, performances, all_sequences, piece_names, interpol_c2os, staff_coords_all,
                        add_per_staff_all, augment=augment, transform=transform, predict_sb=predict_sb)
    elif score_type == "blank_bipage": 
        return BiSequenceDataset(scores, performances, all_sequences, piece_names, interpol_c2os, staff_coords_all,
                        add_per_staff_all, augment=augment, transform=transform, predict_sb=predict_sb)
import cv2 as cv
from matplotlib import pyplot as plt
def iterate_dataset(network, dataloader, criterion, optimizer=None, clip_grads=None,
                    device=torch.device('cuda'), tempo_aug=False):
    train = optimizer is not None
    losses = {}

    piece_stats = {}

    if is_main_process():
        progress_bar = tqdm(total=len(dataloader), ncols=80)
    plt.figure()
    for batch_idx, data in enumerate(dataloader):

        scores = data.scores.to(device, non_blocking=True)
        scale_factors = data.scale_factors.to(device, non_blocking=True)
        targets = data.targets.to(device, non_blocking=True)
        # plt.subplot(211)
        # if batch_idx %10 == 0:
        #     plt.imshow(scores[0][0].cpu().numpy())
        #     plt.scatter(targets[0][2].cpu().numpy()*416, targets[0][3].cpu().numpy()*416)
        #     plt.show()
        # print(len(targets), targets[0])
        # print(scores[0].shape)
        
        perf = [p.to(device, non_blocking=True) for p in data.perf]

        with torch.set_grad_enabled(train):
            inference_out, pred = network(score=scores, perf=perf, tempo_aug=tempo_aug)

            loss_dict = criterion(pred, targets, network)
            loss = loss_dict['loss']
            for key in loss_dict:

                if key not in losses:
                    losses[key] = AverageMeter()

                losses[key].update(loss_dict[key].item())

        # perform update
        if train:
            optimizer.zero_grad()
            loss.backward()

            if clip_grads is not None:

                # only clip gradients of the recurrent network
                if hasattr(network, "conditioning_network"):
                    clip_grad_norm_(network.conditioning_network.seq_model.parameters(), clip_grads)
                else:
                    # distributed data parallel
                    clip_grad_norm_(network.module.conditioning_network.seq_model.parameters(), clip_grads)

            optimizer.step()

        unscaled_targets = data.unscaled_targets.to(device, non_blocking=True)

        inference_out = inference_out.detach()

        pred_boxes = get_max_box(inference_out)
        pred_boxes *= scale_factors
        
        # continue
        piece_stats = compute_batch_stats(pred_boxes, unscaled_targets,
                                          piece_stats, data.file_names, data.interpols, data.add_per_staff)

        for class_id in range(1, network.nc):
            piece_stats, _ = eval_class(inference_out, unscaled_targets, piece_stats,
                                        data.file_names, scale_factors, class_id=class_id)

        if is_main_process():
            progress_bar.update(1)

    # summarize statistics
    stats = {'piece_stats': {}}

    frame_diffs = []
    for key in piece_stats:
        stat = piece_stats[key]

        assert key not in stats['piece_stats']
        stats['piece_stats'][key] = {'frame_diff': stat['frame_diff']}

        frame_diffs.extend(stat['frame_diff'])

        for i in range(1, network.nc):
            stats['piece_stats'][key][CLASS_MAPPING[i]+"_accuracy"] = float(np.mean(stat[i]))

    stats['frame_diffs_mean'] = float(np.mean(frame_diffs))

    for i in range(1, network.nc):
        stats[CLASS_MAPPING[i]+"_accuracy"] = float(np.mean([stats['piece_stats'][key][CLASS_MAPPING[i]+"_accuracy"]
                                                             for key in stats['piece_stats'].keys()]))

    # add losses to statistics
    for key in losses:
        stats[key] = losses[key].avg

    if is_main_process():
        progress_bar.close()

    return stats

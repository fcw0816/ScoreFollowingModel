import cv2
import os

import numpy as np

from collections import Counter
from cyolo_score_following.utils.general import load_wav, xywh2xyxy
from scipy import interpolate

from repeatsigncropping.cropping import RepeatCropping
SAMPLE_RATE = 22050
FRAME_SIZE = 2048
HOP_SIZE = 1102
FPS = SAMPLE_RATE/HOP_SIZE




def load_piece(path, piece_name):
    npzfile = np.load(os.path.join(path, piece_name + '.npz'), allow_pickle=True)
    # print(path)

    scores = npzfile["sheets"]
    coords, systems, bars = list(npzfile["coords"]), list(npzfile['systems']), list(npzfile['bars'])

    synthesized = npzfile['synthesized'].item()
    n_pages, h, w = scores.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # Determine padding
    pad = ((0, 0), (0, 0), (pad1, pad2))

    # Add padding
    padded_scores = np.pad(scores, pad, mode="constant", constant_values=255)

    wav_path = os.path.join(path, piece_name + '.wav')
    signal = load_wav(wav_path, sr=SAMPLE_RATE)

    onsets = []
        
    for i in range(len(coords)):
        if coords[i]['note_x'] > 0:
            coords[i]['note_x'] += pad1

        # onset time to frame
        coords[i]['onset'] = int(coords[i]['onset'] * FPS)
        onsets.append(coords[i]['onset'])

    for i in range(len(systems)):
        systems[i]['x'] += pad1

    for i in range(len(bars)):
        bars[i]['x'] += pad1

    onsets = np.asarray(onsets, dtype=np.int)

    onsets = np.unique(onsets)
    coords_new = []
    for onset in onsets:
        onset_coords = list(filter(lambda x: x['onset'] == onset, coords))

        onset_coords_merged = {}
        for entry in onset_coords:
            for key in entry:
                if key not in onset_coords_merged:
                    onset_coords_merged[key] = []
                onset_coords_merged[key].append(entry[key])

        # get system and page with most notes in it
        system_idx = int(Counter(onset_coords_merged['system_idx']).most_common(1)[0][0])
        note_x = np.mean(
            np.asarray(onset_coords_merged['note_x'])[np.asarray(onset_coords_merged['system_idx']) == system_idx])
        page_nr = int(Counter(onset_coords_merged['page_nr']).most_common(1)[0][0])
        bar_idx = int(Counter(onset_coords_merged['bar_idx']).most_common(1)[0][0])

        # set y to staff center
        note_y = -1.0
        if note_x > 0:
            note_y = systems[system_idx]['y']
        coords_new.append([note_y, note_x, system_idx, bar_idx, page_nr])
    coords_new = np.asarray(coords_new)

    # we want to match the frames to the coords of the previous onset, as the notes at the next coord position
    # aren't played yet
    interpol_fnc = interpolate.interp1d(onsets, coords_new.T, kind='previous', bounds_error=False,
                                        fill_value=(coords_new[0, :], coords_new[-1, :]))

    return padded_scores, scores, onsets, coords_new, bars, systems, interpol_fnc, signal, pad1, synthesized

def load_masked_piece(path, piece_name):
    # print(path, piece_name)
    npzfile = np.load(os.path.join(path, piece_name + '.npz'), allow_pickle=True)
    # print(path)

    scores = npzfile["sheets"]
    coords, systems, bars = list(npzfile["coords"]), list(npzfile['systems']), list(npzfile['bars'])
    
    # null_page = np.ones_like(scores[:1, :, :]) * 255
    # scores = np.concatenate((scores, null_page), axis=0)
    if "room" in piece_name:
        piece_name = piece_name[:-4] + "synth"
        path = path[:-2] + "test"
        print(path, piece_name)
    rpcrop = RepeatCropping(
        msmd_test=path,
        msmd_test_repeat=r"D:\scorefollowersystem\cyolo_score_following\data\msmd\repeat_subset\msmd_test_image",
        piece=piece_name 
    )


    if rpcrop.score_mask == [] or "Kruetzer__lodiska__lodiska_synth" == piece_name:
        
        print(f"No repeat sign piece")
    else:
       

        masked_score = []
        for page, mask in rpcrop.score_mask:
            tmp = scores[page].copy()
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i, j] == 0:
                        tmp[i, j] = 255

            masked_score.append(tmp)
        
        scores = np.array(masked_score)

    synthesized = npzfile['synthesized'].item()
    n_pages, h, w = scores.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # Determine padding
    pad = ((0, 0), (0, 0), (pad1, pad2))

    # Add padding
    padded_scores = np.pad(scores, pad, mode="constant", constant_values=255)

    wav_path = os.path.join(path, piece_name + '.wav')
    signal = load_wav(wav_path, sr=SAMPLE_RATE)

    onsets = []
    for i in range(len(coords)):
        if coords[i]['note_x'] > 0:
            coords[i]['note_x'] += pad1

        # onset time to frame
        coords[i]['onset'] = int(coords[i]['onset'] * FPS)
        onsets.append(coords[i]['onset'])

    for i in range(len(systems)):
        systems[i]['x'] += pad1

    for i in range(len(bars)):
        bars[i]['x'] += pad1

    onsets = np.asarray(onsets, dtype=np.int)

    onsets = np.unique(onsets)
    coords_new = []
    for onset in onsets:
        onset_coords = list(filter(lambda x: x['onset'] == onset, coords))

        onset_coords_merged = {}
        for entry in onset_coords:
            for key in entry:
                if key not in onset_coords_merged:
                    onset_coords_merged[key] = []
                onset_coords_merged[key].append(entry[key])

        # get system and page with most notes in it
        system_idx = int(Counter(onset_coords_merged['system_idx']).most_common(1)[0][0])
        note_x = np.mean(
            np.asarray(onset_coords_merged['note_x'])[np.asarray(onset_coords_merged['system_idx']) == system_idx])
        page_nr = int(Counter(onset_coords_merged['page_nr']).most_common(1)[0][0])
        bar_idx = int(Counter(onset_coords_merged['bar_idx']).most_common(1)[0][0])

        # set y to staff center
        note_y = -1.0
        if note_x > 0:
            note_y = systems[system_idx]['y']
        coords_new.append([note_y, note_x, system_idx, bar_idx, page_nr])
    coords_new = np.asarray(coords_new)

    # we want to match the frames to the coords of the previous onset, as the notes at the next coord position
    # aren't played yet
    interpol_fnc = interpolate.interp1d(onsets, coords_new.T, kind='previous', bounds_error=False,
                                        fill_value=(coords_new[0, :], coords_new[-1, :]))
    
    return padded_scores, scores, onsets, coords_new, bars, systems, interpol_fnc, signal, pad1, synthesized, rpcrop.score_mask

def load_bipiece(path, piece_name):
    npzfile = np.load(os.path.join(path, piece_name + '.npz'), allow_pickle=True)

    scores = npzfile["sheets"]
    coords, systems, bars = list(npzfile["coords"]), list(npzfile['systems']), list(npzfile['bars'])

    synthesized = npzfile['synthesized'].item()
    
    null_page = np.ones_like(scores[:1, :, :]) * 255
    
    scores = np.concatenate((scores, null_page), axis=0)
    # print(scores.shape, piece_name)
    bipage = []
    
    for i in range(len(scores)-1):
        # print(i, np.concatenate((scores[i], scores[i+1]), axis=-1).shape)
        bipage.append(np.concatenate((scores[i], scores[i+1]), axis=-1))  
    bipage = np.array(bipage)
    
    # print(bipage.shape, piece_name)  
    n_pages, h, w = bipage.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((0, 0), (pad1, pad2), (0, 0))

    # Add padding
    padded_scores = np.pad(bipage, pad, mode="constant", constant_values=255)
    wav_path = os.path.join(path, piece_name + '.wav')
    signal = load_wav(wav_path, sr=SAMPLE_RATE)

    onsets = []
    for i in range(len(coords)):
        if coords[i]['note_y'] > 0:
            coords[i]['note_y'] += pad1

        # onset time to frame
        coords[i]['onset'] = int(coords[i]['onset'] * FPS)
        onsets.append(coords[i]['onset'])

    for i in range(len(systems)):
        systems[i]['y'] += pad1

    for i in range(len(bars)):
        bars[i]['y'] += pad1

    onsets = np.asarray(onsets, dtype=np.int)

    onsets = np.unique(onsets)
    coords_new = []
    for onset in onsets:
        onset_coords = list(filter(lambda x: x['onset'] == onset, coords))

        onset_coords_merged = {}
        for entry in onset_coords:
            for key in entry:
                if key not in onset_coords_merged:
                    onset_coords_merged[key] = []
                onset_coords_merged[key].append(entry[key])

        # get system and page with most notes in it
        system_idx = int(Counter(onset_coords_merged['system_idx']).most_common(1)[0][0])
        note_x = np.mean(
            np.asarray(onset_coords_merged['note_x'])[np.asarray(onset_coords_merged['system_idx']) == system_idx])
        page_nr = int(Counter(onset_coords_merged['page_nr']).most_common(1)[0][0])
        bar_idx = int(Counter(onset_coords_merged['bar_idx']).most_common(1)[0][0])

        # set y to staff center
        note_y = -1.0
        if note_x > 0:
            note_y = systems[system_idx]['y']
        # print([note_y, note_x, system_idx, bar_idx, page_nr])
        coords_new.append([note_y, note_x, system_idx, bar_idx, page_nr])
    coords_new = np.asarray(coords_new)

    # we want to match the frames to the coords of the previous onset, as the notes at the next coord position
    # aren't played yet
    interpol_fnc = interpolate.interp1d(onsets, coords_new.T, kind='previous', bounds_error=False,
                                        fill_value=(coords_new[0, :], coords_new[-1, :]))

    return padded_scores, bipage, onsets, coords_new, bars, systems, interpol_fnc, signal, pad1, synthesized

def load_masked_bipiece(path, piece_name):
    npzfile = np.load(os.path.join(path, piece_name + '.npz'), allow_pickle=True)
    # print(path)

    scores = npzfile["sheets"]
    coords, systems, bars = list(npzfile["coords"]), list(npzfile['systems']), list(npzfile['bars'])
    
    synthesized = npzfile['synthesized'].item()
       
    rpcrop = RepeatCropping(
        msmd_test=path,
        msmd_test_repeat=r"D:\scorefollowersystem\cyolo_score_following\data\msmd\repeat_subset\msmd_test_image",
        piece=piece_name 
    )
    # print(f"{piece_name}", len(rpcrop.score_mask))
    masked_score = []

    for page, mask in rpcrop.score_mask:

        # print(len(rpcrop.score_mask), page, np.sum(mask))
        tmp = scores[page].copy()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 0:
                    tmp[i, j] = 255
        # if piece_name in d:
        #     print(sum(sum(tmp)))
        #     plt.imshow(tmp)
        #     plt.show()
        masked_score.append(tmp)
    
    masked_score = np.array(masked_score)
    
    
    null_page = np.ones_like(masked_score[:1, :, :]) * 255
    
    scores = np.concatenate((masked_score, null_page), axis=0)

    masked_bipage = []
    
    for i in range(len(scores)-1):
        # print(i, np.concatenate((scores[i], scores[i+1]), axis=-1).shape)
        masked_bipage.append(np.concatenate((scores[i], scores[i+1]), axis=-1))  
    masked_bipage = np.array(masked_bipage)

    n_pages, h, w = masked_bipage.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((0, 0), (pad1, pad2), (0, 0))

    # Add padding
    padded_scores = np.pad(masked_bipage, pad, mode="constant", constant_values=255)

    wav_path = os.path.join(path, piece_name + '.wav')
    signal = load_wav(wav_path, sr=SAMPLE_RATE)

    onsets = []
    for i in range(len(coords)):
        if coords[i]['note_y'] > 0:
            coords[i]['note_y'] += pad1

        # onset time to frame
        coords[i]['onset'] = int(coords[i]['onset'] * FPS)
        onsets.append(coords[i]['onset'])

    for i in range(len(systems)):
        systems[i]['y'] += pad1

    for i in range(len(bars)):
        bars[i]['y'] += pad1

    onsets = np.asarray(onsets, dtype=np.int)

    onsets = np.unique(onsets)
    coords_new = []
    for onset in onsets:
        onset_coords = list(filter(lambda x: x['onset'] == onset, coords))

        onset_coords_merged = {}
        for entry in onset_coords:
            for key in entry:
                if key not in onset_coords_merged:
                    onset_coords_merged[key] = []
                onset_coords_merged[key].append(entry[key])

        # get system and page with most notes in it
        system_idx = int(Counter(onset_coords_merged['system_idx']).most_common(1)[0][0])
        note_x = np.mean(
            np.asarray(onset_coords_merged['note_x'])[np.asarray(onset_coords_merged['system_idx']) == system_idx])
        page_nr = int(Counter(onset_coords_merged['page_nr']).most_common(1)[0][0])
        bar_idx = int(Counter(onset_coords_merged['bar_idx']).most_common(1)[0][0])

        # set y to staff center
        note_y = -1.0
        if note_x > 0:
            note_y = systems[system_idx]['y']
        # print([note_y, note_x, system_idx, bar_idx, page_nr])
        coords_new.append([note_y, note_x, system_idx, bar_idx, page_nr])
    coords_new = np.asarray(coords_new)

    # we want to match the frames to the coords of the previous onset, as the notes at the next coord position
    # aren't played yet
    interpol_fnc = interpolate.interp1d(onsets, coords_new.T, kind='previous', bounds_error=False,
                                        fill_value=(coords_new[0, :], coords_new[-1, :]))

    return padded_scores, masked_bipage, onsets, coords_new, bars, systems, interpol_fnc, signal, pad1, synthesized, rpcrop.score_mask

def load_blank_bipiece(path, piece_name):
    npzfile = np.load(os.path.join(path, piece_name + '.npz'), allow_pickle=True)

    scores = npzfile["sheets"]
    coords, systems, bars = list(npzfile["coords"]), list(npzfile['systems']), list(npzfile['bars'])

    synthesized = npzfile['synthesized'].item()
    
    null_page = np.ones_like(scores[0, :, :]) * 255
    
    bipage = []
    
    for i in range(len(scores)):
        # print(i, np.concatenate((scores[i], scores[i+1]), axis=-1).shape)
        # print(scores[i].shape, null_page.shape)
        bipage.append(np.concatenate((scores[i], null_page), axis=-1))  
    bipage = np.array(bipage)
    # print(bipage.shape, piece_name)  
    n_pages, h, w = bipage.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((0, 0), (pad1, pad2), (0, 0))

    # Add padding
    padded_scores = np.pad(bipage, pad, mode="constant", constant_values=255)
    wav_path = os.path.join(path, piece_name + '.wav')
    signal = load_wav(wav_path, sr=SAMPLE_RATE)

    onsets = []
    for i in range(len(coords)):
        if coords[i]['note_y'] > 0:
            coords[i]['note_y'] += pad1

        # onset time to frame
        coords[i]['onset'] = int(coords[i]['onset'] * FPS)
        onsets.append(coords[i]['onset'])

    for i in range(len(systems)):
        systems[i]['y'] += pad1

    for i in range(len(bars)):
        bars[i]['y'] += pad1

    onsets = np.asarray(onsets, dtype=np.int)

    onsets = np.unique(onsets)
    coords_new = []
    for onset in onsets:
        onset_coords = list(filter(lambda x: x['onset'] == onset, coords))

        onset_coords_merged = {}
        for entry in onset_coords:
            for key in entry:
                if key not in onset_coords_merged:
                    onset_coords_merged[key] = []
                onset_coords_merged[key].append(entry[key])

        # get system and page with most notes in it
        system_idx = int(Counter(onset_coords_merged['system_idx']).most_common(1)[0][0])
        note_x = np.mean(
            np.asarray(onset_coords_merged['note_x'])[np.asarray(onset_coords_merged['system_idx']) == system_idx])
        page_nr = int(Counter(onset_coords_merged['page_nr']).most_common(1)[0][0])
        bar_idx = int(Counter(onset_coords_merged['bar_idx']).most_common(1)[0][0])

        # set y to staff center
        note_y = -1.0
        if note_x > 0:
            note_y = systems[system_idx]['y']
        # print([note_y, note_x, system_idx, bar_idx, page_nr])
        coords_new.append([note_y, note_x, system_idx, bar_idx, page_nr])
    coords_new = np.asarray(coords_new)

    # we want to match the frames to the coords of the previous onset, as the notes at the next coord position
    # aren't played yet
    interpol_fnc = interpolate.interp1d(onsets, coords_new.T, kind='previous', bounds_error=False,
                                        fill_value=(coords_new[0, :], coords_new[-1, :]))

    return padded_scores, bipage, onsets, coords_new, bars, systems, interpol_fnc, signal, pad1, synthesized

def load_sequences(params):

    piece_idx = params['i']
    path = params['path']
    piece_name = params['piece_name']
    scale_width = params['scale_width']
    load_audio = params.get('load_audio', True)

    
    
    padded_scores, _, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized = load_piece(path, piece_name)
    # padded_scores, _, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized, score_masks = load_maskd_piece(path, piece_name)
    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

    scale_factor = padded_scores.shape[1] / scale_width

    scaled_score = []
    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    scores = np.stack(scaled_score)

    duration = signal.shape[0]
    n_frames = int(np.ceil(FPS * duration/ SAMPLE_RATE))
    piece_sequences = []

    interpol_c2o = {}
    staff_coords = {}
    add_per_staff = {}
    valid_pages = np.unique(coords_new[:, -1])

    for page_nr in valid_pages:
        page_coords = coords_new[coords_new[:, -1] == page_nr]
        page_onsets = onsets[coords_new[:, -1] == page_nr]
        unrolled_coords_x = []
        coords_per_staff = []

        # only add 0 for first staff
        max_xes = [0]
        staff_coords[page_nr] = sorted(np.unique(page_coords[:, 0]))

        for c in staff_coords[page_nr]:

            cs_staff = page_coords[page_coords[:, 0] == c, :-1]
            max_x = max(cs_staff[:, 1])
            coords_per_staff.append(cs_staff)
            max_xes.append(max_x)

        # last entry not needed
        add_per_staff[page_nr] = np.cumsum(max_xes)[:-1]
        for idx in range(len(staff_coords[page_nr])):
            unrolled_coords_x.append(coords_per_staff[idx][:, 1] + add_per_staff[page_nr][idx])

        unrolled_coords_x = np.concatenate(unrolled_coords_x)

        # if we are in between two coord position it will match it to the closest onset
        interpol_c2o[page_nr] = interpolate.interp1d(unrolled_coords_x, page_onsets, kind='nearest', bounds_error=False,
                                                     fill_value=(page_onsets[0], page_onsets[-1]))

    start_frame = 0
    curr_page = 0

    page_systems = {}
    page_bars = {}

    for page_idx in valid_pages:
        page_systems[page_idx] = list(filter(lambda x: x['page_nr'] == page_idx, systems))
        page_bars[page_idx] = list(filter(lambda bar: bar['page_nr'] == page_idx, bars))

    for frame in range(n_frames):
        frame
        true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)
        bar_idx = true_position[3]
        system_idx = true_position[2]

        # figure out at which frame we change pages
        if true_position[-1] != curr_page:
            curr_page = true_position[-1]
            start_frame = frame

        bar = bars[bar_idx]
        system = systems[system_idx]

        true_bar = np.asarray([bar['x'], bar['y'], bar['w'], bar['h']], dtype=np.float)
        true_system = np.asarray([system['x'], system['y'], system['w'], system['h']], dtype=np.float)

        systems_xywh = np.asarray([[x['x'], x['y'], x['w'], x['h']] for x in page_systems[curr_page]])
        systems_xyxy = xywh2xyxy(systems_xywh)

        max_x_shift = (-(int(systems_xyxy[:, 0].min() - 50)),
                       int(padded_scores.shape[2] - systems_xyxy[:, 2].max() - 50))
        max_y_shift = (min(0, -int((systems_xyxy[:, 1].min() - 50))),
                       max(1, int(padded_scores.shape[1] - systems_xyxy[:, 3].max() - 50)))

        piece_sequences.append({'piece_id': piece_idx,
                                'is_onset': frame in onsets,
                                'start_frame': start_frame,
                                'frame': frame,
                                'max_x_shift': max_x_shift,
                                'max_y_shift': max_y_shift,
                                'true_position': true_position,
                                'true_system': true_system,
                                'true_bar': true_bar,
                                'height': system['h'],
                                'synthesized': synthesized,
                                'scale_factor': scale_factor,
                                })

    if not load_audio:
        signal = os.path.join(path, piece_name + '.wav')

    return piece_idx, scores, signal, piece_name, piece_sequences, interpol_c2o, staff_coords, add_per_staff

def load_msseq(params):
    
    piece_idx = params['i']
    path = params['path']
    piece_name = params['piece_name']
    scale_width = params['scale_width']
    load_audio = params.get('load_audio', True)

    padded_scores, _, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized, score_masks = load_masked_piece(path, piece_name)
    
    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

    scale_factor = padded_scores.shape[1] / scale_width

    scaled_score = []
    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    scores = np.stack(scaled_score)

    duration = signal.shape[0]
    n_frames = int(np.ceil(FPS * duration / SAMPLE_RATE))
    piece_sequences = []

    interpol_c2o = {}
    staff_coords = {}
    add_per_staff = {}
    valid_pages = np.unique(coords_new[:, -1])
    # print(valid_pages, len(score_masks), score_masks[0])
    # for page_nr in valid_pages:
    for mask_id in range(len(score_masks)):
        page_nr, _ = score_masks[mask_id]
        
        page_coords = coords_new[coords_new[:, -1] == page_nr]
        page_onsets = onsets[coords_new[:, -1] == page_nr]
        unrolled_coords_x = []
        coords_per_staff = []

        # only add 0 for first staff
        max_xes = [0]
        staff_coords[mask_id] = sorted(np.unique(page_coords[:, 0]))

        for c in staff_coords[mask_id]:

            cs_staff = page_coords[page_coords[:, 0] == c, :-1]
            max_x = max(cs_staff[:, 1])
            coords_per_staff.append(cs_staff)
            max_xes.append(max_x)

        # last entry not needed
        add_per_staff[mask_id] = np.cumsum(max_xes)[:-1]
        for idx in range(len(staff_coords[mask_id])):
            unrolled_coords_x.append(coords_per_staff[idx][:, 1] + add_per_staff[mask_id][idx])

        unrolled_coords_x = np.concatenate(unrolled_coords_x)

        # if we are in between two coord position it will match it to the closest onset
        interpol_c2o[mask_id] = interpolate.interp1d(unrolled_coords_x, page_onsets, kind='nearest', bounds_error=False,
                                                     fill_value=(page_onsets[0], page_onsets[-1]))
    # print("interpolate", len(interpol_c2o))
    
    curr_page = 0

    page_systems = {}
    page_bars = {}

    for page_idx in valid_pages:
        page_systems[page_idx] = list(filter(lambda x: x['page_nr'] == page_idx, systems))
        page_bars[page_idx] = list(filter(lambda bar: bar['page_nr'] == page_idx, bars))

    for mask_id, score_mask in enumerate(score_masks):
        page, mask = score_mask
        start_frame = -1
        for frame in range(n_frames):

            true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)

            bar_idx = true_position[3]
            system_idx = true_position[2]

            bar = bars[bar_idx]
            system = systems[system_idx]

            true_bar = np.asarray([bar['x'], bar['y'], bar['w'], bar['h']], dtype=np.float)
            true_system = np.asarray([system['x'], system['y'], system['w'], system['h']], dtype=np.float)

            systems_xywh = np.asarray([[x['x'], x['y'], x['w'], x['h']] for x in page_systems[curr_page]])
            systems_xyxy = xywh2xyxy(systems_xywh)

            max_x_shift = (-(int(systems_xyxy[:, 0].min() - 50)),
                        int(padded_scores.shape[2] - systems_xyxy[:, 2].max() - 50))
            max_y_shift = (min(0, -int((systems_xyxy[:, 1].min() - 50))),
                        max(1, int(padded_scores.shape[1] - systems_xyxy[:, 3].max() - 50)))

            bar_mask = np.zeros_like(mask)
            bar_mask[int(bar['y']-bar['h']/2):int(bar['y']+bar['h']/2), int(bar['x']-pad-bar['w']/2):int(bar['x']-pad+bar['w']/2)] = 1
            # figure out at which frame we change pages
            if true_position[-1] == page and np.logical_and(mask, bar_mask).sum() > 0:
                
                if start_frame == -1:
                    start_frame = frame
                    
                # if frame % 40 == 0:
                #     print(mask_id, bar_idx, frame, np.logical_and(mask, bar_mask).sum())
                #     plt.subplot(131)
                #     plt.imshow(mask)
                #     plt.subplot(132)
                #     plt.imshow(bar_mask)
                #     plt.subplot(133)
                #     plt.imshow(scores[mask_id])
                #     plt.show()
                
                
                piece_sequences.append({'piece_id': piece_idx,
                                        'is_onset': frame in onsets,
                                        'start_frame': start_frame,
                                        'frame': frame,
                                        'max_x_shift': max_x_shift,
                                        'max_y_shift': max_y_shift,
                                        'true_position': true_position,
                                        'true_system': true_system,
                                        'true_bar': true_bar,
                                        'height': system['h'],
                                        'synthesized': synthesized,
                                        'scale_factor': scale_factor,
                                        'mask_id': mask_id
                                        })

    if not load_audio:
        signal = os.path.join(path, piece_name + '.wav')

    return piece_idx, scores, signal, piece_name, piece_sequences, interpol_c2o, staff_coords, add_per_staff

def load_bipage_sequences(params):
    # load full page
    piece_idx = params['i']
    path = params['path']
    piece_name = params['piece_name']
    scale_width = params['scale_width']
    load_audio = params.get('load_audio', True)

    padded_scores, _, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized = load_bipiece(path, piece_name)
    # print(piece_name, padded_scores.shape)
    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

    scale_factor = padded_scores.shape[1] / scale_width

    scaled_score = []
    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    scores = np.stack(scaled_score)

    duration = signal.shape[0]
    n_frames = int(np.ceil(FPS * duration / SAMPLE_RATE))
    piece_sequences = []

    interpol_c2o = {}
    staff_coords = {}
    add_per_staff = {}
    valid_pages = np.unique(coords_new[:, -1])

    for page_nr in valid_pages:
        page_coords = coords_new[coords_new[:, -1] == page_nr]
        page_onsets = onsets[coords_new[:, -1] == page_nr]
        unrolled_coords_x = []
        coords_per_staff = []

        # only add 0 for first staff
        max_xes = [0]
        staff_coords[page_nr] = sorted(np.unique(page_coords[:, 0]))

        for c in staff_coords[page_nr]:

            cs_staff = page_coords[page_coords[:, 0] == c, :-1]
            max_x = max(cs_staff[:, 1])
            coords_per_staff.append(cs_staff)
            max_xes.append(max_x)

        # last entry not needed
        add_per_staff[page_nr] = np.cumsum(max_xes)[:-1]
        for idx in range(len(staff_coords[page_nr])):
            unrolled_coords_x.append(coords_per_staff[idx][:, 1] + add_per_staff[page_nr][idx])

        unrolled_coords_x = np.concatenate(unrolled_coords_x)

        # if we are in between two coord position it will match it to the closest onset
        interpol_c2o[page_nr] = interpolate.interp1d(unrolled_coords_x, page_onsets, kind='nearest', bounds_error=False,
                                                     fill_value=(page_onsets[0], page_onsets[-1]))

    # page_systems = {}
    # page_bars = {}

    # for page_idx in valid_pages:
    #     page_systems[page_iinterpol_fncdx] = list(filter(lambda x: x['page_nr'] == page_idx, systems))
    #     page_bars[page_idx] = list(filter(lambda bar: bar['page_nr'] == page_idx, bars))
        
    bipage_id = 0
    n, h, w = padded_scores.shape
    # print(n, h, w)
    for bipage_n in range(n):
        
        start_frame = None
        curr_page = 0
        
        for frame in range(n_frames):

            true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)

            bar_idx = true_position[3]
            system_idx = true_position[2]
            
            
            # figure out at which frame we change pages
            if true_position[-1] != curr_page:
                curr_page = true_position[-1]
                
            bar = bars[bar_idx]
            system = systems[system_idx]
            if bipage_n == curr_page:
                if start_frame == None:
                    
                    start_frame = frame
                

                true_bar = np.asarray([bar['x'], bar['y'], bar['w'], bar['h']], dtype=np.float)
                true_system = np.asarray([system['x'], system['y'], system['w'], system['h']], dtype=np.float)

                
                max_x_shift = (-20, 20)
                max_y_shift = (-180, 180)
                piece_sequences.append({'piece_id': piece_idx,
                                        'is_onset': frame in onsets,
                                        'start_frame': start_frame,
                                        'frame': frame,
                                        'max_x_shift': max_x_shift,
                                        'max_y_shift': max_y_shift,
                                        'true_position': true_position,
                                        'true_system': true_system,
                                        'true_bar': true_bar,
                                        'height': system['h'],
                                        'synthesized': synthesized,
                                        'scale_factor': scale_factor,
                                        })
            elif bipage_n == curr_page-1 and start_frame != None:
                

                true_position[1] += w/2 
                true_position[-1] = bipage_n
                true_bar = np.asarray([bar['x'] + w/2, bar['y'], bar['w'], bar['h']], dtype=np.float)
                true_system = np.asarray([system['x'] + w/2, system['y'], system['w'], system['h']], dtype=np.float)

                max_x_shift = (-20, 20)
                max_y_shift = (-180, 180)
                # print(true_position)
                piece_sequences.append({'piece_id': piece_idx,
                                        'is_onset': frame in onsets,
                                        'start_frame': start_frame,
                                        'frame': frame,
                                        'max_x_shift': max_x_shift,
                                        'max_y_shift': max_y_shift,
                                        'true_position': true_position,
                                        'true_system': true_system,
                                        'true_bar': true_bar,
                                        'height': system['h'],
                                        'synthesized': synthesized,
                                        'scale_factor': scale_factor,
                                        
                                        })

    if not load_audio:
        signal = os.path.join(path, piece_name + '.wav')
    # print(scores.shape, len(piece_sequences))
    return piece_idx, scores, signal, piece_name, piece_sequences, interpol_c2o, staff_coords, add_per_staff

def load_masked_bipage_sequences(params):
    # load full page
    piece_idx = params['i']
    path = params['path']
    piece_name = params['piece_name']
    scale_width = params['scale_width']
    load_audio = params.get('load_audio', True)

    padded_scores, _, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized, score_masks = load_masked_bipiece(path, piece_name)
    # print(piece_name, padded_scores.shape)
    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

    scale_factor = padded_scores.shape[1] / scale_width

    scaled_score = []
    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    scores = np.stack(scaled_score)

    duration = signal.shape[0]
    n_frames = int(np.ceil(FPS * duration / SAMPLE_RATE))
    piece_sequences = []

    interpol_c2o = {}
    staff_coords = {}
    add_per_staff = {}
    valid_pages = np.unique(coords_new[:, -1])

    for mask_id in range(len(score_masks)):
        page_nr, _ = score_masks[mask_id]
        
        page_coords = coords_new[coords_new[:, -1] == page_nr]
        page_onsets = onsets[coords_new[:, -1] == page_nr]
        unrolled_coords_x = []
        coords_per_staff = []

        # only add 0 for first staff
        max_xes = [0]
        staff_coords[mask_id] = sorted(np.unique(page_coords[:, 0]))

        for c in staff_coords[mask_id]:

            cs_staff = page_coords[page_coords[:, 0] == c, :-1]
            max_x = max(cs_staff[:, 1])
            coords_per_staff.append(cs_staff)
            max_xes.append(max_x)

        # last entry not needed
        add_per_staff[mask_id] = np.cumsum(max_xes)[:-1]
        for idx in range(len(staff_coords[mask_id])):
            unrolled_coords_x.append(coords_per_staff[idx][:, 1] + add_per_staff[mask_id][idx])

        unrolled_coords_x = np.concatenate(unrolled_coords_x)

        # if we are in between two coord position it will match it to the closest onset
        interpol_c2o[mask_id] = interpolate.interp1d(unrolled_coords_x, page_onsets, kind='nearest', bounds_error=False,
                                                     fill_value=(page_onsets[0], page_onsets[-1]))

    page_systems = {}
    page_bars = {}

    for page_idx in valid_pages:
        page_systems[page_idx] = list(filter(lambda x: x['page_nr'] == page_idx, systems))
        page_bars[page_idx] = list(filter(lambda bar: bar['page_nr'] == page_idx, bars))
        

    n, h, w = padded_scores.shape
    # print(n, h, w)
    # print(piece_name, n, len(score_masks))
    
    # plt.figure()
    for bipage_n in range(n):
        
        start_frame = None
        # last_lframe = None
        mask_page, l_mask = score_masks[bipage_n]

        for frame in range(n_frames):

            true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)

            bar_idx = true_position[3]
            system_idx = true_position[2]
            
            
            bar = bars[bar_idx]
            system = systems[system_idx]       
            
            
            
                
            true_bar = np.asarray([bar['x'], bar['y'], bar['w'], bar['h']], dtype=np.float)
            true_system = np.asarray([system['x'], system['y'], system['w'], system['h']], dtype=np.float)
        
            bar_mask = np.zeros_like(l_mask)
            bar_mask[int(bar['y']-pad-bar['h']/2):int(bar['y']-pad+bar['h']/2), int(bar['x']-bar['w']/2):int(bar['x']+bar['w']/2)] = 1
            
            if np.logical_and(l_mask, bar_mask).sum() > 0 and true_position[-1] == mask_page:
                # print(frame, bar_mask.shape, mask_l.shape, bar['x'], bar['y']-pad, bar['w'], bar['h'], pad)
                # if frame % 40 == 0:
                #     print("left", frame, true_position, bipage_n, mask_page)
                #     plt.subplot(211)
                #     plt.imshow(bar_mask)
                #     plt.subplot(212)
                #     plt.imshow(l_mask)
                #     plt.show()
                if start_frame == None:
                
                    start_frame = frame
                # last_lframe = frame
                
                bar = bars[bar_idx]
                system = systems[system_idx]

                true_bar = np.asarray([bar['x'], bar['y'], bar['w'], bar['h']], dtype=np.float)
                true_system = np.asarray([system['x'], system['y'], system['w'], system['h']], dtype=np.float)
        
                true_position[-1] = mask_page
                # print("L", start_frame, frame)
                # os.system("pause")
                piece_sequences.append({'piece_id': piece_idx,
                                        'is_onset': frame in onsets,
                                        'start_frame': start_frame,
                                        'frame': frame,
                                        'max_x_shift': 0,
                                        'max_y_shift': 0,
                                        'true_position': true_position,
                                        'true_system': true_system,
                                        'true_bar': true_bar,
                                        'height': system['h'],
                                        'synthesized': synthesized,
                                        'scale_factor': scale_factor,
                                        'mask_score': bipage_n,
                                        'loc': "L"
                                        })
        
        # if bipage_n+1 == n:
        #     mask_page, r_mask = score_masks[bipage_n]
        #     mask_page += 1
        #     # print(r_mask, r_mask.shape)
        #     r_mask = np.ones_like(r_mask)
        # else:
        #     mask_page, r_mask = score_masks[bipage_n+1]
            
        # for frame in range(n_frames):

        #     true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)

        #     bar_idx = true_position[3]
        #     system_idx = true_position[2]
            
        #     bar = bars[bar_idx]
        #     system = systems[system_idx]       
            
                
        #     true_position[1] += w/2 
            
        #     if len(start_frame) == 1:
            
        #         start_frame.append(frame)
                
        #     true_bar = np.asarray([bar['x'] + w/2, bar['y'], bar['w'], bar['h']], dtype=np.float)
        #     true_system = np.asarray([system['x'] + w/2, system['y'], system['w'], system['h']], dtype=np.float)

        #     bar_mask = np.zeros_like(r_mask)
        #     bar_mask[int(bar['y']-pad-bar['h']/2):int(bar['y']-pad+bar['h']/2), int(bar['x']-bar['w']/2):int(bar['x']+bar['w']/2)] = 1
            
        #     if np.logical_and(r_mask, bar_mask).sum() > 0 and true_position[-1] == mask_page:
        #         # print(frame, bar_mask.shape, mask_l.shape, bar['x'], bar['y']-pad, bar['w'], bar['h'], pad)
        #         # if frame % 40 == 0:
        #         #     print("right", frame, true_position, bipage_n, mask_page)
        #         #     plt.subplot(211)
        #         #     plt.imshow(bar_mask)
        #         #     plt.subplot(212)
        #         #     plt.imshow(r_mask)
        #         #     plt.show()
                    
        #         bar = bars[bar_idx]
        #         system = systems[system_idx]

        #         true_bar = np.asarray([bar['x'], bar['y'], bar['w'], bar['h']], dtype=np.float)
        #         true_system = np.asarray([system['x'], system['y'], system['w'], system['h']], dtype=np.float)
        #         true_position[-1] = mask_page
        #         # print("R", start_frame, [last_lframe, frame])
        #         piece_sequences.append({'piece_id': piece_idx,
        #                                 'is_onset': frame in onsets,
        #                                 'start_frame': start_frame,
        #                                 'frame': [last_lframe, frame],
        #                                 'max_x_shift': 0,
        #                                 'max_y_shift': 0,
        #                                 'true_position': true_position,
        #                                 'true_system': true_system,
        #                                 'true_bar': true_bar,
        #                                 'height': system['h'],
        #                                 'synthesized': synthesized,
        #                                 'scale_factor': scale_factor,
        #                                 'mask_score': bipage_n,
        #                                 'loc': "R"
        #                                 })

    if not load_audio:
        signal = os.path.join(path, piece_name + '.wav')
    # print(scores.shape, len(piece_sequences))
    return piece_idx, scores, signal, piece_name, piece_sequences, interpol_c2o, staff_coords, add_per_staff

def load_blank_bipage_sequences(params):
    # load full page
    piece_idx = params['i']
    path = params['path']
    piece_name = params['piece_name']
    scale_width = params['scale_width']
    load_audio = params.get('load_audio', True)

    padded_scores, _, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized = load_blank_bipiece(path, piece_name)
    # print(piece_name, padded_scores.shape)
    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

    scale_factor = padded_scores.shape[1] / scale_width

    scaled_score = []
    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    scores = np.stack(scaled_score)

    duration = signal.shape[0]
    n_frames = int(np.ceil(FPS * duration / SAMPLE_RATE))
    piece_sequences = []

    interpol_c2o = {}
    staff_coords = {}
    add_per_staff = {}
    valid_pages = np.unique(coords_new[:, -1])

    for page_nr in valid_pages:
        page_coords = coords_new[coords_new[:, -1] == page_nr]
        page_onsets = onsets[coords_new[:, -1] == page_nr]
        unrolled_coords_x = []
        coords_per_staff = []

        # only add 0 for first staff
        max_xes = [0]
        staff_coords[page_nr] = sorted(np.unique(page_coords[:, 0]))

        for c in staff_coords[page_nr]:

            cs_staff = page_coords[page_coords[:, 0] == c, :-1]
            max_x = max(cs_staff[:, 1])
            coords_per_staff.append(cs_staff)
            max_xes.append(max_x)

        # last entry not needed
        add_per_staff[page_nr] = np.cumsum(max_xes)[:-1]
        for idx in range(len(staff_coords[page_nr])):
            unrolled_coords_x.append(coords_per_staff[idx][:, 1] + add_per_staff[page_nr][idx])

        unrolled_coords_x = np.concatenate(unrolled_coords_x)

        # if we are in between two coord position it will match it to the closest onset
        interpol_c2o[page_nr] = interpolate.interp1d(unrolled_coords_x, page_onsets, kind='nearest', bounds_error=False,
                                                     fill_value=(page_onsets[0], page_onsets[-1]))

    
    
    page_systems = {}
    page_bars = {}

    for page_idx in valid_pages:
        page_systems[page_idx] = list(filter(lambda x: x['page_nr'] == page_idx, systems))
        # page_bars[page_idx] = list(filter(lambda bar: bar['page_nr'] == page_idx, bars))
        
    bipage_id = 0
    n, h, w = padded_scores.shape
    # print(n, h, w)
    for bipage_n in range(n):
        
        start_frame = None
        curr_page = 0
        
        for frame in range(n_frames):

            true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)

            bar_idx = true_position[3]
            system_idx = true_position[2]
            
            
            # figure out at which frame we change pages
            if true_position[-1] != curr_page:
                curr_page = true_position[-1]
                
            
            if bipage_n == curr_page:
                if start_frame == None:
                    
                    start_frame = frame
                bar = bars[bar_idx]
                system = systems[system_idx]

                true_bar = np.asarray([bar['x'], bar['y'], bar['w'], bar['h']], dtype=np.float)
                true_system = np.asarray([system['x'], system['y'], system['w'], system['h']], dtype=np.float)

                
                max_x_shift = (-20, 20)
                max_y_shift = (-180, 180)
                piece_sequences.append({'piece_id': piece_idx,
                                        'is_onset': frame in onsets,
                                        'start_frame': start_frame,
                                        'frame': frame,
                                        'max_x_shift': max_x_shift,
                                        'max_y_shift': max_y_shift,
                                        'true_position': true_position,
                                        'true_system': true_system,
                                        'true_bar': true_bar,
                                        'height': system['h'],
                                        'synthesized': synthesized,
                                        'scale_factor': scale_factor,
                                        })
            
    if not load_audio:
        signal = os.path.join(path, piece_name + '.wav')
    # print(scores.shape, len(piece_sequences))
    return piece_idx, scores, signal, piece_name, piece_sequences, interpol_c2o, staff_coords, add_per_staff


def calculate_iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)
    x_inter2 = min(x1 + w1, x2 + w2)
    y_inter2 = min(y1 + h1, y2 + h2)

    # 计算交集区域的面积
    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)

    # 计算两个矩形的面积
    box1_area = w1 * h1
    box2_area = w2 * h2

    # 计算并集区域的面积
    union_area = box1_area + box2_area - inter_area

    # 计算 IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou

def find_system_ys(org_img, thicken_lines=False):
    img = np.asarray(cv2.cvtColor(org_img * 255, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
    img = 1 - cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)/255

    if thicken_lines:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.dilate(img, kernel, iterations=1)

    pxl = img.sum(-1)

    kernel_size = 10
    kernel = np.ones(kernel_size)

    # peaks = scipy.signal.find_peaks(pxl, height=np.max(pxl) / 2)[0]
    pxl_th = pxl.max() / 2
    peaks = np.argwhere(pxl > pxl_th).flatten()
    pxl[:] = 0
    pxl[peaks] = 1

    staff_indicator = np.convolve(pxl, kernel, mode="same")
    staff_indicator[staff_indicator < 1] = 0
    staff_indicator[staff_indicator >= 1] = 1

    diff_values = np.diff(staff_indicator)
    sys_start = np.argwhere(diff_values == 1).flatten()
    sys_end = np.argwhere(diff_values == -1).flatten()

    j = 0

    staffs = []
    for i in range(len(sys_start)):
        s = int(sys_start[i])

        while s >= sys_end[j] and j < len(sys_end):
            j += 1

        e = int(sys_end[j])
        local_peaks = np.argwhere(pxl[s:e + 1] == 1).flatten()
        n_peaks = len(local_peaks)

        # staff has to contain at least 3 peaks and needs to be at least 15 pixel high
        if n_peaks < 3 or local_peaks[-1] - local_peaks[0] < 15:

            staff_indicator[s:e + 1] = 0
            pxl[s:e + 1] = 0
        else:
            pxl[s:e + 1][:local_peaks[0]] = 0
            staff_indicator[s:e + 1][:local_peaks[0]] = 0

            pxl[s:e + 1][local_peaks[-1] + 1:] = 0
            staff_indicator[s:e + 1][local_peaks[-1] + 1:] = 0

            staffs.append((s + local_peaks[0], s + local_peaks[-1]))

    i = 0
    systems = []
    while i + 1 < len(staffs):
        s1 = staffs[i]
        s2 = staffs[i + 1]

        # system has to be at least 30 pixel high
        if s2[1] - s1[0] <= 30:
            i += 1
            continue

        systems.append((s1[0], s2[1]))

        i += 2

    return np.asarray(systems)

def load_piece_for_pageturning_metrics(path, piece_name, scale_width, mode):
    
    masks = []
    if mode == "fp":
        padded_scores, org_scores, onsets, _, bars, systems, interpol_fnc, signal, pad, _ = load_piece(path, piece_name)
    elif mode == "mfp":
        padded_scores, org_scores, onsets, _, bars, systems, interpol_fnc, signal, pad, _, masks = load_masked_piece(path, piece_name)
        # print(piece_name, org_scores.shape, len(masks))
    elif mode == "bp":
        padded_scores, org_scores, onsets, _, bars, systems, interpol_fnc, signal, pad, _ = load_bipiece(path, piece_name)
    elif mode == "mbp":
        padded_scores, org_scores, onsets, _, bars, systems, interpol_fnc, signal, pad, _, masks = load_masked_bipiece(path, piece_name)
    elif mode == "bp":
        padded_scores, org_scores, onsets, _, bars, systems, interpol_fnc, signal, pad, _ = load_bipiece(path, piece_name)
        
    if padded_scores.shape[0] <= 1:
        return []
    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.
    
    # scale scores
    scaled_score = []
    scale_factor = scores[0].shape[0] / scale_width

    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    score = np.stack(scaled_score)

    org_scores_rgb = []
    system_ys = []

    for org_score in org_scores:
        # print(org_score.shape)
        org_score = cv2.cvtColor(np.array(org_score, dtype=np.float32) / 255., cv2.COLOR_GRAY2BGR)
        # print(org_score.shape)
        if mode == "fp":
            system_ys.append(find_system_ys(org_score))
        elif mode == "mfp":
            system_ys.append(find_system_ys(org_score))
        elif mode == "bp":
            system_ys.append(find_system_ys(org_score[:, :org_score.shape[1]//2]))
        elif mode == "mbp":
            system_ys.append(find_system_ys(org_score[:, :org_score.shape[1]//2]))
            
        org_scores_rgb.append(org_score)
 
        
    return org_scores_rgb, system_ys, score, signal, systems, bars, interpol_fnc, pad, scale_factor, onsets, masks

def load_piece_for_testing(path, piece_name, scale_width):

    padded_scores, org_scores, onsets, _, bars, systems, interpol_fnc, signal, pad, _ = load_piece(path, piece_name)

    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

    # scale scores
    scaled_score = []
    scale_factor = scores[0].shape[0] / scale_width

    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    score = np.stack(scaled_score)

    org_scores_rgb = []
    system_ys = []
    for org_score in org_scores:
        # print(org_score.shape)
        org_score = cv2.cvtColor(np.array(org_score, dtype=np.float32) / 255., cv2.COLOR_GRAY2BGR)
        # print(org_score.shape)
        system_ys.append(find_system_ys(org_score))
        org_scores_rgb.append(org_score)
        

    return org_scores_rgb, system_ys, score, signal, systems, bars, interpol_fnc, pad, scale_factor, onsets

def load_maskscore_for_testing(path, piece_name, scale_width):

    padded_scores, org_scores, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized, score_masks = load_masked_piece(path, piece_name)
    
    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

    # scale scores
    scaled_score = []
    scale_factor = scores[0].shape[0] / scale_width

    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    score = np.stack(scaled_score)

    org_scores_rgb = []
    for org_score in org_scores:
        org_score = np.array(org_score, dtype=np.float32) / 255.

        org_scores_rgb.append(cv2.cvtColor(org_score, cv2.COLOR_GRAY2BGR))
        
    duration = signal.shape[0]
    n_frames = int(np.ceil(FPS * duration / SAMPLE_RATE))
    maskrange = []
    
    for page, mask in score_masks:
        tmp = []
        for frame in range(n_frames):
            
            true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)
            bar_idx = true_position[3]
            bar = bars[bar_idx]
            
            bar_mask = np.zeros_like(mask)
            bar_mask[int(bar['y']-bar['h']/2):int(bar['y']+bar['h']/2), int(bar['x']-pad-bar['w']/2):int(bar['x']-pad+bar['w']/2)] = 1
            # figure out at which frame we change pages
            if true_position[-1] == page and np.logical_and(mask, bar_mask).sum() > 0:
                # print(frame, , np.logical_and(mask, bar_mask).sum()/np.array(bar_mask).sum())
                tmp.append(frame)
        maskrange.append([min(tmp), max(tmp)])

    return org_scores_rgb, score, signal, systems, bars, interpol_fnc, pad, scale_factor, onsets, score_masks, maskrange

def load_bipiece_for_testing(path, piece_name, scale_width):
   
    padded_scores, org_scores, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized = load_blank_bipiece(path, piece_name)
    
    
    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.
    # scale scores
    scaled_score = []
    scale_factor = scores[0].shape[0] / scale_width

    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    score = np.stack(scaled_score)

    org_scores_rgb = []
    for org_score in org_scores:
        org_score = np.array(org_score, dtype=np.float32) / 255.

        org_scores_rgb.append(cv2.cvtColor(org_score, cv2.COLOR_GRAY2BGR))

    return org_scores_rgb, score, signal, systems, bars, interpol_fnc, pad, scale_factor, onsets


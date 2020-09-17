import os
import math
import torch
import librosa
import numpy as np
import scipy.signal

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from torch_complex.tensor import ComplexTensor
from functools import lru_cache

import torchaudio


def load_audio(path):
    
    if isinstance(path, str):
        sound = np.memmap(path, dtype='h', mode='r', offset=44)
        # print("np.memmap    ", sound.shape, len(sound), sound)
    else:
        sound = np.frombuffer(path.getvalue(), dtype=np.int16, offset=44) # offset=44
        # print("np.frombuffer", sound.shape, len(sound), sound)
    
    sound = sound.astype('float32') / 32767

    assert len(sound)

    sound = torch.from_numpy(sound).view(-1, 1).type(torch.FloatTensor)
    sound = sound.numpy()

    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average

    return sound


class SpectrogramDataset(Dataset):
    def __init__(self, audio_conf, dataset_path, data_list, char2index, sos_id, eos_id, normalize=False):
        super(SpectrogramDataset, self).__init__()
        """
        Dataset loads data from a list contatining wav_name, transcripts, speaker_id by dictionary.
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds.
        :param data_list: List of dictionary. key : "wav", "text", "speaker_id"
        :param char2index: Dictionary mapping character to index value.
        :param sos_id: Start token index.
        :param eos_id: End token index.
        :param normalize: Normalized by instance-wise standardazation.
        """
        self.audio_conf = audio_conf
        self.data_list = data_list
        self.size = len(self.data_list)
        self.char2index = char2index
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.PAD = 0
        self.normalize = normalize
        self.dataset_path = dataset_path
        
        # self.normalize_type = 'instance'
        self.normalize_type = 'utterance_mvn' # or 'utterance_mvn' or 'global_mvn'
        print("normalize_type : ", self.normalize_type)

        self.stft_conf = dict(
            n_fft = 512,
            win_length = 512,
            hop_length = 128,
            center = True,
            window = torch.hann_window(window_length=512,
                                       dtype=torch.float32,
                                       device=torch.device('cpu')),
            normalized = False,
            onesided = True
        )

#         self.stft_conf = dict(
#             n_fft = 320,
#             win_length = 320,
#             hop_length = 160,
#             center = True,
#             window = torch.hamming_window(window_length=320,
#                                           dtype=torch.float32,
#                                           device=torch.device('cpu')),
#             normalized = False,
#             onesided = True
#         )
        
        mel_conf = dict(
            sr = 16000,
            n_fft = 512,
            n_mels = 80,
            fmin = 0,
            fmax = 16000 / 2,
            htk = False
        )
        
        melmat = librosa.filters.mel(**mel_conf)
        self.melmat = torch.from_numpy(melmat.T).float()

    @lru_cache(maxsize=100000)
    def __getitem__(self, index):
        wav_name = self.data_list[index]['wav']
        audio_path = os.path.join(self.dataset_path, wav_name)
        
        transcript = self.data_list[index]['text']
        # spect = self.parse_audio(index, audio_path)
        
        spect = self.parse_audio_melfbank_from_espnet(index, audio_path)
        # spect = self.parse_audio_melfbank_from_kaldi(index, audio_path)
        transcript = self.parse_transcript(transcript)
        return spect, transcript

    def parse_audio(self, index, audio_path):
        y = load_audio(audio_path)

        n_fft = int(self.audio_conf['sample_rate'] * self.audio_conf['window_size'])
        window_size = n_fft
        stride_size = int(self.audio_conf['sample_rate'] * self.audio_conf['window_stride'])

        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=stride_size, win_length=window_size, window=scipy.signal.hamming)
        spect, phase = librosa.magphase(D)

        # spect.shape = (161, Seq_Len)
        # S = log(S+1) 
        spect = np.log1p(spect)
        if self.normalize:
            if self.normalize_type == 'instance':
                mean = np.mean(spect)
                std = np.std(spect)
                spect -= mean
                spect /= std
            elif self.normalize_type == 'utterance_mvn':
                mean = np.mean(spect, axis=-1, keepdims=True)
                std = np.std(spect, axis=-1, keepdims=True)
                std = np.clip(std, a_min=1e-20, a_max=None)
                spect -= mean
                spect /= std
            else:
                raise NotImplementedError
        spect = torch.FloatTensor(spect)

        return spect
    
    def parse_audio_melfbank_from_espnet(self, index, audio_path):
        y = load_audio(audio_path)
        
        # 1. PCM --> STFT
        y = torch.from_numpy(y)
        D = torch.stft(y, **self.stft_conf) # D.shape = (Freq, Frames, 2)
        D = D.transpose(0, 1) # D.shape = (Frames, Freq, 2)
        
        # 2. STFT --> Power Spectrum
        input_stft = D
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        input_power = (input_stft.real ** 2) + (input_stft.imag ** 2)
        
        # 3. Power Spectrum --> Log Mel-Fbank
        # feat: (T, D1) x melmat: (D1, D2) -> mel_feat: (T, D2)
        mel_feat = torch.matmul(input_power, self.melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)
        logmel_feat = mel_feat.log()
        
        feat = logmel_feat
        
        # 4. Utt-MVN (Utterance Mean Variance Normalization)
        if self.normalize:
            if self.normalize_type == 'instance':
                mean = torch.mean(feat)
                std = torch.std(feat)
                feat -= mean
                feat /= std
            elif self.normalize_type == 'utterance_mvn':
                mean = torch.mean(feat, dim=-1, keepdim=True)
                std = torch.std(feat, dim=-1, keepdim=True)
                std = torch.clamp(std, min=1e-20)

                feat -= mean
                feat /= std
        
        feat = feat.transpose(0, 1).contiguous()
        
        return feat

    def parse_audio_melfbank_from_kaldi(self, index, audio_path):
        y = load_audio(audio_path)
        y = torch.FloatTensor(y).unsqueeze(0)
        
        # |feat| = (seq_len, feat_dim)
        feat = torchaudio.compliance.kaldi.fbank(y,
                                                 dither= 0.0,
                                                 frame_length=25.0,
                                                 frame_shift=10.0,
                                                 sample_frequency=16000.0,
                                                 num_mel_bins=80)
        
        # 4. Utt-MVN (Utterance Mean Variance Normalization)
        if self.normalize:
            if self.normalize_type == 'instance':
                mean = torch.mean(feat)
                std = torch.std(feat)
                feat -= mean
                feat /= std
            elif self.normalize_type == 'utterance_mvn':
                mean = torch.mean(feat, dim=-1, keepdim=True)
                # std = torch.std(feat, dim=-1, keepdim=True)
                # std = torch.clamp(std, min=1e-20)

                feat -= mean
                # feat /= std

        feat = feat.transpose(0, 1).contiguous()
        
        return feat
        

    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.char2index.get(x) for x in list(transcript)]))
        transcript = [self.sos_id] + transcript + [self.eos_id]
        return transcript

    def __len__(self):
        return self.size


def _collate_fn(batch):
    def seq_length_(p):
        return p[0].size(1)
    def target_length_(p):
        return len(p[1])

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    seq_lengths    = [s[0].size(1) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_size = max(seq_lengths)
    max_target_size = max(target_lengths)

    feat_size = batch[0][0].size(0)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, 1, feat_size, max_seq_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        seqs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    return seqs, targets, seq_lengths, target_lengths


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)

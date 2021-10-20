import torch

import cargan


###############################################################################
# cargan dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):

    def __init__(self, name):
        self.name = name
        self.cache = cargan.CACHE_DIR / name
        if name == 'cumsum':
            self.files = list(self.cache.glob('input-*'))
        else:
            self.files = list(self.cache.rglob('*.wav'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        with cargan.data.chdir(self.cache):
            
            # Load item of cumsum dataset
            if self.name == 'cumsum':
                cumsum_input = torch.load(file)
                cumsum_output = torch.load(
                    file.parent / f'output-{file.stem[-6:]}.pt')
                return cumsum_input, cumsum_output, None, None
        
            # Load audio
            audio = cargan.load.audio(file)

            # Load features
            features = torch.load(file.parent / f'{file.stem}-mels.pt')
            pitch = torch.load(file.parent / f'{file.stem}-pitch.pt')
            periodicity = torch.load(
                file.parent / f'{file.stem}-periodicity.pt')

            # Maybe add features
            if cargan.PITCH_FEATURE:
                features = torch.cat([features, pitch], dim=1)
            if cargan.PERIODICITY_FEATURE:
                features = torch.cat([features, periodicity], dim=1)

            return features, audio, pitch, periodicity

    def speaker(self, index):
        """Retrieve the speaker corresponding to the given index"""
        if self.name != 'vctk':
            raise ValueError('Only vctk dataset has speaker information')
        return self.files[index].parent.name

    def speakers(self):
        """Retrieve a list of all speakers"""
        if self.name != 'vctk':
            raise ValueError('Only vctk dataset has speaker information')
        return [speaker_dir.name for speaker_dir in self.cache.glob('*')]

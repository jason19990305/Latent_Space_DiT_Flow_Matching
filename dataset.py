import os
import urllib.request
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 10 High-Quality Centered Frontal Face Portrait URLs (CelebA / FFHQ standard alignment)
SAMPLE_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=500&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=500&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=500&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=500&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=500&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=500&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=500&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=500&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?w=500&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=500&auto=format&fit=crop&q=80",
]


class CelebADataset(Dataset):
    def __init__(self, root_dir=None, img_size=256, num_samples=10):
        if root_dir is None:
            root_dir = os.path.join(os.path.dirname(__file__), "data", f"celeba_{num_samples}")
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.image_paths = []

        os.makedirs(self.root_dir, exist_ok=True)
        self._prepare_data()

        # Preserve true aspect ratio using Resize(256) + CenterCrop(256)
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _prepare_data(self):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        for i in range(min(self.num_samples, len(SAMPLE_IMAGE_URLS))):
            save_path = os.path.join(self.root_dir, f"sample_{i+1}.jpg")
            if not os.path.exists(save_path):
                try:
                    req = urllib.request.Request(SAMPLE_IMAGE_URLS[i], headers=headers)
                    with urllib.request.urlopen(req) as resp, open(save_path, 'wb') as f:
                        f.write(resp.read())
                except Exception as e:
                    print(f"  Download failed for sample {i+1}: {e}. Creating placeholder.")
                    Image.new('RGB', (256, 256), color=((i*25)%255, (i*45)%255, (i*65)%255)).save(save_path)
            self.image_paths.append(save_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.image_paths[idx]).convert("RGB"))


def get_celeba_dataloader(num_samples=10, img_size=256, batch_size=None, root_dir=None):
    dataset = CelebADataset(root_dir=root_dir, img_size=img_size, num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size or num_samples, shuffle=True)

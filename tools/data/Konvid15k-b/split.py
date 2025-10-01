import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil

# Paths
videos_dir = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\k150kb'
train_dir = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\videos_train'
val_dir   = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\videos_val'
ann_dir   = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\annotations'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(ann_dir, exist_ok=True)

# Load CSV
scores = pd.read_csv(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\k150kb_scores.csv')  # video_name, mos, video_score
scores['label'] = scores['mos']  # Use MOS as regression label

# Split 80/20
train_df, val_df = train_test_split(scores, test_size=0.2, random_state=42)

# Move videos to respective folders
for _, row in train_df.iterrows():
    src = os.path.join(videos_dir, row['video_name'])
    dst = os.path.join(train_dir, row['video_name'])
    shutil.copy(src, dst)  # or os.rename(src,dst) to move

for _, row in val_df.iterrows():
    src = os.path.join(videos_dir, row['video_name'])
    dst = os.path.join(val_dir, row['video_name'])
    shutil.copy(src, dst)

# Train annotations
train_txt = os.path.join(ann_dir, 'train.txt')
train_df.to_csv(train_txt, sep=' ', index=False, header=False, columns=['video_name','label'])

# Val annotations
val_txt = os.path.join(ann_dir, 'val.txt')
val_df.to_csv(val_txt, sep=' ', index=False, header=False, columns=['video_name','label'])

# Test annotations (for now, use val as test)
test_txt = os.path.join(ann_dir, 'test.txt')
val_df.to_csv(test_txt, sep=' ', index=False, header=False, columns=['video_name','label'])

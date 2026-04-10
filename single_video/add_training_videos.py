

video = 'video.mp4'

with open('data/training_videos.txt', 'r') as reader:
    training_videos = reader.read().split()

if video not in training_videos:
    with open('data/training_videos.txt', 'a') as writer:
        writer.write(video + '\n')


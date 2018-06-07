# instrument-prediction
There are two instrument prediction models inside this repo. One is trained on MusicNet dataset and one is trained on Musescore dataset.
### MusicNet model
Contains 7 kinds of instrument - Piano, Violin, Viola, Cello, Clarinet, Horn and Bassoon. Since it's trained on classical music, it will be more efficient when testing on classical music.
### Musescore model
Contains 9 kinds of instrument - Piano, Acoustic Guitar, Electrical Guitar, Drum, Trumpet, Saxphone, Bass, String, Flute. Musecore model is trained on synthetic music, but it can work on real-life music too. Domain adaptation is under process to make it perfect.
### File structure
- data: store the model parameters
- function: store the model structure
- mp3: folder to put the testing mp3 files
- plot: folder to store the result graphic
- result: folder to store the result raw data

### Requirement
- librosa==0.6.0
- matplotlib==2.2.0
- numpy==1.14.2
- pytorch==0.3.1

## Model Testing Process
1. Put MP3/WAV files in the "mp3" folder
2. Run the prediction python file with the name of the song as the first arg
```
python prediction.py test_song.mp3
```
3. Prediction result will be shown as a picture and stored in the "plot" folder. Prediction raw data will be stored in the "result" folder



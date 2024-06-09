import os
# import scipy.io.matlab
import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import soundfile as sf
import sys
import pickle
import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense,Dropout,Bidirectional,LSTM
import torch
import wave
import gradio as gr
from pydub import AudioSegment

torch.set_num_threads(1)

FRAME_TIME=80*10**(-3) # to try 60ms, 80ms,100ms, 110, 120,
HOP_TIME=10*10**(-3)
S_FRAME_TIME=10*10**(-3) # changed from 20ms to 32 ms to adjust FFT length
S_HOP_TIME=4.1*10**(-3) # 4.5for 80ms,4 for 100ms, 4 for 110ms,4.1 for 120
THRESHOLD_TIME=FRAME_TIME
SAMPLING_RATE=16000
N_MFCC=20
BREATH_THRESHOLD=100*10**(-3)
BREATH_TO_BREATH_TIME=150*10**(-3)
VAD_THRESHOLD=0.1
join=0
remove=1
classifier_threshold=0.5
# Specify the path to your pickle file
pickle_file_path = 'Normalisation_parameters_2018_full_data.pickle'
ModelWeightFilepath='Breath_detection_3BILSTM_2018_full_data_80ms_10ms_10ms_best_weights.hdf5'
# global model, utils, original_task_model, get_speech_timestamps, read_audio, Feature_mean, Feature_std
# ***********************************************
# Initialisation
# ***********************************************
print("Reading normalisation parameters")
try:
    # Open the file in binary read mode
    with open(pickle_file_path, 'rb') as file:
        # Load the object from the file
        Feature_mean,Feature_std = pickle.load(file)
    print("Object loaded successfully!")
    print(Feature_mean.shape,Feature_std.shape)
except Exception as e:
    print(f"An error occurred: {e}")

print("Initialising the Breath Detection model")
lstm_1= 24
l2_1= 0.02
drop_1= 0.25
lstm_2= 8
l2_2= 0.04
drop_2= 0.3
lstm_3= 24
l2_3= 0.03
drop_3= 0.45
lr= 0.0001

input = Input(shape=Feature_mean.shape)
# print(input.shape)
lay1=Bidirectional(LSTM(lstm_1,activation='tanh',kernel_regularizer=tf.keras.regularizers.l2(l2_1),
                        return_sequences=True))(input)
lay1=Dropout(drop_1)(lay1)
# print(lay1.shape)

lay2=Bidirectional(LSTM(lstm_2,activation='tanh',kernel_regularizer=tf.keras.regularizers.l2(l2_2),
                        return_sequences=True))(lay1)
lay2=Dropout(drop_2)(lay2)
# print(lay2.shape)

lay3=Bidirectional(LSTM(lstm_3,activation='tanh',kernel_regularizer=tf.keras.regularizers.l2(l2_3),
                        return_sequences=False))(lay2)
lay3=Dropout(drop_3)(lay3)
# print(lay3.shape)

output=Dense(1,activation='sigmoid')(lay3)
# print(output.shape)
original_task_model=Model(inputs=input,outputs=output,name='BILSTM_model')
# original_task_model.summary()
original_task_model.load_weights(ModelWeightFilepath)

print("Initialising Voice Activity Detection Model")
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                            model='silero_vad',
                            force_reload=True)

(get_speech_timestamps,_, read_audio,*_) = utils 

def speech_feature_melspect(speech_seg,Fs,frame_length,hop_length,s_frame_length,s_hop_length):
    Feat=[]
    Feature_min=[]
    Feature_max=[]
    index_start=0;
    index_end=frame_length;
    fft_length=int(2**np.ceil(np.log(int(s_frame_length))/np.log(2)))
    speech_seg = lb.effects.preemphasis(speech_seg)
    while index_end<len(speech_seg):
        s_frame=speech_seg[range(index_start,index_end)]
        cepst=lb.feature.melspectrogram(y=s_frame.reshape((-1,)),sr=Fs,n_fft=fft_length,win_length=s_frame_length,
                                        hop_length=s_hop_length,window='hann',n_mels=60,power=1)
        cepst=lb.power_to_db(cepst, ref=np.max)
        Feat.append(cepst)
        index_start += hop_length;
        index_end += hop_length;    
    Feat=np.array(Feat)
    return Feat

def read_speech_derive_vad (speech_file_path,sampling_rate,original_task_model,Feature_mean,Feature_std):  
     
    # sampling_rate = SAMPLING_RATE # also accepts 8000
    wav = read_audio(speech_file_path, sampling_rate=SAMPLING_RATE)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    index_vad = []
    for item in speech_timestamps:
        index_vad.extend([item['start'],item['end']])
    if index_vad[0] != 0:
        index_vad = [0] + index_vad
    if index_vad[-1] != len(wav):
        index_vad.append(len(wav))
    else:
        index_vad = index_vad[:-1]       
    index_vad = np.array(index_vad)
    speech,Fs=lb.load(speech_file_path,sr=sampling_rate)
    speech_scaled=speech/max(abs(speech))
    return speech,speech_scaled,index_vad

def remove_small_breaths(index_b,threshold_breath,speech_b_detect):
    for i in range(int(np.size(index_b)/2)):
        b_length=index_b[0,2*i+1]-index_b[0,2*i]
        if b_length <= threshold_breath:
            speech_b_detect[range(int(index_b[0,2*i]),int(index_b[0,2*i+1])+1)]=0
     
    index_b=np.argwhere(abs(np.diff(speech_b_detect))==1)
    if speech_b_detect[0]==1:
        index_b=np.insert(index_b,0,0)
    
    if speech_b_detect[-1]==1:
        index_b=np.append(index_b,len(speech_b_detect))
    index_b=np.reshape(index_b,(1,-1))
    
    return index_b,speech_b_detect
                   
def join_close_breaths(index_b,threshold_breath_to_breath,speech_b_detect):
    for i in range(int(np.size(index_b)/2)-1):
        bb_length=index_b[0,2*i+2]-index_b[0,2*i+1]
        if bb_length <= threshold_breath_to_breath:
            speech_b_detect[range(int(index_b[0,2*i+1]),int(index_b[0,2*i+2])+1)]=1
     
    index_b=np.argwhere(abs(np.diff(speech_b_detect))==1)
    if speech_b_detect[0]==1:
        index_b=np.insert(index_b,0,0)
    
    if speech_b_detect[-1]==1:
        index_b=np.append(index_b,len(speech_b_detect))
    index_b=np.reshape(index_b,(1,-1))
    frame_length=int(np.floor(FRAME_TIME*SAMPLING_RATE))
    hop_length=int(np.floor(HOP_TIME*SAMPLING_RATE))
    offset = frame_length - hop_length
    for i in range(int(np.size(index_b)/2)):
        index_b[2*i+1] = index_b[2*i+1] + offset
        speech_b_detect[range(int(index_b[0,2*i]),int(index_b[0,2*i+1])+1)]=1
    
    return index_b,speech_b_detect

# ***********************************************
def detect_breath_from_speed_vad(speech,index_vad):
    index_vad=np.reshape(index_vad,(1,-1))
    
    frame_length=int(np.floor(FRAME_TIME*SAMPLING_RATE))
    hop_length=int(np.floor(HOP_TIME*SAMPLING_RATE))
    s_frame_length=int(np.floor(S_FRAME_TIME*SAMPLING_RATE))
    s_hop_length=int(np.floor(S_HOP_TIME*SAMPLING_RATE))

    speech_b_detect=np.zeros(np.size(speech))
    
    for vi in range(int(np.size(index_vad)/2)):        
        index_start=index_vad[0,2*vi]
        index_end=index_vad[0,2*vi+1]
        speech_seg=speech[index_start:index_end]
        if (len(speech_seg)> frame_length+1):
            feature=speech_feature_melspect(speech_seg, SAMPLING_RATE,
                                            frame_length, hop_length,
                                            s_frame_length, s_hop_length)
            feature=(feature-Feature_mean)/Feature_std
            prediction=original_task_model.predict(feature)
            y_pred=np.array(list(map(int,prediction>classifier_threshold)))
            if sum(y_pred)>2:
                detect_point=np.argwhere(y_pred==1)
                speech_b_detect[int(index_start+detect_point[0]*hop_length):int(index_start+(detect_point[-1]+1)*hop_length)]=1

    index_b=np.argwhere(abs(np.diff(speech_b_detect))==1)
    if speech_b_detect[0]==1:
        index_b=np.insert(index_b,0,0)
        
    if speech_b_detect[-1]==1:
        index_b=np.append(index_b,len(speech_b_detect))
    index_b=np.reshape(index_b,(1,-1))
    index_b1=index_b.copy()
    threshold_breath=BREATH_THRESHOLD*SAMPLING_RATE
    threshold_breath_to_breath=BREATH_TO_BREATH_TIME*SAMPLING_RATE

    frame_length=int(np.floor(FRAME_TIME*SAMPLING_RATE))
    hop_length=int(np.floor(HOP_TIME*SAMPLING_RATE))
    offset = frame_length - hop_length
    print(f"Number of breaths detected: {np.size(index_b)/2}")
    for i in range(int(np.size(index_b)/2)):
        index_b[0,2*i+1] = index_b[0,2*i+1] + offset
        if (index_b[0,2*i+1] > len(speech)):
            index_b[0,2*i+1]=len(speech)
        speech_b_detect[range(int(index_b[0,2*i]),int(index_b[0,2*i+1])+1)]=1
    
    # if join==1:
    #     index_b,speech_b_detect=join_close_breaths(index_b,threshold_breath_to_breath,speech_b_detect)
    # if remove==1:
    #     index_b,speech_b_detect=remove_small_breaths(index_b,threshold_breath,speech_b_detect)
    
   
    return speech_b_detect

def detect_breath_from_speed(speech_file_path,original_task_model,Feature_mean,Feature_std):
    print("Finding Voice Activity Deteciton")
    speech,speech_scaled,index_vad=read_speech_derive_vad(speech_file_path,SAMPLING_RATE,original_task_model,Feature_mean,Feature_std)
    print(f"Number of Non-Voice regions: {len(index_vad)/2}")
    print("Detecting Breath sound in speech")
    speech_b_detect=detect_breath_from_speed_vad(speech,index_vad)
    return speech,speech_b_detect

def plot_waveform(speech,SAMPLING_RATE,speech_b_detect):
    # Read the audio file
    # Create the X values based on the length of the speech data and the sampling rate
    X = np.divide(range(0, len(speech)), SAMPLING_RATE)

    # Create a figure
    plt.figure(figsize=(8, 2))

    # Define font size
    font_size = 24

    # Second subplot: Speech, Detected breath, and True breath
    # plt.subplot(3, 1, 2)
    plt.plot(X, 0.5*speech, label="Speech", color='blue', linewidth=2)
    plt.plot(X, 0.2 * speech_b_detect, label="Detected breath", color='red', linewidth=3)
    plt.title(f"Speech and detected breaths", fontsize=24)
    plt.legend(fontsize=12)
    plt.xlabel("Time (seconds)", fontsize=20)
    plt.ylabel("Amplitude", fontsize=20)
    plt.grid(True)
    
    # Save to a file
    output_image_file = "waveform.png"
    plt.savefig(output_image_file)
    plt.close()
        
    return output_image_file
# if __name__ == "__main__":
#     speech_file_path = 'DATA\Introductory\C1W1L01.wav'
    # original_task_model,Feature_mean,Feature_std = initialisation()
    

# def gradio_interface(image_file,input_audio_file):
def gradio_interface(text,input_audio_file):
    print("Gradio Interface audio file:",input_audio_file)
    # Load the audio file
    audio = AudioSegment.from_file(input_audio_file)    
    # Process the audio (e.g., normalize)
    processed_audio = audio.normalize()    
    # Export the processed audio to a file
    speech_file_path = "input_audio.wav"
    processed_audio.export(speech_file_path, format="wav")
    speech,speech_b_detect = detect_breath_from_speed(speech_file_path,original_task_model,Feature_mean,Feature_std)
    breath_output = 10*np.multiply(speech,speech_b_detect)
    breath_enhanced_speech = speech + breath_output
    print("Writing output file")
    output_audio_file = "Breath_v1.wav"
    print(f"Output file path : {output_audio_file}")
    sf.write(output_audio_file, breath_enhanced_speech, samplerate= SAMPLING_RATE,format='WAV')
    output_image_file = plot_waveform(speech,SAMPLING_RATE,speech_b_detect)
    return output_image_file,output_audio_file

# Example speech links for download
example_links = """
<a href="https://github.com/kmiyasararafath/BreathSoundDetection/blob/main/Example%20Speech/C1W1L01_half.wav" download>Download Example Speech 1</a><br>
<a href="https://github.com/kmiyasararafath/BreathSoundDetection/blob/main/Example%20Speech/aljazeera_only_report.wav" download>Download Example Speech 2</a><br>
<a href="https://github.com/kmiyasararafath/BreathSoundDetection/blob/main/Example%20Speech/nz_pm_speech.wav" download>Download Example Speech 3</a>
"""
# Create the Gradio interface
# default_image = "Text.png"
examples=gr.Markdown(f"Upload a speech audio file to process. You can also download example speeches below.\n{example_links}")
audio_input = gr.Audio(sources="upload", type="filepath", label="Upload your speech file")
iface = gr.Interface(
    fn=gradio_interface,
    # inputs=[gr.Image(type="filepath", value=default_image,interactive=False),gr.Audio(sources=["microphone","upload"], type="filepath",format='wav')],
    inputs=[examples,audio_input],
    outputs=[gr.Image(type="filepath"),gr.Audio(type="filepath")],
    title="Breath sound Detector",
    description="Record your speech reading the given paragraph. The audio will be processed and the breath detection will be performed. The detected breath will be displayed in the image and the breath enhanced speech can be heard.",
)

# Launch the Gradio interface
iface.launch()
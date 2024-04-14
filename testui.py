import customtkinter
import cv2
import mediapipe as mp
import numpy as np
import time
import itertools
import sys
from pynput import keyboard 
import os  # Import the os module to handle path operations
import sounddevice as sd
import speech_recognition as sr
import difflib

# def voice_code(file_name):

#     def recognize_speech_from_mic(recognizer, microphone):
#         """Transcribe speech from recorded from `microphone`."""
#         with microphone as source:
#             recognizer.adjust_for_ambient_noise(source)
#             audio = recognizer.listen(source)

#         try:
#             # Recognize speech using Google Web Speech API
#             transcription = recognizer.recognize_google(audio)
#         except sr.RequestError:
#             # API was unreachable or unresponsive
#             return "API unavailable"
#         except sr.UnknownValueError:
#             # speech was unintelligible
#             return "Unable to recognize speech"

#         return transcription

#     def compare_texts(recognized_text, file_text):
#         """Compare two texts and return similarity ratio."""
#         return difflib.SequenceMatcher(None, recognized_text, file_text).ratio()

#     # Main execution

#     # Set up the recognizer and microphone objects
#     script_directory = os.path.dirname(os.path.abspath(__file__))

#     # List all files in this directory
#     files = os.listdir(script_directory)
#     success=0
#     file_name=file_name+".txt"
#     # Print the files
#     for file in files:
#         if file==file_name:
#             success=1
#     file_name=script_directory+"/"+file_name

#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     # Recognize speech
#     recognized_text = recognize_speech_from_mic(recognizer, microphone)
#     print(f"Recognized Speech: {recognized_text}")

#     # Read the text file content
#     with open(file_name, "r") as file:
#         file_text = file.read().strip()

#     # Compare the recognized text with the file text
#     similarity = compare_texts(recognized_text, file_text)
#     print(f"Text similarity: {similarity:.2%}")


#     if similarity>0.7:
#         show_success_window()
#     else:
#         show_failure_window()



def show_success_window():
    # Create a new top-level window
    success_window = customtkinter.CTkToplevel(root)
    success_window.geometry("200x100")
    success_window.title("Success")

    # Configure the window color
    success_window.configure(fg_color="#0f3")  # A bright green color

    # Add a label to indicate success
    success_label = customtkinter.CTkLabel(success_window, text="Login Successful!", text_color="#fff", fg_color="#0f3")
    success_label.pack(expand=True, fill="both")

def show_failure_window():
    # Create a new top-level window for failure notification
    failure_window = customtkinter.CTkToplevel(root)
    failure_window.geometry("200x100")
    failure_window.title("Login Failed")

    # Configure the window color to a red shade
    failure_window.configure(fg_color="#f33")  # A shade of red for failure

    # Add a label to indicate failure
    failure_label = customtkinter.CTkLabel(failure_window, text="Login Failed!", text_color="#fff", fg_color="#f33")
    failure_label.pack(expand=True, fill="both")

def pattern_code(file_name):
    
    
    row_1 = 'qwertyuiop'
    row_2 = 'asdfghjkl'
    row_3 = 'zxcvbnm'


    def frequency_from_position(char):
        if char in row_1:
            return 250 + 20 * row_1.index(char)
        elif char in row_2:
            return 300 + 20 * row_2.index(char)  # Slight offset for the second row
        elif char in row_3:
            return 350 + 20 * row_3.index(char) # Further offset for the third row
        return 440 


    def play_tone(frequency=440, duration=0.4, samplerate=44100, amplitude=0.5):
        t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        sd.play(wave, samplerate=samplerate, blocking=False, latency='high')
        sd.wait()  # Wait until the sound is finished


    def read_keystrokes(file_name):
        # Read keystrokes from the given file
        # Get the directory of the current script
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # List all files in this directory
        files = os.listdir(script_directory)
        success=0
        file_name=file_name+".txt"
        # Print the files
        for file in files:
            if file==file_name:
                success=1
        file_name=script_directory+"/"+file_name
        keys = []
        if success:
            print("Found the username")
            with open(file_name, 'r') as file:
                for line in file:
                    key_char, time_stamp = line.strip().split()
                    keys.append((key_char, float(time_stamp)))
        else:
            print("Username not found")
            sys.exit(0)
        return keys

    def compare_keystrokes(original, new):
        correct=1
        i=0
        if len(new) != len(original):
            correct=0
            print("Incorrect Password, not matching in length")
            return

        for ((new_key, new_ts), (saved_key, saved_ts)) in zip(new, original):
            if new_key == saved_key and abs(new_ts - saved_ts) < 0.4:  # Allowing some time variance
                pass
            else:
                correct=0
                i+=1
                print("Incorrect Password from {} element".format(i))
                return correct
        
        return correct

    def capture_keystrokes():
        # Function to capture keystrokes
        keys = []
        recording = False
        start_time = None
        print("To start recording, press Enter...")

        def on_press(key):
            nonlocal recording, start_time

            # Play the keypress sound
            
            
            if key == keyboard.Key.enter:
                if recording:
                    # Stop recording on the second Enter
                    return False
                else:
                    # Start recording on the first Enter
                    recording = True
                    start_time = time.time()
                    print("Recording started. Press ENTER again to stop.")
                    play_tone(duration=0.7)
            elif recording:
                # Record key and timestamp relative to start
                elapsed_time = time.time() - start_time
                keys.append((key.char, elapsed_time))

                frequency=frequency_from_position(key.char)
                play_tone(frequency, duration=0.2)

        # Set up the listener
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()


        return keys

    original_keystrokes = read_keystrokes(file_name)
    new_keystrokes = capture_keystrokes()
    for (key_orig, time_orig), (key_new, time_new) in itertools.zip_longest(original_keystrokes, new_keystrokes, fillvalue=("No Key", "No Time")):
        print(f"{key_orig} {time_orig} | {key_new} {time_new}")
    
    result=compare_keystrokes(original_keystrokes, new_keystrokes)
    
    if result:
        show_success_window()
    else:
        show_failure_window()
    







def motion_code(username):
        
 
# Initialize MediaPipe Hands.
    login_duration_limit=30
    timer=time.time()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # For drawing the hand annotations on the image.
    mp_drawing = mp.solutions.drawing_utils

    # Variable to store the captured hand gesture landmarks
    captured_hand_position = None

    # Timer variables
    gesture_start_time = None
    gesture_hold_duration = 2  # Duration in seconds to hold the gesture

    def capture_hand_position(file_name):
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # List all files in this directory
        files = os.listdir(script_directory)
        success=0
        file_name=file_name+".csv"
        # Print the files
        for file in files:
            if file==file_name:
                success=1
        file_name=script_directory+"/"+file_name
        if success:
            print("Found the username")
            loaded_array = np.loadtxt(file_name, delimiter=',')
        else:
            print("Username not found")
            sys.exit(0)
        return loaded_array
        

    def is_same_gesture(captured_hand_position, current_hand_landmarks):
        if captured_hand_position is None:
            return False

        # Calculate the Euclidean distance between corresponding landmarks
        current_position = np.array([[landmark.x, landmark.y, landmark.z] for landmark in current_hand_landmarks.landmark])
        distances = np.linalg.norm(captured_hand_position - current_position, axis=1)

        # Check if the positions are similar by seeing if the mean distance is below a threshold
        return np.mean(distances) < 0.08  # Threshold, adjust based on testing

    cap = cv2.VideoCapture(0)

    unlocked = False

    captured_hand_position = capture_hand_position(username)

    while cap.isOpened() and not unlocked and (time.time()-timer<login_duration_limit):
        success, image = cap.read()
        if not success:
            continue

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands.
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                

                if is_same_gesture(captured_hand_position, hand_landmarks):
                    if gesture_start_time is None:
                        gesture_start_time = time.time()  # Start the timer when gesture is first detected
                    elif (time.time() - gesture_start_time) > gesture_hold_duration:
                        unlocked = True
                    #cv2.putText(image, 'Gesture Match Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    gesture_start_time = None  # Reset the timer if the gesture does not match

        if unlocked:
            cv2.putText(image, 'Unlocked', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the image.
        cv2.imshow('MediaPipe Hands', image)
        
        # Check for 'ESC' key press to break the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    if unlocked:
        show_success_window()
    else:
        show_failure_window()


customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("400x300")


def voice_login():
    username = entry1.get()
    print("Directing to voice login for",username)
    login_frame.pack_forget()  # Hide the login frame
    
    main_frame.pack(pady=20, padx=60, fill="both", expand=True) 
    # main_frame = customtkinter.CTkFrame(master=root)
    
    main_label = customtkinter.CTkLabel(master=main_frame, text="Welcome to the Voice App!", font=("Roboto", 18))
    main_label.pack(pady=12, padx=10)

    #voice_code(username)

def pattern_login():
    username = entry1.get()
    print("Directing to pattern login for",username)
    login_frame.pack_forget()  # Hide the login frame
    
    main_frame.pack(pady=20, padx=60, fill="both", expand=True) 
    # main_frame = customtkinter.CTkFrame(master=root)
    
    main_label = customtkinter.CTkLabel(master=main_frame, text="Welcome to the Pattern App!", font=("Roboto", 18))
    main_label.pack(pady=12, padx=10)

    pattern_code(username)
    # main_frame = customtkinter.CTkFrame(master=root)
    
    # main_label = customtkinter.CTkLabel(master=main_frame, text="Welcome to the Voice App!", font=("Roboto", 24))
    # main_label.pack(pady=12, padx=10)


def motion_login():
    username = entry1.get()
    print("Directing to motion login for",username)
    login_frame.pack_forget()  # Hide the login frame
    main_frame.pack(pady=20, padx=60, fill="both", expand=True) 
    # main_frame = customtkinter.CTkFrame(master=root)
    
    main_label = customtkinter.CTkLabel(master=main_frame, text="Welcome to the Motion App!", font=("Roboto", 18))
    main_label.pack(pady=12, padx=10)

    motion_code(username)

    # main_frame = customtkinter.CTkFrame(master=root)
    
    # main_label = customtkinter.CTkLabel(master=main_frame, text="Welcome to the Voice App!", font=("Roboto", 24))
    # main_label.pack(pady=12, padx=10)


def login():
    username = entry1.get()
    print("Testing",username)
    login_frame.pack_forget()  # Hide the login frame
    
    # main_frame = customtkinter.CTkFrame(master=root)
    
    # main_label = customtkinter.CTkLabel(master=main_frame, text="Welcome to the Voice App!", font=("Roboto", 24))
    # main_label.pack(pady=12, padx=10)

# Login Frame
login_frame = customtkinter.CTkFrame(master=root)
login_frame.pack(pady=20, padx=60, fill="both", expand=True)

# Set the font using the 'font' argument
label = customtkinter.CTkLabel(master=login_frame, text="Login System", font=("Roboto", 24))
label.pack(pady=12, padx=10)

entry1 = customtkinter.CTkEntry(master=login_frame, placeholder_text="Username")
entry1.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=login_frame, text="Motion Password", command=motion_login)
button.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=login_frame, text="Voice Password", command=voice_login)
button.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=login_frame, text="Pattern Password", command=pattern_login)
button.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=login_frame, text="Login", command=login)
button.pack(pady=12, padx=10)

checkbox = customtkinter.CTkCheckBox(master=login_frame, text="Remember Me")  # Fixed typo
checkbox.pack(pady=12, padx=10)

# Main Application Frame (initially hidden)
main_frame = customtkinter.CTkFrame(master=root)

# main_label = customtkinter.CTkLabel(master=main_frame, text="Welcome to the Main App!", font=("Roboto", 18))
# main_label.pack(pady=12, padx=10)

root.mainloop()

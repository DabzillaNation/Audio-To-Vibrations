# Credits
This is based on: https://github.com/abstract-creations/subwoofer

# Description
Converts desktop audio (or any audio device) into vibration commands for sex toys through initface in real time.

Alternative to funscripts for vibrators.

# Usage Guide (For executable build)
Download the latest release from the releases tab on the right.
1. Run initface and start the server
2. Double click AudioToVibrations.exe
3. Select and confirm an audio device. The audio visualizer, debug window and control panel will open after.
4. The AudioToVibration app should automatically connect with initface. It will display debug messages in the debug window.
6. You can now use the control panel to edit settings in real time. The effect of the settings will be shown in the Live Audio View.

# Usage Guide (For Running Rust Script Directly)
Clone and Extract this repo. (Download and extract the zip file)

Navigate to the project directory: Open your terminal or command prompt and change your current directory to the root of your Rust project (the directory containing Cargo.toml, Cargo.lock, and the src folder).

Run the project: Once you are in the project's root directory, execute the following command:
```cargo run```

if you don't have rust installed you will get this error:
```cargo : The term 'cargo' is not recognized as the name of a cmdlet, function, script file, or operable program. ```

Go to the official Rust website: https://www.rust-lang.org/tools/install
Follow the on-screen instructions. It will usually ask you to proceed with the default installation, which is generally fine. This process will install Rust, Cargo, and set up the necessary environment variables.

Close and Reopen your Terminal/PowerShell: After installing Rust, the changes to the PATH variable might only take effect in new terminal sessions.

Then try running ```cargo run``` again in your project directory. (The first time it runs it will download the necessary dependencies)

(While downloading dependencies you may get the error "**The process cannot access the file because it is being used by another process. (os error 32)**" Simply close and reopen the console terminal and type ```cargo run``` again)

It will ask you to select an audio device in the GUI and to confirm it.

Once you select a device, the audio visualizer and control panel will open.

The visualizer just lets you see the audio waveform and you can edit various settings in realtime from the control panel.

Open initface and start the server, you will see debug messages in the console letting you know it has connected to initface and with your device.

# Added Features:
- More seamless connection, continuously rescans for initface on connection loss
- Audio Signal Smoothing
- Setup window for Audio Device Selection
- Control Panel
  - slider for vibration intensity
  - slider for minimum instruction delay
  - slider for threshold cutoff value
  - slider for audio smoothing value
  - toggle for low pass filter
  - Button to reset settings to defaults

<img width="395" height="206" alt="image" src="https://github.com/user-attachments/assets/06e0ab62-2faf-46b7-9eb6-69468e5c227b" />

<img width="474" height="235" alt="image" src="https://github.com/user-attachments/assets/60f336df-1f49-4385-92d5-5422fb78d590" />

<img width="1279" height="742" alt="image" src="https://github.com/user-attachments/assets/08ecc4ac-2066-4abe-ba8a-abc6af6dcbd7" />




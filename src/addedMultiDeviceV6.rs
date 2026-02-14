// This attribute ensures that a console window is created on Windows for debug messages.
#![cfg_attr(windows, windows_subsystem = "console")]

//==================================================================================
//  IMPORTS (Dependencies)
//
//  'use' statements bring external code (libraries, or "crates" in Rust) into our
//  program's scope, so we can use their functions and types.
//==================================================================================

// 'anyhow::Result' is used for a more ergonomic and simple way of handling errors.
use anyhow::Result;
// These two imports are from the 'audio_visualizer' crate. They help us create
// a window that displays the audio waveform in real-time.
use audio_visualizer::dynamic::live_input::AudioDevAndCfg;
use audio_visualizer::dynamic::window_top_btm::{open_window_connect_audio, TransformFn};
// These are from the 'buttplug' crate, which allows our program to communicate with
// sex toys and vibrating devices through a server like Intiface.
use buttplug::{
    client::{ButtplugClient, ScalarValueCommand},
    core::connector::new_json_ws_client_connector,
};
// 'cpal' is the cross-platform audio library. It lets us find audio devices
// (like microphones or system loopbacks) and capture their signals.
use cpal::{
    traits::{DeviceTrait, HostTrait},
    Device,
};
// 'eframe' and 'egui' are used to create the graphical user interface (GUI)
// control panel window with sliders and toggles.
use eframe::egui;
// A simple low-pass filter function to isolate bass frequencies.
use lowpass_filter::lowpass_filter;
// Standard library imports for handling errors, user input, and concurrency.
use std::error::Error;
use std::sync::{mpsc as std_mpsc, Arc, Mutex, OnceLock}; // 'Arc', 'Mutex', and 'OnceLock' are crucial for safe multi-threading.
use std::time::Duration;
// 'tokio' is an asynchronous runtime for Rust. We use it to handle networking (to Buttplug)
// and timing without freezing the application.
use tokio::{sync::mpsc, time};

//==================================================================================
//  CONSTANTS
//
//  Constants are values that are fixed and cannot be changed during the program's execution.
//==================================================================================

/// We collect audio samples in batches to average them out. This prevents the vibration
/// from being too jerky. This constant defines the maximum number of samples in a batch.
const SAMPLE_LIMIT: usize = 16;
const ALL_DEVICES_LABEL: &str = "All Devices";

//==================================================================================
//  GLOBAL STATE & STATIC VARIABLES
//
//  This section defines data structures that need to be accessed from different
//  parts of our program, especially from different threads.
//==================================================================================

/// This struct holds the state required by the audio processing callback.
/// The audio callback runs on a separate, high-priority thread managed by the audio library,
/// and it has strict limitations on what it can do. We package everything it needs here.
struct AudioProcessorState {
    /// `tx` is a "transmitter". It's the sending end of a channel that we use to pass
    /// the calculated vibration values from the audio thread to our main vibration logic thread.
    tx: mpsc::Sender<f64>,
    /// This is a reference to the shared application settings, so the audio thread
    /// can see the latest values from the GUI (like intensity, threshold, etc.).
    settings: Arc<Mutex<AppSettings>>,
}

/// `static` means this variable will live for the entire duration of the program.
/// `OnceLock` is a special type that ensures the variable can be initialized exactly once.
///
/// WHY IS THIS NEEDED?
/// The `audio_visualizer` library requires a function pointer (`fn`) for its audio processing.
/// A normal `fn` cannot "capture" or borrow variables from its environment.
/// By using a global `static OnceLock`, we create a single, safe place to store our state
/// that the `fn` can access from anywhere, solving the environment capture problem.
static AUDIO_STATE: OnceLock<AudioProcessorState> = OnceLock::new();

/// We need to store the last smoothed sample between calls to the audio callback.
/// A `static Mutex` is a safe way to maintain this state across threads and function calls,
/// especially since the callback is a `fn` pointer and cannot capture its environment.
static LAST_SMOOTHED_SAMPLE: Mutex<f32> = Mutex::new(0.0);


//==================================================================================
//  APPLICATION SETTINGS STRUCT
//
//  This struct defines the configuration options that can be changed by the user
//  through the GUI control panel.
//==================================================================================

/// Using `#[derive(Debug)]` allows us to easily print this struct for debugging purposes.
#[derive(Debug)]
struct AppSettings {
    intensity: f64,
    delay_ms: u64,
    threshold: f64,
    use_lowpass_filter: bool, // Toggle to enable or disable the bass filter.
    smoothing_ms: f64,        // The decay time in milliseconds.
    /// The name of the device currently targeted for vibration.
    target_device: String,
    /// A list of names of devices currently discovered by the Buttplug client.
    available_devices: Vec<String>,
    /// A flag to signal the vibration thread to perform a new device scan.
    refresh_requested: bool,
}

/// This `impl` block provides a `default` constructor for `AppSettings`.
/// This is what the settings will be when the program first starts.
impl Default for AppSettings {
    fn default() -> Self {
        Self {
            intensity: 10.0,
            delay_ms: 35,
            threshold: 0.005,
            use_lowpass_filter: true, // Default to using the filter.
            smoothing_ms: 0.0,      // Default to a 0ms decay time.
            target_device: ALL_DEVICES_LABEL.to_string(),
            available_devices: Vec::new(),
            refresh_requested: false,
        }
    }
}

//==================================================================================
//  GUI (CONTROL PANEL) IMPLEMENTATION
//
//  This section defines the structure and behavior of the control panel window.
//==================================================================================

struct ControlPanelApp {
    /// The GUI needs access to the settings so it can display and modify them.
    /// `Arc<Mutex<T>>` is a pattern for sharing data between threads safely.
    /// - `Arc` (Atomic Reference Counting) lets multiple parts of the program "own" the data.
    /// - `Mutex` (Mutual Exclusion) ensures that only one thread can write to the data at a time,
    ///   preventing data corruption.
    settings: Arc<Mutex<AppSettings>>,
}

impl ControlPanelApp {
    /// A simple constructor to create a new instance of our GUI app.
    fn new(settings: Arc<Mutex<AppSettings>>) -> Self {
        Self { settings }
    }
}

/// This `impl` block is where we tell `eframe` how to draw our application.
/// The `eframe::App` trait requires an `update` method.
impl eframe::App for ControlPanelApp {
    /// The `update` method is called repeatedly to draw the GUI, handle user input, etc.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Create a central panel to hold all our widgets.
        egui::CentralPanel::default().show(ctx, |ui| {
            // Some minor styling.
            ui.heading("Vibration Controls");
            ui.separator();

            // To access the settings, we must 'lock()' the Mutex.
            // This blocks other threads from accessing it until we're done.
            // The lock is automatically released when 'settings' goes out of scope.
            let mut settings = self.settings.lock().unwrap();

            // Create a single instance of the default settings to compare against.
            let default_settings = AppSettings::default();

            // FIX: Clone the list of devices so we can iterate over them without 
            // blocking the mutable access needed to change 'target_device'.
            let device_list = settings.available_devices.clone();

            // --- Device Selection Dropdown ---
            ui.horizontal(|ui| {
                ui.label("Target Device:");
                egui::ComboBox::from_id_source("device_selector")
                    .selected_text(&settings.target_device)
                    .width(250.0)
                    .show_ui(ui, |ui| {
                        // Always provide the option to vibrate everything.
                        ui.selectable_value(&mut settings.target_device, ALL_DEVICES_LABEL.to_string(), ALL_DEVICES_LABEL);
                        
                        // List individual devices found during the scan.
                        for device_name in device_list {
                            ui.selectable_value(&mut settings.target_device, device_name.clone(), &device_name);
                        }
                    });

                // Refresh Button: Sets a flag that the background thread will notice.
                if ui.button("🔄").on_hover_text("Re-scan for devices").clicked() {
                    settings.refresh_requested = true;
                }
            });
            ui.add_space(8.0);

            // We use a horizontal layout for each control.
            // To place buttons "left and right of the number fields", we separate the slider
            // from its value display and reconstruct the row manually.
            
            // --- Vibration Intensity ---
            ui.horizontal(|ui| {
                // Reset Button
                if ui.button("🔄").on_hover_text("Reset Intensity").clicked() {
                    settings.intensity = default_settings.intensity;
                }
                
                // Slider (Track Only) - .show_value(false) hides the number and text
                ui.style_mut().spacing.slider_width = 250.0; // Set slider width inside the layout
                ui.add(egui::Slider::new(&mut settings.intensity, 0.0..=1000.0).show_value(false));

                // --- REDUCE SPACING ---
                // Reduce the gap between items for the buttons and text field
                ui.style_mut().spacing.item_spacing.x = 4.0;

                // Minus button (Left of number)
                if ui.add(egui::Button::new("-").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.intensity = (settings.intensity - 1.0).max(0.0);
                }

                // Number Field (DragValue)
                ui.add_sized([55.0, 18.0], egui::DragValue::new(&mut settings.intensity).speed(1.0).range(0.0..=1000.0));

                ui.style_mut().spacing.item_spacing.x = 8.0; // Reset spacing to default

                // Plus button (Right of number)
                if ui.add(egui::Button::new("+").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.intensity += 1.0;
                }

                // Label
                ui.label("Vibration Intensity");
            });

            // --- Instruction Delay ---
            ui.horizontal(|ui| {
                // Reset Button
                if ui.button("🔄").on_hover_text("Reset Delay").clicked() {
                    settings.delay_ms = default_settings.delay_ms;
                }
                
                // Slider (Track Only)
                ui.style_mut().spacing.slider_width = 250.0; // Set slider width inside the layout
                ui.add(egui::Slider::new(&mut settings.delay_ms, 5..=200).show_value(false));

                // --- REDUCE SPACING ---
                // Reduce the gap between items for the buttons and text field
                ui.style_mut().spacing.item_spacing.x = 4.0;

                // Minus button
                if ui.add(egui::Button::new("-").min_size(egui::vec2(20.0, 0.0))).clicked() {
                     if settings.delay_ms >= 10 {
                        settings.delay_ms -= 5;
                    } else {
                        settings.delay_ms = 5;
                    }
                }

                // Number Field
                ui.add_sized([55.0, 18.0], egui::DragValue::new(&mut settings.delay_ms).speed(1.0).range(5..=200).suffix(" ms"));

                ui.style_mut().spacing.item_spacing.x = 8.0; // Reset spacing to default

                // Plus button
                if ui.add(egui::Button::new("+").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.delay_ms += 5;
                }

                // Label
                ui.label("Instruction Delay (ms)");
            });

            // --- Minimum Threshold ---
            ui.horizontal(|ui| {
                // Reset Button
                if ui.button("🔄").on_hover_text("Reset Threshold").clicked() {
                    settings.threshold = default_settings.threshold;
                }
                
                // Slider (Track Only)
                ui.style_mut().spacing.slider_width = 250.0; // Set slider width inside the layout
                ui.add(egui::Slider::new(&mut settings.threshold, 0.0..=1.0).show_value(false));

                // --- REDUCE SPACING ---
                // Reduce the gap between items for the buttons and text field
                ui.style_mut().spacing.item_spacing.x = 4.0;
                
                // Minus button
                if ui.add(egui::Button::new("-").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.threshold = (settings.threshold - 0.005).max(0.0);
                }

                // Number Field
                ui.add_sized([55.0, 18.0], egui::DragValue::new(&mut settings.threshold).speed(0.005).range(0.0..=1.0));

                ui.style_mut().spacing.item_spacing.x = 8.0; // Reset spacing to default
                
                // Plus button
                if ui.add(egui::Button::new("+").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.threshold = (settings.threshold + 0.005).min(1.0);
                }

                // Label
                ui.label("Minimum Threshold");
            });
            
            // --- Smooth Decay Time ---
            ui.horizontal(|ui| {
                // Reset Button
                if ui.button("🔄").on_hover_text("Reset Decay Time").clicked() {
                    settings.smoothing_ms = default_settings.smoothing_ms;
                }
                
                // Slider (Track Only)
                ui.style_mut().spacing.slider_width = 250.0; // Set slider width inside the layout
                ui.add(egui::Slider::new(&mut settings.smoothing_ms, 0.0..=2000.0).show_value(false));

                // --- REDUCE SPACING ---
                // Reduce the gap between items for the buttons and text field
                ui.style_mut().spacing.item_spacing.x = 4.0;
                
                // Minus button
                if ui.add(egui::Button::new("-").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.smoothing_ms = (settings.smoothing_ms - 1.0).max(0.0);
                }

                // Number Field
                ui.add_sized([55.0, 18.0], egui::DragValue::new(&mut settings.smoothing_ms).speed(1.0).range(0.0..=2000.0).suffix(" ms"));

                ui.style_mut().spacing.item_spacing.x = 8.0; // Reset spacing to default

                // Plus button
                if ui.add(egui::Button::new("+").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.smoothing_ms += 1.0;
                }

                // Label
                ui.label("Smooth Decay Time");
            });
            
            ui.separator();
            
            // This is the toggle switch for the low-pass filter.
            // It's bound to the `use_lowpass_filter` boolean field.
            ui.toggle_value(&mut settings.use_lowpass_filter, "Use Low-Pass Filter");

            // Add the global reset button. When clicked, it replaces all current settings
            // with a new instance of the default settings.
            if ui.button("Reset All to Defaults").clicked() {
                *settings = AppSettings::default();
            }

            ui.separator();
            ui.label("Close this window and the visualizer to exit.");
        });
    }
}

//==================================================================================
//  GUI (AUDIO DEVICE SELECTOR) IMPLEMENTATION
//
//  This section defines the GUI window that prompts the user to select an audio device.
//==================================================================================

struct DeviceSelectorApp {
    devices: Vec<(String, Device)>,
    selected_device_index: Option<usize>,
    // This 'sender' will pass the chosen device from the GUI thread back to the main thread.
    sender: std_mpsc::Sender<Device>,
}

impl DeviceSelectorApp {
    /// Creates a new instance of the device selector GUI.
    fn new(sender: std_mpsc::Sender<Device>) -> Self {
        Self {
            devices: list_output_devs(), // Populate the list of devices.
            selected_device_index: None,
            sender,
        }
    }
}

impl eframe::App for DeviceSelectorApp {
    /// This update function draws the device selection window.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Select an Audio Device");
            ui.separator();
            ui.label("Please choose the audio output device you want to monitor:");

            // Determine the text for the combo box based on the current selection.
            let selected_text = self.selected_device_index
                .and_then(|index| self.devices.get(index))
                .map(|(name, _)| name.clone())
                .unwrap_or_else(|| "Click to select...".to_string());

            // A Combo Box provides a nice dropdown menu for device selection.
            egui::ComboBox::from_label("Audio Device")
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    for (i, (name, _device)) in self.devices.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_device_index, Some(i), name);
                    }
                });

            ui.add_space(10.0);

            // The confirm button is only enabled if a device has been selected.
            ui.add_enabled_ui(self.selected_device_index.is_some(), |ui| {
                 if ui.button("Confirm and Start").clicked() {
                    // Use take() to get the index and set the Option to None.
                    // This prevents the logic from running twice if the UI redraws before closing.
                    if let Some(index) = self.selected_device_index.take() {
                        // Move the selected device out of the vector.
                        let (_, device) = self.devices.remove(index);
                        self.sender.send(device).expect("Failed to send device");
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            });

            ui.separator();

            ui.label("Reminder: Open initface and start the server to connect to devices.");
        });
    }
}

/// This function launches the device selection GUI and waits until the user makes a choice.
fn select_device_gui() -> Result<Device, Box<dyn Error>> {
    // We use a standard (blocking) channel here because the main thread will wait for the GUI.
    let (tx, rx) = std_mpsc::channel();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([400.0, 180.0]),
        ..Default::default()
    };

    // Run the device selector GUI. This blocks the main thread until the GUI window is closed.
    eframe::run_native(
        "Select Audio Device",
        native_options,
        Box::new(move |_cc| Ok(Box::new(DeviceSelectorApp::new(tx)))),
    )?;

    // Wait to receive the device from the GUI thread.
    // If the user closes the window without confirming, `recv` will return an error.
    rx.recv().map_err(|e| e.into())
}

//==================================================================================
//  MAIN FUNCTION
//
//  This is the entry point of our Rust program.
//==================================================================================

fn main() -> std::result::Result<(), Box<dyn Error>> {
    // Run device selection GUI first.
    // The program will not proceed until a device is selected or the window is closed.
    let selected_device = match select_device_gui() {
        Ok(device) => device,
        Err(_) => {
            println!("[Info] No device selected. Exiting program.");
            return Ok(()); // Exit gracefully if the selection window was closed by the user.
        }
    };

    // 1. Create the application settings inside the Arc/Mutex for thread-safe sharing.
    let settings = Arc::new(Mutex::new(AppSettings::default()));
    // Create a "clone" of the Arc pointer to pass to the vibration logic thread.
    // This increases the reference count, it does not copy the data itself.
    let settings_clone = Arc::clone(&settings);

    // 2. Spawn a new OS thread to run our core logic (audio processing and vibration).
    // This prevents the core logic from blocking the GUI, keeping it responsive.
    std::thread::spawn(move || {
        // Create a Tokio runtime to execute our `async` functions.
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        // Block the new thread until our async vibration logic completes or errors.
        rt.block_on(async {
            // Pass ownership of the device chosen by the user into the main logic function.
            if let Err(e) = run_vibration_logic(settings_clone, selected_device).await {
                eprintln!("Vibration logic failed: {}", e);
            }
        });
    });

    // 3. Set up the native window options for our GUI control panel.
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([535.0, 240.0]) // Increased height slightly to accommodate the new dropdown
            .with_always_on_top(),
        ..Default::default()
    };

    // 4. Run the eframe GUI application. This function takes over the main thread.
    // It passes the original `settings` Arc to the GUI.
    eframe::run_native(
        "AudioToVibrations Control Panel",
        native_options,
        Box::new(|_cc| Ok(Box::new(ControlPanelApp::new(settings)))),
    )?;

    Ok(())
}


//==================================================================================
//  AUDIO PROCESSING CALLBACK
//
//  This function is called repeatedly by the audio library with new chunks of audio data.
//==================================================================================

/// This is the function pointer that gets passed to the audio visualizer.
/// It processes audio data and returns a modified version for visualization.
/// It also sends vibration data to the main logic loop.
fn audio_transform_fn(direct_values: &[f32], sampling_rate: f32) -> Vec<f32> {
    // Get the global state. This will panic if it hasn't been initialized, which is a safeguard.
    let state = AUDIO_STATE.get().expect("AUDIO_STATE not initialized");

    // Get a snapshot of the current settings from the GUI.
    let (intensity, threshold, use_lowpass_filter, smoothing_ms) = {
        let s = state.settings.lock().unwrap();
        (s.intensity, s.threshold, s.use_lowpass_filter, s.smoothing_ms)
    };

    // Create a mutable copy of the incoming audio data to work with.
    let mut processed_values = direct_values.to_vec();

    // Conditionally apply the low-pass filter based on the GUI toggle.
    if use_lowpass_filter {
        // This function modifies the `processed_values` in-place.
        lowpass_filter(&mut processed_values, sampling_rate, 80.0); // 80.0 Hz is a good bass cutoff.
    }

    // --- ASYMMETRIC SMOOTHING LOGIC (FAST ATTACK, SLOW DECAY) ---
    // We check if smoothing_ms is meaningfully greater than zero to avoid division by zero
    // and to disable smoothing when the user sets the slider to 0.
    if smoothing_ms > 1.0 {
        // Convert decay time in ms to a time constant 'tau' in seconds. This is the foundation
        // of making the slider intuitive.
        let tau = smoothing_ms / 1000.0;
        
        // Calculate the per-sample smoothing factor 'alpha' from tau and the sampling rate.
        // The formula exp(-Δt / τ) correctly maps a linear time slider to the non-linear
        // alpha values required for a per-sample filter. Δt here is 1.0 / sampling_rate.
        let alpha = (-1.0 / (sampling_rate as f64 * tau)).exp() as f32;

        let mut last_sample_val = LAST_SMOOTHED_SAMPLE.lock().unwrap();
        for current_sample in processed_values.iter_mut() {
            let new_sample;
            // Compare the absolute values (amplitude) of the signals.
            if current_sample.abs() > last_sample_val.abs() {
                // FAST ATTACK: The new signal is louder. Immediately adopt its value,
                // ignoring the previous state. This makes the vibration ramp up instantly.
                new_sample = *current_sample;
            } else {
                // SLOW DECAY: The new signal is quieter. Apply the EMA formula using our calculated
                // alpha to smoothly transition downwards from the last peak.
                new_sample = *current_sample * (1.0 - alpha) + *last_sample_val * alpha;
            }
            // Update the sample in the buffer that will be sent to the visualizer.
            *current_sample = new_sample;
            // Update our state for the *next* sample in this batch.
            *last_sample_val = new_sample;
        }
    }

    // --- VIBRATION VALUE CALCULATION ---
    // Take the last sample from the processed chunk as our representative value.
    let mut vibration_value = *processed_values.last().unwrap_or(&0.0) as f64;
    vibration_value = f64::abs(vibration_value); // Vibration can't be negative.

    // Apply the threshold: if the sound is too quiet, ignore it.
    if vibration_value < threshold {
        vibration_value = 0.0;
    }
    vibration_value *= intensity; // Apply the intensity multiplier.

    // Send the final vibration value through the channel to the vibration logic loop.
    // `try_send` is non-blocking; it won't wait if the channel is full.
    let _ = state.tx.try_send(vibration_value);


    // --- VISUALIZER DATA MODIFICATION ---
    // Modify the entire dataset that will be returned for plotting on the graph.
    // This ensures the graph reflects the settings in real-time.
    for sample in processed_values.iter_mut() {
        let sample_abs = sample.abs() as f64;

        if sample_abs < threshold {
            *sample = 0.0; // Apply threshold visually, flattening small waves.
        } else {
            // Apply intensity visually, making waves taller or shorter.
            *sample *= intensity as f32;
        }
    }

    // Return the modified vector, which will now be plotted by the visualizer.
    processed_values
}


//==================================================================================
//  VIBRATION AND NETWORKING LOGIC
//
//  This async function runs in the background and handles everything related to
//  audio capture, Buttplug connection, and sending vibration commands.
//==================================================================================
async fn run_vibration_logic(settings: Arc<Mutex<AppSettings>>, audio_device: Device) -> Result<()> {
    // 1. Create a "Multi-Producer, Single-Consumer" (MPSC) channel.
    // The audio callback (`tx`) is the producer, and the loop below (`rx`) is the consumer.
    let (tx, mut rx) = mpsc::channel::<f64>(SAMPLE_LIMIT);

    // 2. Initialize our global `AUDIO_STATE` with the transmitter and settings.
    let initial_state = AudioProcessorState {
        tx,
        settings: Arc::clone(&settings),
    };
    if AUDIO_STATE.set(initial_state).is_err() {
        // This should never happen if the logic is correct.
        panic!("AUDIO_STATE was already initialized");
    }

    // 3. Set up the audio capture using the device chosen from the GUI.
    let default_out_config = audio_device.default_output_config().unwrap().config();
    println!("[Audio] Using audio device: {}", audio_device.name()?);

    // Spawn another async task specifically for the audio visualizer window.
    // This lets it run independently without blocking our Buttplug logic.
    tokio::spawn(async move {
        open_window_connect_audio(
            "Live Audio View",
            None, None, None, None,
            "time (seconds)",
            "Amplitude (After Processing)",
            // Pass ownership of the selected device and its config to the visualizer.
            AudioDevAndCfg::new(Some(audio_device), Some(default_out_config)),
            TransformFn::Basic(audio_transform_fn), // Pass our processing function here.
        );
    });

    // 4. Main reconnection loop. The label `'reconnection_loop` lets us `continue`
    // from the beginning if the connection to the device is ever lost.
    'reconnection_loop: loop {
        println!("[Buttplug] Attempting to connect to server at ws://localhost:12345...");

        // Create a Buttplug client and try to connect to the server (e.g., Intiface Desktop).
        let client = ButtplugClient::new("AudioToVibrations Client");
        let connector = new_json_ws_client_connector("ws://localhost:12345/buttplug");

        if let Err(e) = client.connect(connector).await {
            eprintln!("[Buttplug] Connection failed: {}. Retrying in 5 seconds...", e);
            time::sleep(Duration::from_secs(5)).await;
            continue 'reconnection_loop; // Retry connection.
        }

        println!("[Buttplug] Connected! Scanning for devices...");
        client.start_scanning().await?;
        time::sleep(Duration::from_secs(2)).await; // Scan for a couple of seconds.
        client.stop_scanning().await?;

        // After scanning, update the shared available_devices list so the GUI can see them.
        {
            let mut s = settings.lock().unwrap();
            s.available_devices = client.devices().iter().map(|d| d.name().clone()).collect();
        }

        // Check if we actually found anything.
        if client.devices().is_empty() {
            eprintln!("[Buttplug] No devices found. Retrying in 10 seconds...");
            let _ = client.disconnect().await;
            time::sleep(Duration::from_secs(10)).await;
            continue 'reconnection_loop;
        };

        println!("[Buttplug] {} device(s) found.", client.devices().len());
        println!("[Vibration] Starting vibration loop...");

        // 5. Inner operational loop. This runs as long as the device is connected.
        loop {
            // --- RE-SCAN LOGIC ---
            // Check if the GUI has requested a device refresh.
            let refresh_needed = {
                let mut s = settings.lock().unwrap();
                if s.refresh_requested {
                    s.refresh_requested = false; // Reset the flag.
                    true
                } else {
                    false
                }
            };

            if refresh_needed {
                println!("[Buttplug] Refreshing device list...");
                let _ = client.start_scanning().await;
                time::sleep(Duration::from_secs(2)).await;
                let _ = client.stop_scanning().await;
                {
                    let mut s = settings.lock().unwrap();
                    s.available_devices = client.devices().iter().map(|d| d.name().clone()).collect();
                }
                println!("[Buttplug] Refresh complete. Found {} device(s).", client.devices().len());
            }

            let mut collected_values: Vec<f64> = Vec::with_capacity(SAMPLE_LIMIT);
            // Wait to receive a batch of values from the audio thread.
            // If `recv_many` returns 0, the channel was closed, meaning the app is shutting down.
            if rx.recv_many(&mut collected_values, SAMPLE_LIMIT).await == 0 {
                println!("[Audio] Audio stream closed. Shutting down.");
                client.disconnect().await?;
                return Ok(()); // Exit the function and the thread.
            }

            // Average the received values to smooth out the vibration signal.
            let collected_length = collected_values.len();
            let mean_value: f64 = if collected_length > 0 {
                collected_values.iter().sum::<f64>() / collected_length as f64
            } else {
                0.0
            };

            // Clamp the value between 0.0 and 1.0, as required by the Buttplug protocol.
            let computed_intensity = f64::min(mean_value, 1.0);

            // Get current target device and delay from settings.
            let (target, delay) = {
                let s = settings.lock().unwrap();
                (s.target_device.clone(), s.delay_ms)
            };

            // Iterate over ALL connected devices.
            let all_connected = client.devices();
            let mut command_failed = false;

            for device in all_connected {
                // If "All Devices" is selected, or if the specific device name matches the selection.
                // FIX: Added &target to match the return type of device.name().
                if target == ALL_DEVICES_LABEL || device.name() == &target {
                    if let Err(e) = device.vibrate(&ScalarValueCommand::ScalarValue(computed_intensity)).await {
                        eprintln!("[Buttplug] Vibrate command failed for {}: {}.", device.name(), e);
                        command_failed = true;
                    }
                }
            }

            // If a command failed, we assume a connection issue and restart the client.
            if command_failed && !client.connected() {
                break; // Break inner loop to trigger reconnection.
            }

            // Wait for a short duration, determined by the user's "Delay" setting.
            // This acts as a rate-limiter to not flood the device with commands.
            time::sleep(Duration::from_millis(delay)).await;
        }

        // If we're here, the inner loop broke due to a connection error.
        println!("[Buttplug] Disconnected. Will attempt to reconnect...");
        let _ = client.disconnect().await; // Clean up the old session.
    }
}


//==================================================================================
//  HELPER FUNCTIONS
//
//  This function assists with finding audio devices for the GUI selector.
//==================================================================================

/// Returns a list of all available audio output devices on the system.
/// This is called by the device selector GUI.
pub fn list_output_devs() -> Vec<(String, Device)> {
    let host = cpal::default_host();
    type DeviceName = String;
    let mut devs: Vec<(DeviceName, Device)> = host
        .output_devices()
        .unwrap()
        .map(|dev| {
            (
                dev.name().unwrap_or_else(|_| String::from("<unknown>")),
                dev,
            )
        })
        .collect();
    devs.sort_by(|(n1, _), (n2, _)| n1.cmp(n2));
    devs
}
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
use std::io::{stdin, BufRead};
use std::sync::{Arc, Mutex, OnceLock}; // 'Arc', 'Mutex', and 'OnceLock' are crucial for safe multi-threading.
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
            ui.style_mut().spacing.slider_width = 250.0;
            ui.heading("Vibration Controls");
            ui.separator();

            // To access the settings, we must 'lock()' the Mutex.
            // This blocks other threads from accessing it until we're done.
            // The lock is automatically released when 'settings' goes out of scope.
            let mut settings = self.settings.lock().unwrap();

            // Add the GUI widgets (sliders and a toggle switch).
            // Each widget is bound to a field in the `AppSettings` struct.
            ui.add(egui::Slider::new(&mut settings.intensity, 0.0..=1000.0).text("Vibration Intensity"));
            ui.add(egui::Slider::new(&mut settings.delay_ms, 5..=200).text("Instruction Delay (ms)").suffix(" ms"));
            ui.add(egui::Slider::new(&mut settings.threshold, 0.0..=1.0).text("Minimum Threshold"));

            // MODIFIED: This slider now controls the decay time in milliseconds, which is much
            // more intuitive for the user.
            ui.add(egui::Slider::new(&mut settings.smoothing_ms, 0.0..=200.0).text("Decay Time").suffix(" ms"));

            // This is the toggle switch for the low-pass filter.
            // It's bound to the `use_lowpass_filter` boolean field.
            ui.toggle_value(&mut settings.use_lowpass_filter, "Use Low-Pass Filter");

            // Add the reset button. When clicked, it replaces the current settings
            // with a new instance of the default settings.
            if ui.button("Reset to Defaults").clicked() {
                *settings = AppSettings::default();
            }

            ui.separator();
            ui.label("Close this window and the visualizer to exit.");
        });
    }
}


//==================================================================================
//  MAIN FUNCTION
//
//  This is the entry point of our Rust program.
//==================================================================================

fn main() -> std::result::Result<(), Box<dyn Error>> {
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
            if let Err(e) = run_vibration_logic(settings_clone).await {
                eprintln!("Vibration logic failed: {}", e);
            }
        });
    });

    // 3. Set up the native window options for our GUI control panel.
    let native_options = eframe::NativeOptions {
        // Increased height to fit the new smoothing slider comfortably.
        viewport: egui::ViewportBuilder::default().with_inner_size([440.0, 200.0]),
        ..Default::default()
    };

    // 4. Run the eframe GUI application. This function takes over the main thread.
    // It passes the original `settings` Arc to the GUI.
    eframe::run_native(
        "Subwoofer Control Panel",
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

    // --- MODIFIED: ASYMMETRIC SMOOTHING LOGIC (FAST ATTACK, SLOW DECAY) ---
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
async fn run_vibration_logic(settings: Arc<Mutex<AppSettings>>) -> Result<()> {
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

    // 3. Set up the audio capture.
    let default_out_dev = select_output_dev(); // Prompts user to select an audio device.
    let default_out_config = default_out_dev.default_output_config().unwrap().config();
    println!("[Audio] Using audio device: {}", default_out_dev.name()?);

    // Spawn another async task specifically for the audio visualizer window.
    // This lets it run independently without blocking our Buttplug logic.
    tokio::spawn(async move {
        open_window_connect_audio(
            "Live Audio View",
            None, None, None, None,
            "time (seconds)",
            "Amplitude (After Processing)",
            AudioDevAndCfg::new(Some(default_out_dev), Some(default_out_config)),
            TransformFn::Basic(audio_transform_fn), // Pass our processing function here.
        );
    });

    // 4. Main reconnection loop. The label `'reconnection_loop` lets us `continue`
    // from the beginning if the connection to the device is ever lost.
    'reconnection_loop: loop {
        println!("[Buttplug] Attempting to connect to server at ws://localhost:12345...");

        // Create a Buttplug client and try to connect to the server (e.g., Intiface Desktop).
        let client = ButtplugClient::new("subwoofer");
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

        // ******** THE FIX IS HERE ********
        // We must first store the list of devices in a variable (`all_devices`).
        // This ensures the list itself lives long enough for us to borrow from it.
        let all_devices = client.devices();

        // Now, we can safely get the first item from `all_devices`. `client_device` will
        // be a reference to an item inside `all_devices`, which is perfectly fine
        // because `all_devices` is still in scope.
        let Some(client_device) = all_devices.first() else {
            eprintln!("[Buttplug] No device found. Retrying in 10 seconds...");
            let _ = client.disconnect().await;
            time::sleep(Duration::from_secs(10)).await;
            continue 'reconnection_loop;
        };
        // ******** END OF FIX ********

        println!("[Buttplug] Device connected: {}", client_device.name());
        println!("[Vibration] Starting vibration loop...");

        // 5. Inner operational loop. This runs as long as the device is connected.
        loop {
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

            // Send the final vibrate command to the device.
            if let Err(e) = client_device.vibrate(&ScalarValueCommand::ScalarValue(computed_intensity)).await {
                eprintln!("[Buttplug] Vibrate command failed: {}. Connection lost.", e);
                break; // Break the inner loop to trigger reconnection.
            }

            // Wait for a short duration, determined by the user's "Delay" setting.
            // This acts as a rate-limiter to not flood the device with commands.
            let delay = { settings.lock().unwrap().delay_ms };
            time::sleep(Duration::from_millis(delay)).await;
        }

        // If we're here, the inner loop broke due to a connection error.
        println!("[Buttplug] Disconnected. Will attempt to reconnect...");
        let _ = client.disconnect().await; // Clean up the old session.
    }
}


//==================================================================================
//  HELPER FUNCTIONS (UNCHANGED)
//
//  These functions assist with audio device selection.
//==================================================================================

/// Returns a list of all available audio output devices on the system.
pub fn list_output_devs() -> Vec<(String, cpal::Device)> {
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

/// Prompts the user to select an audio device from a list printed to the console.
fn select_output_dev() -> cpal::Device {
    let mut devs = list_output_devs();
    assert!(!devs.is_empty(), "no output devices found!");
    if devs.len() == 1 {
        return devs.remove(0).1;
    }
    println!("Please select the audio device to monitor:");
    devs.iter().enumerate().for_each(|(i, (name, _))| {
        println!("  [{}] {}", i, name);
    });
    loop {
        let mut input = String::new();
        if stdin().lock().read_line(&mut input).is_err() {
            println!("Failed to read line, please try again.");
            continue;
        }
        if let Ok(index) = input.trim().parse::<usize>() {
            if index < devs.len() {
                return devs.remove(index).1;
            }
        }
        println!("Invalid input. Please enter a number from the list.");
    }
}

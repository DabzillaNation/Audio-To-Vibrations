// This attribute ensures that a console window is created on Windows for debug messages.
#![cfg_attr(windows, windows_subsystem = "console")]

//==================================================================================
//  IMPORTS (Dependencies)
//==================================================================================

use anyhow::Result;
// These two imports are from the 'audio_visualizer' crate.
use audio_visualizer::dynamic::live_input::AudioDevAndCfg;
use audio_visualizer::dynamic::window_top_btm::{open_window_connect_audio, TransformFn};
// These are from the 'buttplug' crate.
use buttplug::{
    client::{ButtplugClient, ScalarValueCommand},
    core::connector::new_json_ws_client_connector,
};
// 'cpal' is the cross-platform audio library.
use cpal::{
    traits::{DeviceTrait, HostTrait},
    Device,
};
// 'eframe' and 'egui' for the GUI.
use eframe::egui;
// Low-pass filter.
use lowpass_filter::lowpass_filter;
// Standard library imports.
use std::error::Error;
use std::sync::{mpsc as std_mpsc, Arc, Mutex, OnceLock};
use std::time::Duration;
// 'tokio' for async.
use tokio::{sync::mpsc, time};

//==================================================================================
//  CONSTANTS
//==================================================================================

const SAMPLE_LIMIT: usize = 16;
const ALL_DEVICES_LABEL: &str = "All Devices";

//==================================================================================
//  GLOBAL STATE & STATIC VARIABLES
//==================================================================================

struct AudioProcessorState {
    tx: mpsc::Sender<f64>,
    settings: Arc<Mutex<AppSettings>>,
}

static AUDIO_STATE: OnceLock<AudioProcessorState> = OnceLock::new();
static LAST_SMOOTHED_SAMPLE: Mutex<f32> = Mutex::new(0.0);

//==================================================================================
//  APPLICATION SETTINGS STRUCT
//==================================================================================

#[derive(Debug)]
struct AppSettings {
    intensity: f64,
    delay_ms: u64,
    threshold: f64,
    use_lowpass_filter: bool,
    smoothing_ms: f64,
    target_device: String,
    available_devices: Vec<String>,
    refresh_requested: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            intensity: 10.0,
            delay_ms: 35,
            threshold: 0.005,
            use_lowpass_filter: true,
            smoothing_ms: 0.0,
            target_device: ALL_DEVICES_LABEL.to_string(),
            available_devices: Vec::new(),
            refresh_requested: false,
        }
    }
}

//==================================================================================
//  GUI (CONTROL PANEL) IMPLEMENTATION
//==================================================================================

struct ControlPanelApp {
    settings: Arc<Mutex<AppSettings>>,
}

impl ControlPanelApp {
    fn new(settings: Arc<Mutex<AppSettings>>) -> Self {
        Self { settings }
    }
}

impl eframe::App for ControlPanelApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Vibration Controls");
            ui.separator();

            let mut settings = self.settings.lock().unwrap();
            let default_settings = AppSettings::default();

            // Clone device list to avoid borrowing issues
            let device_list = settings.available_devices.clone();

            // --- Device Selection Dropdown ---
            ui.horizontal(|ui| {
                ui.label("Target Device:");
                egui::ComboBox::from_id_source("device_selector")
                    .selected_text(&settings.target_device)
                    .width(250.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut settings.target_device, ALL_DEVICES_LABEL.to_string(), ALL_DEVICES_LABEL);
                        for device_name in device_list {
                            ui.selectable_value(&mut settings.target_device, device_name.clone(), &device_name);
                        }
                    });

                if ui.button("🔄").on_hover_text("Re-scan for devices").clicked() {
                    settings.refresh_requested = true;
                }
            });
            ui.add_space(8.0);

            // --- Vibration Intensity ---
            ui.horizontal(|ui| {
                if ui.button("🔄").on_hover_text("Reset Intensity").clicked() {
                    settings.intensity = default_settings.intensity;
                }
                ui.style_mut().spacing.slider_width = 250.0;
                ui.add(egui::Slider::new(&mut settings.intensity, 0.0..=1000.0).show_value(false));

                ui.style_mut().spacing.item_spacing.x = 4.0;
                if ui.add(egui::Button::new("-").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.intensity = (settings.intensity - 1.0).max(0.0);
                }
                ui.add_sized([55.0, 18.0], egui::DragValue::new(&mut settings.intensity).speed(1.0).range(0.0..=1000.0));
                ui.style_mut().spacing.item_spacing.x = 8.0;
                if ui.add(egui::Button::new("+").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.intensity += 1.0;
                }
                ui.label("Vibration Intensity");
            });

            // --- Instruction Delay ---
            ui.horizontal(|ui| {
                if ui.button("🔄").on_hover_text("Reset Delay").clicked() {
                    settings.delay_ms = default_settings.delay_ms;
                }
                ui.style_mut().spacing.slider_width = 250.0;
                ui.add(egui::Slider::new(&mut settings.delay_ms, 5..=200).show_value(false));

                ui.style_mut().spacing.item_spacing.x = 4.0;
                if ui.add(egui::Button::new("-").min_size(egui::vec2(20.0, 0.0))).clicked() {
                     if settings.delay_ms >= 10 { settings.delay_ms -= 5; } else { settings.delay_ms = 5; }
                }
                ui.add_sized([55.0, 18.0], egui::DragValue::new(&mut settings.delay_ms).speed(1.0).range(5..=200).suffix(" ms"));
                ui.style_mut().spacing.item_spacing.x = 8.0;
                if ui.add(egui::Button::new("+").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.delay_ms += 5;
                }
                ui.label("Instruction Delay (ms)");
            });

            // --- Minimum Threshold ---
            ui.horizontal(|ui| {
                if ui.button("🔄").on_hover_text("Reset Threshold").clicked() {
                    settings.threshold = default_settings.threshold;
                }
                ui.style_mut().spacing.slider_width = 250.0;
                ui.add(egui::Slider::new(&mut settings.threshold, 0.0..=1.0).show_value(false));

                ui.style_mut().spacing.item_spacing.x = 4.0;
                if ui.add(egui::Button::new("-").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.threshold = (settings.threshold - 0.005).max(0.0);
                }
                ui.add_sized([55.0, 18.0], egui::DragValue::new(&mut settings.threshold).speed(0.005).range(0.0..=1.0));
                ui.style_mut().spacing.item_spacing.x = 8.0;
                if ui.add(egui::Button::new("+").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.threshold = (settings.threshold + 0.005).min(1.0);
                }
                ui.label("Minimum Threshold");
            });

            // --- Smooth Decay Time ---
            ui.horizontal(|ui| {
                if ui.button("🔄").on_hover_text("Reset Decay Time").clicked() {
                    settings.smoothing_ms = default_settings.smoothing_ms;
                }
                ui.style_mut().spacing.slider_width = 250.0;
                ui.add(egui::Slider::new(&mut settings.smoothing_ms, 0.0..=2000.0).show_value(false));

                ui.style_mut().spacing.item_spacing.x = 4.0;
                if ui.add(egui::Button::new("-").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.smoothing_ms = (settings.smoothing_ms - 1.0).max(0.0);
                }
                ui.add_sized([55.0, 18.0], egui::DragValue::new(&mut settings.smoothing_ms).speed(1.0).range(0.0..=2000.0).suffix(" ms"));
                ui.style_mut().spacing.item_spacing.x = 8.0;
                if ui.add(egui::Button::new("+").min_size(egui::vec2(20.0, 0.0))).clicked() {
                    settings.smoothing_ms += 1.0;
                }
                ui.label("Smooth Decay Time");
            });

            ui.separator();
            ui.toggle_value(&mut settings.use_lowpass_filter, "Use Low-Pass Filter");

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
//==================================================================================

struct DeviceSelectorApp {
    devices: Vec<(String, Device)>,
    selected_device_index: Option<usize>,
    sender: std_mpsc::Sender<Device>,
}

impl DeviceSelectorApp {
    fn new(sender: std_mpsc::Sender<Device>) -> Self {
        Self {
            devices: list_output_devs(),
            selected_device_index: None,
            sender,
        }
    }
}

impl eframe::App for DeviceSelectorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Select an Audio Device");
            ui.separator();
            ui.label("Please choose the audio output device you want to monitor:");

            let selected_text = self.selected_device_index
                .and_then(|index| self.devices.get(index))
                .map(|(name, _)| name.clone())
                .unwrap_or_else(|| "Click to select...".to_string());

            egui::ComboBox::from_label("Audio Device")
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    for (i, (name, _device)) in self.devices.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_device_index, Some(i), name);
                    }
                });

            ui.add_space(10.0);

            ui.add_enabled_ui(self.selected_device_index.is_some(), |ui| {
                 if ui.button("Confirm and Start").clicked() {
                    if let Some(index) = self.selected_device_index.take() {
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

fn select_device_gui() -> Result<Device, Box<dyn Error>> {
    let (tx, rx) = std_mpsc::channel();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([400.0, 180.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Select Audio Device",
        native_options,
        Box::new(move |_cc| Ok(Box::new(DeviceSelectorApp::new(tx)))),
    )?;
    rx.recv().map_err(|e| e.into())
}

//==================================================================================
//  MAIN FUNCTION
//==================================================================================

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let selected_device = match select_device_gui() {
        Ok(device) => device,
        Err(_) => {
            println!("[Info] No device selected. Exiting program.");
            return Ok(());
        }
    };

    let settings = Arc::new(Mutex::new(AppSettings::default()));
    let settings_clone = Arc::clone(&settings);

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        rt.block_on(async {
            if let Err(e) = run_vibration_logic(settings_clone, selected_device).await {
                eprintln!("Vibration logic failed: {}", e);
            }
        });
    });

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([535.0, 240.0])
            .with_always_on_top(),
        ..Default::default()
    };

    eframe::run_native(
        "AudioToVibrations Control Panel",
        native_options,
        Box::new(|_cc| Ok(Box::new(ControlPanelApp::new(settings)))),
    )?;

    Ok(())
}

//==================================================================================
//  AUDIO PROCESSING CALLBACK
//==================================================================================

fn audio_transform_fn(direct_values: &[f32], sampling_rate: f32) -> Vec<f32> {
    let state = AUDIO_STATE.get().expect("AUDIO_STATE not initialized");
    let (intensity, threshold, use_lowpass_filter, smoothing_ms) = {
        let s = state.settings.lock().unwrap();
        (s.intensity, s.threshold, s.use_lowpass_filter, s.smoothing_ms)
    };

    let mut processed_values = direct_values.to_vec();

    if use_lowpass_filter {
        lowpass_filter(&mut processed_values, sampling_rate, 80.0);
    }

    if smoothing_ms > 1.0 {
        let tau = smoothing_ms / 1000.0;
        let alpha = (-1.0 / (sampling_rate as f64 * tau)).exp() as f32;
        let mut last_sample_val = LAST_SMOOTHED_SAMPLE.lock().unwrap();
        for current_sample in processed_values.iter_mut() {
            let new_sample;
            if current_sample.abs() > last_sample_val.abs() {
                new_sample = *current_sample;
            } else {
                new_sample = *current_sample * (1.0 - alpha) + *last_sample_val * alpha;
            }
            *current_sample = new_sample;
            *last_sample_val = new_sample;
        }
    }

    let mut vibration_value = *processed_values.last().unwrap_or(&0.0) as f64;
    vibration_value = f64::abs(vibration_value);

    if vibration_value < threshold {
        vibration_value = 0.0;
    }
    vibration_value *= intensity;

    let _ = state.tx.try_send(vibration_value);

    for sample in processed_values.iter_mut() {
        let sample_abs = sample.abs() as f64;
        if sample_abs < threshold {
            *sample = 0.0;
        } else {
            *sample *= intensity as f32;
        }
    }
    processed_values
}

//==================================================================================
//  VIBRATION AND NETWORKING LOGIC
//==================================================================================

async fn run_vibration_logic(settings: Arc<Mutex<AppSettings>>, audio_device: Device) -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<f64>(SAMPLE_LIMIT);

    let initial_state = AudioProcessorState {
        tx,
        settings: Arc::clone(&settings),
    };
    if AUDIO_STATE.set(initial_state).is_err() {
        panic!("AUDIO_STATE was already initialized");
    }

    let default_out_config = audio_device.default_output_config().unwrap().config();
    println!("[Audio] Using audio device: {}", audio_device.name()?);

    tokio::spawn(async move {
        open_window_connect_audio(
            "Live Audio View",
            None, None, None, None,
            "time (seconds)",
            "Amplitude (After Processing)",
            AudioDevAndCfg::new(Some(audio_device), Some(default_out_config)),
            TransformFn::Basic(audio_transform_fn),
        );
    });

    'reconnection_loop: loop {
        println!("[Buttplug] Attempting to connect to server at ws://localhost:12345...");

        let client = ButtplugClient::new("AudioToVibrations Client");
        let connector = new_json_ws_client_connector("ws://localhost:12345/buttplug");

        if let Err(e) = client.connect(connector).await {
            eprintln!("[Buttplug] Connection failed: {}. Retrying in 5 seconds...", e);
            time::sleep(Duration::from_secs(5)).await;
            continue 'reconnection_loop;
        }

        println!("[Buttplug] Connected! Scanning for devices...");
        client.start_scanning().await?;
        time::sleep(Duration::from_secs(2)).await;
        client.stop_scanning().await?;

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

        // 5. Inner operational loop.
        loop {
            // --- RE-SCAN LOGIC ---
            let refresh_needed = {
                let mut s = settings.lock().unwrap();
                if s.refresh_requested {
                    s.refresh_requested = false;
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
            if rx.recv_many(&mut collected_values, SAMPLE_LIMIT).await == 0 {
                println!("[Audio] Audio stream closed. Shutting down.");
                client.disconnect().await?;
                return Ok(());
            }

            let collected_length = collected_values.len();
            let mean_value: f64 = if collected_length > 0 {
                collected_values.iter().sum::<f64>() / collected_length as f64
            } else {
                0.0
            };

            let computed_intensity = f64::min(mean_value, 1.0);

            let (target, delay) = {
                let s = settings.lock().unwrap();
                (s.target_device.clone(), s.delay_ms)
            };

            let all_connected = client.devices();
            
            // FIX: If the list is empty (devices disconnected silently), trigger reconnection.
            if all_connected.is_empty() {
                println!("[Buttplug] Device list is empty. Triggering re-scan/reconnect.");
                break; 
            }

            let mut command_failed = false;

            for device in all_connected {
                if target == ALL_DEVICES_LABEL || device.name() == &target {
                    if let Err(e) = device.vibrate(&ScalarValueCommand::ScalarValue(computed_intensity)).await {
                        eprintln!("[Buttplug] Vibrate command failed for {}: {}.", device.name(), e);
                        command_failed = true;
                    }
                }
            }

            // FIX: If a command failed, we assume the device is gone.
            // We REMOVED the "&& !client.connected()" check because the server 
            // often stays connected even if the device drops.
            if command_failed {
                println!("[Buttplug] Command failed. Device lost? Restarting loop...");
                break;
            }

            time::sleep(Duration::from_millis(delay)).await;
        }

        println!("[Buttplug] Disconnected. Will attempt to reconnect...");
        let _ = client.disconnect().await;
    }
}

//==================================================================================
//  HELPER FUNCTIONS
//==================================================================================

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

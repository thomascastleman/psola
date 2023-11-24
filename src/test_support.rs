#![cfg(test)]

use crate::Psola;
use anyhow::Result;
use dasp::{signal, Signal};
use pitch_detection::{detector::yin::YINDetector, detector::PitchDetector, Pitch};
use std::fs;
use std::sync::Once;

pub static ONCE: Once = Once::new();

// TODO(tcastleman) Still tuning this error value.
/// The allowable error (Hz) between the actual and expected output frequency in a pure
/// sign shifting test. This accounts for both imprecision in pitch detection as well
/// as near-imperceptible differences in pitch.
const ALLOWED_ERROR_HZ: f64 = 3.0;

/// Detect the pitch of a given input signal.
pub fn detect_pitch(signal: &[f64], sample_rate: usize) -> Option<Pitch<f64>> {
    const POWER_THRESHOLD: f64 = 5.0;
    const CLARITY_THRESHOLD: f64 = 0.7;
    YINDetector::new(signal.len(), signal.len() / 2).get_pitch(
        signal,
        sample_rate,
        POWER_THRESHOLD,
        CLARITY_THRESHOLD,
    )
}

#[must_use]
pub enum RoughlyEqResult {
    Equal,
    NotEqual {
        expected: f64,
        actual: f64,
        diff: f64,
    },
}

impl RoughlyEqResult {
    pub fn unwrap(&self) {
        if let RoughlyEqResult::NotEqual { .. } = self {
            panic!("{}", self.display());
        }
    }

    fn display(&self) -> String {
        match self {
            RoughlyEqResult::Equal => "Equal".into(),
            RoughlyEqResult::NotEqual {
                expected,
                actual,
                diff,
            } => {
                format!(
                    "Not Equal\n{:<11} {}\n{:<11} {}\n{:<11} {} (> {})",
                    "Expected:", expected, "Actual:", actual, "Diff:", diff, ALLOWED_ERROR_HZ
                )
            }
        }
    }
}

/// Asserts that two values are within `ALLOWED_ERROR_HZ` of each other, printing both values
/// and their difference upon failure.
pub fn roughly_eq(expected: f64, actual: f64) -> RoughlyEqResult {
    let diff = (expected - actual).abs();
    if diff > ALLOWED_ERROR_HZ {
        RoughlyEqResult::NotEqual {
            expected,
            actual,
            diff,
        }
    } else {
        RoughlyEqResult::Equal
    }
}

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{fs::File, io::Write};

#[derive(Serialize, Deserialize)]
struct TestFailureData {
    input: Vec<f64>,
    output: Vec<f64>,
    ideal_output: Vec<f64>,
    analysis_peaks: Vec<usize>,
    synthesis_peaks: Vec<usize>,
    input_frequency: f64,
    target_frequency: f64,
    detected_output_frequency: Option<f64>,
    diff: Option<f64>,
    buffer_size: usize,
    sample_rate: usize,
}

static FAILURES_DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/generated/failures");

fn export_test_failure_information(test_failure_data: &TestFailureData, test_name: &str) {
    let serialized = serde_json::to_string(test_failure_data).unwrap();
    let failures_directory = Path::new(FAILURES_DIRECTORY);
    let output_path = failures_directory.join(test_name);

    let mut output_file = File::create(output_path).unwrap();
    output_file.write_all(serialized.as_bytes()).unwrap();
}

pub fn assert_correctly_shifts_pure_sine_wave(
    test_name: &str,
    input_frequency: f64,
    target_frequency: f64,
    buffer_size: usize,
    sample_rate: usize,
) {
    // Construct a pure sine wave at the input frequency
    let input: Vec<_> = signal::rate(sample_rate as f64)
        .const_hz(input_frequency)
        .sine()
        .take(buffer_size)
        .collect();

    let ideal_output: Vec<_> = signal::rate(sample_rate as f64)
        .const_hz(target_frequency)
        .sine()
        .take(buffer_size)
        .collect();

    // Shift to the target frequency
    let psola = Psola::new(&input, sample_rate as f32, input_frequency as f32).unwrap();
    let mut output = vec![0.0; buffer_size];
    psola.shift(target_frequency as f32, &mut output);

    let detected_output_frequency = detect_pitch(&output, sample_rate).map(|pitch| pitch.frequency);

    let synthesis_peaks = psola.calculate_synthesis_peaks(target_frequency as f32);
    let analysis_peaks = psola.analysis_peaks;

    match detected_output_frequency {
        Some(detected_output_frequency) => {
            // Compare detected pitch of shifted output with target frequency
            let eq_result = roughly_eq(target_frequency, detected_output_frequency);

            if let RoughlyEqResult::NotEqual { diff, .. } = eq_result {
                export_test_failure_information(
                    &TestFailureData {
                        analysis_peaks,
                        synthesis_peaks,
                        input: input.clone(),
                        output,
                        ideal_output,
                        input_frequency,
                        target_frequency,
                        detected_output_frequency: Some(detected_output_frequency),
                        diff: Some(diff),
                        buffer_size,
                        sample_rate,
                    },
                    test_name,
                );
            }

            // Fail test if not roughly equal
            eq_result.unwrap();
        }
        None => {
            export_test_failure_information(
                &TestFailureData {
                    analysis_peaks,
                    synthesis_peaks,
                    input: input.clone(),
                    output,
                    input_frequency,
                    ideal_output,
                    target_frequency,
                    detected_output_frequency: None,
                    diff: None,
                    buffer_size,
                    sample_rate,
                },
                test_name,
            );

            panic!("No pitch detected in PSOLA output");
        }
    };
}

pub fn clear_failures_directory() -> Result<()> {
    for entry in fs::read_dir(Path::new(FAILURES_DIRECTORY))? {
        fs::remove_file(entry?.path())?;
    }
    Ok(())
}

/// Generates a test case which checks that PSOLA correctly shifts a pure sine wave of a
/// chosen frequency to a given target frequency.
macro_rules! test_pure_sine {
    ($test_fn_name:ident, $input_frequency:expr, $target_frequency:expr, $buffer_size:expr, $sample_rate:expr) => {
        #[test]
        fn $test_fn_name() {
            // Ensure that the failures directory is cleared of previous test output before generating any more
            $crate::test_support::ONCE.call_once(|| clear_failures_directory().unwrap());

            assert_correctly_shifts_pure_sine_wave(
                stringify!($test_fn_name),
                $input_frequency,
                $target_frequency,
                $buffer_size,
                $sample_rate,
            );
        }
    };
}

pub(crate) use test_pure_sine;

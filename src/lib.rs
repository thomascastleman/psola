use anyhow::{anyhow, Result};
use dasp::Sample;
use std::collections::HashMap;

pub struct Psola<'a, S>
where
    S: Sample,
{
    audio: &'a [S],
    sample_rate: f32,
    pitch_period: usize,
    analysis_peaks: Vec<usize>,
    hann: Vec<f32>,
}

impl<'a, S> Psola<'a, S>
where
    S: Sample,
{
    pub fn new(audio: &'a [S], sample_rate: f32, frequency: f32) -> Result<Psola<'a, S>> {
        // Pitch period is the number of samples it takes for the wave to repeat itself.
        let pitch_period = Self::period(sample_rate, frequency);
        let half_period = pitch_period / 2;
        let one_and_half_period = pitch_period + half_period;

        // Pre-compute the values of the Hann window function for a whole window.
        let mut hann = vec![0.0; 2 * pitch_period + 1];
        let half_window = pitch_period as isize;
        for i in -half_window..=half_window {
            hann[(i + half_window) as usize] = Self::hann_window(i, half_window);
        }

        // We expect to see roughly (# samples in buffer) / (# samples per cycle)
        // peaks, since each cycle has one peak.
        let mut analysis_peaks = Vec::with_capacity(audio.len() / pitch_period);

        // We will move a window of width pitch_period across the samples,
        // looking for local maxima and using these as our analysis peaks.
        let mut left_bound = 0;
        let mut right_bound = pitch_period;

        loop {
            // If at end of buffer, break
            if right_bound >= audio.len() {
                break;
            }

            // Look for a local max in the window of samples [left_bound, right_bound)
            let mut local_max = left_bound;
            for i in left_bound..right_bound {
                if audio[i] > audio[local_max] {
                    local_max = i;
                }
            }

            // Add the local max as an analysis peak
            analysis_peaks.push(local_max);

            // Move along the window by a half period
            left_bound = local_max + half_period;
            right_bound = local_max + one_and_half_period;
        }

        // No peaks were found
        if analysis_peaks.is_empty() {
            Err(anyhow!("No analysis peaks found"))
        } else {
            Ok(Self {
                audio,
                pitch_period,
                sample_rate,
                hann,
                analysis_peaks,
            })
        }
    }

    /// Calculates the pitch period of a given frequency at a given sample rate. The
    /// pitch period is the number of samples in a single cycle of the signal.
    ///
    /// # Panics
    /// Panics if the given frequency is 0.
    fn period(sample_rate: f32, frequency: f32) -> usize {
        assert!(frequency != 0.0, "frequency cannot be 0");
        // samples/sec / cycles/sec --> samples/cycle
        f32::floor(sample_rate / frequency) as usize
    }

    /// The [Hann window function]. `n` is the current shift in the
    /// window, `w` is half of the window size (window ranges from
    /// `-w` to `w`).
    ///
    /// [Hann window function]: https://en.wikipedia.org/wiki/Hann_function
    fn hann_window(n: isize, w: isize) -> f32 {
        use std::f32::consts::PI;
        0.5 * (1.0 - ((2.0 * PI * (n + w) as f32) / (2 * w) as f32).cos())
    }

    /// The absolute value of the difference between pitch peaks.
    #[inline]
    fn distance(peak1: usize, peak2: usize) -> isize {
        (peak1 as isize - peak2 as isize).abs()
    }

    /// Given a target frequency to shift the audio signal to, `calculate_synthesis_peaks`
    /// returns a vector of indices into the audio buffer which represent synthesis peaks.
    /// These peaks indicate where peaks in the target audio should occur.
    ///
    /// # Panics
    /// Panics if the `Psola` is in an invalid state and has
    /// no analysis peaks.
    fn calculate_synthesis_peaks(&self, target_frequency: f32) -> Vec<usize> {
        // Find pitch period of the frequency we are trying to shift to
        let target_period = Self::period(self.sample_rate, target_frequency);

        let mut initial_peak = *self
            .analysis_peaks
            .first()
            .expect("valid analyzed audio must have >0 analysis peaks");

        // Adjust the initial synthesis peak as early as possible in the audio
        // but so that synthesis peaks still line up with the first analysis peak
        initial_peak %= target_period;

        // Number of synthesis peaks is the buffer size after the initial peak
        // divided by # of samples per cycle, then +1 for the initial peak.
        let num_synth_peaks =
            ((self.audio.len() - initial_peak) as f32 / target_period as f32).ceil() as usize;
        let mut synthesis_peaks = Vec::with_capacity(num_synth_peaks);

        // Add synthesis peaks every multiple of the target period,
        // starting at the initially chosen peak.
        for i in 0..num_synth_peaks {
            synthesis_peaks.push(initial_peak + (i * target_period));
        }

        synthesis_peaks
    }

    /// Given the synthesis peaks of the target frequency, `map_synthesis_to_analysis`
    /// constructs a mapping from synthesis peaks to analysis peaks from which
    /// to get the actual audio that will be used in each synthesis peak position.
    fn map_synthesis_to_analysis(&self, synthesis_peaks: &[usize]) -> HashMap<usize, usize> {
        // Each synthesis peak will be mapped
        let mut map = HashMap::with_capacity(synthesis_peaks.len());
        let mut previous_closest = 0;

        for &synth_peak in synthesis_peaks {
            // Assume the closest analysis peak to this synthesis peak
            // is the one that was closest to the previous synthesis peak
            let mut closest: usize = previous_closest;

            for i in previous_closest..(self.analysis_peaks.len()) {
                // If this peak is closer to the current synthesis peak,
                // update the closest.
                if Self::distance(self.analysis_peaks[i], synth_peak)
                    < Self::distance(self.analysis_peaks[closest], synth_peak)
                {
                    closest = i;
                }

                // Once we have past the synthesis peak, do not look any further
                // for close analysis peaks, as they only get further away.
                if self.analysis_peaks[i] >= synth_peak {
                    break;
                }
            }

            // Map this synthesis peak to its closest analysis peak
            map.insert(synth_peak, self.analysis_peaks[closest]);

            // Record the closest analysis peak from this round
            previous_closest = closest;
        }

        map
    }

    /// Given synthesis peaks, a mapping from synthesis to analysis peaks,
    /// and an output buffer to write samples to, `overlap_and_add` constructs
    /// audio data in the output buffer by copying the audio from the
    /// analysis peaks in the input into the positions of the synthesis peaks
    /// in the output. The copied audio is tapered around the edges by a window
    /// function, so they transition smoothly.
    ///
    /// # Panics
    /// This panics if the given output buffer does not match the size
    /// of the audio buffer of this `AnalyzedAudio`, or if the given
    /// `synthesis_peaks` contains a peak that lacks an entry in the
    /// `synthesis_to_analysis` map.
    fn overlap_and_add(
        &self,
        synthesis_peaks: &[usize],
        synthesis_to_analysis: HashMap<usize, usize>,
        out: &mut [S],
    ) {
        assert!(out.len() == self.audio.len());
        let half_window = self.pitch_period as isize;

        for &synth_peak in synthesis_peaks {
            // Get the corresponding analysis peak, whose audio will be
            // to create the artifical synthesis peak.
            let &analysis_peak = synthesis_to_analysis
                .get(&synth_peak)
                .expect("synthesis peak was unmapped");

            // For each sample in the window
            for shift in -half_window..=half_window {
                // s: index of current sample in window relative to synthesis peak
                // a: same, but for corresponding analysis peak
                let s = (synth_peak as isize + shift) as usize;
                let a = (analysis_peak as isize + shift) as usize;

                if s > 0 && a > 0 && s < self.audio.len() && a < self.audio.len() {
                    // Fill in the analysis peak's audio in the output buffer at
                    // the position of the synthesis peak, scaled by the window function.
                    let hann_scaler =
                        Sample::from_sample(self.hann[(shift + half_window) as usize]);
                    let scaled_audio = self.audio[a].mul_amp(hann_scaler);
                    out[s] = out[s].add_amp(scaled_audio.to_signed_sample());
                }
            }
        }
    }

    /// Given a target frequency to shift the analyzed audio to, `shift`
    /// writes pitch shifted audio into the given buffer.
    ///
    /// # Panics
    /// Panics if the given target frequency is 0, or if the given
    /// output buffer does not have the same size as the input audio buffer, or
    /// if the `Psola` is in an invalid state and has no analysis peaks
    pub fn shift(&self, target_frequency: f32, output: &mut [S]) {
        assert!(output.len() == self.audio.len());
        assert!(target_frequency != 0.0);
        assert!(!self.analysis_peaks.is_empty());
        let synthesis_peaks = self.calculate_synthesis_peaks(target_frequency);
        let map = self.map_synthesis_to_analysis(&synthesis_peaks);
        self.overlap_and_add(&synthesis_peaks, map, output);
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use dasp::{signal, Signal};
    use pitch_detection::{
        detector::{yin::YINDetector, PitchDetector},
        Pitch,
    };

    #[test]
    fn small_buffer_same_source_and_target_frequency() {
        let audio = [
            0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0,
        ];
        let psola = Psola::new(&audio, 1.0, 0.25).unwrap();

        let mut output = vec![0.0; audio.len()];
        psola.shift(0.25, &mut output);

        // NOTE: The -0.5 on the left is due to the tapering of the Hann window, and
        // there is no peak to the left of the initial analysis peak which can contribute
        // audio.
        //
        // But maybe there should be?.. Should we allow synthesis peaks to extend beyond the
        // bounds of the buffer, as long as *some* portion of their windowed value would
        // contribute to the inside region?
        assert_eq!(
            output,
            vec![0.0, -0.5, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0,]
        );
    }

    #[test]
    /// Checks that our signal generation and pitch detection mechanisms function correctly.
    fn pitch_detection_sanity_check() {
        const FREQUENCY: f64 = 440.0;
        const BUFFER_SIZE: usize = 1024;
        const SAMPLE_RATE: usize = 44_100;
        let signal = signal::rate(SAMPLE_RATE as f64).const_hz(FREQUENCY).sine();
        let buffer: Vec<f64> = signal.take(BUFFER_SIZE).collect();
        let detected_pitch = detect_pitch(&buffer, SAMPLE_RATE).unwrap();
        roughly_eq(FREQUENCY, detected_pitch.frequency).unwrap();
    }

    // TODO(tcastleman) Test that works with other sample types

    // TODO(tcastleman) Still tuning this error value.
    /// The allowable error (Hz) between the actual and expected output frequency in a pure
    /// sign shifting test. This accounts for both imprecision in pitch detection as well
    /// as near-imperceptible differences in pitch.
    const ALLOWED_ERROR_HZ: f64 = 3.0;

    /// Detect the pitch of a given input signal.
    fn detect_pitch(signal: &[f64], sample_rate: usize) -> Option<Pitch<f64>> {
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
    enum RoughlyEqResult {
        Equal,
        NotEqual {
            expected: f64,
            actual: f64,
            diff: f64,
        },
    }

    impl RoughlyEqResult {
        fn unwrap(&self) {
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
    fn roughly_eq(expected: f64, actual: f64) -> RoughlyEqResult {
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

    fn export_test_failure_information(
        psola: Psola<'_, f64>,
        test_name: &str,
        input: Vec<f64>,
        output: Vec<f64>,
        input_frequency: f64,
        target_frequency: f64,
        detected_output_frequency: Option<f64>,
        diff: Option<f64>,
        buffer_size: usize,
        sample_rate: usize,
    ) {
        let synthesis_peaks = psola.calculate_synthesis_peaks(target_frequency as f32);
        let analysis_peaks = psola.analysis_peaks;

        let failure_data = TestFailureData {
            input,
            output,
            analysis_peaks,
            synthesis_peaks,
            input_frequency,
            target_frequency,
            detected_output_frequency,
            diff,
            buffer_size,
            sample_rate,
        };

        let serialized = serde_json::to_string(&failure_data).unwrap();
        let failures_directory = Path::new(FAILURES_DIRECTORY);
        let output_path = failures_directory.join(test_name);

        let mut output_file = File::create(output_path).unwrap();
        output_file.write_all(serialized.as_bytes()).unwrap();
    }

    fn assert_correctly_shifts_pure_sine_wave(
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

        // Shift to the target frequency
        let psola = Psola::new(&input, sample_rate as f32, input_frequency as f32).unwrap();
        let mut output = vec![0.0; buffer_size];
        psola.shift(target_frequency as f32, &mut output);

        let detected_output_frequency =
            detect_pitch(&output, sample_rate).map(|pitch| pitch.frequency);

        match detected_output_frequency {
            Some(detected_output_frequency) => {
                // Compare detected pitch of shifted output with target frequency
                let eq_result = roughly_eq(target_frequency, detected_output_frequency);

                if let RoughlyEqResult::NotEqual { diff, .. } = eq_result {
                    export_test_failure_information(
                        psola,
                        test_name,
                        input.clone(),
                        output,
                        input_frequency,
                        target_frequency,
                        Some(detected_output_frequency),
                        Some(diff),
                        buffer_size,
                        sample_rate,
                    );
                }

                // Fail test if not roughly equal
                eq_result.unwrap();
            }
            None => {
                export_test_failure_information(
                    psola,
                    test_name,
                    input.clone(),
                    output,
                    input_frequency,
                    target_frequency,
                    None,
                    None,
                    buffer_size,
                    sample_rate,
                );

                panic!("No pitch detected in PSOLA output");
            }
        };
    }

    use std::fs;
    use std::sync::Once;
    static ONCE: Once = Once::new();

    fn clear_failures_directory() -> Result<()> {
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
                ONCE.call_once(|| clear_failures_directory().unwrap());

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

    /// Module for automatically generated test cases. To re-generate, run
    /// ```no_run
    /// ./scripts/generate_tests.py >./generated/test_cases.rs
    /// ```
    mod generated {
        use super::*;

        include!("../generated/test_cases.rs");
    }
}

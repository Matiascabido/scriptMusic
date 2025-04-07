import { createWriteStream } from 'fs';

// audio-classifier.ts
import fs from 'fs';
import path from 'path';
import ffmpeg from 'fluent-ffmpeg';
import { decode } from 'node-wav';
import Meyda from 'meyda';

const INPUT_DIR = './incoming';
const REFERENCE_DIR = './reference';
const OUTPUT_DIR = './classified';
const TEMP_DIR = './converted';
const logStream = createWriteStream('log.txt', { flags: 'a' });

function log(msg: string) {
    const timestamp = new Date().toISOString();
    logStream.write(`[${timestamp}] ${msg}\n`);
    console.log(msg);
}

// Ensure directories exist
for (const dir of [OUTPUT_DIR, TEMP_DIR]) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir);
}

function convertToWav(inputPath: string, outputPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .toFormat('wav')
      .on('end', () => resolve())
      .on('error', reject)
      .save(outputPath);
  });
}

async function extractFeatures(wavPath: string) {
  const buffer = fs.readFileSync(wavPath);
  const decoded = await decode(buffer);
  const signal = decoded.channelData[0];

  const meydaFeatures = Meyda.extract(['mfcc', 'spectralCentroid'], signal);
  const peaks = getPeaks(signal);
  const bpm = getBPMFromPeaks(peaks);

  return { mfcc: meydaFeatures?.mfcc || [], centroid: meydaFeatures?.spectralCentroid || 0, peaks, bpm };
}

function getPeaks(signal: Float32Array): number[] {
  const threshold = 0.3;
  const peaks = [];
  for (let i = 1; i < signal.length - 1; i++) {
    if (signal[i] > threshold && signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
      peaks.push(i);
    }
  }
  return peaks;
}

function getBPMFromPeaks(peaks: number[]): number {
  const intervals = [];
  for (let i = 1; i < peaks.length; i++) {
    intervals.push(peaks[i] - peaks[i - 1]);
  }
  const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  const sampleRate = 44100; // suposición estándar
  return Math.round(60 / (avgInterval / sampleRate));
}

function compareFeatures(a: any, b: any): number {
  const mfccDist = euclideanDistance(a.mfcc, b.mfcc);
  const peakMatch = 1 - Math.abs(a.peaks.length - b.peaks.length) / Math.max(a.peaks.length, b.peaks.length);
  const bpmMatch = Math.abs(a.bpm - b.bpm) <= a.bpm * 0.05 ? 1 : 0;
  return (1 - mfccDist / 100) * 0.6 + peakMatch * 0.3 + bpmMatch * 0.1;
}

function euclideanDistance(a: number[], b: number[]): number {
  const length = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

async function process() {
  const referenceFiles = fs.readdirSync(REFERENCE_DIR).filter(f => f.endsWith('.wav'));
  const referenceFeatures = await Promise.all(referenceFiles.map(f => extractFeatures(path.join(REFERENCE_DIR, f))));

  const inputFiles = fs.readdirSync(INPUT_DIR);
  for (const file of inputFiles) {
    const fullInputPath = path.join(INPUT_DIR, file);
    const wavPath = path.join(TEMP_DIR, path.parse(file).name + '.wav');

    await convertToWav(fullInputPath, wavPath);
    const features = await extractFeatures(wavPath);

    const scores = referenceFeatures.map(ref => compareFeatures(features, ref));
    const bestScore = Math.max(...scores);

    let category = 'other';
    if (bestScore > 0.9) category = 'very_similar';
    else if (bestScore > 0.7) category = 'similar';
    else if (bestScore > 0.5) category = 'somewhat_similar';

    const outputPath = path.join(OUTPUT_DIR, category);
    if (!fs.existsSync(outputPath)) fs.mkdirSync(outputPath, { recursive: true });
    fs.copyFileSync(fullInputPath, path.join(outputPath, file));

    log(`${file} → ${category} (score: ${bestScore.toFixed(2)})`);
  }
}

process().catch(error => log(error));
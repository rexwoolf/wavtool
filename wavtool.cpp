// WavTool — a tiny single-file C++17 audio utility for WAV files
// ---------------------------------------------------------------
// Features:
//   • info: print WAV metadata
//   • gain <dB>: apply linear gain in dB
//   • normalize: peak-normalize to -1 dBFS
//   • fade <in_ms> <out_ms>: apply linear fade in/out
//   • lowpass <cutoff_hz> [Q]: simple biquad LPF (default Q=0.707)
//   • synth-sine <freq_hz> <seconds> [samplerate] [amplitude 0..1]: generate a sine wave
//
// Supported formats: PCM 16-bit, mono or stereo, little-endian .wav
//
// Build:  g++ -std=gnu++17 -O2 -o wavtool wavtool.cpp
// Usage examples:
//   ./wavtool info input.wav
//   ./wavtool gain -6 input.wav output.wav
//   ./wavtool normalize input.wav output.wav
//   ./wavtool fade 50 300 input.wav output.wav
//   ./wavtool lowpass 8000 input.wav output.wav       # Q defaults to 0.707
//   ./wavtool lowpass 1200 0.5 input.wav output.wav
//   ./wavtool synth-sine 440 2.5 48000 0.5 out.wav
//
// Copyright (c) 2025 Rex Woolf (MIT License)
// ---------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace std;

// ---------- Utility ----------
static inline int16_t clamp16(int x){
    if(x>32767) return 32767; if(x<-32768) return -32768; return (int16_t)x;
}

static inline float db_to_linear(float dB){ return powf(10.0f, dB/20.0f); }

// ---------- WAV Structures ----------
struct WavHeader {
    // RIFF header
    char riff[4];       // "RIFF"
    uint32_t chunkSize; // 36 + Subchunk2Size
    char wave[4];       // "WAVE"
    // fmt subchunk
    char fmt[4];        // "fmt "
    uint32_t subchunk1Size; // 16 for PCM
    uint16_t audioFormat;   // 1 = PCM
    uint16_t numChannels;   // 1 or 2 supported here
    uint32_t sampleRate;    // e.g., 44100
    uint32_t byteRate;      // sampleRate * numChannels * bitsPerSample/8
    uint16_t blockAlign;    // numChannels * bitsPerSample/8
    uint16_t bitsPerSample; // 16 supported here
    // data subchunk
    char data[4];       // "data"
    uint32_t subchunk2Size; // numSamples * numChannels * bitsPerSample/8
};

struct AudioBuffer{
    uint32_t sampleRate = 0;
    uint16_t channels = 0; // 1 or 2
    vector<int16_t> interleaved; // PCM 16-bit interleaved
};

static bool read_wav(const string& path, AudioBuffer& out, string& err){
    ifstream f(path, ios::binary);
    if(!f){ err = "Cannot open file: " + path; return false; }

    // Read RIFF header
    WavHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(WavHeader));
    if(!f){ err = "Failed to read WAV header (" + path + ")"; return false; }

    if(strncmp(h.riff, "RIFF", 4)!=0 || strncmp(h.wave, "WAVE", 4)!=0){
        err = "Not a RIFF/WAVE file."; return false;
    }

    // The above struct assumes canonical layout (fmt then data). Some WAVs have extra chunks.
    // Robust parse: rewind after RIFF/WAVE and scan chunks.
    f.clear();
    f.seekg(12, ios::beg); // after RIFF/WAVE

    uint16_t audioFormat=0, numChannels=0, bitsPerSample=0; uint32_t sampleRate=0;
    uint32_t dataSize=0; streampos dataPos = -1;

    while(f && !f.eof()){
        char id[4]; uint32_t size=0; f.read(id,4); f.read(reinterpret_cast<char*>(&size),4);
        if(!f) break;
        string sid(id,4);
        if(sid=="fmt "){
            vector<char> buf(size); f.read(buf.data(), size);
            if(size < 16){ err = "Invalid fmt chunk"; return false; }
            audioFormat   = *reinterpret_cast<uint16_t*>(&buf[0]);
            numChannels   = *reinterpret_cast<uint16_t*>(&buf[2]);
            sampleRate    = *reinterpret_cast<uint32_t*>(&buf[4]);
            // uint32_t byteRate = *reinterpret_cast<uint32_t*>(&buf[8]);
            uint16_t blockAlign = *reinterpret_cast<uint16_t*>(&buf[12]);
            bitsPerSample = *reinterpret_cast<uint16_t*>(&buf[14]);
            (void)blockAlign;
        } else if(sid=="data"){
            dataPos = f.tellg();
            dataSize = size;
            f.seekg(size, ios::cur);
        } else {
            // skip unknown chunk
            f.seekg(size, ios::cur);
        }
        // Chunks are word-aligned
        if(size % 2 == 1) f.seekg(1, ios::cur);
    }

    if(audioFormat!=1){ err = "Only PCM (format 1) is supported."; return false; }
    if(bitsPerSample!=16){ err = "Only 16-bit PCM is supported."; return false; }
    if(numChannels!=1 && numChannels!=2){ err = "Only mono or stereo supported."; return false; }
    if(dataPos == streampos(-1)){ err = "Missing data chunk."; return false; }

    // Read samples
    out.sampleRate = sampleRate;
    out.channels = numChannels;
    out.interleaved.resize(dataSize/2);
    ifstream f2(path, ios::binary); f2.seekg(dataPos);
    f2.read(reinterpret_cast<char*>(out.interleaved.data()), dataSize);
    if(!f2){ err = "Failed reading sample data."; return false; }
    return true;
}

static bool write_wav(const string& path, const AudioBuffer& in, string& err){
    ofstream f(path, ios::binary);
    if(!f){ err = "Cannot open for write: " + path; return false; }
    uint32_t dataSize = (uint32_t)(in.interleaved.size()*sizeof(int16_t));

    // Prepare canonical header
    f.write("RIFF",4);
    uint32_t chunkSize = 36 + dataSize; f.write(reinterpret_cast<char*>(&chunkSize),4);
    f.write("WAVE",4);

    // fmt
    f.write("fmt ",4);
    uint32_t sc1 = 16; f.write(reinterpret_cast<char*>(&sc1),4);
    uint16_t fmt = 1; f.write(reinterpret_cast<char*>(&fmt),2);
    f.write(reinterpret_cast<const char*>(&in.channels),2);
    f.write(reinterpret_cast<const char*>(&in.sampleRate),4);
    uint32_t byteRate = in.sampleRate * in.channels * 2; f.write(reinterpret_cast<char*>(&byteRate),4);
    uint16_t blockAlign = in.channels * 2; f.write(reinterpret_cast<char*>(&blockAlign),2);
    uint16_t bps = 16; f.write(reinterpret_cast<char*>(&bps),2);

    // data
    f.write("data",4);
    f.write(reinterpret_cast<char*>(&dataSize),4);
    f.write(reinterpret_cast<const char*>(in.interleaved.data()), dataSize);

    if(!f){ err = "Write error."; return false; }
    return true;
}

// ---------- DSP helpers ----------
struct Biquad {
    // Direct Form I
    float a0=1, a1=0, a2=0, b1=0, b2=0; // normalized so a0 is used directly
    float z1L=0, z2L=0, z1R=0, z2R=0;    // state per channel

    void setLowpass(float sr, float cutoff, float Q){
        cutoff = max(10.0f, min(cutoff, sr*0.45f));
        float w0 = 2.0f * float(M_PI) * (cutoff / sr);
        float alpha = sinf(w0)/(2.0f*Q);
        float cosw0 = cosf(w0);
        float b0 = (1 - cosw0)/2;
        float b1n = 1 - cosw0;
        float b2 = (1 - cosw0)/2;
        float a0n = 1 + alpha;
        float a1n = -2*cosw0;
        float a2n = 1 - alpha;
        // normalize
        a0 = b0/a0n; a1 = b1n/a0n; a2 = b2/a0n; b1 = a1n/a0n; b2 = a2n/a0n;
        // reset state optional
        z1L=z2L=z1R=z2R=0;
    }

    inline float processSample(float x, bool right){
        float& z1 = right? z1R : z1L;
        float& z2 = right? z2R : z2L;
        float y = a0*x + z1;
        z1 = a1*x - b1*y + z2;
        z2 = a2*x - b2*y;
        return y;
    }
};

// ---------- Operations ----------
static void apply_gain(AudioBuffer& buf, float dB){
    float g = db_to_linear(dB);
    for(auto &s : buf.interleaved){
        int v = (int)lroundf((float)s * g);
        s = clamp16(v);
    }
}

static void normalize_peak(AudioBuffer& buf, float targetDbFS=-1.0f){
    int16_t peak = 0;
    for(auto s: buf.interleaved) peak = max<int16_t>(peak, (int16_t)abs((int)s));
    if(peak==0) return;
    float target = db_to_linear(targetDbFS) * 32767.0f;
    float g = target / (float)peak;
    if(g <= 0) return;
    for(auto &s: buf.interleaved){ s = clamp16((int)lroundf((float)s*g)); }
}

static void fade(AudioBuffer& buf, float in_ms, float out_ms){
    size_t frames = buf.interleaved.size() / buf.channels;
    size_t in_samps = (size_t) ((in_ms/1000.0f) * buf.sampleRate);
    size_t out_samps = (size_t) ((out_ms/1000.0f) * buf.sampleRate);

    for(size_t n=0; n<frames; ++n){
        float g = 1.0f;
        if(in_samps>0 && n<in_samps){ g = (float)n / (float)in_samps; }
        if(out_samps>0 && n>=frames-out_samps){
            float t = (float)(frames - n) / (float)out_samps;
            g = min(g, max(0.0f, t));
        }
        for(int c=0;c<buf.channels;++c){
            size_t idx = n*buf.channels + c;
            buf.interleaved[idx] = clamp16((int)lroundf((float)buf.interleaved[idx]*g));
        }
    }
}

static void lowpass(AudioBuffer& buf, float cutoff, float Q){
    Biquad biq; biq.setLowpass((float)buf.sampleRate, cutoff, Q);
    size_t frames = buf.interleaved.size() / buf.channels;
    for(size_t n=0;n<frames;++n){
        // process L
        float xL = (float)buf.interleaved[n*buf.channels+0] / 32768.0f;
        float yL = biq.processSample(xL, false);
        buf.interleaved[n*buf.channels+0] = clamp16((int)lroundf(yL*32767.0f));
        if(buf.channels==2){
            float xR = (float)buf.interleaved[n*buf.channels+1] / 32768.0f;
            float yR = biq.processSample(xR, true);
            buf.interleaved[n*buf.channels+1] = clamp16((int)lroundf(yR*32767.0f));
        }
    }
}

static AudioBuffer synth_sine(float freq, float seconds, uint32_t sr=44100, float amp=0.8f){
    AudioBuffer buf; buf.sampleRate=sr; buf.channels=1;
    size_t frames = (size_t) llround(seconds * sr);
    buf.interleaved.resize(frames);
    double phase=0.0, dphi = 2.0*M_PI*freq/(double)sr;
    for(size_t n=0;n<frames;++n){
        float s = sin(phase) * amp;
        phase += dphi;
        buf.interleaved[n] = clamp16((int)lroundf(s*32767.0f));
    }
    return buf;
}

// ---------- CLI ----------
static void print_info(const AudioBuffer& b){
    size_t frames = b.interleaved.size()/b.channels;
    cout << "Sample rate: " << b.sampleRate << " Hz\n";
    cout << "Channels   : " << b.channels << "\n";
    cout << "Format     : PCM 16-bit" << "\n";
    cout << "Frames     : " << frames << "\n";
    cout << "Duration   : " << fixed << setprecision(3)
         << (double)frames / (double)b.sampleRate << " s\n";
}

int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if(argc < 2){
        cerr << "WavTool — simple WAV utility\n";
        cerr << "Usage:\n"
             << "  wavtool info <in.wav>\n"
             << "  wavtool gain <dB> <in.wav> <out.wav>\n"
             << "  wavtool normalize <in.wav> <out.wav>\n"
             << "  wavtool fade <in_ms> <out_ms> <in.wav> <out.wav>\n"
             << "  wavtool lowpass <cutoff_hz> [Q] <in.wav> <out.wav>\n"
             << "  wavtool synth-sine <freq_hz> <seconds> [sr] [amp 0..1] <out.wav>\n";
        return 1;
    }

    string cmd = argv[1];
    string err;

    try{
        if(cmd=="info"){
            if(argc!=3){ cerr << "info requires <in.wav>\n"; return 1; }
            AudioBuffer b; if(!read_wav(argv[2], b, err)){ cerr << err << "\n"; return 1; }
            print_info(b);
            return 0;
        }
        else if(cmd=="gain"){
            if(argc!=5){ cerr << "gain requires <dB> <in.wav> <out.wav>\n"; return 1; }
            float dB = stof(argv[2]);
            AudioBuffer b; if(!read_wav(argv[3], b, err)){ cerr << err << "\n"; return 1; }
            apply_gain(b, dB);
            if(!write_wav(argv[4], b, err)){ cerr << err << "\n"; return 1; }
            return 0;
        }
        else if(cmd=="normalize"){
            if(argc!=4 && argc!=5){ /* keep old note, but enforce 4 */ }
            if(argc!=4){ cerr << "normalize requires <in.wav> <out.wav>\n"; return 1; }
            AudioBuffer b; if(!read_wav(argv[2], b, err)){ cerr << err << "\n"; return 1; }
            normalize_peak(b, -1.0f);
            if(!write_wav(argv[3], b, err)){ cerr << err << "\n"; return 1; }
            return 0;
        }
        else if(cmd=="fade"){
            if(argc!=6){ cerr << "fade requires <in_ms> <out_ms> <in.wav> <out.wav>\n"; return 1; }
            float in_ms = stof(argv[2]);
            float out_ms = stof(argv[3]);
            AudioBuffer b; if(!read_wav(argv[4], b, err)){ cerr << err << "\n"; return 1; }
            fade(b, in_ms, out_ms);
            if(!write_wav(argv[5], b, err)){ cerr << err << "\n"; return 1; }
            return 0;
        }
        else if(cmd=="lowpass"){
            if(argc!=5 && argc!=6 && argc!=7){ cerr << "lowpass <cutoff_hz> [Q] <in.wav> <out.wav>\n"; return 1; }
            float cutoff = stof(argv[2]);
            float Q = 0.707f; int argi = 3;
            if(argc==7){ Q = stof(argv[3]); argi = 4; }
            else if(argc==6){ // could be either Q provided or not
                // Heuristic: if argv[3] ends with .wav, then no Q
                string s3 = argv[3];
                bool looksNum = !s3.empty() && (isdigit(s3[0]) || s3[0]=='0' || s3[0]=='.');
                if(looksNum) { Q = stof(argv[3]); argi = 4; }
            }
            string inpath = argv[argi]; string outpath = argv[argi+1];
            AudioBuffer b; if(!read_wav(inpath, b, err)){ cerr << err << "\n"; return 1; }
            lowpass(b, cutoff, Q);
            if(!write_wav(outpath, b, err)){ cerr << err << "\n"; return 1; }
            return 0;
        }
        else if(cmd=="synth-sine"){
            if(argc<5 || argc>7){ cerr << "synth-sine <freq_hz> <seconds> [sr] [amp] <out.wav>\n"; return 1; }
            float freq = stof(argv[2]);
            float secs = stof(argv[3]);
            uint32_t sr = 44100; float amp = 0.8f; string outpath;
            if(argc==6){ sr = (uint32_t)stoul(argv[4]); outpath = argv[5]; }
            else if(argc==7){ sr = (uint32_t)stoul(argv[4]); amp = stof(argv[5]); outpath = argv[6]; }
            else { outpath = argv[4]; }
            AudioBuffer b = synth_sine(freq, secs, sr, amp);
            if(!write_wav(outpath, b, err)){ cerr << err << "\n"; return 1; }
            return 0;
        }
        else {
            cerr << "Unknown command: " << cmd << "\n";
            return 1;
        }
    } catch(const exception& e){
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

// ---------------------------------------------------------------
//
// TODO:
//   • Support 24-bit and floating-point WAV
//   • Add highpass/shelving/peaking filters and a simple EQ
//   • Implement DC offset removal and dither
//   • Add simple delay/chorus/reverb with comb/allpass
//   • Add RMS/LUFS meter and loudness normalization
//   • Add a tiny unit-test file and CI with GitHub Actions
//   • Port to a header-only library + minimal CLI
//   • Create a JUCE-based GUI wrapper later as a separate repo

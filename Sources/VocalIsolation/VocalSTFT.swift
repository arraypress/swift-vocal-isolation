//
//  VocalSTFT.swift
//  VocalIsolation
//
//  Created by David Sherlock on 2026.
//

import Accelerate
import Foundation

/// The STFT/ISTFT that bracket the MDX-Net core, reproduced in vDSP to match
/// the model's PyTorch `STFT` class bit-for-bit.
///
/// Contract: `n_fft = 8192`, `hop = 1024` (8× overlap), periodic Hann,
/// `center = true` (reflect-pad `n_fft/2` each side, trimmed on inverse). The
/// spectrogram is `[4, dimF, frames]` — channels `[Lᵣ, Lᵢ, Rᵣ, Rᵢ]` — with the
/// Nyquist bin dropped (`dimF = 4096`, not 4097), exactly like the model.
final class VocalSTFT {

    let nFFT = 8192
    let hop = 1024
    let dimF = 4096          // truncated; the model drops bin 4096 (Nyquist)
    private let pad: Int      // n_fft / 2, the center pad
    private let window: [Float]
    private let fwd: vDSP_DFT_Setup
    private let inv: vDSP_DFT_Setup

    init?() {
        pad = nFFT / 2
        // Periodic Hann: 0.5 − 0.5·cos(2πn/N).
        var w = [Float](repeating: 0, count: nFFT)
        for n in 0..<nFFT { w[n] = 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(nFFT)) }
        window = w
        guard let f = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD),
              let i = vDSP_DFT_zop_CreateSetup(f, vDSP_Length(nFFT), .INVERSE) else { return nil }
        fwd = f; inv = i
    }

    deinit { vDSP_DFT_DestroySetup(inv); vDSP_DFT_DestroySetup(fwd) }

    func frameCount(forSamples t: Int) -> Int { 1 + t / hop }

    // MARK: - Forward: stereo audio → spectrogram [4, dimF, frames]

    /// `channels` is `[left, right]`, each `count` samples (a multiple of `hop`).
    func forward(_ channels: [[Float]]) -> (spec: [Float], frames: Int) {
        let t = channels[0].count
        let frames = frameCount(forSamples: t)
        var spec = [Float](repeating: 0, count: 4 * dimF * frames)
        var frameReal = [Float](repeating: 0, count: nFFT)
        let frameImag = [Float](repeating: 0, count: nFFT)
        var outR = [Float](repeating: 0, count: nFFT)
        var outI = [Float](repeating: 0, count: nFFT)

        for ch in 0..<2 {
            let padded = reflectPadded(channels[ch])
            let reBase = (2 * ch) * dimF * frames        // channel Lᵣ/Rᵣ
            let imBase = (2 * ch + 1) * dimF * frames     // channel Lᵢ/Rᵢ
            window.withUnsafeBufferPointer { win in
                padded.withUnsafeBufferPointer { pp in
                    for i in 0..<frames {
                        vDSP_vmul(pp.baseAddress! + i * hop, 1, win.baseAddress!, 1, &frameReal, 1, vDSP_Length(nFFT))
                        vDSP_DFT_Execute(fwd, frameReal, frameImag, &outR, &outI)
                        for f in 0..<dimF {
                            spec[reBase + f * frames + i] = outR[f]
                            spec[imBase + f * frames + i] = outI[f]
                        }
                    }
                }
            }
        }
        return (spec, frames)
    }

    // MARK: - Inverse: spectrogram [4, dimF, frames] → stereo audio

    func inverse(_ spec: [Float], frames: Int) -> [[Float]] {
        let t = (frames - 1) * hop                       // original length (hop-multiple)
        let paddedLen = (frames - 1) * hop + nFFT
        let scale = 1 / Float(nFFT)
        var out = [[Float]](repeating: [Float](repeating: 0, count: t), count: 2)
        var fullR = [Float](repeating: 0, count: nFFT)
        var fullI = [Float](repeating: 0, count: nFFT)
        var timeR = [Float](repeating: 0, count: nFFT)
        var timeI = [Float](repeating: 0, count: nFFT)

        for ch in 0..<2 {
            let reBase = (2 * ch) * dimF * frames
            let imBase = (2 * ch + 1) * dimF * frames
            var recon = [Float](repeating: 0, count: paddedLen)
            var wenv = [Float](repeating: 0, count: paddedLen)
            for i in 0..<frames {
                for f in 0..<dimF { fullR[f] = spec[reBase + f * frames + i]; fullI[f] = spec[imBase + f * frames + i] }
                fullR[dimF] = 0; fullI[dimF] = 0          // Nyquist bin (dropped by the model)
                for k in (dimF + 1)..<nFFT { fullR[k] = fullR[nFFT - k]; fullI[k] = -fullI[nFFT - k] } // Hermitian
                vDSP_DFT_Execute(inv, fullR, fullI, &timeR, &timeI)
                let off = i * hop
                for n in 0..<nFFT {
                    let w = window[n]
                    recon[off + n] += timeR[n] * scale * w
                    wenv[off + n] += w * w
                }
            }
            // torch.istft normalizes by the squared-window envelope, then trims
            // the center padding.
            for n in 0..<t {
                let e = wenv[pad + n]
                out[ch][n] = e > 1e-8 ? recon[pad + n] / e : 0
            }
        }
        return out
    }

    // MARK: - Reflect padding (torch `center=True` default), pad = n_fft/2

    private func reflectPadded(_ x: [Float]) -> [Float] {
        let n = x.count
        var out = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<pad { out[i] = x[pad - i] }                 // left: x[pad]…x[1]
        for i in 0..<n { out[pad + i] = x[i] }
        for j in 0..<pad { out[pad + n + j] = x[n - 2 - j] }     // right: x[n-2]…x[n-1-pad]
        return out
    }
}

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
///
/// The hot paths (spectrogram gather/scatter, overlap-add) are vectorized with
/// BLAS/vDSP; the Hermitian mirror stays a scalar loop (cheap, and negative
/// BLAS strides alias the buffer).
final class VocalSTFT {

    let nFFT = 8192
    let hop = 1024
    let dimF = 4096          // truncated; the model drops bin 4096 (Nyquist)
    private let pad: Int      // n_fft / 2, the center pad
    private let window: [Float]
    private let scaledWindow: [Float]   // window / n_fft — for the inverse DFT scale
    private let windowSq: [Float]       // window² — the overlap normalization
    private let fwd: vDSP_DFT_Setup
    private let inv: vDSP_DFT_Setup

    init?() {
        pad = nFFT / 2
        var w = [Float](repeating: 0, count: nFFT)
        for n in 0..<nFFT { w[n] = 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(nFFT)) }
        window = w
        let invN = 1 / Float(nFFT)
        scaledWindow = w.map { $0 * invN }
        windowSq = w.map { $0 * $0 }
        guard let f = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD),
              let i = vDSP_DFT_zop_CreateSetup(f, vDSP_Length(nFFT), .INVERSE) else { return nil }
        fwd = f; inv = i
    }

    deinit { vDSP_DFT_DestroySetup(inv); vDSP_DFT_DestroySetup(fwd) }

    func frameCount(forSamples t: Int) -> Int { 1 + t / hop }

    // MARK: - Forward: stereo audio → spectrogram [4, dimF, frames]

    func forward(_ channels: [[Float]]) -> (spec: [Float], frames: Int) {
        let t = channels[0].count
        let frames = frameCount(forSamples: t)
        var spec = [Float](repeating: 0, count: 4 * dimF * frames)
        var frameReal = [Float](repeating: 0, count: nFFT)
        let frameImag = [Float](repeating: 0, count: nFFT)
        var outR = [Float](repeating: 0, count: nFFT)
        var outI = [Float](repeating: 0, count: nFFT)

        spec.withUnsafeMutableBufferPointer { sp in
            for ch in 0..<2 {
                let padded = reflectPadded(channels[ch])
                let reBase = (2 * ch) * dimF * frames
                let imBase = (2 * ch + 1) * dimF * frames
                window.withUnsafeBufferPointer { win in
                    padded.withUnsafeBufferPointer { pp in
                        for i in 0..<frames {
                            vDSP_vmul(pp.baseAddress! + i * hop, 1, win.baseAddress!, 1, &frameReal, 1, vDSP_Length(nFFT))
                            vDSP_DFT_Execute(fwd, frameReal, frameImag, &outR, &outI)
                            // scatter bins into the strided spectrogram (stride = frames)
                            cblas_scopy(Int32(dimF), outR, 1, sp.baseAddress! + reBase + i, Int32(frames))
                            cblas_scopy(Int32(dimF), outI, 1, sp.baseAddress! + imBase + i, Int32(frames))
                        }
                    }
                }
            }
        }
        return (spec, frames)
    }

    // MARK: - Inverse: spectrogram [4, dimF, frames] → stereo audio

    func inverse(_ spec: [Float], frames: Int) -> [[Float]] {
        let t = (frames - 1) * hop
        let paddedLen = (frames - 1) * hop + nFFT
        var out = [[Float]](repeating: [Float](repeating: 0, count: t), count: 2)
        var fullR = [Float](repeating: 0, count: nFFT)
        var fullI = [Float](repeating: 0, count: nFFT)
        var timeR = [Float](repeating: 0, count: nFFT)
        var timeI = [Float](repeating: 0, count: nFFT)
        var tmp = [Float](repeating: 0, count: nFFT)

        spec.withUnsafeBufferPointer { sp in
            for ch in 0..<2 {
                let reBase = (2 * ch) * dimF * frames
                let imBase = (2 * ch + 1) * dimF * frames
                var recon = [Float](repeating: 0, count: paddedLen)
                var wenv = [Float](repeating: 0, count: paddedLen)
                recon.withUnsafeMutableBufferPointer { rp in
                    wenv.withUnsafeMutableBufferPointer { wp in
                        for i in 0..<frames {
                            // gather the strided bins
                            cblas_scopy(Int32(dimF), sp.baseAddress! + reBase + i, Int32(frames), &fullR, 1)
                            cblas_scopy(Int32(dimF), sp.baseAddress! + imBase + i, Int32(frames), &fullI, 1)
                            fullR[dimF] = 0; fullI[dimF] = 0                      // Nyquist (dropped)
                            for k in (dimF + 1)..<nFFT { fullR[k] = fullR[nFFT - k]; fullI[k] = -fullI[nFFT - k] }
                            vDSP_DFT_Execute(inv, fullR, fullI, &timeR, &timeI)
                            // recon[off..] += timeR · (window/nFFT);  wenv[off..] += window²
                            vDSP_vmul(timeR, 1, scaledWindow, 1, &tmp, 1, vDSP_Length(nFFT))
                            let off = i * hop
                            vDSP_vadd(rp.baseAddress! + off, 1, tmp, 1, rp.baseAddress! + off, 1, vDSP_Length(nFFT))
                            vDSP_vadd(wp.baseAddress! + off, 1, windowSq, 1, wp.baseAddress! + off, 1, vDSP_Length(nFFT))
                        }
                    }
                }
                for n in 0..<t {
                    let e = wenv[pad + n]
                    out[ch][n] = e > 1e-8 ? recon[pad + n] / e : 0
                }
            }
        }
        return out
    }

    // MARK: - Reflect padding (torch `center=True` default), pad = n_fft/2

    private func reflectPadded(_ x: [Float]) -> [Float] {
        let n = x.count
        var out = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<pad { out[i] = x[pad - i] }
        for i in 0..<n { out[pad + i] = x[i] }
        for j in 0..<pad { out[pad + n + j] = x[n - 2 - j] }
        return out
    }
}
